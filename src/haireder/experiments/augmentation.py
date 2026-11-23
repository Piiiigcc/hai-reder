"""Persistent, validated pair generation shared by initial and refresh training."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from haireder.augmentation.llm_wiring import (
    build_llm_components,
    resolve_augmentation_config,
)
from haireder.taxonomy import LABELS, load_taxonomy
from haireder.utils.config import resolve_path
from haireder.utils.io import read_jsonl, write_json, write_jsonl
from .review import fingerprint, review_queue


def generate_many(client, prompts, limit, temperature):
    if hasattr(client, "generate_many"):
        return client.generate_many(
            prompts, max_new_tokens=limit, temperature=temperature
        )
    return [
        client.generate(p, max_new_tokens=limit, temperature=temperature)
        for p in prompts
    ]


def presented(row, limit):
    out = dict(row)
    for key in ("issue_i", "issue_j"):
        txt = row[key]["text"]
        if len(txt) > limit:
            txt = txt[: limit * 3 // 4] + " […] " + txt[-limit // 4 :]
        out[key] = dict(row[key], text=txt)
    return out


def generate_pairs(
    rows,
    directory,
    config,
    *,
    sample_goal=None,
    proposal_attempts=None,
    batch_size=4,
    seed=42,
    human_sample=0,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = resolve_augmentation_config(config)
    llm = cfg["llm"]
    prefix = "" if proposal_attempts else "full_"
    gen_prompt = resolve_path(
        cfg.get(prefix + "prompt_template", cfg["prompt_template"])
    ).read_text()
    val_prompt = resolve_path(
        cfg.get(
            prefix + "validation_prompt_template", cfg["validation_prompt_template"]
        )
    ).read_text()
    identity = fingerprint(
        [
            rows,
            llm,
            gen_prompt,
            val_prompt,
            seed,
            batch_size,
            sample_goal,
            proposal_attempts,
        ]
    )
    identity_file = directory / "generation.json"
    if (
        identity_file.exists()
        and json.loads(identity_file.read_text())["fingerprint"] != identity
    ):
        raise ValueError(
            "Generation inputs have changed; select a new output directory"
        )
    write_json(
        identity_file,
        {
            "fingerprint": identity,
            "models": [llm["model_name"], llm["validator_model_name"]],
            "sample_goal": sample_goal,
            "proposal_attempts": proposal_attempts,
            "seed": seed,
            "batch_size": batch_size,
        },
    )
    by_label = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["pair_id"]):
        by_label[row["label"]].append(row)
    before = Counter(r["label"] for r in rows)
    if not before:
        raise ValueError("Empty augmentation corpus")
    target = max(before.values())
    deficits = {k: target - before[k] for k in LABELS}
    if not sample_goal and any(deficits[k] and not before[k] for k in LABELS):
        raise ValueError(
            "Missing a class seed; pair-preserving generation cannot introduce an absent class"
        )
    labels = [k for k in LABELS if deficits[k] and before[k]]
    rng = random.Random(seed)
    for k in labels:
        rng.shuffle(by_label[k])
    index = {r["pair_id"]: r for r in rows}
    refs = [presented(r, 190) for r in rows]
    eligible_texts = {
        fingerprint([r[k]["text"] for k in ("issue_i", "issue_j")]) for r in rows
    }
    counts = Counter()
    attempts = Counter()
    accepted = []
    proposals = []

    def record(entries):
        for entry in entries:
            attempts[entry["label"]] += 1
            if entry.get("proposal"):
                proposals.append(entry["proposal"])
            row = entry.get("qualified")
            if row is not None:
                accepted.append(row)
                counts[row["label"]] += 1
                eligible_texts.add(
                    fingerprint([row[k]["text"] for k in ("issue_i", "issue_j")])
                )

    from haireder.data.formal import file_hash
    from haireder.utils.model_versions import revision_kwargs

    write_json(
        directory / "model_revisions.json",
        {
            name: revision_kwargs(name)
            for name in (
                llm["model_name"],
                llm["validator_model_name"],
                llm["sbert_model"],
            )
        },
    )
    taxonomy = load_taxonomy()
    protocol = dict(
        target_definitions={k: taxonomy.label_definitions[k] for k in labels},
        model_revisions={
            name: revision_kwargs(name)
            for name in (
                llm["model_name"],
                llm["validator_model_name"],
                llm["sbert_model"],
            )
        },
        output_tokens=150 if proposal_attempts else int(llm.get("max_new_tokens", 512)),
        validation_output_tokens=80 if proposal_attempts else 256,
    )
    protocol_path = directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(
            "Generation models, class definitions or token budget have changed"
        )
    write_json(protocol_path, protocol)
    ledger_path = directory / "batches.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    batches = [directory / name for name in sorted(ledger)]
    for i, path in enumerate(batches):
        if path.name != f"batch_{i:06}.jsonl":
            raise ValueError("Interrupted generation ledger")
        if ledger[path.name] != file_hash(path):
            raise ValueError(f"Altered generation batch {path.name}")
        record(read_jsonl(path))
    goal = sample_goal or sum(deficits.values())
    # The pilot has a fixed sampling frame. A resumed training run gets another
    # bounded block of attempts, retaining every prior decision and unmet quota.
    limit_attempts = proposal_attempts or (
        sum(attempts.values()) + max(32, (goal - len(accepted)) * 8)
    )
    components = None
    batch_number = len(batches)
    while (proposal_attempts or len(accepted) < goal) and sum(
        attempts.values()
    ) < limit_attempts:
        if components is None:
            components = build_llm_components(llm, gen_prompt, val_prompt, taxonomy)
        gen, val, emb = components
        jobs = []
        scheduled = Counter()
        for _ in range(
            min(
                batch_size,
                limit_attempts - sum(attempts.values()),
                goal if proposal_attempts else goal - len(accepted),
            )
        ):
            options = [k for k in labels if counts[k] + scheduled[k] < deficits[k]]
            if not options:
                break
            k = min(
                options,
                key=lambda label: (
                    (attempts[label] + scheduled[label])
                    / (
                        deficits[label] if not (sample_goal or proposal_attempts) else 1
                    ),
                    label,
                ),
            )
            n = attempts[k] + scheduled[k]
            source = by_label[k][n % len(by_label[k])]
            variant = n // len(by_label[k]) + 1
            jobs.append((source, variant))
            scheduled[k] += 1
        if not jobs:
            break
        if llm.get("backend") == "mlx":
            import mlx.core as mx

            mx.random.seed(seed + batch_number)
        else:
            from transformers import set_seed

            set_seed(seed + batch_number)
        prompts = [
            gen._build_prompt(presented(r, 1800), r["label"], v, refs) for r, v in jobs
        ]
        outputs = generate_many(
            gen.llm_client,
            prompts,
            protocol["output_tokens"],
            float(llm.get("temperature", 0.7)),
        )
        entries = []
        candidates = []
        for (r, v), raw in zip(jobs, outputs, strict=True):
            parsed = gen._parse_llm_output(raw)
            e = dict(seed=r["pair_id"], label=r["label"], variant=v, raw_generation=raw)
            entries.append(e)
            if parsed and parsed.get("target_label", r["label"]) == r["label"]:
                new = gen._build_augmented_row(r, parsed, r["label"], v)
                if all(
                    isinstance(parsed.get("augmented_" + k + "_text"), str)
                    and new[k]["text"] != r[k]["text"]
                    for k in ("issue_i", "issue_j")
                ):
                    e["proposal"] = dict(new)
                    candidates.append((e, r, new))
        checks = [
            val._build_prompt(presented(r, 2200), c, r["label"]).replace(
                "{{target_label_definition}}", taxonomy.label_definitions[r["label"]]
            )
            for _, r, c in candidates
        ]
        decisions = (
            generate_many(
                val.llm_client,
                checks,
                protocol["validation_output_tokens"],
                float(llm.get("validation_temperature", 0.3)),
            )
            if checks
            else []
        )
        for (entry, r, c), raw in zip(candidates, decisions, strict=True):
            result = val._parse_llm_output(raw)
            entry["raw_validation"] = raw
            if result["is_valid"] and result["label_matches"]:
                filt = emb.filter(r, c)
                entry["embedding_filter"] = filt
                digest = fingerprint([c[k]["text"] for k in ("issue_i", "issue_j")])
                if filt["is_valid"] and digest not in eligible_texts:
                    c["augmentation"].update(
                        validator_model=llm["validator_model_name"],
                        batch_fingerprint=identity,
                        validation=dict(
                            is_valid=True,
                            label_matches=True,
                            llm_validation=result,
                            embedding_filter_result=filt,
                        ),
                    )
                    eligible_texts.add(digest)
                    entry["qualified"] = c
        path = write_jsonl(directory / f"batch_{batch_number:06}.jsonl", entries)
        ledger[path.name] = file_hash(path)
        write_json(ledger_path, ledger)
        record(entries)
        batch_number += 1
        print(
            f"Augmentation: {len(proposals)} well-formed, {len(accepted)} qualified, {sum(attempts.values())} attempts; "
            + (
                f"pilot frame {proposal_attempts} attempts"
                if proposal_attempts
                else f"training target {goal} qualified"
            ),
            flush=True,
        )
    if components is not None:
        from haireder.utils.llm_client import _MODEL_CACHE

        _MODEL_CACHE.clear()
    after = before + counts
    summary = dict(
        counts_before={k: before[k] for k in LABELS},
        counts_after={k: after[k] for k in LABELS},
        target_per_class=target,
        accepted=len(accepted),
        attempts=sum(attempts.values()),
        missing={k: deficits[k] - counts[k] for k in LABELS if deficits[k] > counts[k]},
        sample_goal=sample_goal,
        proposal_attempts=proposal_attempts,
        well_formed_generated=len(proposals),
    )
    write_json(directory / "balance.json", summary)
    write_jsonl(directory / "qualified.jsonl", accepted)
    write_jsonl(directory / "proposals.jsonl", proposals)
    if not proposal_attempts and len(accepted) < goal:
        raise RuntimeError(f"Augmentation shortfall: {summary}")
    if human_sample:
        review_queue(
            directory / "human_review",
            identity,
            "augmentation",
            proposals,
            seeds=index,
            sample_size=human_sample,
            seed=seed,
        )
    return accepted
