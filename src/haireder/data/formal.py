"""Construct the paper cohort on the fixed UTC calendar."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import tempfile

from .pair_construction import (
    mine_hard_negatives_for_window,
    stratified_sample_positive,
)
from .windowing import (
    build_positive_pairs,
    build_fixed_sprints,
    parse_jira_datetime,
    requirement_text,
)
from haireder.taxonomy import LABELS
from haireder.utils.io import write_json, write_jsonl


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formal(
    config, taxonomy, root, output, config_path, taxonomy_path, backend=None,
    *, shared_test_dir=None,
):
    cfg = config["formal_run"]
    repos = config["target_repositories"]
    backend = backend or config["pair_construction"]["negative_mining_backend"]
    model = config["pair_construction"]["sbert_model_name"]
    canonical_dir = Path(shared_test_dir) if shared_test_dir is not None else root / "data/processed/splits"
    shared_test = canonical_dir / "formal_test.jsonl"
    if backend == "random" and (
        not shared_test.exists() or output.resolve() == shared_test.parent.resolve()
    ):
        raise ValueError(
            "Random training negatives require a separate directory and a frozen canonical test split"
        )

    def portable(path):
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return str(path)

    source = root / config["raw_sources"]["processed_pairs_json"]
    history = root / config["raw_sources"]["creation_history"]
    if not history.exists():
        raise FileNotFoundError(f"Build creation-time issue histories first: {history}")
    if backend == "random":
        canonical = json.loads((canonical_dir / "formal_manifest.json").read_text())
        if (
            canonical["artifacts"].get(shared_test.name) != file_hash(shared_test)
            or any(
                canonical["inputs"].get(portable(path)) != file_hash(path)
                for path in (source, history, config_path, taxonomy_path)
            )
        ):
            raise ValueError("Unverified shared test cohort")
    with source.open() as handle:
        payload = json.load(handle)
    requirements = {
        (r["source"], r["issue_key"]): r for r in payload.pop("requirements")
    }
    eligible = defaultdict(list)
    recorded = defaultdict(set)
    seen = {}
    duplicates = 0
    with history.open() as handle:
        for line in handle:
            snap = json.loads(line)
            repo, key = snap["source"], snap["issue_key"]
            if repo not in repos:
                continue
            recorded[repo].update((key, target) for target in snap["recorded_targets"])
            ident = (repo, key)
            fingerprint = sha256(
                json.dumps(
                    [snap["created"], snap["creation_text"]], sort_keys=True
                ).encode()
            ).digest()
            if ident in seen:
                if seen[ident] != fingerprint:
                    raise ValueError(f"Conflicting raw issue versions {ident}")
                duplicates += 1
                continue
            seen[ident] = fingerprint
            if ident in requirements and snap["creation_text"]:
                row = requirements[ident]
                if parse_jira_datetime(row["created"]) != parse_jira_datetime(
                    snap["created"]
                ):
                    raise ValueError(f"Creation date mismatch for {ident}")
                eligible[repo].append(
                    dict(
                        source=repo,
                        issue_key=key,
                        project_key=row["project_key"],
                        created=snap["created"],
                        creation_text=snap["creation_text"],
                    )
                )
    del requirements, seen
    verified_edges = defaultdict(list)
    known = {repo: set(recorded[repo]) for repo in repos}
    for edge in payload.pop("dependency_pairs"):
        repo = edge["source_source"]
        if repo not in known or repo != edge["target_source"]:
            continue
        pair = (edge["source_issue_key"], edge["target_issue_key"])
        known[repo].add(pair)
        if pair in recorded[repo] or pair[::-1] in recorded[repo]:
            verified_edges[repo].append(edge)
    del payload, recorded
    _, calendar = build_fixed_sprints(
        [], cfg["start_date"], cfg["end_date"], cfg["sprint_days"], cfg
    )
    declared = sum(
        (cfg[f"{role}_windows"] for role in ("train", "validation", "online", "test")),
        [],
    )
    if sorted(declared) != sorted(w.window for w in calendar):
        raise ValueError("Every calendar window must have exactly one split")
    positives = {}
    metadata = {}
    for repo in repos:
        rows, metadata[repo] = build_positive_pairs(
            dict(requirements=eligible[repo], dependency_pairs=verified_edges[repo]),
            repo,
            taxonomy,
        )
        positives[repo], _ = build_fixed_sprints(
            rows, cfg["start_date"], cfg["end_date"], cfg["sprint_days"], cfg
        )
    by_window = defaultdict(list)
    for rows in positives.values():
        for row in rows:
            by_window[row["window"]].append(row)
    cap = cfg.get("max_pairs_per_window")
    ratio = cfg["negative_ratio"]
    if ratio != 1:
        raise ValueError("Paper cohort requires one negative for each positive")
    rng = random.Random(cfg["seed"])
    selected = defaultdict(list)
    for window, rows in by_window.items():
        for row in stratified_sample_positive(rows, cap // 2, rng) if cap else rows:
            selected[row["repo"], window].append(row)
    result = []
    for repo in repos:
        issues = []
        for r in eligible[repo]:
            issues.append(
                (
                    parse_jira_datetime(r["created"]),
                    dict(
                        id=r["issue_key"],
                        created=r["created"],
                        project=r["project_key"],
                        text=requirement_text(r),
                    ),
                )
            )
        issues.sort(key=lambda item: (item[0], item[1]["id"]))
        times = [x[0] for x in issues]
        from bisect import bisect_left

        for w in calendar:
            if backend == "random" and w.role == "test":
                continue
            pos = selected[repo, w.window]
            if not pos:
                continue
            lo = bisect_left(times, parse_jira_datetime(w.start_at))
            hi = bisect_left(times, parse_jira_datetime(w.end_at))
            current = [v for _, v in issues[lo:hi]]
            limit = cfg.get("max_negative_candidate_issues_per_window")
            if limit and len(current) > limit:
                current = rng.sample(current, limit)
            pool = {v["id"]: v for v in current}
            anchors = set(pool)
            args = (w.window, w.role, pos, pool, known[repo], len(pos), rng)

            def mine(method):
                return mine_hard_negatives_for_window(
                    *args, backend=method, sbert_model_name=model, anchor_ids=anchors
                )

            actual = "sbert" if backend == "random" and w.role == "test" else backend
            # The random-negative ablation reuses the published held-out file.
            negatives = mine(actual)
            if len(negatives) < len(pos):
                for _, issue in reversed(issues[:lo]):
                    pool[issue["id"]] = issue
                    if len(pool) >= len(anchors) + max(128, 2 * len(pos)):
                        break
                negatives = mine(actual)
            if len(negatives) != len(pos):
                raise ValueError(
                    f"{repo} {w.window}: only {len(negatives)} unlinked negatives for {len(pos)} positives"
                )
            for negative in negatives:
                if any(negative[side]["id"] not in anchors for side in ("issue_i","issue_j")):
                    negative["negative_mining"]["candidate_source"] = "current_and_prior_repository_requirements"
            result.extend(pos + negatives)
        print(f"{repo}: {len(positives[repo])} eligible directed positives", flush=True)
    if backend == "random":
        with shared_test.open() as handle:
            result.extend(json.loads(line) for line in handle)
    result.sort(key=lambda row: (row["pair_available_at"], row["pair_id"]))
    if len(result) != len({r["pair_id"] for r in result}):
        raise ValueError("Duplicate entity identity")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        mode="formal",
        identical_raw_duplicates=duplicates,
        data_version=config["data_version"],
        dataset_config=portable(config_path),
        processed_pairs_json=portable(source),
        negative_mining_backend=backend,
        sbert_model=model,
        repositories=repos,
        window_count=len(calendar),
        window_strategy="fixed_sprints",
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        sprint_days=cfg["sprint_days"],
        split_config=cfg,
        seed=cfg["seed"],
        repository_reports=metadata,
        total_positive_pairs=len(result) // 2,
        total_negative_pairs=len(result) // 2,
        total_pairs=len(result),
        split_counts={},
        class_counts={},
        artifacts={},
        inputs={
            portable(p): file_hash(p)
            for p in (
                source,
                history,
                config_path,
                taxonomy_path,
                root / config["raw_sources"]["public_jira_zip"],
            )
        },
        windows=[],
    )
    if backend == "sbert":
        from .pair_construction import sentence_encoder

        backbone = sentence_encoder(model)[0].auto_model
        manifest["sbert_revision"] = getattr(backbone.config, "_commit_hash", None)
    for w in calendar:
        counts = Counter(r["label"] for r in result if r["window"] == w.window)
        manifest["windows"].append(
            dict(asdict(w), count=sum(counts.values()), label_counts=dict(counts))
        )
    with tempfile.TemporaryDirectory(dir=output) as directory:
        tmp = Path(directory)
        write_jsonl(tmp / "formal_pairs.jsonl", result)
        for split in ("train", "validation", "online", "test"):
            rows = [r for r in result if r["split"] == split]
            write_jsonl(tmp / f"formal_{split}.jsonl", rows)
            manifest["split_counts"][split] = len(rows)
            counts = Counter(r["label"] for r in rows)
            manifest["class_counts"][split] = {label: counts[label] for label in LABELS}
        for path in sorted(tmp.glob("*.jsonl")):
            manifest["artifacts"][path.name] = file_hash(path)
            path.replace(output / path.name)
        write_json(tmp / "formal_manifest.json", manifest).replace(
            output / "formal_manifest.json"
        )
    print(json.dumps(manifest["split_counts"]), flush=True)
    return manifest
