"""Real role inference, grounded supervision, and warm-started adapter fitting."""

from collections import Counter
import gc
import json
import math
from pathlib import Path

import numpy as np

from haireder.agents import (
    parse_proposer_output,
    parse_critic_output,
    parse_judge_output,
    render_proposer_prompt,
    render_critic_prompt,
    render_judge_prompt,
)
from haireder.agents.judge import compact_transcript_for_judge
from haireder.confidence.signals import compute_inter_agent_agreement_signal
from haireder.data.formal import file_hash
from haireder.data.pair_construction import sentence_encoder
from haireder.debate.engine import run_arguments
from haireder.online_learning.retrieval import (
    format_examples_for_role,
    _apply_role_ranking,
)
from haireder.training.context import fit_prompt
from haireder.training.lora_trainer import (
    train_role_adapter,
    load_multi_role_bundle,
    select_role_bundle,
    generate_role_output,
)
from haireder.utils.config import resolve_path
from haireder.utils.io import write_json, write_jsonl
from haireder.utils.llm_client import create_llm_client
from .augmentation import presented
from .review import fingerprint


PARSERS = {
    "proposer": parse_proposer_output,
    "critic": parse_critic_output,
    "judge": parse_judge_output,
}


def verify_quotes(payload, row):
    for key, side in (("quote_a", "issue_i"), ("quote_b", "issue_j")):
        quote = payload.get(key)
        if (
            not isinstance(quote, str)
            or not any(c.isalnum() for c in quote)
            or quote not in row[side]["text"]
        ):
            raise ValueError(f"{key} must be an exact text excerpt from {side}")


def agreed_label(transcript):
    if transcript.get("termination_reason") not in {"agreement", "critic_concede"}:
        return None
    last = transcript["rounds"][-1]
    label = last["proposer"]["parsed_output"]["label"]
    if last["critic"]["parsed_output"]["label"] != label:
        raise ValueError("Agreement requires the same label from both agents")
    return label


class ExampleIndex:
    def __init__(self, pool, model_name, k=5):
        self.pool = list(pool)
        self.k = k
        self.encoder = sentence_encoder(model_name)
        self.matrix = (
            self.encoder.encode(
                [self.text(r) for r in pool],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if pool
            else None
        )

    @staticmethod
    def text(row):
        # Give each endpoint representation space in the sentence encoder.
        return " [SEP] ".join(row[k]["text"][:450] for k in ("issue_i", "issue_j"))

    def query(self, row):
        if not self.pool:
            return [], 0.0, []
        vector = self.encoder.encode(
            self.text(row), normalize_embeddings=True, show_progress_bar=False
        )
        ordered = sorted(
            (
                (record, float(sim))
                for record, sim in zip(self.pool, self.matrix @ vector)
                if record["pair_id"] != row["pair_id"]
            ),
            key=lambda x: (-x[1], x[0]["pair_id"]),
        )
        nearest = ordered[: self.k]
        return (
            ordered,
            max(0.0, min(1.0, float(np.mean([s for _, s in nearest]))))
            if nearest
            else 0.0,
            [
                {
                    "pair_id": r["pair_id"],
                    "window": r["window"],
                    "source": r.get("review_source"),
                    "cosine": s,
                }
                for r, s in nearest
            ],
        )


class RoleSystem:
    def __init__(self, adapters, model_config, config, *, single=False):
        self.bundle = load_multi_role_bundle(adapters, model_config=model_config)
        self.templates = {
            r: resolve_path(f"prompts/agents/{r}_prompt.md").read_text()
            for r in PARSERS
        }
        self.verbal = config["confidence"]["verbal_confidence_map"]
        self.weighted = config["confidence"].get("s2_round_weighted", False)
        self.rounds = config["debate"]["max_rounds"]
        self.single = single

    def prompt(self, role, row, argument=None, examples=""):
        if role == "critic":
            return render_critic_prompt(
                row,
                json.dumps(argument, ensure_ascii=False),
                self.templates[role],
                retrieval_examples=examples,
            )
        if role == "judge":
            return render_judge_prompt(
                row,
                compact_transcript_for_judge(argument),
                self.templates[role],
                retrieval_examples=examples,
            )
        prompt = render_proposer_prompt(
            row, self.templates[role], retrieval_examples=examples
        )
        if argument:
            prompt += (
                "\nRespond to the previous Critic argument:\n"
                + json.dumps(argument, ensure_ascii=False)
                + "\nJSON Output:\n"
            )
        return prompt

    def turn(self, role, row, argument=None, ranked=()):
        bundle = select_role_bundle(self.bundle, role)
        selected = [
            presented(r, 240) for r, _ in _apply_role_ranking(list(ranked), role)[:3]
        ]
        while True:
            prompt = self.prompt(
                role, row, argument, format_examples_for_role(selected, role)
            )
            try:
                prompt = fit_prompt(
                    prompt, bundle["tokenizer"], bundle["max_seq_length"] - 256
                )
                break
            except ValueError:
                if not selected:
                    raise
                selected.pop()
        errors = []
        for _ in range(3):
            prompt = fit_prompt(
                prompt, bundle["tokenizer"], bundle["max_seq_length"] - 224
            )
            raw = generate_role_output(bundle, prompt, max_new_tokens=224)
            parsed = PARSERS[role](raw, self.verbal)
            if parsed["error"] is None:
                try:
                    verify_quotes(parsed["parsed_output"], row)
                    if role == "judge":
                        agreed = agreed_label(argument)
                        if agreed is not None and parsed["predicted_label"] != agreed:
                            raise ValueError(
                                f"Judge must formalise the agreed label {agreed}"
                            )
                except ValueError as exc:
                    parsed["error"] = str(exc)
            if parsed["error"] is None:
                return dict(
                    parsed,
                    retrieved_pair_ids=[r["pair_id"] for r in selected],
                    prompt_hash=fingerprint(prompt),
                    invalid_generation_attempts=errors,
                )
            errors.append(
                dict(
                    error=parsed["error"],
                    raw_output=raw,
                    prompt_hash=fingerprint(prompt),
                )
            )
            prompt += (
                "\nCorrect this output error and return the complete JSON object: "
                + parsed["error"]
                + "\n"
            )
        raise ValueError(
            f"{row['pair_id']} {role}: invalid model output after three attempts: "
            + json.dumps(errors, ensure_ascii=False)
        )

    def arguments(self, row, ranked=()):
        def propose(r, i, t):
            previous = t["rounds"][-1]["critic"]["parsed_output"] if i else None
            return self.turn("proposer", r, previous, ranked)

        def critique(r, p, i, t):
            return self.turn("critic", r, p["parsed_output"], ranked)

        return run_arguments(row, propose, critique, self.rounds)

    def infer(self, row, index):
        # Only the evaluation layer joins historical labels to predictions.
        input_row = {
            k: row[k]
            for k in ("pair_id", "repo", "window", "split", "issue_i", "issue_j")
        }
        ranked, s3, neighbors = (
            index.query(input_row) if index is not None else ([], 0.0, [])
        )
        if self.single:
            decision = self.turn("proposer", input_row, ranked=ranked)
            transcript = {
                k: input_row[k] for k in ("pair_id", "repo", "window", "split")
            }
            transcript.update(proposer=decision, mode="single_proposer")
            s2 = 0.0
        else:
            transcript = self.arguments(input_row, ranked)
            decision = self.turn("judge", input_row, transcript, ranked)
            transcript["judge"] = decision
            s2, _ = compute_inter_agent_agreement_signal(
                transcript, round_weighted=self.weighted
            )
        return dict(
            input_row,
            predicted_label=decision["predicted_label"],
            rationale=decision["parsed_output"].get("justification")
            or decision["parsed_output"].get("rationale"),
            confidence=decision["confidence"],
            s1=decision["confidence"],
            s2=s2,
            s3=s3,
            s3_neighbors=neighbors,
            transcript=transcript,
            method="proposer_only" if self.single else "three_role_debate",
        )


def excerpt_options(text, original, prefix):
    spans = list(
        dict.fromkeys(
            fragment
            for line in text.replace(" […] ", "\n").splitlines()
            if (fragment := line.strip()[:144].strip())
            and any(c.isalnum() for c in fragment)
            and fragment in original
        )
    )
    selected = spans if len(spans) <= 4 else spans[:2] + spans[-2:]
    return {f"{prefix}{i}": span for i, span in enumerate(selected, 1)}


def grounded_target(teacher, row, role, argument=None):
    text = presented(row, 1500)
    instruction = (
        "Explain the supplied training label using evidence from each issue. Choose an exact short substring from each given "
        "issue for quote_a and quote_b. Return JSON with quote_a, quote_b, explanation, verbal_confidence "
        "(very_low, low, medium, high, very_high). Discuss how the evidence supports the relation."
        " For the Critic evaluate the candidate argument. For the Judge compare the actual arguments. "
    )
    source = {
        "role": role,
        "gold_training_label": row["label"],
        "issue_a": text["issue_i"]["text"],
        "issue_b": text["issue_j"]["text"],
    }
    if argument:
        source["arguments"] = argument
    if row.get("human_reviewed") and (row.get("review_detail") or {}).get("rationale"):
        source["analyst_rationale"] = row["review_detail"]["rationale"]
    prompt = instruction + json.dumps(source, ensure_ascii=False)
    invalid_attempts = []
    options = None
    for _ in range(3):
        raw = teacher.generate(prompt, max_new_tokens=256, temperature=0.3)
        try:
            obj = json.JSONDecoder().raw_decode(raw[raw.index("{") :])[0]
            selection = None
            if options is None:
                qa, qb = obj["quote_a"], obj["quote_b"]
                if not isinstance(qa, str) or qa not in source["issue_a"]:
                    raise ValueError("quote_a must be copied from issue_a")
                if not isinstance(qb, str) or qb not in source["issue_b"]:
                    raise ValueError("quote_b must be copied from issue_b")
            else:
                selection = {
                    field: obj[field] for field in ("quote_a_id", "quote_b_id")
                }
                qa = options["options_a"][selection["quote_a_id"]]
                qb = options["options_b"][selection["quote_b_id"]]
            reason, verbal = obj["explanation"], obj["verbal_confidence"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("empty or non-text explanation")
            if verbal not in ("very_low", "low", "medium", "high", "very_high"):
                raise ValueError("invalid confidence")
            rationale = f"“{qa}” and “{qb}”. {reason}"
            target = dict(verbal_confidence=verbal, quote_a=qa, quote_b=qb)
            verify_quotes(target, row)
            if role == "judge":
                target.update(final_label=row["label"], justification=rationale)
            else:
                target.update(label=row["label"], rationale=rationale)
                if role == "critic":
                    target["stance"] = (
                        "endorse" if argument["label"] == row["label"] else "challenge"
                    )
            return target, dict(
                teacher=teacher.model_name,
                quote_a=qa,
                quote_b=qb,
                raw_teacher_output=raw,
                prompt_hash=fingerprint(prompt),
                excerpt_selection=selection,
                excerpt_options=options,
                invalid_generation_attempts=invalid_attempts,
            )
        except (TypeError, ValueError, KeyError) as exc:
            invalid_attempts.append(
                dict(error=str(exc), raw_output=raw, prompt_hash=fingerprint(prompt))
            )
            if options is None:
                options = {
                    "options_a": excerpt_options(
                        source["issue_a"], row["issue_i"]["text"], "A"
                    ),
                    "options_b": excerpt_options(
                        source["issue_b"], row["issue_j"]["text"], "B"
                    ),
                }
            prompt = (
                "Explain the supplied training label using an excerpt from each issue. "
                "Select one exact ID from options_a as quote_a_id, and one exact ID from options_b "
                "as quote_b_id. In explanation, connect the selected excerpts to the label. "
                "If the texts give limited support, state the limitation in the explanation. "
                "For the Critic evaluate the candidate argument, and for the Judge compare the arguments. "
                "Return a JSON object with quote_a_id, quote_b_id, explanation, and verbal_confidence "
                "(very_low, low, medium, high, very_high). Prior output error was "
                + str(exc)
                + ".\n"
                + json.dumps(dict(source, **options), ensure_ascii=False)
            )
    raise ValueError(
        f"Unable to ground {role} supervision for {row['pair_id']}: "
        + json.dumps(invalid_attempts, ensure_ascii=False)
    )


def train_roles(rows, directory, model_cfg, cfg, *, parents=None, single=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Role training requires unique labelled pairs")
    llm = cfg["training"]["augmentation"]["llm"]
    templates = {
        r: resolve_path(f"prompts/agents/{r}_prompt.md").read_text() for r in PARSERS
    }
    adapters = {}
    for role in ["proposer"] if single else list(PARSERS):
        arguments = {}
        evidence = {}
        if role != "proposer":
            system = RoleSystem(adapters, model_cfg, cfg)
            for r in rows:
                prior = r.get("transcript") or {}
                prior_rounds = (
                    prior.get("rounds") if not r.get("is_augmented") else None
                )
                if role == "critic":
                    if prior_rounds:
                        t = dict(prior_rounds[0]["proposer"])
                        t["predicted_label"] = t["parsed_output"]["label"]
                        t["confidence"] = system.verbal[
                            t["parsed_output"]["verbal_confidence"]
                        ]
                    else:
                        t = system.turn("proposer", r)
                    evidence[r["pair_id"]] = t
                    # Keep all errors/uncertain cases and a reproducible subset of confident endorsements.
                    if (
                        t["predicted_label"] != r["label"]
                        or t["confidence"] <= 0.5
                        or r.get("review_source") == "low_review"
                        or any(
                            turn["critic"]["parsed_output"]["stance"] == "challenge"
                            for turn in prior_rounds or []
                        )
                        or int(fingerprint(r["pair_id"])[:8], 16) % 5 == 0
                    ):
                        arguments[r["pair_id"]] = t["parsed_output"]
                else:
                    t = prior if prior_rounds else system.arguments(r)
                    evidence[r["pair_id"]] = t
                    if agreed_label(t) in (None, r["label"]):
                        arguments[r["pair_id"]] = t
            write_jsonl(
                directory / f"{role}_observations.jsonl",
                [
                    dict(
                        pair_id=k,
                        observation=v,
                        selected_for_supervision=k in arguments,
                    )
                    for k, v in evidence.items()
                ],
            )
            del system
            gc.collect()
        teacher = create_llm_client(
            backend=llm["backend"], model_name=llm["validator_model_name"]
        )
        examples = []
        for r in rows:
            if role != "proposer" and r["pair_id"] not in arguments:
                continue
            arg = arguments.get(r["pair_id"])
            if role == "proposer":
                prompt = render_proposer_prompt(r, templates[role])
            elif role == "critic":
                prompt = render_critic_prompt(
                    r, json.dumps(arg, ensure_ascii=False), templates[role]
                )
            else:
                prompt = render_judge_prompt(
                    r, compact_transcript_for_judge(arg), templates[role]
                )
            target, proof = grounded_target(
                teacher,
                r,
                role,
                compact_transcript_for_judge(arg) if role == "judge" else arg,
            )
            examples.append(
                dict(
                    pair_id=r["pair_id"],
                    window=r["window"],
                    gold_label=r["label"],
                    prompt_text=prompt,
                    target_text=json.dumps(target, ensure_ascii=False),
                    supervision=proof,
                )
            )
        del teacher
        from haireder.utils.llm_client import _MODEL_CACHE

        _MODEL_CACHE.clear()
        gc.collect()
        if not examples:
            raise ValueError(f"No authentic training examples for {role}")
        write_jsonl(directory / f"{role}_examples.jsonl", examples)
        path = directory / role
        meta = train_role_adapter(
            examples,
            model_cfg,
            path,
            max_steps=model_cfg["training"].get("max_steps"),
            parent_adapter=parents[role] if parents else None,
            resume_from_checkpoint=None,
        )
        loss = meta.get("train_metrics", {}).get("train_loss")
        weights = path / "adapter_model.safetensors"
        if (
            (meta.get("optimizer_steps") or 0) <= 0
            or not isinstance(loss, (float, int))
            or not math.isfinite(loss)
            or not (path / "adapter_config.json").is_file()
            or not weights.is_file()
            or file_hash(weights) != meta.get("adapter_sha256")
        ):
            raise ValueError(f"No verified trained adapter for {role} in {path}")
        adapters[role] = path
        write_json(
            directory / f"{role}_training.json",
            dict(
                meta,
                labels=dict(Counter(r["gold_label"] for r in examples)),
                examples_hash=fingerprint(examples),
            ),
        )
    return adapters
