from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haireder.agents.critic import build_critic_target_payload, render_critic_prompt
from haireder.agents.judge import build_judge_target_payload, compact_transcript_for_judge, render_judge_prompt
from haireder.agents.proposer import build_proposer_target_payload, render_proposer_prompt
from haireder.taxonomy import LABELS


def load_prompt_template(path: str | Path) -> str:
    return Path(path).expanduser().resolve().read_text(encoding="utf-8")


def _bootstrap_wrong_label(gold_label: str) -> str:
    gold_index = LABELS.index(gold_label)
    return LABELS[(gold_index + 1) % len(LABELS)]


def select_stratified_training_examples(
    examples: list[dict[str, Any]],
    *,
    max_examples: int | None,
    key_fields: tuple[str, ...] = ("gold_label",),
) -> list[dict[str, Any]]:
    if max_examples is None or len(examples) <= max_examples:
        return list(examples)
    if max_examples <= 0:
        return []

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    key_order: list[tuple[Any, ...]] = []
    for example in examples:
        key = tuple(example.get(field) for field in key_fields)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append(example)

    selected: list[dict[str, Any]] = []
    while len(selected) < max_examples:
        added = False
        for key in key_order:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added = True
            if len(selected) == max_examples:
                break
        if not added:
            break
    return selected


def build_proposer_training_examples(
    rows: list[dict[str, Any]],
    prompt_template: str,
    allowed_splits: tuple[str, ...] = ("train",),
    rationales_by_pair_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("split") not in allowed_splits:
            continue
        rationale_override = None
        if rationales_by_pair_id is not None:
            rationale_override = rationales_by_pair_id.get(str(row["pair_id"]))
        prompt_text = render_proposer_prompt(row, prompt_template)
        payload = build_proposer_target_payload(row, rationale=rationale_override)
        provenance = "generated_rationale" if rationale_override else "bootstrap_rationale"
        payload["_provenance"] = provenance
        target_text = json.dumps(payload, ensure_ascii=False)
        examples.append(
            {
                "pair_id": row["pair_id"],
                "repo": row.get("repo"),
                "split": row.get("split"),
                "gold_label": row["label"],
                "rationale_provenance": provenance,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "full_text": f"{prompt_text}\n\n{target_text}",
            }
        )
    return examples


def build_critic_training_examples(
    rows: list[dict[str, Any]],
    prompt_template: str,
    allowed_splits: tuple[str, ...] = ("train",),
    rationales_by_pair_id: dict[str, str] | None = None,
    *,
    failure_rows: list[dict[str, Any]] | None = None,
    data_source: str = "bootstrap",
) -> list[dict[str, Any]]:
    """Build role-specific training examples for the Critic agent.

    Args:
        rows: Training data rows with gold labels.
        prompt_template: The critic prompt template.
        allowed_splits: Which data splits to include.
        rationales_by_pair_id: Optional override map of pair_id → rationale text.
        failure_rows: Real proposer failure rows (pair_id, predicted_label, gold_label, ...).
            When provided and data_source="real", these replace the bootstrap wrong-label
            simulation with authentic failure patterns.
        data_source: "bootstrap" for synthetic wrong labels, "real" for failure_rows.

    Returns:
        List of critic training examples.
    """
    examples = []

    if data_source == "real" and failure_rows:
        failure_index = 0
        for row in rows:
            if row.get("split") not in allowed_splits:
                continue
            gold_label = str(row["label"])
            pair_id = str(row["pair_id"])

            # Try to find a matching real failure for this pair
            matching_failure = None
            for frow in failure_rows:
                if str(frow.get("pair_id")) == pair_id:
                    matching_failure = frow
                    break

            if matching_failure is not None:
                proposer_label = str(matching_failure.get("predicted_label") or matching_failure.get("label") or gold_label)
                source = "real_failure"
            else:
                # Fallback: alternate endorse/challenge using bootstrap wrong label
                proposer_label = gold_label if failure_index % 2 == 0 else _bootstrap_wrong_label(gold_label)
                source = "bootstrap_failure"
            failure_index += 1

            rationale_override = None
            if rationales_by_pair_id is not None:
                rationale_override = rationales_by_pair_id.get(pair_id)
            proposer_payload = build_proposer_target_payload({**row, "label": proposer_label}, rationale=rationale_override)
            proposer_payload["_provenance"] = "generated_rationale" if rationale_override else source
            proposer_output_json = json.dumps(proposer_payload, ensure_ascii=False)
            prompt_text = render_critic_prompt(row, proposer_output_json, prompt_template)
            target_payload = build_critic_target_payload(row, proposer_label, rationale=rationale_override)
            target_payload["stance"] = "endorse" if proposer_label == gold_label else "challenge"
            provenance = "generated_rationale" if rationale_override else source
            target_payload["_provenance"] = provenance
            target_text = json.dumps(target_payload, ensure_ascii=False)
            examples.append(
                {
                    "pair_id": pair_id,
                    "repo": row.get("repo"),
                    "split": row.get("split"),
                    "gold_label": gold_label,
                    "critic_scenario": target_payload["stance"],
                    "rationale_provenance": provenance,
                    "failure_source": source,
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "full_text": f"{prompt_text}\n\n{target_text}",
                }
            )
    else:
        # Original bootstrap behavior (unchanged)
        for index, row in enumerate(rows):
            if row.get("split") not in allowed_splits:
                continue
            gold_label = str(row["label"])
            proposer_label = gold_label if index % 2 == 0 else _bootstrap_wrong_label(gold_label)
            rationale_override = None
            if rationales_by_pair_id is not None:
                rationale_override = rationales_by_pair_id.get(str(row["pair_id"]))
            proposer_payload = build_proposer_target_payload({**row, "label": proposer_label}, rationale=rationale_override)
            proposer_payload["_provenance"] = "generated_rationale" if rationale_override else "bootstrap_rationale"
            proposer_output_json = json.dumps(proposer_payload, ensure_ascii=False)
            prompt_text = render_critic_prompt(row, proposer_output_json, prompt_template)
            target_payload = build_critic_target_payload(row, proposer_label, rationale=rationale_override)
            target_payload["stance"] = "endorse" if proposer_label == gold_label else "challenge"
            provenance = "generated_rationale" if rationale_override else "bootstrap_rationale"
            target_payload["_provenance"] = provenance
            target_text = json.dumps(target_payload, ensure_ascii=False)
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "repo": row.get("repo"),
                    "split": row.get("split"),
                    "gold_label": gold_label,
                    "critic_scenario": target_payload["stance"],
                    "rationale_provenance": provenance,
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "full_text": f"{prompt_text}\n\n{target_text}",
                }
            )
    return examples


def build_judge_training_examples(
    rows: list[dict[str, Any]],
    prompt_template: str,
    allowed_splits: tuple[str, ...] = ("train",),
    rationales_by_pair_id: dict[str, str] | None = None,
    *,
    real_transcripts: list[dict[str, Any]] | None = None,
    data_source: str = "bootstrap",
) -> list[dict[str, Any]]:
    """Build role-specific training examples for the Judge agent.

    Args:
        rows: Training data rows with gold labels.
        prompt_template: The judge prompt template.
        allowed_splits: Which data splits to include.
        rationales_by_pair_id: Optional override map of pair_id → rationale text.
        real_transcripts: Real debate transcript rows (pair_id, proposer_parsed,
            critic_parsed, ...). When provided and data_source="real", these replace
            the synthetic single-round transcripts with authentic debate reasoning.
        data_source: "bootstrap" for synthetic transcripts, "real" for real_transcripts.

    Returns:
        List of judge training examples.
    """
    examples = []

    if data_source == "real" and real_transcripts:
        transcript_by_pair_id = {str(t["pair_id"]): t for t in real_transcripts}
        for row in rows:
            if row.get("split") not in allowed_splits:
                continue
            gold_label = str(row["label"])
            pair_id = str(row["pair_id"])

            if pair_id in transcript_by_pair_id:
                rt = transcript_by_pair_id[pair_id]
                proposer_parsed = rt["proposer_parsed"]
                critic_parsed = rt["critic_parsed"]
                example = build_judge_training_example_from_real_transcript(
                    row, proposer_parsed, critic_parsed, prompt_template,
                )
                example["transcript_source"] = "real_model_inference"
                examples.append(example)
            else:
                # Fallback: synthetic single-round transcript
                termination_reason = "round_budget"
                proposer_label = _bootstrap_wrong_label(gold_label)
                critic_label = gold_label
                critic_stance = "challenge"
                rationale_override = None
                if rationales_by_pair_id is not None:
                    rationale_override = rationales_by_pair_id.get(pair_id)
                provenance = "generated_rationale" if rationale_override else "bootstrap_rationale"
                proposer_target = build_proposer_target_payload({**row, "label": proposer_label}, rationale=rationale_override)
                critic_target = build_critic_target_payload(row, proposer_label, rationale=rationale_override)
                transcript = {
                    "pair_id": pair_id,
                    "repo": row["repo"],
                    "window": row["window"],
                    "split": row["split"],
                    "rounds": [
                        {
                            "round_index": 0,
                            "proposer": {
                                "raw_output": json.dumps(proposer_target, ensure_ascii=False),
                                "parsed_output": proposer_target,
                            },
                            "critic": {
                                "raw_output": json.dumps(critic_target, ensure_ascii=False),
                                "parsed_output": {
                                    "stance": critic_stance,
                                    "label": critic_label,
                                    "rationale": critic_target["rationale"],
                                    "verbal_confidence": critic_target["verbal_confidence"],
                                },
                            },
                        }
                    ],
                    "termination_reason": termination_reason,
                }
                transcript_json = compact_transcript_for_judge(transcript)
                prompt_text = render_judge_prompt(row, transcript_json, prompt_template)
                target_payload = build_judge_target_payload(row, termination_reason, justification=rationale_override)
                target_payload["_provenance"] = provenance
                target_text = json.dumps(target_payload, ensure_ascii=False)
                examples.append(
                    {
                        "pair_id": pair_id,
                        "repo": row.get("repo"),
                        "split": row.get("split"),
                        "gold_label": gold_label,
                        "termination_reason": termination_reason,
                        "rationale_provenance": provenance,
                        "debate_transcript_json": transcript_json,
                        "transcript_source": "bootstrap_synthetic",
                        "prompt_text": prompt_text,
                        "target_text": target_text,
                        "full_text": f"{prompt_text}\n\n{target_text}",
                    }
                )
    else:
        # Original bootstrap behavior (unchanged)
        for index, row in enumerate(rows):
            if row.get("split") not in allowed_splits:
                continue
            gold_label = str(row["label"])
            termination_reason = "agreement" if index % 2 == 0 else "round_budget"
            proposer_label = gold_label if termination_reason == "agreement" else _bootstrap_wrong_label(gold_label)
            critic_label = gold_label
            critic_stance = "endorse" if termination_reason == "agreement" else "challenge"
            rationale_override = None
            if rationales_by_pair_id is not None:
                rationale_override = rationales_by_pair_id.get(str(row["pair_id"]))
            provenance = "generated_rationale" if rationale_override else "bootstrap_rationale"
            proposer_target = build_proposer_target_payload({**row, "label": proposer_label}, rationale=rationale_override)
            critic_target = build_critic_target_payload(row, proposer_label, rationale=rationale_override)
            transcript = {
                "pair_id": row["pair_id"],
                "repo": row["repo"],
                "window": row["window"],
                "split": row["split"],
                "rounds": [
                    {
                        "round_index": 0,
                        "proposer": {
                            "raw_output": json.dumps(proposer_target, ensure_ascii=False),
                            "parsed_output": proposer_target,
                        },
                        "critic": {
                            "raw_output": json.dumps(critic_target, ensure_ascii=False),
                            "parsed_output": {
                                "stance": critic_stance,
                                "label": critic_label,
                                "rationale": critic_target["rationale"],
                                "verbal_confidence": critic_target["verbal_confidence"],
                            },
                        },
                    }
                ],
                "termination_reason": termination_reason,
            }
            transcript_json = compact_transcript_for_judge(transcript)
            prompt_text = render_judge_prompt(row, transcript_json, prompt_template)
            target_payload = build_judge_target_payload(row, termination_reason, justification=rationale_override)
            target_payload["_provenance"] = provenance
            target_text = json.dumps(target_payload, ensure_ascii=False)
            examples.append(
                {
                    "pair_id": row["pair_id"],
                    "repo": row.get("repo"),
                    "split": row.get("split"),
                    "gold_label": gold_label,
                    "termination_reason": termination_reason,
                    "rationale_provenance": provenance,
                    "debate_transcript_json": transcript_json,
                    "prompt_text": prompt_text,
                    "target_text": target_text,
                    "full_text": f"{prompt_text}\n\n{target_text}",
                }
            )
    return examples


def collect_proposer_failures(
    rows: list[dict[str, Any]],
    proposer_fn: Any,
    prompt_template: str,
) -> list[dict[str, Any]]:
    """Run a proposer function on rows and collect cases where it predicts incorrectly.

    This enables training the Critic on authentic failure patterns rather than
    synthetic bootstrap wrong labels.

    Args:
        rows: Input rows (must contain pair_id, label, issue_i, issue_j).
        proposer_fn: Callable(prompt_text) -> dict with at least 'predicted_label'.
        prompt_template: The proposer prompt template.

    Returns:
        List of failure rows, each augmented with 'predicted_label' and 'confidence'.
    """
    failures: list[dict[str, Any]] = []
    for row in rows:
        gold_label = str(row.get("label") or row.get("gold_label") or "")
        if not gold_label:
            continue
        pair_id = str(row.get("pair_id") or "")
        prompt_text = render_proposer_prompt(row, prompt_template)
        try:
            parsed = proposer_fn(prompt_text)
        except Exception:
            continue
        predicted = str(parsed.get("predicted_label") or "")
        if not predicted or predicted == gold_label:
            continue
        failures.append({
            "pair_id": pair_id,
            "predicted_label": predicted,
            "gold_label": gold_label,
            "confidence": float(parsed.get("confidence") or 0.0),
            "rationale": str(parsed.get("parsed_output", {}).get("rationale") or ""),
        })
    return failures


def collect_real_transcripts(
    rows: list[dict[str, Any]],
    proposer_fn: Any,
    critic_fn: Any,
    proposer_prompt_template: str,
    critic_prompt_template: str,
    confidence_map: dict[str, float],
) -> list[dict[str, Any]]:
    """Run real proposer + critic inference and collect debate transcripts.

    This enables training the Judge on authentic debate reasoning rather than
    synthetic single-round transcripts.

    Args:
        rows: Input rows (must contain pair_id, label, issue_i, issue_j).
        proposer_fn: Callable(prompt_text) -> dict with 'parsed_output'.
        critic_fn: Callable(prompt_text) -> dict with 'parsed_output'.
        proposer_prompt_template: The proposer prompt template.
        critic_prompt_template: The critic prompt template.
        confidence_map: Mapping from verbal confidence to numeric values.

    Returns:
        List of transcript dicts, each with 'pair_id', 'proposer_parsed', 'critic_parsed'.
    """
    transcripts: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        proposer_prompt = render_proposer_prompt(row, proposer_prompt_template)
        try:
            proposer_result = proposer_fn(proposer_prompt)
            proposer_parsed = proposer_result.get("parsed_output")
            if not proposer_parsed:
                continue
        except Exception:
            continue

        proposer_output_json = json.dumps(proposer_parsed, ensure_ascii=False)
        critic_prompt = render_critic_prompt(row, proposer_output_json, critic_prompt_template)
        try:
            critic_result = critic_fn(critic_prompt)
            critic_parsed = critic_result.get("parsed_output")
            if not critic_parsed:
                continue
        except Exception:
            continue

        transcripts.append({
            "pair_id": pair_id,
            "proposer_parsed": proposer_parsed,
            "critic_parsed": critic_parsed,
        })
    return transcripts


def build_judge_training_example_from_real_transcript(
    row: dict[str, Any],
    proposer_parsed: dict[str, Any],
    critic_parsed: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    critic_stance = str(critic_parsed.get("stance") or "challenge")
    termination_reason = "agreement" if critic_stance == "endorse" else "round_budget"

    transcript = {
        "pair_id": row["pair_id"],
        "repo": row["repo"],
        "window": row["window"],
        "split": row["split"],
        "rounds": [
            {
                "round_index": 0,
                "proposer": {
                    "raw_output": json.dumps(proposer_parsed, ensure_ascii=False),
                    "parsed_output": proposer_parsed,
                },
                "critic": {
                    "raw_output": json.dumps(critic_parsed, ensure_ascii=False),
                    "parsed_output": critic_parsed,
                },
            }
        ],
        "termination_reason": termination_reason,
    }
    transcript_json = compact_transcript_for_judge(transcript)
    prompt_text = render_judge_prompt(row, transcript_json, prompt_template)
    target_payload = build_judge_target_payload(row, termination_reason)
    target_payload["_provenance"] = "real_transcript"
    target_text = json.dumps(target_payload, ensure_ascii=False)
    return {
        "pair_id": row["pair_id"],
        "repo": row.get("repo"),
        "split": row.get("split"),
        "gold_label": str(row["label"]),
        "termination_reason": termination_reason,
        "debate_transcript_json": transcript_json,
        "transcript_source": "real_model_inference",
        "rationale_provenance": "real_transcript",
        "prompt_text": prompt_text,
        "target_text": target_text,
        "full_text": f"{prompt_text}\n\n{target_text}",
    }
