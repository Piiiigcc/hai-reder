from __future__ import annotations

import json
from typing import Any

from haireder.taxonomy import LABELS
from .prompts import render_role_template


VALID_CONFIDENCE_LEVELS = ["very_low", "low", "medium", "high", "very_high"]


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    try:
        _, end = json.JSONDecoder().raw_decode(text[start:])
        return text[start:start + end]
    except ValueError:
        return text[start:].strip()



def render_judge_prompt(
    row: dict[str, Any],
    debate_transcript_json: str,
    prompt_template: str,
    tfidf_baseline_hint: str = "",
    retrieval_examples: str = "",
) -> str:
    return render_role_template(prompt_template, row, debate_transcript_json=debate_transcript_json,
                                tfidf_baseline_hint=tfidf_baseline_hint, retrieval_examples=retrieval_examples)


def compact_transcript_for_judge(transcript: dict[str, Any]) -> str:
    compact = {
        "pair_id": transcript["pair_id"],
        "repo": transcript["repo"],
        "window": transcript["window"],
        "split": transcript["split"],
        "rounds": [
            {
                "round_index": round_data["round_index"],
                "proposer": {
                    "label": round_data["proposer"]["parsed_output"].get("label"),
                    "rationale": round_data["proposer"]["parsed_output"].get("rationale"),
                    **{k: round_data["proposer"]["parsed_output"][k] for k in ("quote_a", "quote_b") if k in round_data["proposer"]["parsed_output"]},
                    "verbal_confidence": round_data["proposer"]["parsed_output"].get("verbal_confidence"),
                },
                "critic": {
                    "stance": round_data["critic"]["parsed_output"].get("stance"),
                    "label": round_data["critic"]["parsed_output"].get("label"),
                    "rationale": round_data["critic"]["parsed_output"].get("rationale"),
                    **{k: round_data["critic"]["parsed_output"][k] for k in ("quote_a", "quote_b") if k in round_data["critic"]["parsed_output"]},
                    "verbal_confidence": round_data["critic"]["parsed_output"].get("verbal_confidence"),
                },
            }
            for round_data in transcript["rounds"]
        ],
        "termination_reason": transcript["termination_reason"],
    }
    return json.dumps(compact, ensure_ascii=False)


def build_judge_target_payload(
    row: dict[str, Any],
    termination_reason: str,
    justification: str | None = None,
) -> dict[str, str]:
    gold_label = str(row["label"])
    if gold_label not in LABELS:
        raise ValueError(f"Unknown label: {gold_label}")
    verbal_confidence = "high" if termination_reason == "agreement" else "medium"
    if justification is None:
        justification = f"The final label is {gold_label} after comparing the proposer and critic arguments."
    return {
        "final_label": gold_label,
        "justification": justification,
        "verbal_confidence": verbal_confidence,
    }


def build_fallback_judge_decision(transcript: dict[str, Any]) -> dict[str, Any]:
    last_round = transcript["rounds"][-1]
    critic_output = last_round["critic"]["parsed_output"]
    proposer_output = last_round["proposer"]["parsed_output"]
    final_label = critic_output.get("label") or proposer_output.get("label")
    if final_label not in LABELS:
        final_label = proposer_output.get("label")
    if final_label not in LABELS:
        raise ValueError("Could not recover final label from debate transcript")
    return {
        "final_label": final_label,
        "justification": "Recovered from malformed judge output using the last debate state.",
        "verbal_confidence": "very_low",
        "repair_applied": True,
    }


def parse_judge_output(raw_output: str, confidence_map: dict[str, float]) -> dict[str, Any]:
    parsed_output = None
    predicted_label = None
    confidence = None
    error = None
    candidate_text = _first_json_object(raw_output.strip())
    try:
        parsed_output = json.loads(candidate_text)
        if not isinstance(parsed_output, dict):
            raise ValueError("parsed output is not a JSON object")
        predicted_label = str(parsed_output.get("final_label") or "")
        if predicted_label not in LABELS:
            raise ValueError(f"invalid final label: {predicted_label}")
        verbal_confidence = str(parsed_output.get("verbal_confidence") or "")
        if verbal_confidence not in VALID_CONFIDENCE_LEVELS or verbal_confidence not in confidence_map:
            raise ValueError(f"invalid verbal confidence: {verbal_confidence}")
        if not isinstance(parsed_output.get("justification"), str) or not parsed_output["justification"].strip():
            raise ValueError("missing justification")
        confidence = float(confidence_map[verbal_confidence])
    except (TypeError, ValueError, KeyError) as exc:
        error = str(exc)
        parsed_output = predicted_label = confidence = None
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "parsed_output": parsed_output,
        "error": error,
        "raw_output": raw_output,
    }
