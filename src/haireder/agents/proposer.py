from __future__ import annotations

import json
from typing import Any

from haireder.taxonomy import LABELS
from .prompts import render_role_template


VALID_CONFIDENCE_LEVELS = ["very_low", "low", "medium", "high", "very_high"]


def render_proposer_prompt(
    row: dict[str, Any],
    prompt_template: str,
    tfidf_baseline_hint: str = "",
    retrieval_examples: str = "",
) -> str:
    return render_role_template(prompt_template, row, tfidf_baseline_hint=tfidf_baseline_hint,
                                retrieval_examples=retrieval_examples)


def build_proposer_target_payload(row: dict[str, Any], rationale: str | None = None) -> dict[str, str]:
    label = str(row["label"])
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label}")
    verbal_confidence = "medium" if label == "None" else "high"
    if rationale is None:
        rationale = f"The pair is labeled {label} based on the requirement texts provided in the training data."
    return {
        "label": label,
        "rationale": rationale,
        "verbal_confidence": verbal_confidence,
    }


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text.strip()
    try:
        _, end = json.JSONDecoder().raw_decode(text[start:])
        return text[start:start + end]
    except ValueError:
        return text[start:].strip()



def parse_proposer_output(raw_output: str, confidence_map: dict[str, float]) -> dict[str, Any]:
    parsed_output = None
    predicted_label = None
    confidence = None
    error = None
    candidate_text = _first_json_object(raw_output.strip())
    try:
        parsed_output = json.loads(candidate_text)
        if not isinstance(parsed_output, dict):
            raise ValueError("parsed output is not a JSON object")
        predicted_label = str(parsed_output.get("label") or "")
        if predicted_label not in LABELS:
            raise ValueError(f"invalid predicted label: {predicted_label}")
        verbal_confidence = str(parsed_output.get("verbal_confidence") or "")
        if verbal_confidence not in VALID_CONFIDENCE_LEVELS or verbal_confidence not in confidence_map:
            raise ValueError(f"invalid verbal confidence: {verbal_confidence}")
        if not isinstance(parsed_output.get("rationale"), str) or not parsed_output["rationale"].strip():
            raise ValueError("missing rationale")
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
