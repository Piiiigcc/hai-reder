"""Simulated analyst review helpers."""

from __future__ import annotations

from typing import Any

from haireder.taxonomy.labels import LABELS


def _require_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown accepted label: {label}")
    return label


def build_accepted_decision(
    row: dict[str, Any],
    *,
    accepted_label: str,
    decision_path: str,
    review_source: str,
    analyst_status: str,
    promoted_to_high: bool = False,
) -> dict[str, Any]:
    accepted = dict(row)
    accepted_label = _require_label(str(accepted_label))
    accepted.update(
        {
            "accepted_label": accepted_label,
            "decision_path": decision_path,
            "review_source": review_source,
            "analyst_status": analyst_status,
            "human_reviewed": False,
            "promoted_to_high": bool(promoted_to_high),
            "analyst_simulation": True,
            "analyst_method": "gold_label_oracle",
        }
    )
    return accepted


def simulate_analyst_review(
    row: dict[str, Any],
    *,
    decision_path: str,
    review_source: str,
) -> dict[str, Any]:
    """Use gold_label as the simulated analyst correction for reproducible experiments.

    Every accepted row from this function is tagged with analyst_simulation=True
    and analyst_method='gold_label_oracle' so downstream results cannot masquerade
    as real human review.
    """
    gold_label = _require_label(str(row.get("gold_label") or row.get("label") or ""))
    predicted_label = str(row.get("predicted_label") or "")
    analyst_status = "confirmed" if predicted_label == gold_label else "corrected"
    return build_accepted_decision(
        row,
        accepted_label=gold_label,
        decision_path=decision_path,
        review_source=review_source,
        analyst_status=analyst_status,
    )
