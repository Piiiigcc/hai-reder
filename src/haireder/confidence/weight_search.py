"""Grid search for optimal confidence fusion weights on a validation set.

Implements the weight tuning described in Section 3.4:
    w1*s1 + w2*s2 + w3*s3  where  w1,w2,w3 in [0,1], sum(w) = 1
"""
from __future__ import annotations

from typing import Any

from haireder.confidence.signals import compute_raw_confidence
from haireder.evaluation.metrics import expected_calibration_error


def _generate_weight_combinations(step: float = 0.1) -> list[dict[str, float]]:
    """Generate all (w1, w2, w3) combinations where each is in [0,1] and sum = 1."""
    values = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    combinations = []
    for w1 in values:
        for w2 in values:
            w3 = round(1.0 - w1 - w2, 2)
            if 0.0 <= w3 <= 1.0:
                combinations.append({
                    "self_evaluation": w1,
                    "inter_agent_agreement": w2,
                    "retrieval_similarity": w3,
                })
    return combinations


def search_confidence_weights(
    validation_rows: list[dict[str, Any]],
    *,
    step: float = 0.1,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Search for optimal confidence fusion weights by minimizing validation ECE.

    Args:
        validation_rows: Rows with s1, s2, s3 signals and gold_label + predicted_label.
        step: Grid step size for weight search (default 0.1 → ~66 combinations).
        n_bins: Number of bins for ECE calculation.

    Returns:
        Dict with 'best_weights', 'best_ece', 'n_evaluated', 'n_rows'.
    """
    if not validation_rows:
        return {
            "best_weights": {"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33},
            "best_ece": None,
            "n_evaluated": 0,
            "n_rows": 0,
            "note": "no_validation_rows",
        }

    combinations = _generate_weight_combinations(step=step)
    best_weights = None
    best_ece = float("inf")
    best_brier = float("inf")

    for weights in combinations:
        raw_confidences = []
        for row in validation_rows:
            s1 = float(row.get("s1") or 0.0)
            s2 = float(row.get("s2") or 0.0)
            s3 = float(row.get("s3") or 0.0)
            raw_confidences.append(compute_raw_confidence(s1=s1, s2=s2, s3=s3, weights=weights))

        gold_labels = [str(row.get("gold_label") or "") for row in validation_rows]
        predicted_labels = [str(row.get("predicted_label") or "") for row in validation_rows]

        ece_result = expected_calibration_error(gold_labels, predicted_labels, raw_confidences, n_bins=n_bins)
        ece = ece_result["ece"]

        brier = sum((c - float(g == p))**2 for c,g,p in zip(raw_confidences,gold_labels,predicted_labels)) / len(validation_rows)
        if ece is not None and (ece, brier) < (best_ece, best_brier):
            best_ece, best_brier = ece, brier
            best_weights = dict(weights)

    if best_weights is None:
        best_weights = {"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33}
        best_ece = None

    return {
        "best_weights": best_weights,
        "best_ece": best_ece,
        "tie_break_brier": best_brier,
        "n_evaluated": len(combinations),
        "n_rows": len(validation_rows),
    }
