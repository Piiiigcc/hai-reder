from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CalibrationModel:
    mode: str
    estimator: LogisticRegression | None = None
    reason: str | None = None



def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))



def fit_confidence_calibrator(
    raw_confidences: Sequence[float],
    gold_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> CalibrationModel:
    if not (
        len(raw_confidences) == len(gold_labels) == len(predicted_labels)
    ):
        raise ValueError("raw confidences, gold labels, and predicted labels must align")

    if not raw_confidences:
        return CalibrationModel(
            mode="fallback",
            reason="insufficient correctness variation for calibration",
        )

    correctness = [int(gold == predicted) for gold, predicted in zip(gold_labels, predicted_labels)]
    if len(set(correctness)) < 2:
        return CalibrationModel(
            mode="fallback",
            reason="insufficient correctness variation for calibration",
        )

    features = np.array([[_clamp_probability(value)] for value in raw_confidences], dtype=float)
    estimator = LogisticRegression(random_state=0)
    estimator.fit(features, correctness)
    return CalibrationModel(mode="platt", estimator=estimator, reason="fitted_platt")



def apply_confidence_calibration(raw_confidence: float, calibrator: CalibrationModel) -> float:
    confidence = _clamp_probability(raw_confidence)
    if calibrator.mode != "platt" or calibrator.estimator is None:
        return confidence

    calibrated = calibrator.estimator.predict_proba(np.array([[confidence]], dtype=float))[0][1]
    return _clamp_probability(float(calibrated))


def resolve_confidence_weights(
    config: dict[str, Any] | None = None,
    *,
    validation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Resolve confidence fusion weights from config or weight search.

    If weight_search_enabled is True and validation_rows are provided, runs
    grid search to find optimal weights by minimizing validation ECE.
    Otherwise, returns the config's fusion_weights (or defaults).

    Args:
        config: Experiment config dict with 'confidence' section.
        validation_rows: Validation prediction rows for weight search.

    Returns:
        Dict with 'self_evaluation', 'inter_agent_agreement', 'retrieval_similarity'.
    """
    default_weights = {"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33}

    if config is None:
        return dict(default_weights)

    confidence_config = config.get("confidence", {})
    fusion_weights = confidence_config.get("fusion_weights", default_weights)
    weight_search_enabled = confidence_config.get("weight_search_enabled", False)

    if not weight_search_enabled or not validation_rows:
        return {
            "self_evaluation": float(fusion_weights.get("self_evaluation", default_weights["self_evaluation"])),
            "inter_agent_agreement": float(fusion_weights.get("inter_agent_agreement", default_weights["inter_agent_agreement"])),
            "retrieval_similarity": float(fusion_weights.get("retrieval_similarity", default_weights["retrieval_similarity"])),
        }

    from haireder.confidence.weight_search import search_confidence_weights

    search_config = confidence_config.get("weight_search", {})
    step = float(search_config.get("step", 0.1))
    n_bins = int(search_config.get("n_bins", 10))

    result = search_confidence_weights(validation_rows, step=step, n_bins=n_bins)
    return result["best_weights"]
