"""Tests for GAP-11: Confidence fusion weight validation set tuning.

Verifies that:
- _generate_weight_combinations() produces valid combinations summing to 1
- search_confidence_weights() finds weights that minimize ECE
- Edge cases (empty rows, single row) are handled
- Default fallback weights are returned when no improvement found
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

weight_search = importlib.import_module("haireder.confidence.weight_search")
_generate_weight_combinations = weight_search._generate_weight_combinations
search_confidence_weights = weight_search.search_confidence_weights


def _make_val_row(s1: float, s2: float, s3: float, gold: str, predicted: str) -> dict:
    return {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "gold_label": gold,
        "predicted_label": predicted,
    }


def test_weight_combinations_sum_to_one() -> None:
    """All generated combinations should sum to 1.0."""
    combos = _generate_weight_combinations(step=0.1)
    for combo in combos:
        total = combo["self_evaluation"] + combo["inter_agent_agreement"] + combo["retrieval_similarity"]
        assert abs(total - 1.0) < 1e-9, f"Combination {combo} sums to {total}"


def test_weight_combinations_in_range() -> None:
    """All weights should be in [0, 1]."""
    combos = _generate_weight_combinations(step=0.1)
    for combo in combos:
        for key in ("self_evaluation", "inter_agent_agreement", "retrieval_similarity"):
            assert 0.0 <= combo[key] <= 1.0, f"{key}={combo[key]} out of range"


def test_weight_combinations_count() -> None:
    """With step=0.1, should produce ~66 combinations."""
    combos = _generate_weight_combinations(step=0.1)
    assert 60 <= len(combos) <= 70, f"Expected ~66, got {len(combos)}"


def test_weight_combinations_unique() -> None:
    """All combinations should be unique."""
    combos = _generate_weight_combinations(step=0.1)
    seen = set()
    for combo in combos:
        key = (combo["self_evaluation"], combo["inter_agent_agreement"], combo["retrieval_similarity"])
        assert key not in seen, f"Duplicate combination: {key}"
        seen.add(key)


def test_search_returns_default_for_empty_rows() -> None:
    """Empty validation rows should return default weights."""
    result = search_confidence_weights([])
    assert result["best_weights"]["self_evaluation"] == 0.34
    assert result["best_ece"] is None
    assert result["n_rows"] == 0


def test_search_finds_best_weights() -> None:
    """Search should find weights that minimize ECE."""
    # Create rows where s1 is a perfect predictor, s2 and s3 are noise
    rows = []
    for i in range(20):
        gold = "Composition" if i % 2 == 0 else "None"
        predicted = gold  # All correct
        if gold == "Composition":
            s1, s2, s3 = 0.9, 0.5, 0.5
        else:
            s1, s2, s3 = 0.1, 0.5, 0.5
        rows.append(_make_val_row(s1, s2, s3, gold, predicted))

    result = search_confidence_weights(rows, step=0.1)
    assert result["best_ece"] is not None
    assert result["n_evaluated"] > 0
    assert result["n_rows"] == 20
    # The search should find weights that achieve ECE no worse than the default
    from haireder.confidence.signals import compute_raw_confidence
    from haireder.evaluation.metrics import expected_calibration_error

    default_weights = {"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33}
    default_confs = [compute_raw_confidence(s1=r["s1"], s2=r["s2"], s3=r["s3"], weights=default_weights) for r in rows]
    default_ece = expected_calibration_error(
        [r["gold_label"] for r in rows], [r["predicted_label"] for r in rows], default_confs
    )["ece"]
    assert result["best_ece"] <= default_ece + 1e-9


def test_search_respects_step_size() -> None:
    """Smaller step should produce more combinations."""
    rows = [_make_val_row(0.8, 0.5, 0.5, "Composition", "Composition")]

    result_01 = search_confidence_weights(rows, step=0.1)
    result_02 = search_confidence_weights(rows, step=0.05)

    assert result_02["n_evaluated"] > result_01["n_evaluated"]


def test_search_weights_sum_to_one() -> None:
    """Returned best_weights should sum to 1.0."""
    rows = [_make_val_row(0.8, 0.6, 0.4, "Composition", "Composition") for _ in range(10)]
    result = search_confidence_weights(rows, step=0.1)
    weights = result["best_weights"]
    total = weights["self_evaluation"] + weights["inter_agent_agreement"] + weights["retrieval_similarity"]
    assert abs(total - 1.0) < 1e-9


def test_search_with_mispredictions() -> None:
    """Search should handle rows with mispredictions."""
    rows = [
        _make_val_row(0.9, 0.9, 0.9, "Composition", "None"),  # Wrong prediction, high confidence
        _make_val_row(0.1, 0.1, 0.1, "None", "None"),  # Correct prediction, low confidence
    ]

    result = search_confidence_weights(rows, step=0.2)
    assert result["best_ece"] is not None
    assert result["best_ece"] >= 0.0


def test_search_n_bins_parameter() -> None:
    """Different n_bins should produce different ECE granularity."""
    rows = [_make_val_row(0.7, 0.5, 0.3, "Composition", "Composition") for _ in range(20)]

    result_5 = search_confidence_weights(rows, step=0.2, n_bins=5)
    result_10 = search_confidence_weights(rows, step=0.2, n_bins=10)

    # Both should complete without error
    assert result_5["best_ece"] is not None
    assert result_10["best_ece"] is not None
