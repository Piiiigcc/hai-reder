"""Tests for GAP-15 (Qwen formal config) and GAP-16 (longitudinal metrics).

Verifies that:
- formal_qwen_qlora.yaml is valid YAML with correct Qwen-2.5-7B settings
- compute_metrics_by_window() groups by window correctly
- compute_refresh_round_metrics() computes before/after delta
- compute_cumulative_metrics() tracks progressive improvement
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

metrics = importlib.import_module("haireder.evaluation.metrics")
compute_metrics_by_window = metrics.compute_metrics_by_window
compute_refresh_round_metrics = metrics.compute_refresh_round_metrics
compute_cumulative_metrics = metrics.compute_cumulative_metrics


# ---------------------------------------------------------------------------
# GAP-15: Formal Qwen config tests
# ---------------------------------------------------------------------------
def test_formal_qwen_config_exists() -> None:
    """formal_qwen_qlora.yaml should exist."""
    config_path = PROJECT_ROOT / "configs" / "models" / "formal_qwen_qlora.yaml"
    assert config_path.exists(), f"Config not found: {config_path}"


def test_formal_qwen_config_valid_yaml() -> None:
    """formal_qwen_qlora.yaml should be valid YAML."""
    config_path = PROJECT_ROOT / "configs" / "models" / "formal_qwen_qlora.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config is not None


def test_formal_qwen_config_model() -> None:
    """Should use Qwen-2.5-7B-Instruct as primary model."""
    config_path = PROJECT_ROOT / "configs" / "models" / "formal_qwen_qlora.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["base_model"]["primary"] == "Qwen/Qwen2.5-7B-Instruct"


def test_formal_qwen_config_quantization() -> None:
    """Should have 4-bit QLoRA quantization enabled."""
    config_path = PROJECT_ROOT / "configs" / "models" / "formal_qwen_qlora.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    q = config["quantization"]
    assert q["enabled"] is True
    assert q["load_in_4bit"] is True
    assert q["bnb_4bit_quant_type"] == "nf4"


def test_formal_qwen_config_lora_params() -> None:
    """Should have adjusted LoRA params (r=16, alpha=32)."""
    config_path = PROJECT_ROOT / "configs" / "models" / "formal_qwen_qlora.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32


def test_minimal_config_unchanged() -> None:
    """minimal_lora.yaml should still use SmolLM2-135M."""
    config_path = PROJECT_ROOT / "configs" / "models" / "minimal_lora.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    assert config["base_model"]["primary"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------------------------
# GAP-16: Longitudinal metrics tests
# ---------------------------------------------------------------------------
def _make_pred(window: str, gold: str, predicted: str, confidence: float = 0.8) -> dict:
    return {
        "pair_id": f"p-{gold}-{predicted}-{window}",
        "window": window,
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": confidence,
    }


def test_compute_metrics_by_window_groups() -> None:
    """Should compute metrics per window."""
    predictions = [
        _make_pred("W1", "Composition", "Composition"),
        _make_pred("W1", "None", "None"),
        _make_pred("W2", "Composition", "None"),  # Wrong
        _make_pred("W2", "None", "None"),
    ]

    result = compute_metrics_by_window(predictions)

    assert "overall" in result
    assert "by_window" in result
    assert "W1" in result["by_window"]
    assert "W2" in result["by_window"]
    # W1: both correct → accuracy 1.0
    assert result["by_window"]["W1"]["accuracy"] == 1.0
    # W2: one wrong → accuracy 0.5
    assert result["by_window"]["W2"]["accuracy"] == 0.5


def test_compute_metrics_by_window_empty() -> None:
    """Empty predictions should return empty overall."""
    result = compute_metrics_by_window([])
    assert result["overall"]["count"] == 0
    assert result["by_window"] == {}


def test_compute_metrics_by_window_with_ece() -> None:
    """Should compute ECE when requested."""
    predictions = [
        _make_pred("W1", "Composition", "Composition", 0.9),
        _make_pred("W1", "None", "None", 0.1),
    ]

    result = compute_metrics_by_window(predictions, compute_ece=True)
    assert "ece" in result["by_window"]["W1"]


def test_refresh_round_metrics_delta() -> None:
    """Should compute delta between before and after."""
    before = [
        _make_pred("W1", "Composition", "None"),  # Wrong
        _make_pred("W1", "None", "None"),  # Correct
    ]
    after = [
        _make_pred("W1", "Composition", "Composition"),  # Now correct
        _make_pred("W1", "None", "None"),  # Still correct
    ]

    result = compute_refresh_round_metrics(before, after)

    assert "before" in result
    assert "after" in result
    assert "delta" in result
    # After should be better
    assert result["delta"]["macro_f1_change"] > 0


def test_refresh_round_metrics_empty() -> None:
    """Empty predictions should not crash."""
    result = compute_refresh_round_metrics([], [])
    assert result["before"]["count"] == 0
    assert result["after"]["count"] == 0


def test_cumulative_metrics_progression() -> None:
    """Should track cumulative metrics across sprints."""
    predictions_by_sprint = {
        "W5": [_make_pred("W5", "Composition", "None")],  # Wrong
        "W6": [_make_pred("W6", "Composition", "Composition")],  # Correct
        "W7": [_make_pred("W7", "None", "None")],  # Correct
    }

    result = compute_cumulative_metrics(predictions_by_sprint)

    assert "per_sprint" in result
    assert "cumulative" in result
    assert "sprint_order" in result
    assert len(result["per_sprint"]) == 3
    assert result["sprint_order"] == ["W5", "W6", "W7"]
    # Cumulative: 2/3 correct
    assert result["cumulative"]["count"] == 3


def test_cumulative_metrics_empty() -> None:
    """Empty sprints should return empty results."""
    result = compute_cumulative_metrics({})
    assert result["cumulative"]["count"] == 0
    assert result["per_sprint"] == {}


def test_cumulative_metrics_with_ece() -> None:
    """Should compute ECE for cumulative metrics."""
    predictions_by_sprint = {
        "W1": [_make_pred("W1", "Composition", "Composition", 0.9)],
    }

    result = compute_cumulative_metrics(predictions_by_sprint, compute_ece=True)
    assert "ece" in result["cumulative"]
