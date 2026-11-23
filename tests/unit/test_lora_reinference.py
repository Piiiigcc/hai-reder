from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.online_learning.triage import (  # noqa: E402
    _format_pool_examples,
    _parse_reinfer_output,
    create_lora_reinfer_fn,
    run_uncertainty_first_triage,
)


CONFIDENCE_MAP = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}

PROMPT_TEMPLATE = """Classify the dependency between Requirement A and Requirement B.

Requirement A:
{{issue_i_text}}

Requirement B:
{{issue_j_text}}

{{tfidf_baseline_hint}}
"""


def _make_prediction_row(pair_id: str, label: str, confidence: float, region: str = "medium") -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "issue_i": {"id": f"{pair_id}-A", "text": f"Login page error for {pair_id}"},
        "issue_j": {"id": f"{pair_id}-B", "text": f"Authentication failed for {pair_id}"},
        "predicted_label": label,
        "label": label,
        "gold_label": label,
        "confidence": confidence,
        "confidence_region": region,
        "verbal_confidence": "medium",
    }


def _make_corrected_context() -> list[dict]:
    return [
        {
            "pair_id": "CORR-1",
            "issue_i": {"id": "C1-A", "text": "User login session expired"},
            "issue_j": {"id": "C1-B", "text": "Password validation failed"},
            "accepted_label": "Composition",
            "label": "Composition",
        },
        {
            "pair_id": "CORR-2",
            "issue_i": {"id": "C2-A", "text": "Database connection timeout"},
            "issue_j": {"id": "C2-B", "text": "SQL query performance issue"},
            "accepted_label": "Temporal/Causal",
            "label": "Temporal/Causal",
        },
    ]


def test_format_pool_examples():
    context = _make_corrected_context()
    result = _format_pool_examples(context)
    assert "Example 1:" in result
    assert "User login session expired" in result
    assert "Label: Composition" in result
    assert "Example 2:" in result


def test_format_pool_examples_respects_max():
    context = _make_corrected_context()
    result = _format_pool_examples(context, max_examples=1)
    assert "Example 1:" in result
    assert "Example 2:" not in result


def test_format_pool_examples_empty():
    result = _format_pool_examples([])
    assert result == ""


def test_parse_reinfer_output_valid():
    raw = '{"label": "Composition", "rationale": "test", "verbal_confidence": "high"}'
    result = _parse_reinfer_output(raw, CONFIDENCE_MAP)
    assert result["predicted_label"] == "Composition"
    assert result["confidence"] == 0.7
    assert result["error"] is None


def test_parse_reinfer_output_invalid():
    raw = "not json"
    result = _parse_reinfer_output(raw, CONFIDENCE_MAP)
    assert result["predicted_label"] is None
    assert result["error"] is not None


def test_create_lora_reinfer_fn_returns_callable():
    fn = create_lora_reinfer_fn(
        adapter_dir="/nonexistent",
        prompt_template=PROMPT_TEMPLATE,
        confidence_map=CONFIDENCE_MAP,
    )
    assert callable(fn)


def test_create_lora_reinfer_fn_error_handling():
    fn = create_lora_reinfer_fn(
        adapter_dir="/nonexistent",
        prompt_template=PROMPT_TEMPLATE,
        confidence_map=CONFIDENCE_MAP,
    )
    row = _make_prediction_row("P-1", "Composition", 0.5)
    context = _make_corrected_context()
    result = fn(row, context)
    assert result["re_infer_source"] == "lora_reinference_error"
    assert "re_infer_note" in result


def test_triage_without_reinfer_fn_uses_placeholder():
    rows = [
        _make_prediction_row("P-1", "Composition", 0.3, "low"),
        _make_prediction_row("P-2", "Composition", 0.5, "medium"),
        _make_prediction_row("P-3", "Composition", 0.9, "high"),
    ]
    accepted, pool, summary = run_uncertainty_first_triage(
        rows, tau_low=0.4, tau_high=0.8,
    )
    assert summary["low_review_count"] == 1
    assert summary["medium_review_count"] == 1
    assert summary["promoted_to_high_count"] == 0


def test_triage_with_custom_reinfer_fn():
    rows = [
        _make_prediction_row("P-1", "Composition", 0.3, "low"),
        _make_prediction_row("P-2", "Composition", 0.5, "medium"),
        _make_prediction_row("P-3", "Composition", 0.9, "high"),
    ]

    def mock_reinfer(row, context):
        updated = dict(row)
        updated["re_infer_source"] = "mock_reinference"
        updated["confidence"] = 0.95
        updated["confidence_region"] = "high"
        return updated

    accepted, pool, summary = run_uncertainty_first_triage(
        rows, tau_low=0.4, tau_high=0.8, medium_reinfer_fn=mock_reinfer,
    )
    assert summary["promoted_to_high_count"] == 1
