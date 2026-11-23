"""Tests for GAP-08: Role-specific training data.

Verifies that:
- collect_proposer_failures() correctly identifies incorrect predictions
- collect_real_transcripts() correctly pairs proposer + critic outputs
- build_critic_training_examples() supports failure_rows + data_source="real"
- build_judge_training_examples() supports real_transcripts + data_source="real"
- Bootstrap (default) behavior is unchanged
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

data_builder = importlib.import_module("haireder.training.data_builder")
build_critic_training_examples = data_builder.build_critic_training_examples
build_judge_training_examples = data_builder.build_judge_training_examples
collect_proposer_failures = data_builder.collect_proposer_failures
collect_real_transcripts = data_builder.collect_real_transcripts


def _make_row(pair_id: str, label: str, split: str = "train") -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": split,
        "issue_i": {"id": f"{pair_id}-A", "text": f"Summary: Issue A for {pair_id}"},
        "issue_j": {"id": f"{pair_id}-B", "text": f"Summary: Issue B for {pair_id}"},
        "label": label,
    }


PROMPT_TEMPLATE = "Prompt {{issue_i_text}} {{issue_j_text}}\n\nJSON Output:\n"
CRITIC_PROMPT_TEMPLATE = "Critic {{issue_i_text}} {{issue_j_text}} {{proposer_prediction}}\n\nJSON Output:\n"

CONFIDENCE_MAP = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "very_high": 0.9,
}


def test_collect_proposer_failures_returns_only_mismatches() -> None:
    """Only rows where predicted != gold should be returned."""
    rows = [
        _make_row("p1", "Composition"),
        _make_row("p2", "Temporal/Causal"),
        _make_row("p3", "Duplication"),
    ]

    def fake_proposer(prompt: str) -> dict:
        # p1: correct, p2: wrong, p3: wrong
        if "p1" in prompt:
            return {"predicted_label": "Composition", "confidence": 0.8, "parsed_output": {"label": "Composition"}}
        if "p2" in prompt:
            return {"predicted_label": "None", "confidence": 0.4, "parsed_output": {"label": "None"}}
        return {"predicted_label": "General Relation", "confidence": 0.6, "parsed_output": {"label": "General Relation"}}

    failures = collect_proposer_failures(rows, fake_proposer, PROMPT_TEMPLATE)

    assert len(failures) == 2
    failure_pair_ids = {f["pair_id"] for f in failures}
    assert "p1" not in failure_pair_ids
    assert "p2" in failure_pair_ids
    assert "p3" in failure_pair_ids


def test_collect_proposer_failures_skips_exceptions() -> None:
    """Rows where proposer_fn raises should be skipped."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]

    def fake_proposer(prompt: str) -> dict:
        if "p1" in prompt:
            raise RuntimeError("model error")
        return {"predicted_label": "None", "confidence": 0.3, "parsed_output": {"label": "None"}}

    failures = collect_proposer_failures(rows, fake_proposer, PROMPT_TEMPLATE)

    assert len(failures) == 1
    assert failures[0]["pair_id"] == "p2"


def test_collect_real_transcripts_pairs_outputs() -> None:
    """Transcripts should pair proposer + critic parsed outputs."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]

    def fake_proposer(prompt: str) -> dict:
        return {"parsed_output": {"label": "Composition", "rationale": "r", "verbal_confidence": "high"}}

    def fake_critic(prompt: str) -> dict:
        return {"parsed_output": {"stance": "endorse", "label": "Composition", "rationale": "agree", "verbal_confidence": "high"}}

    transcripts = collect_real_transcripts(
        rows, fake_proposer, fake_critic,
        PROMPT_TEMPLATE, CRITIC_PROMPT_TEMPLATE,
        CONFIDENCE_MAP,
    )

    assert len(transcripts) == 2
    assert transcripts[0]["pair_id"] == "p1"
    assert "proposer_parsed" in transcripts[0]
    assert "critic_parsed" in transcripts[0]


def test_collect_real_transcripts_skips_proposer_failures() -> None:
    """Rows where proposer returns no parsed_output should be skipped."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]

    def fake_proposer(prompt: str) -> dict:
        if "p1" in prompt:
            return {"predicted_label": None, "error": "parse error"}
        return {"parsed_output": {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "medium"}}

    def fake_critic(prompt: str) -> dict:
        return {"parsed_output": {"stance": "endorse", "label": "Temporal/Causal", "rationale": "ok", "verbal_confidence": "high"}}

    transcripts = collect_real_transcripts(
        rows, fake_proposer, fake_critic,
        PROMPT_TEMPLATE, CRITIC_PROMPT_TEMPLATE,
        CONFIDENCE_MAP,
    )

    assert len(transcripts) == 1
    assert transcripts[0]["pair_id"] == "p2"


def test_critic_bootstrap_default_unchanged() -> None:
    """Default data_source='bootstrap' should produce same output as before."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]

    examples = build_critic_training_examples(rows, PROMPT_TEMPLATE)

    assert len(examples) == 2
    # Bootstrap alternates: index 0 = endorse, index 1 = challenge
    assert examples[0]["critic_scenario"] == "endorse"
    assert examples[1]["critic_scenario"] == "challenge"
    assert examples[0]["rationale_provenance"] == "bootstrap_rationale"


def test_critic_real_data_source_uses_failure_rows() -> None:
    """With data_source='real' and failure_rows, critic should use real failures."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]
    failures = [
        {"pair_id": "p1", "predicted_label": "None", "gold_label": "Composition"},
        {"pair_id": "p2", "predicted_label": "General Relation", "gold_label": "Temporal/Causal"},
    ]

    examples = build_critic_training_examples(
        rows, PROMPT_TEMPLATE,
        failure_rows=failures,
        data_source="real",
    )

    assert len(examples) == 2
    # Both should be challenge (predicted != gold)
    assert examples[0]["critic_scenario"] == "challenge"
    assert examples[1]["critic_scenario"] == "challenge"
    assert examples[0]["failure_source"] == "real_failure"


def test_critic_real_data_source_falls_back_to_bootstrap() -> None:
    """When no matching failure_row, should fallback to bootstrap alternation."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]
    failures = [{"pair_id": "p1", "predicted_label": "None", "gold_label": "Composition"}]

    examples = build_critic_training_examples(
        rows, PROMPT_TEMPLATE,
        failure_rows=failures,
        data_source="real",
    )

    assert len(examples) == 2
    # p1: real failure → challenge
    assert examples[0]["critic_scenario"] == "challenge"
    assert examples[0]["failure_source"] == "real_failure"
    # p2: no matching failure → bootstrap fallback
    assert examples[1]["failure_source"] == "bootstrap_failure"


def test_judge_bootstrap_default_unchanged() -> None:
    """Default data_source='bootstrap' should produce same output as before."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]

    examples = build_judge_training_examples(rows, PROMPT_TEMPLATE)

    assert len(examples) == 2
    assert examples[0]["termination_reason"] == "agreement"
    assert examples[1]["termination_reason"] == "round_budget"


def test_judge_real_data_source_uses_transcripts() -> None:
    """With data_source='real' and real_transcripts, judge should use real debate."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]
    transcripts = [
        {
            "pair_id": "p1",
            "proposer_parsed": {"label": "Composition", "rationale": "r1", "verbal_confidence": "high"},
            "critic_parsed": {"stance": "endorse", "label": "Composition", "rationale": "agree", "verbal_confidence": "high"},
        },
        {
            "pair_id": "p2",
            "proposer_parsed": {"label": "Temporal/Causal", "rationale": "r2", "verbal_confidence": "medium"},
            "critic_parsed": {"stance": "challenge", "label": "Duplication", "rationale": "disagree", "verbal_confidence": "low"},
        },
    ]

    examples = build_judge_training_examples(
        rows, PROMPT_TEMPLATE,
        real_transcripts=transcripts,
        data_source="real",
    )

    assert len(examples) == 2
    # p1: critic endorsed → agreement
    assert examples[0]["termination_reason"] == "agreement"
    assert examples[0].get("transcript_source") == "real_model_inference"
    # p2: critic challenged → round_budget
    assert examples[1]["termination_reason"] == "round_budget"
    assert examples[1].get("transcript_source") == "real_model_inference"


def test_judge_real_data_source_falls_back_to_bootstrap() -> None:
    """When no matching real_transcript, should fallback to bootstrap."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]
    transcripts = [
        {
            "pair_id": "p1",
            "proposer_parsed": {"label": "Composition", "rationale": "r1", "verbal_confidence": "high"},
            "critic_parsed": {"stance": "endorse", "label": "Composition", "rationale": "agree", "verbal_confidence": "high"},
        },
    ]

    examples = build_judge_training_examples(
        rows, PROMPT_TEMPLATE,
        real_transcripts=transcripts,
        data_source="real",
    )

    assert len(examples) == 2
    # p1: real transcript
    assert examples[0]["termination_reason"] == "agreement"
    assert examples[0].get("transcript_source") == "real_model_inference"
    # p2: bootstrap fallback
    assert examples[1]["termination_reason"] == "round_budget"
    assert examples[1].get("transcript_source") == "bootstrap_synthetic"


def test_role_training_data_has_different_distributions() -> None:
    """With real data sources, critic and judge should have different distributions."""
    rows = [_make_row("p1", "Composition"), _make_row("p2", "Temporal/Causal")]
    failures = [
        {"pair_id": "p1", "predicted_label": "None", "gold_label": "Composition"},
        {"pair_id": "p2", "predicted_label": "General Relation", "gold_label": "Temporal/Causal"},
    ]
    transcripts = [
        {
            "pair_id": "p1",
            "proposer_parsed": {"label": "Composition", "rationale": "r", "verbal_confidence": "high"},
            "critic_parsed": {"stance": "endorse", "label": "Composition", "rationale": "ok", "verbal_confidence": "high"},
        },
        {
            "pair_id": "p2",
            "proposer_parsed": {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "medium"},
            "critic_parsed": {"stance": "challenge", "label": "Duplication", "rationale": "no", "verbal_confidence": "low"},
        },
    ]

    critic_examples = build_critic_training_examples(
        rows, PROMPT_TEMPLATE,
        failure_rows=failures,
        data_source="real",
    )
    judge_examples = build_judge_training_examples(
        rows, PROMPT_TEMPLATE,
        real_transcripts=transcripts,
        data_source="real",
    )

    # Critic: both are challenge (real failures)
    critic_scenarios = [e["critic_scenario"] for e in critic_examples]
    assert all(s == "challenge" for s in critic_scenarios)

    # Judge: mixed termination reasons from real transcripts
    judge_terminations = [e["termination_reason"] for e in judge_examples]
    assert "agreement" in judge_terminations
    assert "round_budget" in judge_terminations
