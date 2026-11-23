from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.confidence.calibration import CalibrationModel  # noqa: E402
from haireder.agents.judge import render_judge_prompt  # noqa: E402
from haireder.agents.proposer import render_proposer_prompt  # noqa: E402
from haireder.debate.engine import run_debate  # noqa: E402

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "workflow" / "run_phase15_final_evaluation.py"
SPEC = importlib.util.spec_from_file_location("run_phase15_final_evaluation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
run_phase15_final_evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_phase15_final_evaluation)


def test_flag_prediction_collapse_reports_dominant_label() -> None:
    rows = [
        {"gold_label": "None", "predicted_label": "Temporal/Causal"}
        for _ in range(19)
    ] + [{"gold_label": "Composition", "predicted_label": "Composition"}]

    health = run_phase15_final_evaluation._prediction_health_report(rows, dominant_threshold=0.9)

    assert health["status"] == "collapsed"
    assert health["dominant_label"] == "Temporal/Causal"
    assert health["dominant_ratio"] == 0.95


def test_flag_confidence_collapse_reports_single_region() -> None:
    rows = [
        {"confidence_region": "medium", "confidence": 0.5}
        for _ in range(20)
    ]

    health = run_phase15_final_evaluation._confidence_health_report(rows)

    assert health["status"] == "collapsed_single_region"
    assert health["region_counts"] == {"medium": 20}


def test_final_evaluation_blocked_missing_inputs_returns_nonzero(tmp_path, monkeypatch) -> None:
    metrics_out = tmp_path / "metrics.json"
    table_out = tmp_path / "table.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_phase15_final_evaluation.py",
            "--train", str(tmp_path / "missing_train.jsonl"),
            "--online", str(tmp_path / "missing_online.jsonl"),
            "--test", str(tmp_path / "missing_test.jsonl"),
            "--metrics-out", str(metrics_out),
            "--table-out", str(table_out),
        ],
    )

    assert run_phase15_final_evaluation.main() == 1
    assert metrics_out.exists()
    assert "blocked_missing_inputs" in metrics_out.read_text(encoding="utf-8")


def test_unreliable_tiny_validation_calibrator_falls_back_to_raw_confidence() -> None:
    calibrator = CalibrationModel(mode="platt", estimator=object(), reason="fitted_platt")
    calibration_summary = {
        "row_count": 16,
        "confidence_regions": {"medium": 16},
        "calibration": {"mode": "platt", "reason": "fitted_platt"},
    }

    effective_calibrator, reliability = run_phase15_final_evaluation._select_phase15_calibrator(
        calibrator,
        calibration_summary=calibration_summary,
    )

    assert effective_calibrator.mode == "fallback"
    assert effective_calibrator.reason == "unreliable_validation_calibrator"
    assert reliability["status"] == "fallback_to_raw_confidence"
    assert reliability["row_count"] == 16
    assert reliability["confidence_region_count"] == 1


def test_aggregate_triage_summaries_reads_online_triage_payload_shape(tmp_path: Path) -> None:
    summary_path = tmp_path / "formal_online_W5_triage.json"
    summary_path.write_text(
        """
{
  "window": "W5",
  "triage": {
    "total_rows": 10,
    "low_review_count": 2,
    "medium_review_count": 3,
    "total_review_count": 5,
    "promoted_to_high_count": 4
  }
}
""".strip(),
        encoding="utf-8",
    )

    aggregate = run_phase15_final_evaluation._aggregate_triage_summaries([summary_path])

    assert aggregate["windows"] == ["W5"]
    assert aggregate["total_rows"] == 10
    assert aggregate["low_review_count"] == 2
    assert aggregate["medium_review_count"] == 3
    assert aggregate["total_review_count"] == 5
    assert aggregate["promoted_to_high_count"] == 4
    assert aggregate["triage_saving"] == 4 / (3 + 4)


def test_render_proposer_prompt_includes_tfidf_hint_before_json_output() -> None:
    row = {
        "issue_i": {"text": "Requirement A text"},
        "issue_j": {"text": "Requirement B text"},
    }
    template = "Requirement A:\n{{issue_i_text}}\n\n{{tfidf_baseline_hint}}\n\nRequirement B:\n{{issue_j_text}}"

    prompt = render_proposer_prompt(row, template, tfidf_baseline_hint="Baseline classifier hint: None.")

    assert "Baseline classifier hint: None." in prompt
    assert prompt.endswith("JSON Output:\n")
    assert "{{tfidf_baseline_hint}}" not in prompt


def test_render_judge_prompt_includes_tfidf_hint_before_json_output() -> None:
    row = {
        "issue_i": {"text": "Requirement A text"},
        "issue_j": {"text": "Requirement B text"},
    }
    template = "Requirement A:\n{{issue_i_text}}\n\n{{tfidf_baseline_hint}}\n\nRequirement B:\n{{issue_j_text}}\n\nDebate:\n{{debate_transcript_json}}"

    prompt = render_judge_prompt(
        row,
        "{\"pair_id\": \"pair-1\"}",
        template,
        tfidf_baseline_hint="Baseline classifier hint: Temporal/Causal.",
    )

    assert "Baseline classifier hint: Temporal/Causal." in prompt
    assert prompt.endswith("JSON Output:\n")
    assert "{{tfidf_baseline_hint}}" not in prompt


def test_generate_valid_turn_emits_tagged_fallback_after_parse_exhaustion() -> None:
    def always_fail_parse(raw_output, confidence_map):
        return {"predicted_label": None, "confidence": None, "parsed_output": None, "error": "parse_failure"}

    def always_fail_validate(pair_id, parsed):
        raise ValueError("simulated parse failure")

    def fake_generate(bundle, prompt_text, max_new_tokens=128):
        return "bad output"

    result = run_phase15_final_evaluation._generate_valid_turn(
        pair_id="pair-1",
        bundle={},
        prompt_text="test",
        max_new_tokens=16,
        confidence_map={"very_low": 0.1},
        parse_fn=always_fail_parse,
        generate_output_fn=fake_generate,
        validate_fn=always_fail_validate,
    )

    assert result["parsed_output"]["is_fallback_generation"] is True
    assert result["parsed_output"]["fallback_reason"] == "all_parse_attempts_exhausted"
    assert result["parsed_output"]["label"] is None
    assert result["parsed_output"]["verbal_confidence"] == "very_low"
    assert len(result["parsed_output"]["fallback_errors"]) == 2


def test_enrich_prediction_row_includes_s3_backend_in_provenance() -> None:
    from haireder.confidence.reporting import enrich_prediction_row

    row = {
        "pair_id": "pair-1",
        "gold_label": "Temporal/Causal",
        "predicted_label": "Temporal/Causal",
        "confidence": 0.5,
        "method": "3agent_debate",
        "provenance": {"tfidf_hint_used": True},
    }

    enriched = enrich_prediction_row(
        row,
        s1=0.5,
        s2=0.5,
        s3=0.3,
        s3_backend="tfidf_max_similarity",
        raw_confidence=0.45,
        calibrated_confidence=None,
        calibration_mode="fallback",
        tau_low=0.4,
        tau_high=0.8,
    )

    assert enriched["s3_backend"] == "tfidf_max_similarity"
    assert enriched["provenance"]["s3_backend"] == "tfidf_max_similarity"
    assert enriched["provenance"]["_prior_provenance"] == {"tfidf_hint_used": True}


def test_enrich_predictions_from_transcripts_assigns_multi_signal_region() -> None:
    from haireder.confidence.calibration import CalibrationModel
    from haireder.confidence.reporting import enrich_predictions_from_transcripts

    row = {
        "pair_id": "pair-1",
        "repo": "Apache",
        "window": "W5",
        "split": "online",
        "gold_label": "Composition",
        "predicted_label": "Composition",
        "confidence": 0.5,
        "issue_i": {"text": "same text"},
        "issue_j": {"text": "dependency"},
    }
    transcript = {
        "pair_id": "pair-1",
        "rounds": [
            {
                "round_index": 0,
                "proposer": {"parsed_output": {"label": "Composition"}},
                "critic": {"parsed_output": {"label": "Composition"}},
            }
        ],
        "judge": {"parsed_output": {"final_label": "Composition", "verbal_confidence": "high"}},
    }

    enriched = enrich_predictions_from_transcripts(
        [row],
        transcripts_by_pair_id={"pair-1": transcript},
        text_pool=["same text\n\ndependency"],
        confidence_map={"high": 0.9},
        fusion_weights={"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33},
        calibrator=CalibrationModel(mode="fallback"),
        tau_low=0.4,
        tau_high=0.8,
    )

    assert enriched[0]["confidence_region"] == "high"
    assert enriched[0]["s1"] == 0.9
    assert enriched[0]["s2"] == 1.0
    assert enriched[0]["s3"] > 0.0


def test_enrich_predictions_from_transcripts_honors_sbert_backend() -> None:
    from haireder.confidence.calibration import CalibrationModel
    from haireder.confidence.reporting import enrich_predictions_from_transcripts

    class FakeSBERTIndex:
        size = 1

        def knn_average_similarity(self, query_text: str) -> float:
            return 0.42

    row = {
        "pair_id": "pair-1",
        "repo": "Apache",
        "window": "W5",
        "split": "online",
        "gold_label": "Composition",
        "predicted_label": "Composition",
        "confidence": 0.5,
        "issue_i": {"text": "same text"},
        "issue_j": {"text": "dependency"},
    }
    transcript = {
        "pair_id": "pair-1",
        "rounds": [],
        "judge": {"parsed_output": {"final_label": "Composition", "verbal_confidence": "high"}},
    }

    enriched = enrich_predictions_from_transcripts(
        [row],
        transcripts_by_pair_id={"pair-1": transcript},
        text_pool=["ignored by sbert"],
        confidence_map={"high": 0.9},
        fusion_weights={"self_evaluation": 0.34, "inter_agent_agreement": 0.33, "retrieval_similarity": 0.33},
        calibrator=CalibrationModel(mode="fallback"),
        tau_low=0.4,
        tau_high=0.8,
        s3_backend="sbert",
        sbert_index=FakeSBERTIndex(),
    )

    assert enriched[0]["s3"] == 0.42
    assert enriched[0]["s3_backend"] == "sbert_knn"


def test_debate_engine_tags_endorse_mismatch() -> None:
    row = {
        "pair_id": "pair-1",
        "repo": "Apache",
        "window": "W1",
        "split": "train",
        "issue_i": {"id": "A", "text": "Issue A"},
        "issue_j": {"id": "B", "text": "Issue B"},
    }

    def proposer_step(current_row, round_index, transcript):
        return {"raw_output": "{}", "parsed_output": {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"}}

    def critic_step(current_row, proposer_turn, round_index, transcript):
        return {"raw_output": "{}", "parsed_output": {"stance": "endorse", "label": "Composition", "rationale": "r2", "verbal_confidence": "medium"}}

    def judge_step(current_row, transcript):
        return {"raw_output": "{}", "parsed_output": {"final_label": "Temporal/Causal", "justification": "j", "verbal_confidence": "medium"}}

    transcript = run_debate(row, proposer_step, critic_step, judge_step, max_rounds=3)

    assert transcript["termination_reason"] == "round_budget"
    assert transcript["rounds"][0]["critic"]["parsed_output"]["edge_case"] == "endorse_mismatch"


def _phase15_row(pair_id: str) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": "validation",
        "label": "Composition",
        "issue_i": {"id": f"{pair_id}-A", "text": "Login fails after timeout"},
        "issue_j": {"id": f"{pair_id}-B", "text": "Session renewal must handle timeout"},
    }


def _write_phase15_config_and_prompts(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
confidence:
  verbal_confidence_map:
    very_low: 0.1
    low: 0.3
    medium: 0.5
    high: 0.7
    very_high: 0.9
debate:
  max_rounds: 1
""".strip(),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "judge_prompt.md"
    prompt_path.write_text(
        "A: {{issue_i_text}}\nB: {{issue_j_text}}\nDebate: {{debate_transcript_json}}\n{{retrieval_examples}}",
        encoding="utf-8",
    )
    return config_path, prompt_path


def test_adaptive_evaluation_batches_high_confidence_judge_rows(tmp_path, monkeypatch) -> None:
    config_path, prompt_path = _write_phase15_config_and_prompts(tmp_path)
    predictions_out = tmp_path / "predictions.jsonl"
    transcripts_out = tmp_path / "transcripts.jsonl"

    import haireder.training.lora_trainer as lora_trainer

    monkeypatch.setattr(lora_trainer, "load_role_bundle", lambda adapter_path: {"adapter_path": adapter_path})
    batch_calls = []

    def fake_generate_batch(bundle, prompts, max_new_tokens=128, batch_size=None):
        batch_calls.append((len(prompts), batch_size))
        return [
            '{"final_label": "Composition", "justification": "clear", "verbal_confidence": "very_high"}'
            for _ in prompts
        ]

    monkeypatch.setattr(lora_trainer, "generate_role_outputs", fake_generate_batch)

    predictions, transcripts = run_phase15_final_evaluation._run_adaptive_debate_inference(
        test_rows=[_phase15_row("P1"), _phase15_row("P2")],
        auxiliary_by_pair_id={},
        experiment_config_path=config_path,
        predictions_out_path=predictions_out,
        transcripts_out_path=transcripts_out,
        proposer_prompt_path=prompt_path,
        critic_prompt_path=prompt_path,
        judge_prompt_path=prompt_path,
        proposer_adapter_path=tmp_path / "proposer",
        critic_adapter_path=tmp_path / "critic",
        judge_adapter_path=tmp_path / "judge",
        max_new_tokens=16,
        batch_size=2,
        adaptive_high_threshold=0.8,
        force_rerun=True,
    )

    assert batch_calls == [(2, 2)]
    assert [row["pair_id"] for row in predictions] == ["P1", "P2"]
    assert all(row["method"] == "adaptive_judge" for row in predictions)
    assert all(row["provenance"]["full_debate_used"] is False for row in predictions)
    assert all(transcript["termination_reason"] == "adaptive_judge_only" for transcript in transcripts)
    assert predictions_out.exists()
    assert transcripts_out.exists()


def test_adaptive_evaluation_routes_low_confidence_rows_to_full_debate(tmp_path, monkeypatch) -> None:
    config_path, prompt_path = _write_phase15_config_and_prompts(tmp_path)
    predictions_out = tmp_path / "predictions.jsonl"
    transcripts_out = tmp_path / "transcripts.jsonl"

    import haireder.training.lora_trainer as lora_trainer

    monkeypatch.setattr(lora_trainer, "load_role_bundle", lambda adapter_path: {"adapter_path": adapter_path})
    monkeypatch.setattr(
        lora_trainer,
        "generate_role_outputs",
        lambda bundle, prompts, max_new_tokens=128, batch_size=None: [
            '{"final_label": "Composition", "justification": "uncertain", "verbal_confidence": "medium"}'
            for _ in prompts
        ],
    )
    fallback_pair_ids = []

    def fake_full_debate(**kwargs):
        assert kwargs["predictions_out_path"] != predictions_out
        assert kwargs["transcripts_out_path"] != transcripts_out
        assert kwargs["predictions_out_path"].name == "predictions.adaptive_fallback.jsonl"
        assert kwargs["transcripts_out_path"].name == "transcripts.adaptive_fallback.jsonl"
        assert kwargs["force_rerun"] is True
        fallback_pair_ids.extend(row["pair_id"] for row in kwargs["test_rows"])
        row = kwargs["test_rows"][0]
        prediction = {
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "window": row["window"],
            "split": row["split"],
            "gold_label": row["label"],
            "predicted_label": "Composition",
            "confidence": 0.9,
            "method": "3agent_debate",
            "issue_i": row["issue_i"],
            "issue_j": row["issue_j"],
            "raw_output": "{}",
            "parsed_output": {"final_label": "Composition", "verbal_confidence": "very_high"},
            "error": None,
            "provenance": {},
        }
        transcript = {
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "window": row["window"],
            "split": row["split"],
            "rounds": [
                {
                    "round_index": 0,
                    "proposer": {"parsed_output": {"label": "Composition"}},
                    "critic": {"parsed_output": {"label": "Composition", "stance": "endorse"}},
                }
            ],
            "judge": {"parsed_output": {"final_label": "Composition", "verbal_confidence": "very_high"}},
            "termination_reason": "agreement",
        }
        return [prediction], [transcript]

    monkeypatch.setattr(run_phase15_final_evaluation, "_run_test_debate_inference", fake_full_debate)

    predictions, transcripts = run_phase15_final_evaluation._run_adaptive_debate_inference(
        test_rows=[_phase15_row("P1")],
        auxiliary_by_pair_id={},
        experiment_config_path=config_path,
        predictions_out_path=predictions_out,
        transcripts_out_path=transcripts_out,
        proposer_prompt_path=prompt_path,
        critic_prompt_path=prompt_path,
        judge_prompt_path=prompt_path,
        proposer_adapter_path=tmp_path / "proposer",
        critic_adapter_path=tmp_path / "critic",
        judge_adapter_path=tmp_path / "judge",
        max_new_tokens=16,
        batch_size=4,
        adaptive_high_threshold=0.8,
        force_rerun=True,
    )

    assert fallback_pair_ids == ["P1"]
    assert predictions[0]["method"] == "adaptive_full_debate"
    assert predictions[0]["provenance"]["full_debate_used"] is True
    assert transcripts[0]["adaptive_metadata"]["stage"] == "full_debate"
    assert not (tmp_path / "predictions.adaptive_fallback.jsonl").exists()
    assert not (tmp_path / "transcripts.adaptive_fallback.jsonl").exists()


def test_judge_only_confidence_excludes_missing_inter_agent_agreement() -> None:
    confidence = run_phase15_final_evaluation._compute_raw_confidence_without_inter_agent_agreement(
        s1=0.9,
        s3=0.5,
        weights={
            "self_evaluation": 0.34,
            "inter_agent_agreement": 0.33,
            "retrieval_similarity": 0.33,
        },
    )

    assert confidence == pytest.approx(0.7029850746268657)


def test_resolve_phase15_fusion_weights_loads_precomputed_file(tmp_path, monkeypatch) -> None:
    weight_file = tmp_path / "models" / "calibration" / "optimal_weights.json"
    weight_file.parent.mkdir(parents=True)
    weight_file.write_text(
        '{"self_evaluation": 0.2, "inter_agent_agreement": 0.5, "retrieval_similarity": 0.3}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_phase15_final_evaluation, "PROJECT_ROOT", tmp_path)

    experiment_config = {
        "confidence": {
            "weight_search_enabled": True,
            "fusion_weights": {
                "self_evaluation": 0.34,
                "inter_agent_agreement": 0.33,
                "retrieval_similarity": 0.33,
            },
            "weight_search": {"output_path": "models/calibration/optimal_weights.json"},
        }
    }

    result = run_phase15_final_evaluation._resolve_phase15_fusion_weights(experiment_config)

    assert result == {
        "self_evaluation": 0.2,
        "inter_agent_agreement": 0.5,
        "retrieval_similarity": 0.3,
    }


def test_resolve_phase15_fusion_weights_falls_back_to_config_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_phase15_final_evaluation, "PROJECT_ROOT", tmp_path)

    experiment_config = {
        "confidence": {
            "weight_search_enabled": True,
            "fusion_weights": {
                "self_evaluation": 0.11,
                "inter_agent_agreement": 0.22,
                "retrieval_similarity": 0.67,
            },
            "weight_search": {"output_path": "models/calibration/optimal_weights.json"},
        }
    }

    result = run_phase15_final_evaluation._resolve_phase15_fusion_weights(experiment_config)

    assert result == {
        "self_evaluation": 0.11,
        "inter_agent_agreement": 0.22,
        "retrieval_similarity": 0.67,
    }


def test_phase15_help_exposes_adaptive_evaluation_flags() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "--evaluation-mode" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--adaptive-high-threshold" in result.stdout
