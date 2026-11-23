from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "workflow" / "run_formal_smoke_pipeline.py"


def _cleanup_smoke_artifacts() -> None:
    paths = [
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_train.jsonl",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_validation.jsonl",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_online.jsonl",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_test.jsonl",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_manifest.json",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_online_window_W5.jsonl",
        PROJECT_ROOT / "data" / "processed" / "splits" / "smoke_formal_online_window_W6.jsonl",
        PROJECT_ROOT / "outputs" / "metrics" / "smoke_formal_pipeline_summary.json",
    ]
    for path in paths:
        path.unlink(missing_ok=True)


def test_help_shows_smoke_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "--train-steps" in result.stdout
    assert "--timeout-minutes" in result.stdout
    assert "--prepare-only" in result.stdout
    assert "--dry-run" in result.stdout


def test_prepare_only_writes_tiny_formal_splits() -> None:
    _cleanup_smoke_artifacts()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--prepare-only", "--force"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
        train_rows = [json.loads(line) for line in (split_dir / "smoke_formal_train.jsonl").read_text(encoding="utf-8").splitlines()]
        online_rows = [json.loads(line) for line in (split_dir / "smoke_formal_online.jsonl").read_text(encoding="utf-8").splitlines()]
        test_rows = [json.loads(line) for line in (split_dir / "smoke_formal_test.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((split_dir / "smoke_formal_manifest.json").read_text(encoding="utf-8"))

        assert {row["window"] for row in train_rows} == {"W1", "W2", "W3"}
        assert {row["window"] for row in online_rows} == {"W5", "W6"}
        assert {row["window"] for row in test_rows} == {"W29", "W30", "W31", "W32"}
        assert {row["split"] for row in train_rows} == {"train"}
        assert all(row["issue_i"]["text"] and row["issue_j"]["text"] for row in train_rows + online_rows + test_rows)
        assert manifest["mode"] == "smoke_formal"
        assert manifest["split_counts"]["online"] == len(online_rows)
    finally:
        _cleanup_smoke_artifacts()


def test_dry_run_exercises_full_formal_component_graph() -> None:
    _cleanup_smoke_artifacts()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run", "--force", "--train-steps", "1", "--sample-rows", "1"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        output = result.stdout
        assert "scripts/training/run_proposer_lora.py" in output
        assert "scripts/training/run_critic_lora.py" in output
        assert "scripts/training/run_judge_lora.py" in output
        assert "scripts/workflow/run_phase15_final_evaluation.py" in output
        assert "scripts/workflow/fit_platt_calibrator.py" in output
        assert "scripts/debate/run_debate_inference.py" in output
        assert "scripts/workflow/run_online_triage.py" in output
        assert "scripts/workflow/build_refresh_corpus.py" in output
        assert "--dataset-config configs/dataset/formal_6repos.yaml" in output
        assert "models/adapters/smoke_formal/r1/proposer" in output
        assert "--triage-summaries outputs/metrics/smoke_formal_online_W5_triage.json outputs/metrics/smoke_formal_online_W6_triage.json" in output
    finally:
        _cleanup_smoke_artifacts()


def test_smoke_generation_budget_matches_debate_runner_default() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run", "--force", "--train-steps", "1", "--sample-rows", "1"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "--max-new-tokens 128" in result.stdout
    assert "--max-new-tokens 32" not in result.stdout


def test_dry_run_summary_records_stage_elapsed_seconds() -> None:
    _cleanup_smoke_artifacts()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--dry-run", "--force", "--train-steps", "1", "--sample-rows", "1"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        summary_path = PROJECT_ROOT / "outputs" / "metrics" / "smoke_formal_pipeline_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["stage_count"] == len(summary["stages"])
        assert summary["stages"]
        assert all("elapsed_seconds" in stage for stage in summary["stages"])
        assert all(stage["elapsed_seconds"] >= 0 for stage in summary["stages"])
    finally:
        _cleanup_smoke_artifacts()


def test_timing_diagnostic_dry_run_writes_report_and_summary(tmp_path: Path) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "metrics" / f"pytest_timing_{tmp_path.name}"
    if output_dir.exists():
        for path in output_dir.glob("*"):
            path.unlink()
        output_dir.rmdir()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/workflow/diagnose_formal_smoke_timing.py",
                "--dry-run",
                "--force",
                "--stage-filter",
                "debate_W5",
                "--train-steps",
                "1",
                "--sample-rows",
                "1",
                "--output-dir",
                str(output_dir.relative_to(PROJECT_ROOT)),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        report_path = output_dir / "timing_report.jsonl"
        summary_path = output_dir / "timing_summary.json"
        assert report_path.exists()
        assert summary_path.exists()
        records = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert [record["stage"] for record in records] == ["debate_W5"]
        assert records[0]["status"] == "dry_run"
        assert records[0]["elapsed_seconds"] >= 0
        assert "gpu_before" in records[0]
        assert summary["stage_count"] == 1
        assert summary["slowest_stages"][0]["stage"] == "debate_W5"
    finally:
        if output_dir.exists():
            for path in output_dir.glob("*"):
                path.unlink()
            output_dir.rmdir()


def test_debate_latency_profiler_dry_run_writes_profile(tmp_path: Path) -> None:
    _cleanup_smoke_artifacts()
    output_path = PROJECT_ROOT / "outputs" / "metrics" / f"pytest_profile_{tmp_path.name}.json"
    output_path.unlink(missing_ok=True)
    try:
        prep = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--prepare-only", "--force"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert prep.returncode == 0, prep.stderr

        result = subprocess.run(
            [
                sys.executable,
                "scripts/workflow/profile_debate_inference_latency.py",
                "--dry-run",
                "--input",
                "data/processed/splits/smoke_formal_online_window_W5.jsonl",
                "--output",
                str(output_path.relative_to(PROJECT_ROOT)),
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        profile = json.loads(output_path.read_text(encoding="utf-8"))
        assert profile["status"] == "dry_run"
        assert profile["pair_id"].startswith("smoke_formal_W5")
        assert profile["steps"] == []
    finally:
        output_path.unlink(missing_ok=True)
        _cleanup_smoke_artifacts()


def test_timing_diagnostic_preflight_reports_missing_adapter_metadata() -> None:
    import importlib.util

    script_path = PROJECT_ROOT / "scripts" / "workflow" / "diagnose_formal_smoke_timing.py"
    spec = importlib.util.spec_from_file_location("diagnose_formal_smoke_timing", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    missing = module._missing_prerequisites([
        sys.executable,
        "scripts/workflow/run_phase15_final_evaluation.py",
        "--test",
        "data/processed/splits/definitely_missing_test.jsonl",
        "--proposer-adapter",
        "models/adapters/definitely_missing/proposer",
        "--critic-adapter",
        "models/adapters/definitely_missing/critic",
        "--judge-adapter",
        "models/adapters/definitely_missing/judge",
        "--calibrator-in",
        "models/calibration/definitely_missing.pkl",
        "--triage-summaries",
        "outputs/metrics/definitely_missing_triage.json",
    ])

    assert "data/processed/splits/definitely_missing_test.jsonl" in missing
    assert "models/adapters/definitely_missing/judge/metadata.json" in missing
    assert "models/calibration/definitely_missing.pkl" in missing
    assert "outputs/metrics/definitely_missing_triage.json" in missing


def test_final_evaluation_uses_initial_adapters_when_no_refresh_round() -> None:
    import importlib.util
    import argparse

    spec = importlib.util.spec_from_file_location("run_formal_smoke_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    commands = module.build_component_commands(argparse.Namespace(
        experiment_config="configs/experiments/formal_full_pipeline.yaml",
        model_config="configs/models/formal_qwen_qlora.yaml",
        dataset_config="configs/dataset/formal_6repos.yaml",
        train_steps=1,
        sample_rows=1,
        max_new_tokens=64,
        timeout_minutes=40,
        refresh_period=99,
        force=False,
        dry_run=True,
        prepare_only=False,
    ))
    final_cmd = dict(commands)["final_evaluation"]

    assert "models/adapters/smoke_formal/proposer" in final_cmd
    assert "models/adapters/smoke_formal/r0/proposer" not in final_cmd
