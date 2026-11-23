#!/usr/bin/env python
"""Run a tiny end-to-end smoke validation of the formal HAI-ReDeR pipeline.

This diagnostic exercises the earlier component command interfaces with deterministic tiny
splits. The paper workflow with real human handoffs is run_formal_pipeline.py.
It is intended for the server before launching the full formal run.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

LABELS = [
    "None",
    "General Relation",
    "Duplication",
    "Composition",
    "Temporal/Causal",
    "Workflow",
]

SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
SMOKE_PREFIX = "smoke_formal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/formal_qwen_qlora.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/formal_6repos.yaml")
    parser.add_argument("--train-steps", type=int, default=2, help="Max LoRA training steps per smoke training call")
    parser.add_argument("--sample-rows", type=int, default=2, help="Rows sampled by smoke inference/evaluation commands")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation token budget for smoke inference")
    parser.add_argument("--timeout-minutes", type=int, default=40, help="Total smoke run timeout budget")
    parser.add_argument("--refresh-period", type=int, default=1, help="Online windows per refresh round")
    parser.add_argument("--force", action="store_true", help="Overwrite smoke artifacts before running")
    parser.add_argument("--dry-run", action="store_true", help="Print component commands without executing them")
    parser.add_argument("--prepare-only", action="store_true", help="Only write tiny smoke splits and manifest")
    return parser.parse_args()


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _row(window: str, split: str, index: int, label: str) -> dict[str, Any]:
    pair_id = f"{SMOKE_PREFIX}_{window}_{index}_{label.lower().replace('/', '_').replace(' ', '_')}"
    created_day = 1 + index
    return {
        "pair_id": pair_id,
        "repo": "SmokeRepo",
        "window": window,
        "split": split,
        "label": label,
        "gold_label": label,
        "link_type": label,
        "pair_available_at": f"2024-01-{created_day:02d}T00:00:00Z",
        "issue_i": {
            "id": f"{window}-I-{index}",
            "key": f"SMOKE-{window}-I-{index}",
            "text": f"Smoke requirement {index} in {window}: implement {label} source behavior.",
        },
        "issue_j": {
            "id": f"{window}-J-{index}",
            "key": f"SMOKE-{window}-J-{index}",
            "text": f"Smoke requirement {index} in {window}: validate {label} target behavior.",
        },
    }


def _rows_for_window(window: str, split: str, offset: int) -> list[dict[str, Any]]:
    return [
        _row(window, split, offset, LABELS[offset % len(LABELS)]),
        _row(window, split, offset + 1, LABELS[(offset + 1) % len(LABELS)]),
    ]


def build_smoke_splits(force: bool = False) -> dict[str, Path]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": SPLIT_DIR / f"{SMOKE_PREFIX}_train.jsonl",
        "validation": SPLIT_DIR / f"{SMOKE_PREFIX}_validation.jsonl",
        "online": SPLIT_DIR / f"{SMOKE_PREFIX}_online.jsonl",
        "test": SPLIT_DIR / f"{SMOKE_PREFIX}_test.jsonl",
        "manifest": SPLIT_DIR / f"{SMOKE_PREFIX}_manifest.json",
    }
    if not force and all(path.exists() for path in paths.values()):
        return paths

    train_rows = []
    for offset, window in enumerate(("W1", "W2", "W3")):
        train_rows.extend(_rows_for_window(window, "train", offset * 2))

    validation_rows = _rows_for_window("W4", "validation", 10)

    online_rows = []
    for offset, window in enumerate(("W5", "W6"), start=6):
        online_rows.extend(_rows_for_window(window, "online", offset * 2))

    test_rows = []
    for offset, window in enumerate(("W29", "W30", "W31", "W32"), start=12):
        test_rows.extend(_rows_for_window(window, "test", offset * 2))

    _jsonl_write(paths["train"], train_rows)
    _jsonl_write(paths["validation"], validation_rows)
    _jsonl_write(paths["online"], online_rows)
    _jsonl_write(paths["test"], test_rows)

    windows = [
        *(dict(window=w, role="train", count=2) for w in ("W1", "W2", "W3")),
        {"window": "W4", "role": "validation", "count": 2},
        *(dict(window=w, role="online", count=2) for w in ("W5", "W6")),
        *(dict(window=w, role="test", count=2) for w in ("W29", "W30", "W31", "W32")),
    ]
    manifest = {
        "mode": "smoke_formal",
        "description": "Tiny formal-shaped dataset for server pipeline smoke validation.",
        "windows": windows,
        "split_counts": {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "online": len(online_rows),
            "test": len(test_rows),
        },
        "labels": LABELS,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def _cmd_to_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def build_component_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    exp_cfg = args.experiment_config
    model_cfg = args.model_config
    dataset_cfg = args.dataset_config
    train_steps = str(args.train_steps)
    sample_rows = str(args.sample_rows)
    max_new_tokens = str(args.max_new_tokens)
    commands: list[tuple[str, list[str]]] = []

    role_specs = [
        ("proposer", "scripts/training/run_proposer_lora.py", "proposer_prompt.md", []),
        ("critic", "scripts/training/run_critic_lora.py", "critic_prompt.md", ["--data-source", "bootstrap"]),
        ("judge", "scripts/training/run_judge_lora.py", "judge_prompt.md", []),
    ]
    for role, script, prompt, extra in role_specs:
        commands.append((
            f"train_{role}",
            _python_cmd(
                script,
                "--dataset-config", dataset_cfg,
                "--model-config", model_cfg,
                "--train", f"data/processed/splits/{SMOKE_PREFIX}_train.jsonl",
                "--allowed-splits", "train",
                "--prompt", f"prompts/agents/{prompt}",
                "--adapter-out", f"models/adapters/{SMOKE_PREFIX}/{role}",
                "--training-data-out", f"data/processed/splits/{SMOKE_PREFIX}_train_{role}.jsonl",
                "--summary-out", f"outputs/metrics/{SMOKE_PREFIX}_{role}_training.json",
                "--max-steps", train_steps,
                "--checkpoint-save-steps", train_steps,
                "--checkpoint-save-total-limit", "1",
                "--force-retrain",
                *extra,
            ),
        ))

    commands.append((
        "validation_eval",
        _python_cmd(
            "scripts/workflow/run_phase15_final_evaluation.py",
            "--experiment-config", exp_cfg,
            "--model-config", model_cfg,
            "--train", f"data/processed/splits/{SMOKE_PREFIX}_train.jsonl",
            "--online", f"data/processed/splits/{SMOKE_PREFIX}_online.jsonl",
            "--test", f"data/processed/splits/{SMOKE_PREFIX}_validation.jsonl",
            "--predictions-out", f"outputs/predictions/{SMOKE_PREFIX}_val_predictions.jsonl",
            "--transcripts-out", f"outputs/debate_transcripts/{SMOKE_PREFIX}_val_transcripts.jsonl",
            "--metrics-out", f"outputs/metrics/{SMOKE_PREFIX}_val_metrics.json",
            "--table-out", f"outputs/tables/{SMOKE_PREFIX}_val_results.md",
            "--proposer-adapter", f"models/adapters/{SMOKE_PREFIX}/proposer",
            "--critic-adapter", f"models/adapters/{SMOKE_PREFIX}/critic",
            "--judge-adapter", f"models/adapters/{SMOKE_PREFIX}/judge",
            "--proposer-prompt", "prompts/agents/proposer_prompt.md",
            "--critic-prompt", "prompts/agents/critic_prompt.md",
            "--judge-prompt", "prompts/agents/judge_prompt.md",
            "--max-new-tokens", max_new_tokens,
            "--evaluation-mode", "adaptive",
            "--allow-missing-calibrator",
            "--batch-size", "1",
            "--adaptive-high-threshold", "0.8",
            "--sample-rows", sample_rows,
            "--force-rerun",
        ),
    ))
    commands.append((
        "calibration",
        _python_cmd(
            "scripts/workflow/fit_platt_calibrator.py",
            "--validation-predictions", f"outputs/predictions/{SMOKE_PREFIX}_val_predictions.jsonl",
            "--calibrator-out", f"models/calibration/{SMOKE_PREFIX}_platt_scaler.pkl",
            "--calibration-summary-out", f"outputs/metrics/{SMOKE_PREFIX}_confidence_calibration_summary.json",
        ),
    ))

    online_windows = ["W5", "W6"]
    refresh_round = 0
    for i, window in enumerate(online_windows):
        adapter_root = f"models/adapters/{SMOKE_PREFIX}" if refresh_round == 0 else f"models/adapters/{SMOKE_PREFIX}/r{refresh_round}"
        window_input = f"data/processed/splits/{SMOKE_PREFIX}_online_window_{window}.jsonl"
        commands.append((
            f"debate_{window}",
            _python_cmd(
                "scripts/debate/run_debate_inference.py",
                "--experiment-config", exp_cfg,
                "--model-config", model_cfg,
                "--input", window_input,
                "--predictions-out", f"outputs/predictions/{SMOKE_PREFIX}_online_{window}.jsonl",
                "--transcripts-out", f"outputs/debate_transcripts/{SMOKE_PREFIX}_online_{window}.jsonl",
                "--metadata-out", f"outputs/metrics/{SMOKE_PREFIX}_online_{window}_inference.json",
                "--proposer-adapter", f"{adapter_root}/proposer",
                "--critic-adapter", f"{adapter_root}/critic",
                "--judge-adapter", f"{adapter_root}/judge",
                "--max-rows", sample_rows,
                "--sample-rows", sample_rows,
                "--max-new-tokens", max_new_tokens,
                "--force-rerun",
                *([] if i == 0 else ["--pool-path", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_after_{online_windows[i-1]}.jsonl", "--retrieval-k", "1"]),
            ),
        ))
        triage_cmd = _python_cmd(
            "scripts/workflow/run_online_triage.py","--simulate-analyst",
            "--experiment-config", exp_cfg,
            "--window", window,
            "--predictions", f"outputs/predictions/{SMOKE_PREFIX}_online_{window}.jsonl",
            "--transcripts", f"outputs/debate_transcripts/{SMOKE_PREFIX}_online_{window}.jsonl",
            "--pool-out", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_after_{window}.jsonl",
            "--pool-candidates-out", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_candidates_{window}.jsonl",
            "--accepted-out", f"outputs/predictions/{SMOKE_PREFIX}_online_{window}_accepted.jsonl",
            "--summary-out", f"outputs/metrics/{SMOKE_PREFIX}_online_{window}_triage.json",
            "--report-out", f"outputs/tables/{SMOKE_PREFIX}_online_{window}_triage.md",
            "--calibrator-in", f"models/calibration/{SMOKE_PREFIX}_platt_scaler.pkl",
            "--reinference-adapter", f"{adapter_root}/proposer",
            "--role-specific-pool",
        )
        if i > 0:
            triage_cmd.extend(["--previous-pool", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_after_{online_windows[i-1]}.jsonl"])
        commands.append((f"triage_{window}", triage_cmd))

        if (i + 1) % max(1, int(args.refresh_period)) == 0:
            refresh_round += 1
            commands.append((
                f"refresh{refresh_round}_corpus",
                _python_cmd(
                    "scripts/workflow/build_refresh_corpus.py",
                    "--experiment-config", exp_cfg,
                    "--accepted", f"outputs/predictions/{SMOKE_PREFIX}_online_{window}_accepted.jsonl",
                    "--pool", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_after_{window}.jsonl",
                    "--out", f"data/processed/splits/{SMOKE_PREFIX}_refresh_r{refresh_round}.jsonl",
                    "--summary-out", f"outputs/metrics/{SMOKE_PREFIX}_refresh_r{refresh_round}_summary.json",
                    "--report-out", f"outputs/tables/{SMOKE_PREFIX}_refresh_r{refresh_round}.md",
                    "--role-specific",
                    "--disable-augmentation",
                ),
            ))
            for role, script, prompt, extra in role_specs:
                commands.append((
                    f"refresh{refresh_round}_{role}",
                    _python_cmd(
                        script,
                        "--dataset-config", dataset_cfg,
                        "--model-config", model_cfg,
                        "--train", f"data/processed/splits/{SMOKE_PREFIX}_refresh_r{refresh_round}_{role}.jsonl",
                        "--allowed-splits", "online",
                        "--prompt", f"prompts/agents/{prompt}",
                        "--adapter-out", f"models/adapters/{SMOKE_PREFIX}/r{refresh_round}/{role}",
                        "--training-data-out", f"data/processed/splits/{SMOKE_PREFIX}_refresh_r{refresh_round}_train_{role}.jsonl",
                        "--summary-out", f"outputs/metrics/{SMOKE_PREFIX}_refresh_r{refresh_round}_{role}_training.json",
                        "--max-steps", train_steps,
                        "--checkpoint-save-steps", train_steps,
                        "--checkpoint-save-total-limit", "1",
                        "--force-retrain",
                        *extra,
                    ),
                ))

    final_adapter_root = f"models/adapters/{SMOKE_PREFIX}" if refresh_round == 0 else f"models/adapters/{SMOKE_PREFIX}/r{refresh_round}"
    commands.append((
        "final_evaluation",
        _python_cmd(
            "scripts/workflow/run_phase15_final_evaluation.py",
            "--experiment-config", exp_cfg,
            "--model-config", model_cfg,
            "--train", f"data/processed/splits/{SMOKE_PREFIX}_train.jsonl",
            "--online", f"data/processed/splits/{SMOKE_PREFIX}_online.jsonl",
            "--test", f"data/processed/splits/{SMOKE_PREFIX}_test.jsonl",
            "--predictions-out", f"outputs/predictions/{SMOKE_PREFIX}_test_predictions.jsonl",
            "--transcripts-out", f"outputs/debate_transcripts/{SMOKE_PREFIX}_test_transcripts.jsonl",
            "--metrics-out", f"outputs/metrics/{SMOKE_PREFIX}_final_metrics.json",
            "--table-out", f"outputs/tables/{SMOKE_PREFIX}_results.md",
            "--calibrator-in", f"models/calibration/{SMOKE_PREFIX}_platt_scaler.pkl",
            "--calibration-summary", f"outputs/metrics/{SMOKE_PREFIX}_confidence_calibration_summary.json",
            "--proposer-adapter", f"{final_adapter_root}/proposer",
            "--critic-adapter", f"{final_adapter_root}/critic",
            "--judge-adapter", f"{final_adapter_root}/judge",
            "--proposer-prompt", "prompts/agents/proposer_prompt.md",
            "--critic-prompt", "prompts/agents/critic_prompt.md",
            "--judge-prompt", "prompts/agents/judge_prompt.md",
            "--max-new-tokens", max_new_tokens,
            "--evaluation-mode", "adaptive",
            "--batch-size", "1",
            "--adaptive-high-threshold", "0.8",
            "--sample-rows", sample_rows,
            "--triage-summaries", f"outputs/metrics/{SMOKE_PREFIX}_online_W5_triage.json", f"outputs/metrics/{SMOKE_PREFIX}_online_W6_triage.json",
            "--pool-path", f"data/processed/retrieval_pools/{SMOKE_PREFIX}_pool_after_W6.jsonl",
            "--retrieval-k", "1",
            "--force-rerun",
        ),
    ))
    return commands


def _write_online_window_inputs() -> None:
    online_path = SPLIT_DIR / f"{SMOKE_PREFIX}_online.jsonl"
    by_window: dict[str, list[str]] = {"W5": [], "W6": []}
    for line in online_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        window = str(json.loads(line).get("window"))
        if window in by_window:
            by_window[window].append(line)
    for window, lines in by_window.items():
        (SPLIT_DIR / f"{SMOKE_PREFIX}_online_window_{window}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_smoke_outputs() -> None:
    targets = [
        PROJECT_ROOT / "models" / "adapters" / SMOKE_PREFIX,
        PROJECT_ROOT / "models" / "calibration" / f"{SMOKE_PREFIX}_platt_scaler.pkl",
        PROJECT_ROOT / "outputs" / "metrics" / f"{SMOKE_PREFIX}_pipeline_summary.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)


def run_commands(commands: list[tuple[str, list[str]]], *, dry_run: bool, timeout_minutes: int) -> list[dict[str, Any]]:
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for stage, cmd in commands:
        text = _cmd_to_text(cmd)
        stage_started = time.monotonic()
        if dry_run:
            print(f"DRY RUN [{stage}]: {text}")
            results.append({
                "stage": stage,
                "command": text,
                "status": "dry_run",
                "elapsed_seconds": round(time.monotonic() - stage_started, 3),
            })
            continue
        elapsed = time.monotonic() - started
        remaining = timeout_minutes * 60 - elapsed
        if remaining <= 0:
            raise TimeoutError(f"smoke timeout before stage {stage}")
        print(f"RUN [{stage}]: {text}", flush=True)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, timeout=remaining)
        stage_elapsed = time.monotonic() - stage_started
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        results.append({
            "stage": stage,
            "command": text,
            "status": "ok",
            "elapsed_seconds": round(stage_elapsed, 3),
        })
    return results


def write_summary(results: list[dict[str, Any]], *, dry_run: bool) -> Path:
    summary_path = PROJECT_ROOT / "outputs" / "metrics" / f"{SMOKE_PREFIX}_pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "smoke_formal",
        "dry_run": dry_run,
        "stage_count": len(results),
        "stages": results,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()
    if args.train_steps <= 0:
        raise SystemExit("--train-steps must be positive")
    if args.sample_rows <= 0:
        raise SystemExit("--sample-rows must be positive")
    if args.force:
        clean_smoke_outputs()
    paths = build_smoke_splits(force=args.force)
    _write_online_window_inputs()
    print(f"Smoke splits ready: {paths['manifest'].relative_to(PROJECT_ROOT)}")
    if args.prepare_only:
        print("Prepare-only complete.")
        return 0
    commands = build_component_commands(args)
    results = run_commands(commands, dry_run=args.dry_run, timeout_minutes=args.timeout_minutes)
    summary_path = write_summary(results, dry_run=args.dry_run)
    print(f"Smoke summary written: {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
