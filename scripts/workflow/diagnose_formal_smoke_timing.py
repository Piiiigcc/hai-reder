#!/usr/bin/env python
"""Collect per-stage timing diagnostics for the formal smoke pipeline.

This script reuses the exact component command graph from
run_formal_smoke_pipeline.py, but records timing/log artifacts per subprocess so
server runs can identify whether time is dominated by model loading, generation,
SBERT retrieval, or refresh training.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "workflow"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_formal_smoke_pipeline import (  # noqa: E402
    _cmd_to_text,
    _write_online_window_inputs,
    build_component_commands,
    build_smoke_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/formal_qwen_qlora.yaml")
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--sample-rows", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--refresh-period", type=int, default=1)
    parser.add_argument("--timeout-minutes", type=int, default=40)
    parser.add_argument("--stage-filter", default=None, help="Regex for stages to run, e.g. 'debate|eval|triage'")
    parser.add_argument("--output-dir", default="outputs/metrics/formal_smoke_timing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _gpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"timestamp": _utc_now()}
    try:
        import torch

        snapshot["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            snapshot["torch_cuda_device_count"] = torch.cuda.device_count()
            snapshot["torch_cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
            snapshot["torch_cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 2)
            snapshot["torch_cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
    except Exception as exc:  # pragma: no cover - depends on server torch install
        snapshot["torch_error"] = repr(exc)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        snapshot["nvidia_smi_returncode"] = result.returncode
        snapshot["nvidia_smi"] = result.stdout.strip().splitlines()
        if result.stderr.strip():
            snapshot["nvidia_smi_stderr"] = result.stderr.strip()
    except Exception as exc:  # pragma: no cover - nvidia-smi absent locally
        snapshot["nvidia_smi_error"] = repr(exc)
    return snapshot


def _build_smoke_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        experiment_config=args.experiment_config,
        model_config=args.model_config,
        dataset_config="configs/dataset/formal_6repos.yaml",
        train_steps=args.train_steps,
        sample_rows=args.sample_rows,
        max_new_tokens=args.max_new_tokens,
        timeout_minutes=args.timeout_minutes,
        refresh_period=args.refresh_period,
        force=args.force,
        dry_run=args.dry_run,
        prepare_only=False,
    )


def _select_commands(commands: list[tuple[str, list[str]]], stage_filter: str | None) -> list[tuple[str, list[str]]]:
    if not stage_filter:
        return commands
    pattern = re.compile(stage_filter)
    return [(stage, cmd) for stage, cmd in commands if pattern.search(stage)]


def _flag_values(cmd: list[str], flag: str) -> list[str]:
    values: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] != flag:
            i += 1
            continue
        i += 1
        while i < len(cmd) and not cmd[i].startswith("--"):
            values.append(cmd[i])
            i += 1
    return values


def _required_existing_paths(cmd: list[str]) -> list[Path]:
    required: list[Path] = []
    single_file_flags = (
        "--input",
        "--train",
        "--online",
        "--test",
        "--validation-predictions",
        "--predictions",
        "--transcripts",
        "--pool",
        "--pool-path",
        "--calibration-summary",
    )
    for flag in single_file_flags:
        required.extend(PROJECT_ROOT / value for value in _flag_values(cmd, flag) if value)

    if "--allow-missing-calibrator" not in cmd:
        required.extend(PROJECT_ROOT / value for value in _flag_values(cmd, "--calibrator-in") if value)

    required.extend(PROJECT_ROOT / value for value in _flag_values(cmd, "--accepted") if value)
    required.extend(PROJECT_ROOT / value for value in _flag_values(cmd, "--triage-summaries") if value)

    for flag in ("--proposer-adapter", "--critic-adapter", "--judge-adapter", "--reinference-adapter"):
        for adapter_dir in _flag_values(cmd, flag):
            required.append(PROJECT_ROOT / adapter_dir / "metadata.json")
    return required


def _missing_prerequisites(cmd: list[str]) -> list[str]:
    missing = []
    for path in _required_existing_paths(cmd):
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(set(missing))


def _run_stage(stage: str, cmd: list[str], *, output_dir: Path, timeout_seconds: float, dry_run: bool) -> dict[str, Any]:
    command_text = _cmd_to_text(cmd)
    log_path = output_dir / f"{stage}.log"
    record: dict[str, Any] = {
        "stage": stage,
        "command": command_text,
        "started_at": _utc_now(),
        "dry_run": dry_run,
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        "gpu_before": _gpu_snapshot(),
    }
    started = time.perf_counter()
    if dry_run:
        record.update({
            "status": "dry_run",
            "returncode": 0,
            "elapsed_seconds": 0.0,
            "gpu_after": _gpu_snapshot(),
        })
        log_path.write_text(command_text + "\n", encoding="utf-8")
        return record

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {command_text}\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            record["status"] = "timeout"
        finally:
            elapsed = time.perf_counter() - started
            record["elapsed_seconds"] = round(elapsed, 3)
            record["returncode"] = process.returncode
            record["gpu_after"] = _gpu_snapshot()
    if record.get("status") != "timeout":
        record["status"] = "ok" if process.returncode == 0 else "failed"
    return record


def main() -> int:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    build_smoke_splits(force=args.force)
    _write_online_window_inputs()
    commands = _select_commands(build_component_commands(_build_smoke_args(args)), args.stage_filter)
    if not commands:
        raise SystemExit(f"no smoke stages matched --stage-filter={args.stage_filter!r}")

    report_path = output_dir / "timing_report.jsonl"
    summary_path = output_dir / "timing_summary.json"
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with report_path.open("w", encoding="utf-8") as report:
        for stage, cmd in commands:
            remaining = args.timeout_minutes * 60 - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError(f"timing diagnostic timeout before stage {stage}")
            print(f"DIAG [{stage}]: {_cmd_to_text(cmd)}", flush=True)
            missing = [] if args.dry_run else _missing_prerequisites(cmd)
            if missing:
                log_path = output_dir / f"{stage}.log"
                log_path.write_text(
                    "Blocked before execution: missing prerequisite files\n"
                    + "\n".join(missing)
                    + "\n",
                    encoding="utf-8",
                )
                record = {
                    "stage": stage,
                    "command": _cmd_to_text(cmd),
                    "started_at": _utc_now(),
                    "dry_run": args.dry_run,
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                    "status": "blocked_missing_prerequisite",
                    "returncode": None,
                    "elapsed_seconds": 0.0,
                    "missing_prerequisites": missing,
                    "gpu_before": _gpu_snapshot(),
                    "gpu_after": _gpu_snapshot(),
                }
            else:
                record = _run_stage(stage, cmd, output_dir=output_dir, timeout_seconds=remaining, dry_run=args.dry_run)
            records.append(record)
            report.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.flush()
            print(f"  -> {record['status']} in {record['elapsed_seconds']:.1f}s", flush=True)
            if record["status"] not in {"ok", "dry_run"} and not args.continue_on_error:
                break

    summary = {
        "started_at": _utc_now(),
        "dry_run": args.dry_run,
        "stage_filter": args.stage_filter,
        "stage_count": len(records),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
        "slowest_stages": sorted(
            [
                {"stage": record["stage"], "elapsed_seconds": record["elapsed_seconds"], "status": record["status"]}
                for record in records
            ],
            key=lambda item: item["elapsed_seconds"],
            reverse=True,
        ),
        "report_path": str(report_path.relative_to(PROJECT_ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Timing report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Timing summary: {summary_path.relative_to(PROJECT_ROOT)}")
    return 0 if all(record["status"] in {"ok", "dry_run"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
