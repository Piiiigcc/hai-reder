#!/usr/bin/env python3
"""Run or resume a content-verified paper experiment. Human review queues pause individual cases."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from haireder.data.formal import build_formal, file_hash
from haireder.experiments.review import PendingReviews, fingerprint
from haireder.experiments.stages import RunBusy, run_lock
from haireder.experiments.runner import execute, plans
from haireder.taxonomy import load_taxonomy
from haireder.utils.config import load_yaml, resolve_path
from haireder.utils.io import write_json


BASELINES = ("tfidf", "bert", "zero_shot")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-config", default="configs/experiments/formal_full_pipeline.yaml"
    )
    p.add_argument("--dataset-config")
    p.add_argument("--model-config")
    p.add_argument("--dataset-dir", default="data/processed/splits")
    p.add_argument("--run-dir", default="outputs/formal_experiments")
    p.add_argument(
        "--suite",
        choices=["full", "single", "static", "no_augmentation", "ablations", "grid"],
        default="full",
    )
    p.add_argument("--low", type=float)
    p.add_argument("--high", type=float)
    p.add_argument("--period", type=int)
    p.add_argument("--log-dir")
    p.add_argument("--checkpoint-path")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed stage artifacts",
    )
    p.add_argument(
        "--force", action="store_true", help="Use a new timestamped run directory"
    )
    p.add_argument(
        "--from-stage",
        choices=[
            "training",
            "calibration",
            "online_learning",
            "final_evaluation",
            "baselines",
        ],
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plan-out")
    p.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Per-configuration wall time in seconds (0 means unlimited)",
    )
    p.add_argument("--without-baselines", action="store_true")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--aggregate-only", action="store_true")
    return p.parse_args()


def run_subprocess(command, path, timeout):
    with path.open("a", buffering=1) as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=ROOT,
        )
        try:
            return process.wait(timeout=timeout or None)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            return 124


def record_failure(directory, **details):
    path = directory / "status.json"
    try:
        with run_lock(directory):
            if (
                path.exists()
                and json.loads(path.read_text()).get("status") == "complete"
            ):
                return
            write_json(path, details)
    except RunBusy:
        # Another invocation acquired the directory after this attempt stopped.
        pass


def verify_result_stage(directory, identity, name, stage, required):
    artifacts = stage["artifacts"]
    if (
        not required <= artifacts.keys()
        or stage["commit"] != fingerprint([identity, name, stage["inputs"], artifacts])
        or any(
            not (directory / key).is_file() or file_hash(directory / key) != value
            for key, value in artifacts.items()
        )
    ):
        raise ValueError(f"{name} fingerprint mismatch in {directory}")


def aggregate(root, plan, *, require_baselines=False):
    rows = []
    for case in plan:
        directory = root / case["name"]
        status_file = directory / "status.json"
        status = (
            json.loads(status_file.read_text())
            if status_file.exists()
            else {"status": "unexecuted"}
        )
        row = dict(case, status=status["status"])
        if row["status"] == "complete":
            checkpoint = directory / status.get("checkpoint", "checkpoint.json")
            config = directory / "configuration.json"
            if (
                not checkpoint.is_file()
                or not config.is_file()
                or file_hash(checkpoint) != status.get("checkpoint_sha256")
                or file_hash(config) != status.get("configuration_sha256")
            ):
                raise ValueError(f"Unverified complete run in {directory}")
            verify = json.loads(checkpoint.read_text())
            configured = json.loads(config.read_text())
            stages = verify["stages"]
            if (
                verify["identity"] != status["identity"]
                or configured["identity"] != status["identity"]
                or configured["plan"] != case
            ):
                raise ValueError(
                    f"Final evaluation fingerprint mismatch in {directory}"
                )
            verify_result_stage(
                directory,
                status["identity"],
                "final_evaluation",
                stages["final_evaluation"],
                {"summary.json", "test_predictions.jsonl"},
            )
            result = json.loads((directory / "summary.json").read_text())
            if result["plan"] != case:
                raise ValueError(f"Final evaluation plan mismatch in {directory}")
            row.update(
                macro_f1=result["held_out"]["macro_f1"],
                ece=result["held_out"].get("ece"),
                aurc=result["held_out"]["aurc"],
                review_count=result["workload"]["total_review_count"],
                cohort=result["cohort"],
                elapsed_compute_seconds=sum(
                    v["elapsed_seconds"]
                    for k, v in verify["stages"].items()
                    if k not in {"tfidf", "bert", "zero_shot"}
                ),
                refresh_rounds=sum(k.endswith("_refresh") for k in verify["stages"]),
            )
            if require_baselines and case["variant"] == "full":
                missing = []
                for baseline in BASELINES:
                    if baseline not in stages:
                        missing.append(baseline)
                        continue
                    verify_result_stage(
                        directory,
                        status["identity"],
                        baseline,
                        stages[baseline],
                        {
                            f"baselines/{baseline}/{filename}"
                            for filename in (
                                "metrics.json",
                                "predictions.jsonl",
                                "supervision.json",
                            )
                        },
                    )
                if missing:
                    row.update(status="awaiting_baselines", missing_baselines=missing)
        rows.append(row)
    measured = [r for r in rows if r["status"] == "complete"]
    if len({r["cohort"] for r in measured}) > 1:
        raise ValueError("Suite results do not share a held-out cohort")
    for r in measured:
        r["pareto_f1_review_count"] = not any(
            other["macro_f1"] >= r["macro_f1"]
            and other["review_count"] <= r["review_count"]
            and (
                other["macro_f1"] > r["macro_f1"]
                or other["review_count"] < r["review_count"]
            )
            for other in measured
        )
    interactions = []
    if len(plan) == 64:
        lookup = {(r["low"], r["high"], r["period"]): r for r in measured}
        # Difference of threshold effects at two refresh periods; only complete matched four-cell contrasts.
        for axis, levels in (
            ("low", (0.1, 0.2, 0.3, 0.4)),
            ("high", (0.6, 0.7, 0.8, 0.9)),
        ):
            for fixed in (
                (0.6, 0.7, 0.8, 0.9) if axis == "low" else (0.1, 0.2, 0.3, 0.4)
            ):
                for a, b in zip(levels, levels[1:]):
                    for k, j in ((2, 4), (4, 6), (6, 8)):
                        keys = [
                            (t, fixed, n) if axis == "low" else (fixed, t, n)
                            for t, n in ((b, j), (a, j), (b, k), (a, k))
                        ]
                        if all(key in lookup for key in keys):
                            four = [lookup[key] for key in keys]
                            effects = {
                                m: four[0][m] - four[1][m] - four[2][m] + four[3][m]
                                for m in ("macro_f1", "ece", "aurc", "review_count")
                            }
                            interactions.append(
                                dict(
                                    axis=axis,
                                    fixed=fixed,
                                    levels=[a, b],
                                    periods=[k, j],
                                    effects=effects,
                                )
                            )
    return dict(
        configurations=rows,
        completed=len(measured),
        total=len(rows),
        interactions=interactions,
    )


def main():
    args = parse_args()
    cfg = load_yaml(args.experiment_config)
    data_path = resolve_path(args.dataset_config or cfg["dataset_config"])
    model = load_yaml(args.model_config or cfg["model_config"])
    all_plans = plans(
        args.suite,
        args.low if args.low is not None else cfg["confidence"]["tau_low"],
        args.high if args.high is not None else cfg["confidence"]["tau_high"],
        args.period
        if args.period is not None
        else cfg["online_learning"]["refresh_period_sprints"],
    )
    if args.checkpoint_path and len(all_plans) != 1:
        raise ValueError("Use one checkpoint per configuration")
    if args.dry_run:
        doc = dict(status="plan_only", configurations=all_plans)
        if args.plan_out:
            write_json(resolve_path(args.plan_out), doc)
        print(json.dumps(doc, indent=2))
        return 0
    root = resolve_path(args.run_dir)
    if args.force and args.worker is None:
        root = root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=True)
    with_baselines = not args.without_baselines and args.suite != "grid"
    if args.aggregate_only:
        write_json(
            root / (args.suite + "_summary.json"),
            aggregate(root, all_plans, require_baselines=with_baselines),
        )
        return 0
    if args.worker is None:
        codes = []
        for i, case in enumerate(all_plans):
            log = resolve_path(args.log_dir) if args.log_dir else root / "logs"
            log.mkdir(parents=True, exist_ok=True)
            directory = root / case["name"]
            command = [
                sys.executable,
                __file__,
                *sys.argv[1:],
                "--worker",
                str(i),
                "--run-dir",
                str(root),
            ]
            code = run_subprocess(command, log / (case["name"] + ".log"), args.timeout)
            if code == 124:
                record_failure(directory, status="timeout", seconds=args.timeout)
            print(
                f"{case['name']}: exit {code}; {log / (case['name'] + '.log')}",
                flush=True,
            )
            codes.append(code)
        write_json(
            root / (args.suite + "_summary.json"),
            aggregate(root, all_plans, require_baselines=with_baselines),
        )
        return max(codes, default=0)
    case = all_plans[args.worker]
    directory = root / case["name"]
    try:
        from haireder.training.lora_trainer import _prepare_device

        device = _prepare_device(model)
        if model.get("quantization", {}).get("enabled") and device != "cuda":
            raise RuntimeError(
                "Formal NF4 QLoRA requires CUDA. Run on a CUDA host or use a separately identified local control configuration."
            )
        canonical_dir = resolve_path(args.dataset_dir)
        splits = canonical_dir
        if case["negatives"] == "random":
            splits = splits / "formal_random"
        mpath = splits / "formal_manifest.json"
        if not mpath.exists():
            data = load_yaml(data_path)
            history = resolve_path(data["raw_sources"]["creation_history"])
            if not history.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/data/build_creation_text.py"),
                        "--dataset-config",
                        str(data_path),
                        "--out",
                        str(history),
                    ],
                    check=True,
                )
            taxonomy = resolve_path(
                "data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml"
            )
            build_formal(
                data,
                load_taxonomy(taxonomy),
                ROOT,
                splits,
                data_path,
                taxonomy,
                case["negatives"],
                shared_test_dir=canonical_dir,
            )
        m = json.loads(mpath.read_text())
        if file_hash(data_path) not in m["inputs"].values():
            raise ValueError(
                "Requested dataset configuration differs from the published cohort"
            )
        if case["negatives"] == "random" and m["artifacts"][
            "formal_test.jsonl"
        ] != file_hash(canonical_dir / "formal_test.jsonl"):
            raise ValueError(
                "Random ablation must share the selected canonical test cohort"
            )
        execute(
            case,
            directory,
            splits,
            cfg,
            model,
            from_stage=args.from_stage,
            checkpoint=args.checkpoint_path,
            with_baselines=with_baselines,
            device=device,
        )
        return 0
    except RunBusy as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except PendingReviews as exc:
        record_failure(
            directory,
            status="awaiting_human_review",
            csv=str(exc.path),
            pending=exc.pending,
            total=exc.total,
        )
        print(str(exc), flush=True)
        return 3
    except Exception as exc:
        import traceback

        record_failure(directory, status="blocked", reason=str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
