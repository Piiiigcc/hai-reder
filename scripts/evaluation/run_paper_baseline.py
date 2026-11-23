"""Run a paper baseline independently of the debate; trainable comparators require all accepted online decisions."""

import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from haireder.experiments.baselines import run_baseline, zero_mlx_predictions
from haireder.experiments.evaluation import validate_predictions, evaluation
from haireder.experiments.review import fingerprint
from haireder.experiments.runner import dataset_files
from haireder.utils.config import load_yaml
from haireder.utils.io import read_jsonl, write_jsonl, write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=["tfidf", "bert", "zero_shot"])
    p.add_argument("--data", type=Path, default=root / "data/processed/splits")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--accepted", type=Path, nargs="*", default=[])
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "mlx"], default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        help="Diagnostic prefix only; the result records restricted coverage",
    )
    args = p.parse_args()
    _, splits = dataset_files(args.data)
    test = splits["test"]
    total = len(test)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        test = test[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.method == "zero_shot" and args.device == "mlx":
        model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"
        from haireder.utils.model_versions import revision_kwargs
        import json
        import inspect
        from haireder.experiments.baselines import zero_prefix

        identity = fingerprint(
            [
                test,
                model_name,
                revision_kwargs(model_name),
                args.batch_size,
                inspect.getsource(zero_mlx_predictions),
                inspect.getsource(zero_prefix),
            ]
        )
        progress = args.out / "chunks"
        progress.mkdir(exist_ok=True)
        receipt = progress / "cohort.json"
        if (
            receipt.exists()
            and json.loads(receipt.read_text())["hash"] != identity
            and any(progress.glob("*.jsonl"))
        ):
            raise ValueError("Zero-shot inference cohort or model changed")
        write_json(receipt, {"hash": identity})
        rows = []
        details = {}
        for start in range(0, len(test), 16):
            chunk = test[start : start + 16]
            path = progress / f"{start:06}.jsonl"
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists():
                found = read_jsonl(path)
                info = json.loads(meta.read_text())
                if fingerprint(found) != info["hash"]:
                    raise ValueError("Zero-shot artifact content changed")
                details = info["method"]
            else:
                found, details = zero_mlx_predictions(
                    chunk, model_name, batch_size=args.batch_size
                )
                validate_predictions(chunk, found)
                write_jsonl(path, found)
                write_json(meta, {"hash": fingerprint(found), "method": details})
            validate_predictions(chunk, found)
            rows.extend(found)
            print(
                f"Verified {len(rows)}/{len(test)} frozen held-out predictions",
                flush=True,
            )
        validate_predictions(test, rows)
        write_jsonl(args.out / "predictions.jsonl", rows)
        write_json(args.out / "metrics.json", evaluation(rows))
        write_json(
            args.out / "supervision.json",
            dict(
                details,
                test_hash=fingerprint(test),
                test_total=total,
                evaluated=len(rows),
                test_scope="full" if len(rows) == total else "diagnostic_subset",
                training_count=0,
            ),
        )
    else:
        train = splits["train"]
        accepted = [r for path in args.accepted for r in read_jsonl(path)]
        if args.method != "zero_shot":
            online = splits["online"]
            if {r["pair_id"] for r in accepted} != {
                r["pair_id"] for r in online
            } or len(online) != len(accepted):
                raise ValueError(
                    "Cumulative baseline requires all finalized online decisions from the same cohort"
                )
            original = {r["pair_id"]: r for r in online}
            if any(
                r.get(k) != original[r["pair_id"]][k]
                for r in accepted
                for k in ("repo", "window", "split", "issue_i", "issue_j")
            ):
                raise ValueError(
                    "Online supervision does not match the frozen endpoints"
                )
        cfg = (
            load_yaml("configs/models/formal_qwen_qlora.yaml")
            if args.device == "cuda"
            else None
        )
        run_baseline(
            args.method,
            train,
            accepted,
            test,
            args.out,
            device=args.device,
            model_cfg=cfg,
            batch_size=args.batch_size,
            test_total=total,
        )
