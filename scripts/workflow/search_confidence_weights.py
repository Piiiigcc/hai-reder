#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search confidence fusion weights on validation predictions.")
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--validation-predictions", default="outputs/predictions/formal_val_predictions.jsonl")
    parser.add_argument("--weights-out", default="models/calibration/optimal_weights.json")
    parser.add_argument("--summary-out", default="outputs/metrics/formal_weight_search.json")
    return parser.parse_args()


def main() -> int:
    from haireder.confidence.weight_search import search_confidence_weights
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    confidence_config = experiment_config.get("confidence") or {}
    search_config = confidence_config.get("weight_search") or {}
    step = float(search_config.get("step", 0.1))
    n_bins = int(search_config.get("n_bins", 10))

    validation_rows = read_jsonl(resolve_scoped_path(args.validation_predictions, allowed_roots=("outputs",)))
    result = search_confidence_weights(validation_rows, step=step, n_bins=n_bins)
    weights = result["best_weights"]

    write_json(resolve_scoped_path(args.weights_out, allowed_roots=("models",)), weights)
    write_json(resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), {
        "validation_predictions": args.validation_predictions,
        "weights_out": args.weights_out,
        "step": step,
        "n_bins": n_bins,
        **result,
    })

    print(f"Validation rows: {result['n_rows']}")
    print(f"Evaluated weight combinations: {result['n_evaluated']}")
    print(f"Best ECE: {result['best_ece']}")
    print(f"Weights written: {args.weights_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
