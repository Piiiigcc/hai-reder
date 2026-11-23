#!/usr/bin/env python
"""Evaluate HAI-ReDeR prediction JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.evaluation.metrics import compute_metrics_by_split, render_metrics_markdown
from haireder.utils.config import resolve_path
from haireder.utils.io import read_jsonl, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="Prediction JSONL file(s).")
    parser.add_argument("--json-out", default="outputs/metrics/evaluation_metrics.json")
    parser.add_argument("--md-out", default="outputs/tables/evaluation_metrics.md")
    parser.add_argument("--title", default="Evaluation Metrics")
    parser.add_argument(
        "--compute-ece",
        action="store_true",
        help="Compute ECE for methods with calibrated confidence outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.predictions:
        rows.extend(read_jsonl(resolve_path(path)))
    report = compute_metrics_by_split(rows, compute_ece=args.compute_ece)
    write_json(PROJECT_ROOT / args.json_out, report)
    write_text(PROJECT_ROOT / args.md_out, render_metrics_markdown(report, title=args.title))
    print(f"Metrics JSON written: {args.json_out}")
    print(f"Metrics Markdown written: {args.md_out}")
    print("Overall macro-F1:", report["overall"]["macro_f1"])
    print("Overall accuracy:", report["overall"]["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
