#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Platt confidence calibrator from validation predictions.")
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--calibrator-out", required=True)
    parser.add_argument("--calibration-summary-out", required=True)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _row_confidence(row: dict) -> float:
    if row.get("raw_confidence") is not None:
        return float(row["raw_confidence"])
    return float(row.get("confidence") or 0.0)


def main() -> int:
    from haireder.confidence import fit_confidence_calibrator
    from haireder.utils.io import read_jsonl, write_json

    args = parse_args()
    prediction_path = _resolve(args.validation_predictions)
    rows = read_jsonl(prediction_path)
    usable_rows = [
        row for row in rows
        if row.get("gold_label") is not None and row.get("predicted_label") is not None
    ]
    from haireder.experiments.evaluation import fit_bundle, raw_score
    if len(usable_rows) != len(rows):
        raise ValueError("Unusable W4 prediction")
    bundle = fit_bundle(usable_rows)
    raw_confidences = [raw_score(row, bundle["weights"]) for row in usable_rows]
    gold_labels = [str(row["gold_label"]) for row in usable_rows]
    predicted_labels = [str(row["predicted_label"]) for row in usable_rows]
    calibrator = fit_confidence_calibrator(raw_confidences, gold_labels, predicted_labels)

    calibrator_out = _resolve(args.calibrator_out)
    calibrator_out.parent.mkdir(parents=True, exist_ok=True)
    with calibrator_out.open("wb") as handle:
        pickle.dump(calibrator, handle)

    summary = {
        "bundle": bundle,
        "source_predictions": str(prediction_path),
        "row_count": len(usable_rows),
        "confidence_regions": dict(sorted(Counter(str(row.get("confidence_region") or "unknown") for row in usable_rows).items())),
        "calibration": {"mode": calibrator.mode, "reason": calibrator.reason},
    }
    write_json(_resolve(args.calibration_summary_out), summary)
    print(f"Calibrator written: {calibrator_out}")
    print(f"Calibration summary written: {args.calibration_summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
