#!/usr/bin/env python
"""Validate prediction JSONL files against the project prediction schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", help="Prediction JSONL file(s).")
    parser.add_argument("--schema", default="configs/experiments/prediction_schema.json")
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--expected", help="Check one prediction file against its exact labelled input cohort")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with resolve_path(args.schema).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    errors = []
    rows = 0
    bad_confidence = 0
    predicted_labels = set()
    for prediction_path in args.predictions:
        seen = set()
        path = resolve_path(prediction_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    row=json.loads(line)
                except ValueError as exc:
                    errors.append({"path":str(path),"line":line_number,"message":str(exc)})
                    break
                if not isinstance(row,dict) or not isinstance(row.get("pair_id"),str):
                    errors.append({"path":str(path),"line":line_number,"message":"Expected an object with a string pair_id"})
                    break
                key = row.get("pair_id")
                if key in seen:
                    errors.append({"path":str(path),"line":line_number,"message":"Duplicate pair_id"})
                seen.add(key)
                if isinstance(row.get("predicted_label"),str):
                    predicted_labels.add(row["predicted_label"])
                confidence = row.get("confidence")
                if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
                    bad_confidence += 1
                for error in validator.iter_errors(row):
                    errors.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "pair_id": row.get("pair_id"),
                            "message": error.message,
                        }
                    )
                    if len(errors) >= args.max_errors:
                        break
                if len(errors) >= args.max_errors:
                    break
        if not seen:
            errors.append({"path":str(path),"message":"Empty prediction file"})
        if len(errors) >= args.max_errors:
            break

    print("rows", rows)
    print("schema_errors", len(errors))
    print("bad_confidence", bad_confidence)
    print("predicted_labels", sorted(label for label in predicted_labels if label is not None))
    if errors:
        print(json.dumps(errors, ensure_ascii=False, indent=2))
        return 1
    if bad_confidence:
        return 2
    if args.expected:
        from haireder.experiments.evaluation import validate_predictions
        from haireder.utils.io import read_jsonl
        if len(args.predictions)!=1:
            raise ValueError("--expected requires exactly one prediction file")
        validate_predictions(read_jsonl(resolve_path(args.expected)),read_jsonl(resolve_path(args.predictions[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
