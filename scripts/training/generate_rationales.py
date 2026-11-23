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
    parser = argparse.ArgumentParser(description="Generate metadata-aware rationales for training rows.")
    parser.add_argument("--train-plus-online", default="data/processed/splits/minimal_train_plus_online.jsonl")
    parser.add_argument(
        "--out",
        default="data/processed/rationales/train_plus_online_rationales.jsonl",
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def build_rationale(row: dict) -> str:
    label = str(row.get("label") or "Unknown")
    raw_link_types = []
    value = row.get("raw_link_types")
    if isinstance(value, list):
        raw_link_types.extend(str(item) for item in value if item is not None)
    raw_link_type = row.get("raw_link_type")
    if raw_link_type is not None:
        raw_link_types.append(str(raw_link_type))

    unique_link_types = list(dict.fromkeys(raw_link_types))
    if unique_link_types:
        joined = ", ".join(unique_link_types)
        return f"Label {label} based on Jira dependency metadata: raw link type(s) {joined}."
    return f"Label {label} based on Jira dependency metadata."


def main() -> int:
    from haireder.utils.config import resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_jsonl

    args = parse_args()
    rows = read_jsonl(Path(args.train_plus_online))
    out_path = resolve_scoped_path(args.out, allowed_roots=("data",))
    existing_rows = [] if args.force_rerun or not out_path.exists() else read_jsonl(out_path)
    existing_by_pair_id = {str(row.get("pair_id")): row for row in existing_rows}
    output_by_pair_id = dict(existing_by_pair_id)
    new_rows = 0
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id in output_by_pair_id:
            continue
        output_row = dict(row)
        output_row["rationale"] = build_rationale(row)
        output_by_pair_id[pair_id] = output_row
        new_rows += 1

    output_rows = [
        output_by_pair_id[str(row.get("pair_id") or "")]
        for row in rows
        if str(row.get("pair_id") or "") in output_by_pair_id
    ]
    write_jsonl(out_path, output_rows)
    print(f"Wrote {len(output_rows)} rationales to {out_path}")
    print(f"Reused rows: {len(existing_by_pair_id)}")
    print(f"Newly processed rows: {new_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
