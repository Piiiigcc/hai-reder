#!/usr/bin/env python
"""Build capped minimal positive/negative pair JSONL files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.data.pair_construction import (
    build_window_issue_pools_from_requirements,
    construct_minimal_pairs,
    read_jsonl,
    render_pair_distribution_markdown,
)
from haireder.data.windowing import load_processed_payload
from haireder.utils.config import load_yaml, resolve_path
from haireder.utils.io import write_json, write_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument("--positive-in", default="data/processed/splits/minimal_positive_pairs.jsonl")
    parser.add_argument("--pairs-out", default="data/processed/pairs/minimal_pairs.jsonl")
    parser.add_argument("--json-out", default="outputs/logs/minimal_pair_construction.json")
    parser.add_argument("--md-out", default="outputs/tables/minimal_pair_distribution.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(resolve_path(args.dataset_config))
    minimal = config["minimal_run"]
    positive_rows = read_jsonl(resolve_path(args.positive_in))
    processed_path = resolve_path(config["raw_sources"]["processed_pairs_json"], PROJECT_ROOT)
    payload = load_processed_payload(processed_path)
    windows_path = resolve_path("data/interim/sprint_windows/minimal_windows.json")

    with windows_path.open("r", encoding="utf-8") as handle:
        window_report = json.load(handle)

    rng = random.Random(int(minimal["seed"]))
    issue_pools = build_window_issue_pools_from_requirements(
        requirements=payload.get("requirements", []),
        repository=minimal["selected_repository"],
        window_report=window_report,
        max_candidates_per_window=int(minimal["max_negative_candidate_issues_per_window"]),
        rng=rng,
    )
    pairs, report = construct_minimal_pairs(
        positive_rows=positive_rows,
        issue_pools_by_window=issue_pools,
        max_pairs_per_window=int(minimal["max_pairs_per_window"]),
        negative_ratio=float(minimal["negative_ratio"]),
        seed=int(minimal["seed"]),
    )

    write_jsonl(PROJECT_ROOT / args.pairs_out, pairs)
    write_json(PROJECT_ROOT / args.json_out, report)
    write_text(PROJECT_ROOT / args.md_out, render_pair_distribution_markdown(report))

    split_to_rows: dict[str, list[dict]] = {"train": [], "validation": [], "online": [], "test": []}
    for row in pairs:
        split = row["split"]
        if split in split_to_rows:
            split_to_rows[split].append(row)
    for split, rows in split_to_rows.items():
        write_jsonl(PROJECT_ROOT / f"data/processed/splits/minimal_{split}.jsonl", rows)

    print(f"Minimal pairs JSONL written: {args.pairs_out}")
    print(f"Pair construction JSON written: {args.json_out}")
    print(f"Pair distribution Markdown written: {args.md_out}")
    print(f"Total pairs: {report['total_pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
