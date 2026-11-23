#!/usr/bin/env python
"""Build minimal chronological windows from the processed Public Jira pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.data.windowing import (
    assign_quantile_windows,
    build_positive_pairs,
    load_processed_payload,
    render_window_distribution_markdown,
    window_distribution,
)
from haireder.taxonomy import load_taxonomy
from haireder.utils.config import load_yaml, resolve_path
from haireder.utils.io import write_json, write_jsonl, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument(
        "--taxonomy",
        default="data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml",
    )
    parser.add_argument("--windows-out", default="data/interim/sprint_windows/minimal_windows.json")
    parser.add_argument("--pairs-out", default="data/processed/splits/minimal_positive_pairs.jsonl")
    parser.add_argument("--json-out", default="outputs/logs/minimal_window_split.json")
    parser.add_argument("--md-out", default="outputs/tables/minimal_window_distribution.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(resolve_path(args.dataset_config))
    minimal_config = config["minimal_run"]
    repository = minimal_config["selected_repository"]
    processed_path = resolve_path(config["raw_sources"]["processed_pairs_json"], PROJECT_ROOT)
    taxonomy = load_taxonomy(args.taxonomy)
    payload = load_processed_payload(processed_path)

    positive_pairs, metadata = build_positive_pairs(payload, repository, taxonomy)
    windowed_pairs, windows = assign_quantile_windows(
        positive_pairs,
        int(minimal_config["window_count"]),
        minimal_config,
    )
    distribution = window_distribution(windowed_pairs)
    report = {
        "repository": repository,
        "processed_pairs_json": str(processed_path),
        "window_strategy": minimal_config.get("window_strategy"),
        "pair_available_at_policy": minimal_config.get("pair_available_at"),
        "positive_pairs": len(windowed_pairs),
        "metadata": metadata,
        "windows": [window.__dict__ for window in windows],
        "distribution": distribution,
    }

    write_json(PROJECT_ROOT / args.windows_out, report)
    write_json(PROJECT_ROOT / args.json_out, report)
    write_jsonl(PROJECT_ROOT / args.pairs_out, windowed_pairs)
    write_text(PROJECT_ROOT / args.md_out, render_window_distribution_markdown(report))

    print(f"Window split JSON written: {args.json_out}")
    print(f"Window manifest written: {args.windows_out}")
    print(f"Positive pairs JSONL written: {args.pairs_out}")
    print(f"Window report Markdown written: {args.md_out}")
    print(f"Positive pairs: {len(windowed_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
