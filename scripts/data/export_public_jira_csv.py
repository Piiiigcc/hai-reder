#!/usr/bin/env python3
"""Export requirements and dependency pairs into two CSV files per Jira source."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/interim/public_jira/requirement_pairs_dependencies.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/interim/public_jira/per_repository_csv"

REQ_FIELDS = [
    "issue_key", "source", "project_key", "project_name",
    "issuetype", "summary", "status", "priority",
    "created", "resolutiondate",
]

PAIR_FIELDS = [
    "source_issue_key", "target_issue_key", "dependency_type",
    "raw_link_type", "direction",
    "source_source", "target_source",
    "source_issuetype", "target_issuetype",
    "source_project", "target_project",
    "source_summary", "target_summary",
]


def export_csv(input_json: Path, output_dir: Path) -> None:
    with input_json.open(encoding="utf-8") as handle:
        data = json.load(handle)
    requirements = data["requirements"]
    pairs = data["dependency_pairs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted({row["source"] for row in requirements}):
        tables = (
            ("requirements", REQ_FIELDS, (r for r in requirements if r["source"] == source)),
            ("dependency_pairs", PAIR_FIELDS, (p for p in pairs if p["source_source"] == source)),
        )
        for suffix, fields, rows in tables:
            path = output_dir / f"{source}_{suffix}.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    export_csv(args.input_json, args.output_dir)
