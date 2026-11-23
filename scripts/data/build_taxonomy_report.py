#!/usr/bin/env python
"""Build a taxonomy mapping report for local processed Jira pairs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.taxonomy import LABELS, load_taxonomy
from haireder.utils.config import load_yaml, resolve_path
from haireder.utils.io import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument(
        "--taxonomy",
        default="data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml",
    )
    parser.add_argument("--json-out", default="outputs/logs/taxonomy_report.json")
    parser.add_argument("--md-out", default="outputs/tables/taxonomy_mapping_report.md")
    return parser.parse_args()


def render_markdown(report: dict) -> str:
    lines = [
        "# Taxonomy Mapping Report",
        "",
        "## Labels",
        "",
    ]
    for label in report["labels"]:
        lines.append(f"- `{label}`")
    lines.extend(
        [
            "",
            "## Normalized Dependency Type Mapping",
            "",
            "| Normalized type | HAI-ReDeR label | Count in processed pairs |",
            "|---|---|---:|",
        ]
    )
    for dep_type, details in report["dependency_type_mapping"].items():
        lines.append(f"| `{dep_type}` | `{details['label']}` | {details['count']} |")
    lines.extend(
        [
            "",
            "## HAI-ReDeR Label Distribution from Positive Pairs",
            "",
            "| Label | Count |",
            "|---|---:|",
        ]
    )
    for label in report["labels"]:
        lines.append(f"| `{label}` | {report['label_counts'].get(label, 0)} |")
    lines.extend(
        [
            "",
            "## Unmapped Dependency Types",
            "",
        ]
    )
    if report["unmapped_dependency_types"]:
        for dep_type, count in report["unmapped_dependency_types"].items():
            lines.append(f"- `{dep_type}`: {count}")
    else:
        lines.append("No unmapped dependency types found in processed pairs.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset_config = load_yaml(resolve_path(args.dataset_config))
    processed_path = resolve_path(dataset_config["raw_sources"]["processed_pairs_json"], PROJECT_ROOT)
    taxonomy = load_taxonomy(args.taxonomy)

    with processed_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dependency_pairs = payload.get("dependency_pairs", [])
    type_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()

    for pair in dependency_pairs:
        dep_type = str(pair.get("dependency_type") or "UNKNOWN")
        type_counts[dep_type] += 1
        label = taxonomy.map_normalized_type(dep_type)
        if label is None:
            unmapped[dep_type] += 1
        else:
            label_counts[label] += 1

    dependency_type_mapping = {}
    for dep_type, count in sorted(type_counts.items()):
        dependency_type_mapping[dep_type] = {
            "label": taxonomy.map_normalized_type(dep_type),
            "count": count,
        }

    report = {
        "dataset_config": str(resolve_path(args.dataset_config)),
        "taxonomy_path": str(resolve_path(args.taxonomy)),
        "processed_pairs_json": str(processed_path),
        "labels": LABELS,
        "positive_pair_count": len(dependency_pairs),
        "dependency_type_mapping": dependency_type_mapping,
        "label_counts": {label: label_counts.get(label, 0) for label in LABELS},
        "unmapped_dependency_types": dict(sorted(unmapped.items())),
        "none_label_policy": "Reserved for generated negative pairs; positive recorded links must not map to None.",
    }

    write_json(PROJECT_ROOT / args.json_out, report)
    write_text(PROJECT_ROOT / args.md_out, render_markdown(report))
    print(f"Taxonomy report JSON written: {args.json_out}")
    print(f"Taxonomy report Markdown written: {args.md_out}")
    if unmapped:
        print("Unmapped dependency types:", dict(unmapped))
        return 2
    print("All processed dependency types are mapped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
