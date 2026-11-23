#!/usr/bin/env python
"""Audit local Public Jira Dataset sources configured for HAI-ReDeR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.data.audit import audit_markdown, audit_processed_pairs_json, audit_public_jira_zip
from haireder.utils.config import load_yaml, resolve_path
from haireder.utils.io import write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument("--json-out", default="outputs/logs/data_audit.json")
    parser.add_argument("--md-out", default="outputs/tables/data_audit_summary.md")
    parser.add_argument(
        "--sources-out",
        default="data/interim/selected_repositories/minimal_data_sources.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    raw_sources = config.get("raw_sources", {})
    target_repositories = list(config.get("target_repositories", []))

    zip_path = resolve_path(raw_sources.get("public_jira_zip", ""), PROJECT_ROOT)
    processed_json_path = resolve_path(raw_sources.get("processed_pairs_json", ""), PROJECT_ROOT)

    report = {
        "dataset_config_path": str(config_path),
        "target_repositories": target_repositories,
        "public_jira_zip": audit_public_jira_zip(zip_path),
        "processed_pairs_json": audit_processed_pairs_json(processed_json_path, target_repositories),
    }

    sources = {
        "data_version": config.get("data_version"),
        "public_jira_zip": report["public_jira_zip"],
        "processed_pairs_json": {
            key: report["processed_pairs_json"][key]
            for key in [
                "path",
                "exists",
                "size_bytes",
                "requirements_count",
                "dependency_pairs_count",
                "target_repository_coverage",
                "recommended_minimal_repository",
            ]
        },
    }

    write_json(PROJECT_ROOT / args.json_out, report)
    write_json(PROJECT_ROOT / args.sources_out, sources)
    write_text(PROJECT_ROOT / args.md_out, audit_markdown(report))

    print(f"Data audit JSON written: {args.json_out}")
    print(f"Data audit Markdown written: {args.md_out}")
    print(f"Data source manifest written: {args.sources_out}")
    recommended = report["processed_pairs_json"].get("recommended_minimal_repository")
    if recommended:
        print("Recommended minimal repository:", recommended["repository"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
