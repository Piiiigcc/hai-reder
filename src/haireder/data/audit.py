"""Audit local Public Jira Dataset inputs."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def path_status(path: Path) -> dict[str, Any]:
    """Return existence and size details for a path."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def audit_public_jira_zip(zip_path: Path) -> dict[str, Any]:
    """Audit the official Public Jira Dataset zip archive."""
    report: dict[str, Any] = path_status(zip_path)
    report.update(
        {
            "is_zip": False,
            "members_count": 0,
            "data_definition_files": [],
            "mongo_archive": None,
            "read_error": None,
        }
    )
    if not zip_path.exists():
        return report
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            report["is_zip"] = True
            report["members_count"] = len(members)
            report["data_definition_files"] = [
                item.filename
                for item in members
                if "/0. DataDefinition/" in item.filename and not item.is_dir()
            ]
            for item in members:
                if item.filename.endswith("mongodump-JiraReposAnon.archive"):
                    report["mongo_archive"] = {
                        "path_in_zip": item.filename,
                        "size_bytes": item.file_size,
                        "compress_size_bytes": item.compress_size,
                    }
                    break
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["read_error"] = repr(exc)
    return report


def _safe_preview(value: Any, max_chars: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def audit_processed_pairs_json(json_path: Path, target_repositories: list[str]) -> dict[str, Any]:
    """Audit the v1 processed requirement-pair JSON file."""
    report: dict[str, Any] = path_status(json_path)
    report.update(
        {
            "top_level_keys": [],
            "metadata": {},
            "statistics": {},
            "requirements_count": 0,
            "dependency_pairs_count": 0,
            "requirement_fields": [],
            "dependency_pair_fields": [],
            "sample_requirement": None,
            "sample_dependency_pair": None,
            "by_source_from_pairs": {},
            "by_dependency_type_from_pairs": {},
            "by_source_dependency_type": {},
            "target_repository_coverage": {},
            "recommended_minimal_repository": None,
            "read_error": None,
        }
    )
    if not json_path.exists():
        return report

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - diagnostic only
        report["read_error"] = repr(exc)
        return report

    if not isinstance(payload, dict):
        report["read_error"] = "Expected top-level JSON object."
        return report

    requirements = payload.get("requirements", [])
    dependency_pairs = payload.get("dependency_pairs", [])
    report["top_level_keys"] = list(payload.keys())
    report["metadata"] = payload.get("metadata", {})
    report["statistics"] = payload.get("statistics", {})
    report["requirements_count"] = len(requirements) if isinstance(requirements, list) else 0
    report["dependency_pairs_count"] = len(dependency_pairs) if isinstance(dependency_pairs, list) else 0

    if isinstance(requirements, list) and requirements:
        first_requirement = requirements[0]
        if isinstance(first_requirement, dict):
            report["requirement_fields"] = list(first_requirement.keys())
            report["sample_requirement"] = first_requirement

    if isinstance(dependency_pairs, list) and dependency_pairs:
        first_pair = dependency_pairs[0]
        if isinstance(first_pair, dict):
            report["dependency_pair_fields"] = list(first_pair.keys())
            report["sample_dependency_pair"] = first_pair

    by_source: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_source_type: dict[str, Counter[str]] = defaultdict(Counter)

    if isinstance(dependency_pairs, list):
        for pair in dependency_pairs:
            if not isinstance(pair, dict):
                continue
            source = str(pair.get("source_source") or pair.get("source") or "UNKNOWN")
            dep_type = str(pair.get("dependency_type") or "UNKNOWN")
            by_source[source] += 1
            by_type[dep_type] += 1
            by_source_type[source][dep_type] += 1

    report["by_source_from_pairs"] = dict(sorted(by_source.items()))
    report["by_dependency_type_from_pairs"] = dict(sorted(by_type.items()))
    report["by_source_dependency_type"] = {
        source: dict(sorted(counter.items()))
        for source, counter in sorted(by_source_type.items())
    }

    coverage = {}
    for repo in target_repositories:
        dep_count = by_source.get(repo, 0)
        type_counts = by_source_type.get(repo, Counter())
        coverage[repo] = {
            "dependency_pairs": dep_count,
            "dependency_type_count": len(type_counts),
            "dependency_types": dict(sorted(type_counts.items())),
        }
    report["target_repository_coverage"] = coverage

    eligible = [
        (repo, details["dependency_type_count"], details["dependency_pairs"])
        for repo, details in coverage.items()
        if details["dependency_pairs"] > 0
    ]
    if eligible:
        eligible.sort(key=lambda item: (item[1], item[2]), reverse=True)
        report["recommended_minimal_repository"] = {
            "repository": eligible[0][0],
            "dependency_type_count": eligible[0][1],
            "dependency_pairs": eligible[0][2],
            "reason": "Highest dependency-type coverage among configured target repositories.",
        }
    return report


def audit_markdown(report: dict[str, Any]) -> str:
    """Render data audit results as Markdown."""
    zip_report = report["public_jira_zip"]
    processed = report["processed_pairs_json"]
    lines = [
        "# Data Audit Summary",
        "",
        "## Config",
        "",
        f"- Dataset config: `{report['dataset_config_path']}`",
        "",
        "## Official Public Jira Dataset Zip",
        "",
        f"- Path: `{zip_report['path']}`",
        f"- Exists: `{zip_report['exists']}`",
        f"- Size bytes: `{zip_report['size_bytes']}`",
        f"- Is zip: `{zip_report['is_zip']}`",
        f"- Members count: `{zip_report['members_count']}`",
        f"- Mongo archive: `{zip_report['mongo_archive']}`",
        "",
        "### Data Definition Files",
        "",
    ]
    for item in zip_report["data_definition_files"]:
        lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Processed Requirement-Pair JSON",
            "",
            f"- Path: `{processed['path']}`",
            f"- Exists: `{processed['exists']}`",
            f"- Size bytes: `{processed['size_bytes']}`",
            f"- Top-level keys: `{processed['top_level_keys']}`",
            f"- Requirements count: `{processed['requirements_count']}`",
            f"- Dependency pairs count: `{processed['dependency_pairs_count']}`",
            f"- Requirement fields: `{processed['requirement_fields']}`",
            f"- Dependency pair fields: `{processed['dependency_pair_fields']}`",
            "",
            "## Target Repository Coverage",
            "",
            "| Repository | Dependency pairs | Dependency type count | Dependency types |",
            "|---|---:|---:|---|",
        ]
    )
    for repo, details in processed["target_repository_coverage"].items():
        lines.append(
            f"| {repo} | {details['dependency_pairs']} | "
            f"{details['dependency_type_count']} | `{details['dependency_types']}` |"
        )

    lines.extend(
        [
            "",
            "## Dependency Type Distribution",
            "",
            "| Dependency type | Count |",
            "|---|---:|",
        ]
    )
    for dep_type, count in processed["by_dependency_type_from_pairs"].items():
        lines.append(f"| {dep_type} | {count} |")

    lines.extend(
        [
            "",
            "## Recommended Minimal Repository",
            "",
            f"`{processed['recommended_minimal_repository']}`",
            "",
            "## Sample Requirement",
            "",
            "```json",
            _safe_preview(processed["sample_requirement"]),
            "```",
            "",
            "## Sample Dependency Pair",
            "",
            "```json",
            _safe_preview(processed["sample_dependency_pair"]),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
