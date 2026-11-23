#!/usr/bin/env python3
"""Clean Public Jira requirement links using source-scoped issue lookup.

The original v1 extractor built dependency pairs through a global
lookup[issue_key]. This script rebuilds links and dependency pairs with
lookup[(source, issue_key)] so same issue keys from different Jira collections
cannot be cross-wired.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_JSON = ROOT / "data/interim/public_jira/requirement_pairs_dependencies.json"
DEFAULT_ENRICHED_REQUIREMENTS = ROOT / "data/interim/public_jira/requirements_text_enriched.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / "data/processed/public_jira/requirement_pairs_dependencies_clean.json"
DEFAULT_OUTPUT_ENRICHED_REQUIREMENTS = (
    ROOT / "data/processed/public_jira/requirements_text_enriched_clean.jsonl"
)
DEFAULT_OUTPUT_ENRICHED_PAIRS = (
    ROOT / "data/processed/public_jira/dependency_pairs_text_enriched_clean.jsonl"
)
DEFAULT_SUMMARY_OUT = ROOT / "outputs/logs/public_jira_dependency_cleaning_summary.json"


LINK_TYPE_MAPPING = {
    "Backports": "Backport",
    "Blocked": "Block",
    "Blocker": "Block",
    "Blocks": "Block",
    "Causality": "Cause",
    "Cause": "Cause",
    "Caused": "Cause",
    "Problem/Incident": "Cause",
    "Child-Issue": "Parent-Child",
    "Parent Feature": "Parent-Child",
    "Parent/Child": "Parent-Child",
    "multi-level hierarchy [GANTT]": "Parent-Child",
    "Parent-Relation": "Parent-Child",
    "Cloners": "Clone",
    "Cloners (old)": "Clone",
    "Collection": "Incorporate",
    "Container": "Incorporate",
    "Contains(WBSGantt)": "Incorporate",
    "Incorporate": "Incorporate",
    "Incorporates": "Incorporate",
    "Part": "Incorporate",
    "PartOf": "Incorporate",
    "Superset": "Incorporate",
    "Completes": "Fix",
    "Fixes": "Fix",
    "Resolve": "Fix",
    "Depend": "Depend",
    "Dependency": "Depend",
    "Dependent": "Depend",
    "Depends": "Depend",
    "Gantt Dependency": "Depend",
    "dependent": "Depend",
    "Derived": "Derive",
    "Detail": "Detail",
    "Duplicate": "Duplicate",
    "Epic": "Epic",
    "Epic-Relation": "Epic",
    "Follows": "Follow",
    "Sequence": "Follow",
    "Implement": "Implement",
    "Implements": "Implement",
    "Issue split": "Split",
    "Split": "Split",
    "Work Breakdown": "Split",
    "Preceded By": "Precede",
    "Reference": "Relate",
    "Relate": "Relate",
    "Related": "Relate",
    "Relates": "Relate",
    "Relationship": "Relate",
    "Regression": "Breaks",
    "Replacement": "Replace",
    "Required": "Require",
    "Supercedes": "Supercede",
    "Supersede": "Supercede",
    "Supersession": "Supercede",
    "Test": "Test",
    "Tested": "Test",
    "Trigger": "Trigger",
    "Subtask": "Subtask",
    "Bonfire Testing": "Bonfire Testing",
    "Bonfire testing": "Bonfire Testing",
    "Git Code Review": "Bonfire Testing",
    "Testing": "Bonfire Testing",
    "Documentation": "Documented",
    "Documented": "Documented",
    "Finish-to-Finish link (WBSGantt)": "finish-finish",
    "Gantt End to End": "finish-finish",
    "Gantt: finish-finish": "finish-finish",
    "finish-finish [GANTT]": "finish-finish",
    "Gantt End to Start": "finish-start",
    "Gantt: finish-start": "finish-start",
    "finish-start [GANTT]": "finish-start",
    "Gantt Start to Start": "start-start",
    "Gantt: start-finish": "start-finish",
}


DEPENDENCIES = {
    "Block",
    "Depend",
    "Require",
    "Cause",
    "Parent-Child",
    "Epic",
    "Incorporate",
    "Split",
    "Derive",
    "Precede",
    "Fix",
    "Implement",
    "Follow",
    "Clone",
    "Duplicate",
    "Replace",
    "Relate",
    "Breaks",
    "Detail",
    "Supercede",
    "Backport",
    "Bonfire Testing",
    "Documented",
    "Test",
    "Trigger",
    "finish-finish",
    "finish-start",
    "start-finish",
    "start-start",
    "Subtask",
}


def log(message: str) -> None:
    print(message, flush=True)


def open_text(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def source_issue_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source") or ""), str(row.get("issue_key") or "")


def dependency_pair_signature(pair: dict[str, Any]) -> tuple[Any, ...]:
    return (
        pair.get("source_source"),
        pair.get("source_issue_key"),
        pair.get("target_source"),
        pair.get("target_issue_key"),
        pair.get("dependency_type"),
        pair.get("raw_link_type"),
        pair.get("direction"),
    )


def link_signature(link: dict[str, Any]) -> tuple[Any, ...]:
    return (
        link.get("linked_issue_key"),
        link.get("link_type_name"),
        link.get("direction"),
        link.get("inward"),
        link.get("outward"),
    )


def merge_requirement_rows(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate rows for the same (source, issue_key) without widening scope."""
    merged = dict(base)
    if len(str(incoming.get("summary") or "")) > len(str(merged.get("summary") or "")):
        merged["summary"] = incoming.get("summary")
    for field in ["project_name", "project_key", "issuetype", "status", "priority", "created", "resolutiondate"]:
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming.get(field)
    seen_links = {link_signature(link) for link in merged.get("links", []) or []}
    merged_links = list(merged.get("links", []) or [])
    for link in incoming.get("links", []) or []:
        sig = link_signature(link)
        if sig not in seen_links:
            merged_links.append(link)
            seen_links.add(sig)
    merged["links"] = merged_links
    return merged


def deduplicate_requirements(
    requirements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    canonical: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_examples: list[dict[str, Any]] = []
    duplicate_key_counts: Counter[tuple[str, str]] = Counter()

    for row in requirements:
        key = source_issue_key(row)
        if key in canonical:
            duplicate_key_counts[key] += 1
            if len(duplicate_examples) < 20:
                duplicate_examples.append(
                    {
                        "source": key[0],
                        "issue_key": key[1],
                        "kept_summary": canonical[key].get("summary"),
                        "dropped_or_merged_summary": row.get("summary"),
                    }
                )
            canonical[key] = merge_requirement_rows(canonical[key], row)
        else:
            canonical[key] = dict(row)

    clean_requirements = list(canonical.values())
    stats = {
        "requirements_input": len(requirements),
        "requirements_clean": len(clean_requirements),
        "requirements_duplicate_source_issue_keys": len(duplicate_key_counts),
        "requirements_duplicate_extra_rows": sum(duplicate_key_counts.values()),
        "sample_duplicate_source_issue_keys": duplicate_examples,
    }
    return clean_requirements, canonical, stats


def clean_requirement_links(
    requirements: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    stats = {
        "links_input": 0,
        "links_clean_same_source_resolved": 0,
        "links_dropped_unresolved_same_source": 0,
        "links_dropped_duplicate_within_issue": 0,
        "links_clean_non_dependency_type": 0,
        "sample_dropped_unresolved_links": [],
    }

    for req in requirements:
        source = str(req.get("source") or "")
        issue_key = str(req.get("issue_key") or "")
        clean_links: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for link in req.get("links", []) or []:
            stats["links_input"] += 1
            linked_issue_key = str(link.get("linked_issue_key") or "")
            if (source, linked_issue_key) not in lookup:
                stats["links_dropped_unresolved_same_source"] += 1
                if len(stats["sample_dropped_unresolved_links"]) < 20:
                    stats["sample_dropped_unresolved_links"].append(
                        {
                            "source": source,
                            "issue_key": issue_key,
                            "linked_issue_key": linked_issue_key,
                            "raw_link_type": link.get("link_type_name"),
                            "direction": link.get("direction"),
                        }
                    )
                continue
            sig = link_signature(link)
            if sig in seen:
                stats["links_dropped_duplicate_within_issue"] += 1
                continue
            seen.add(sig)
            clean_link = dict(link)
            clean_links.append(clean_link)
            stats["links_clean_same_source_resolved"] += 1
            normalized = LINK_TYPE_MAPPING.get(str(link.get("link_type_name") or ""), str(link.get("link_type_name") or ""))
            if normalized not in DEPENDENCIES:
                stats["links_clean_non_dependency_type"] += 1
        req["links"] = clean_links

    return stats


def build_clean_dependency_pairs(
    requirements: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen_pairs: set[tuple[Any, ...]] = set()
    dependency_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    duplicate_pairs = 0

    for req in requirements:
        source = str(req.get("source") or "")
        for link in req.get("links", []) or []:
            raw_link_type = str(link.get("link_type_name") or "")
            dependency_type = LINK_TYPE_MAPPING.get(raw_link_type, raw_link_type)
            if dependency_type not in DEPENDENCIES:
                continue
            target = lookup[(source, str(link.get("linked_issue_key") or ""))]
            pair = {
                "source_issue_key": req.get("issue_key"),
                "target_issue_key": target.get("issue_key"),
                "dependency_type": dependency_type,
                "raw_link_type": raw_link_type,
                "direction": link.get("direction"),
                "source_source": req.get("source"),
                "target_source": target.get("source"),
                "source_issuetype": req.get("issuetype"),
                "target_issuetype": target.get("issuetype"),
                "source_project": req.get("project_key"),
                "target_project": target.get("project_key"),
                "source_summary": req.get("summary"),
                "target_summary": target.get("summary"),
            }
            sig = dependency_pair_signature(pair)
            if sig in seen_pairs:
                duplicate_pairs += 1
                continue
            seen_pairs.add(sig)
            pairs.append(pair)
            dependency_counts[dependency_type] += 1
            raw_counts[raw_link_type] += 1

    stats = {
        "dependency_pairs_clean": len(pairs),
        "dependency_pairs_dropped_duplicate_signatures": duplicate_pairs,
        "dependency_type_counts": dict(sorted(dependency_counts.items())),
        "raw_link_type_counts_top20": raw_counts.most_common(20),
    }
    return pairs, stats


def build_statistics(
    requirements: list[dict[str, Any]],
    dependency_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    req_by_source = Counter(str(req.get("source") or "") for req in requirements)
    pairs_by_source = Counter(str(pair.get("source_source") or "") for pair in dependency_pairs)
    pairs_by_dependency = Counter(str(pair.get("dependency_type") or "") for pair in dependency_pairs)
    return {
        "requirements_by_source": dict(sorted(req_by_source.items())),
        "dependency_pairs_by_source": dict(sorted(pairs_by_source.items())),
        "dependency_pairs_by_type": dict(sorted(pairs_by_dependency.items())),
    }


def write_clean_main_json(
    input_payload: dict[str, Any],
    requirements: list[dict[str, Any]],
    dependency_pairs: list[dict[str, Any]],
    output_path: Path,
    summary_stats: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(input_payload.get("metadata") or {})
    metadata.update(
        {
            "description": "Cleaned requirement pairs and dependencies from ThePublicJiraDataset",
            "cleaning_generated_at": datetime.now(timezone.utc).isoformat(),
            "cleaning_rules": [
                "Deduplicate requirements by (source, issue_key).",
                "Resolve linked issues only within the same source.",
                "Drop links whose target requirement is absent in the same source.",
                "Rebuild dependency_pairs after cleaning links.",
                "Deduplicate exact dependency pair signatures.",
            ],
            "source_scoped_link_lookup": True,
            "total_requirement_issues": len(requirements),
            "total_dependency_pairs": len(dependency_pairs),
            "dependency_categories": sorted({pair["dependency_type"] for pair in dependency_pairs}),
            "cleaning_summary_path": str(DEFAULT_SUMMARY_OUT),
        }
    )
    payload = {
        "metadata": metadata,
        "statistics": build_statistics(requirements, dependency_pairs),
        "requirements": requirements,
        "dependency_pairs": dependency_pairs,
        "cleaning_summary": summary_stats,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def build_requirement_text(row: dict[str, Any], include_comments: bool = False) -> str:
    parts: list[str] = []
    summary = str(row.get("summary") or "").strip()
    description = str(row.get("description") or "").strip()
    issue_type = str(row.get("issuetype") or "").strip()
    status = str(row.get("status") or "").strip()
    priority = str(row.get("priority") or "").strip()
    if summary:
        parts.append(f"Summary: {summary}")
    if description:
        parts.append(f"Description: {description}")
    if include_comments:
        comment_bodies = [
            str(comment.get("body") or "").strip()
            for comment in row.get("comment_excerpts", []) or []
            if str(comment.get("body") or "").strip()
        ]
        if comment_bodies:
            parts.append("Comments: " + "\n\n".join(comment_bodies))
    if issue_type:
        parts.append(f"Issue type: {issue_type}")
    if status:
        parts.append(f"Status: {status}")
    if priority:
        parts.append(f"Priority: {priority}")
    return "\n".join(parts)


def write_clean_enriched_requirements(
    enriched_input_path: Path,
    enriched_output_path: Path,
    clean_lookup: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    text_index: dict[tuple[str, str], dict[str, Any]] = {}
    written_keys: set[tuple[str, str]] = set()
    stats = {
        "enriched_requirements_input_rows": 0,
        "enriched_requirements_clean_rows": 0,
        "enriched_requirements_skipped_duplicate_rows": 0,
        "enriched_requirements_skipped_missing_clean_key": 0,
        "enriched_requirements_with_description": 0,
        "enriched_requirements_with_comments": 0,
    }

    with open_text(enriched_input_path, "r") as src, open_text(enriched_output_path, "w") as out:
        for line in src:
            stats["enriched_requirements_input_rows"] += 1
            row = json.loads(line)
            key = source_issue_key(row)
            if key not in clean_lookup:
                stats["enriched_requirements_skipped_missing_clean_key"] += 1
                continue
            if key in written_keys:
                stats["enriched_requirements_skipped_duplicate_rows"] += 1
                continue
            written_keys.add(key)
            clean_req = clean_lookup[key]
            for field in [
                "project_name",
                "project_key",
                "issuetype",
                "summary",
                "status",
                "priority",
                "created",
                "resolutiondate",
                "links",
            ]:
                row[field] = clean_req.get(field)
            row["has_description"] = bool(row.get("description"))
            row["has_comments"] = int(row.get("comment_count") or 0) > 0
            row["requirement_text_with_description"] = build_requirement_text(
                row,
                include_comments=False,
            )
            row["requirement_text_with_description_and_comments"] = build_requirement_text(
                row,
                include_comments=True,
            )
            text_index[key] = {
                "text_with_description": row["requirement_text_with_description"],
                "text_with_description_and_comments": row[
                    "requirement_text_with_description_and_comments"
                ],
                "has_description": row["has_description"],
                "comment_count": int(row.get("comment_count") or 0),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["enriched_requirements_clean_rows"] += 1
            stats["enriched_requirements_with_description"] += int(row["has_description"])
            stats["enriched_requirements_with_comments"] += int(row["has_comments"])

    missing_text = set(clean_lookup) - set(text_index)
    if missing_text:
        raise RuntimeError(f"Missing enriched rows for {len(missing_text)} clean requirements.")
    return text_index, stats


def write_clean_enriched_pairs(
    dependency_pairs: list[dict[str, Any]],
    text_index: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    stats = {
        "enriched_pairs_clean_rows": 0,
        "enriched_pairs_source_with_description": 0,
        "enriched_pairs_target_with_description": 0,
        "enriched_pairs_both_with_description": 0,
    }
    with open_text(output_path, "w") as out:
        for pair in dependency_pairs:
            source_key = (
                str(pair.get("source_source") or ""),
                str(pair.get("source_issue_key") or ""),
            )
            target_key = (
                str(pair.get("target_source") or ""),
                str(pair.get("target_issue_key") or ""),
            )
            source_text = text_index[source_key]
            target_text = text_index[target_key]
            row = dict(pair)
            row["source_enrichment_found"] = True
            row["target_enrichment_found"] = True
            row["source_has_description"] = bool(source_text["has_description"])
            row["target_has_description"] = bool(target_text["has_description"])
            row["source_comment_count"] = int(source_text["comment_count"])
            row["target_comment_count"] = int(target_text["comment_count"])
            row["source_text_with_description"] = source_text["text_with_description"]
            row["target_text_with_description"] = target_text["text_with_description"]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["enriched_pairs_clean_rows"] += 1
            stats["enriched_pairs_source_with_description"] += int(
                row["source_has_description"]
            )
            stats["enriched_pairs_target_with_description"] += int(
                row["target_has_description"]
            )
            stats["enriched_pairs_both_with_description"] += int(
                row["source_has_description"] and row["target_has_description"]
            )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--enriched-requirements", type=Path, default=DEFAULT_ENRICHED_REQUIREMENTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-enriched-requirements",
        type=Path,
        default=DEFAULT_OUTPUT_ENRICHED_REQUIREMENTS,
    )
    parser.add_argument(
        "--output-enriched-pairs",
        type=Path,
        default=DEFAULT_OUTPUT_ENRICHED_PAIRS,
    )
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log(f"Loading input JSON: {args.input_json}")
    with args.input_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    input_requirements = payload.get("requirements", [])
    input_pairs = payload.get("dependency_pairs", [])
    if not isinstance(input_requirements, list) or not isinstance(input_pairs, list):
        raise ValueError("Input JSON must contain list requirements and dependency_pairs.")

    clean_requirements, clean_lookup, dedupe_stats = deduplicate_requirements(input_requirements)
    log(f"Deduplicated requirements: {len(input_requirements)} -> {len(clean_requirements)}")

    link_stats = clean_requirement_links(clean_requirements, clean_lookup)
    log(
        "Cleaned links: "
        f"{link_stats['links_input']} -> {link_stats['links_clean_same_source_resolved']}"
    )

    clean_pairs, pair_stats = build_clean_dependency_pairs(clean_requirements, clean_lookup)
    log(f"Rebuilt dependency pairs: {len(input_pairs)} -> {len(clean_pairs)}")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_json": str(args.input_json),
        "output_json": str(args.output_json),
        "input_enriched_requirements": str(args.enriched_requirements),
        "output_enriched_requirements": str(args.output_enriched_requirements),
        "output_enriched_pairs": str(args.output_enriched_pairs),
        "cleaning_rules": [
            "Canonical key is (source, issue_key), not global issue_key.",
            "Requirements duplicate on (source, issue_key) are merged.",
            "Requirement links are retained only when target exists under the same source.",
            "Dependency pairs are rebuilt from cleaned links.",
            "Exact dependency pair signatures are deduplicated.",
        ],
        "dependency_pairs_input": len(input_pairs),
        **dedupe_stats,
        **link_stats,
        **pair_stats,
    }

    log(f"Writing clean main JSON: {args.output_json}")
    write_clean_main_json(payload, clean_requirements, clean_pairs, args.output_json, summary)

    log(f"Writing clean enriched requirements: {args.output_enriched_requirements}")
    text_index, enriched_req_stats = write_clean_enriched_requirements(
        args.enriched_requirements,
        args.output_enriched_requirements,
        clean_lookup,
    )
    summary.update(enriched_req_stats)

    log(f"Writing clean enriched pairs: {args.output_enriched_pairs}")
    enriched_pair_stats = write_clean_enriched_pairs(
        clean_pairs,
        text_index,
        args.output_enriched_pairs,
    )
    summary.update(enriched_pair_stats)

    summary["statistics"] = build_statistics(clean_requirements, clean_pairs)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    log(f"Wrote summary: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
