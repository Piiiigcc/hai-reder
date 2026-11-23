#!/usr/bin/env python3
"""Audit whether extracted Public Jira dependency links are target-matched safely."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_JSON = ROOT / "data/interim/public_jira/requirement_pairs_dependencies.json"
DEFAULT_JSON_OUT = ROOT / "outputs/logs/public_jira_link_integrity_audit.json"


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 6) if whole else 0.0


def sample_requirement(req: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "idx": idx,
        "source": req.get("source"),
        "project_key": req.get("project_key"),
        "issue_key": req.get("issue_key"),
        "issuetype": req.get("issuetype"),
        "summary": req.get("summary"),
        "links_len": len(req.get("links") or []),
    }


def main() -> int:
    args = parse_args()
    with args.current_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    requirements = payload.get("requirements", [])
    dependency_pairs = payload.get("dependency_pairs", [])
    if not isinstance(requirements, list) or not isinstance(dependency_pairs, list):
        raise ValueError("Input JSON must contain list requirements and dependency_pairs.")

    by_key: dict[str, list[int]] = defaultdict(list)
    by_source_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    links_by_source_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    first_summary_by_source_key: dict[tuple[str, str], str] = {}

    for idx, req in enumerate(requirements):
        issue_key = str(req.get("issue_key") or "")
        source = str(req.get("source") or "")
        by_key[issue_key].append(idx)
        by_source_key[(source, issue_key)].append(idx)
        links_by_source_key[(source, issue_key)].extend(req.get("links") or [])
        first_summary_by_source_key.setdefault((source, issue_key), str(req.get("summary") or ""))

    global_duplicate_keys = {key: idxs for key, idxs in by_key.items() if len(idxs) > 1}
    global_duplicate_keys_multi_source = {
        key: idxs
        for key, idxs in global_duplicate_keys.items()
        if len({requirements[idx].get("source") for idx in idxs}) > 1
    }
    duplicate_source_keys = {
        key: idxs for key, idxs in by_source_key.items() if len(idxs) > 1
    }

    pair_issue_keys = {
        str(pair.get("source_issue_key") or "")
        for pair in dependency_pairs
    } | {
        str(pair.get("target_issue_key") or "")
        for pair in dependency_pairs
    }
    pair_ambiguous_keys = {
        key: by_key[key] for key in pair_issue_keys if len(by_key.get(key, [])) > 1
    }
    pair_ambiguous_keys_multi_source = {
        key: idxs
        for key, idxs in pair_ambiguous_keys.items()
        if len({requirements[idx].get("source") for idx in idxs}) > 1
    }

    missing_source_req = 0
    missing_target_req = 0
    source_summary_mismatch = 0
    target_summary_mismatch = 0
    pair_link_not_found_in_source_links = 0
    cross_source_pairs = 0
    cross_source_same_source_target_exists = 0
    cross_source_no_same_source_target = 0
    cross_source_by_source_target: Counter[tuple[str, str]] = Counter()
    cross_source_by_dependency_type: Counter[str] = Counter()
    cross_source_by_raw_link_type: Counter[str] = Counter()
    cross_source_examples: list[dict[str, Any]] = []

    pair_signature_counts: Counter[tuple[Any, ...]] = Counter()

    for pair in dependency_pairs:
        source_source = str(pair.get("source_source") or "")
        target_source = str(pair.get("target_source") or "")
        source_issue_key = str(pair.get("source_issue_key") or "")
        target_issue_key = str(pair.get("target_issue_key") or "")
        raw_link_type = str(pair.get("raw_link_type") or "")
        direction = str(pair.get("direction") or "")
        dependency_type = str(pair.get("dependency_type") or "")

        source_lookup_key = (source_source, source_issue_key)
        target_lookup_key = (target_source, target_issue_key)
        same_source_target_key = (source_source, target_issue_key)

        if source_lookup_key not in by_source_key:
            missing_source_req += 1
        if target_lookup_key not in by_source_key:
            missing_target_req += 1
        if first_summary_by_source_key.get(source_lookup_key, "") != str(
            pair.get("source_summary") or ""
        ):
            source_summary_mismatch += 1
        if first_summary_by_source_key.get(target_lookup_key, "") != str(
            pair.get("target_summary") or ""
        ):
            target_summary_mismatch += 1

        link_found = any(
            str(link.get("linked_issue_key") or "") == target_issue_key
            and str(link.get("link_type_name") or "") == raw_link_type
            and str(link.get("direction") or "") == direction
            for link in links_by_source_key.get(source_lookup_key, [])
        )
        if not link_found:
            pair_link_not_found_in_source_links += 1

        if source_source != target_source:
            cross_source_pairs += 1
            cross_source_by_source_target[(source_source, target_source)] += 1
            cross_source_by_dependency_type[dependency_type] += 1
            cross_source_by_raw_link_type[raw_link_type] += 1
            if same_source_target_key in by_source_key:
                cross_source_same_source_target_exists += 1
            else:
                cross_source_no_same_source_target += 1
            if len(cross_source_examples) < args.sample_limit:
                candidates = [
                    sample_requirement(requirements[idx], idx)
                    for idx in by_key.get(target_issue_key, [])[:5]
                ]
                cross_source_examples.append(
                    {
                        "pair": pair,
                        "same_source_target_exists": same_source_target_key in by_source_key,
                        "global_target_candidates": candidates,
                    }
                )

        pair_signature_counts[
            (
                source_source,
                source_issue_key,
                target_source,
                target_issue_key,
                dependency_type,
                raw_link_type,
                direction,
            )
        ] += 1

    exact_duplicate_pair_signature_count = sum(
        1 for count in pair_signature_counts.values() if count > 1
    )
    exact_duplicate_pair_extra_rows = sum(
        count - 1 for count in pair_signature_counts.values() if count > 1
    )

    duplicate_key_examples = {
        key: [sample_requirement(requirements[idx], idx) for idx in idxs[:5]]
        for key, idxs in list(global_duplicate_keys_multi_source.items())[: args.sample_limit]
    }
    duplicate_source_key_examples = {
        f"{source}:{issue_key}": [
            sample_requirement(requirements[idx], idx) for idx in idxs[:5]
        ]
        for (source, issue_key), idxs in list(duplicate_source_keys.items())[: args.sample_limit]
    }

    report = {
        "current_json": str(args.current_json),
        "requirements_total": len(requirements),
        "dependency_pairs_total": len(dependency_pairs),
        "unique_issue_keys_global": len(by_key),
        "global_duplicate_issue_key_count": len(global_duplicate_keys),
        "global_duplicate_issue_key_extra_rows": sum(
            len(idxs) - 1 for idxs in global_duplicate_keys.values()
        ),
        "global_duplicate_issue_keys_multi_source_count": len(
            global_duplicate_keys_multi_source
        ),
        "duplicate_source_issue_key_count": len(duplicate_source_keys),
        "duplicate_source_issue_key_extra_rows": sum(
            len(idxs) - 1 for idxs in duplicate_source_keys.values()
        ),
        "pair_issue_keys_ambiguous_global_count": len(pair_ambiguous_keys),
        "pair_issue_keys_ambiguous_multi_source_count": len(
            pair_ambiguous_keys_multi_source
        ),
        "pairs_missing_source_req_by_source_key": missing_source_req,
        "pairs_missing_target_req_by_source_key": missing_target_req,
        "pairs_source_summary_mismatch": source_summary_mismatch,
        "pairs_target_summary_mismatch": target_summary_mismatch,
        "pairs_link_not_found_in_source_links": pair_link_not_found_in_source_links,
        "cross_source_pairs": cross_source_pairs,
        "cross_source_pairs_rate_percent": pct(cross_source_pairs, len(dependency_pairs)),
        "cross_source_pairs_where_same_source_target_exists": (
            cross_source_same_source_target_exists
        ),
        "cross_source_pairs_where_same_source_target_missing": (
            cross_source_no_same_source_target
        ),
        "cross_source_pairs_by_source_target": [
            {"source_source": key[0], "target_source": key[1], "count": count}
            for key, count in cross_source_by_source_target.most_common()
        ],
        "cross_source_pairs_by_dependency_type": cross_source_by_dependency_type.most_common(),
        "cross_source_pairs_by_raw_link_type": cross_source_by_raw_link_type.most_common(),
        "exact_duplicate_pair_signature_count": exact_duplicate_pair_signature_count,
        "exact_duplicate_pair_extra_rows": exact_duplicate_pair_extra_rows,
        "judgement": {
            "safe_to_treat_all_pairs_as_clean": cross_source_pairs == 0
            and len(global_duplicate_keys_multi_source) == 0
            and len(duplicate_source_keys) == 0,
            "primary_logic_risk": (
                "Original extraction used lookup[issue_key] globally instead of "
                "lookup[(source, issue_key)], so duplicate issue keys across Jira "
                "collections can map a link target to an unrelated repository."
            ),
            "recommended_action": (
                "Regenerate dependency pairs with source-scoped target lookup, "
                "drop links whose target issue is not present in the same source, "
                "and deduplicate exact duplicate requirement rows."
            ),
        },
        "sample_duplicate_issue_keys_multi_source": duplicate_key_examples,
        "sample_duplicate_source_issue_keys": duplicate_source_key_examples,
        "sample_cross_source_pairs": cross_source_examples,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-json", type=Path, default=DEFAULT_CURRENT_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--sample-limit", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
