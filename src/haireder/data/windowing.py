"""Time-window construction for minimal HAI-ReDeR runs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from typing import Any

from haireder.taxonomy import Taxonomy


def parse_jira_datetime(value: str | None) -> datetime | None:
    """Parse Jira timestamps such as 2022-01-04T09:13:21.000+0000."""
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def pair_identity(repo: str, left: str, right: str) -> str:
    return ":".join(quote(value, safe="") for value in ("jira", repo, left, right))


def format_datetime(value: datetime | None) -> str | None:
    """Serialize datetimes consistently."""
    if value is None:
        return None
    return value.isoformat()


@dataclass(frozen=True)
class WindowSpec:
    """A single chronological window."""

    window: str
    role: str
    start_at: str
    end_at: str
    count: int


def role_for_window(window: str, split_config: dict[str, Any]) -> str:
    """Return the split role for a window ID."""
    if window in split_config.get("train_windows", []):
        return "train"
    if window in split_config.get("validation_windows", []):
        return "validation"
    if window in split_config.get("online_windows", []):
        return "online"
    if window in split_config.get("test_windows", []):
        return "test"
    return "unused"


def load_processed_payload(path: Path) -> dict[str, Any]:
    """Load the v1 processed JSON payload."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level JSON object: {path}")
    return payload


def build_requirement_index(requirements: list[dict[str, Any]], repository: str) -> dict[str, dict[str, Any]]:
    """Index requirements for one Public Jira source/repository."""
    index = {}
    for req in requirements:
        if req.get("source") != repository:
            continue
        issue_key = req.get("issue_key")
        if issue_key:
            index[str(issue_key)] = req
    return index


def requirement_text(requirement: dict[str, Any]) -> str:
    """Use the creation snapshot when supplied by the formal data builder."""
    fields = requirement.get("creation_text", requirement) or {}
    return "\n".join(f"{key.title()}: {fields[key].strip()}" for key in ("summary", "description")
                     if isinstance(fields.get(key), str) and fields[key].strip())


def build_positive_pairs(
    payload: dict[str, Any],
    repository: str,
    taxonomy: Taxonomy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build positive pairs for a selected repository."""
    requirements = payload.get("requirements", [])
    dependency_pairs = payload.get("dependency_pairs", [])
    if not isinstance(requirements, list) or not isinstance(dependency_pairs, list):
        raise ValueError("Processed payload must contain list requirements and dependency_pairs.")

    req_index = build_requirement_index(requirements, repository)
    grouped_pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    output_pairs: list[dict[str, Any]] = []
    skipped = Counter()

    for raw_pair in dependency_pairs:
        if raw_pair.get("source_source") != repository or raw_pair.get("target_source") != repository:
            continue
        source_key = str(raw_pair.get("source_issue_key") or "")
        target_key = str(raw_pair.get("target_issue_key") or "")
        if not source_key or not target_key or source_key == target_key:
            skipped["invalid_issue_key"] += 1
            continue
        source_req = req_index.get(source_key)
        target_req = req_index.get(target_key)
        if source_req is None or target_req is None:
            skipped["missing_requirement"] += 1
            continue
        source_created = parse_jira_datetime(source_req.get("created"))
        target_created = parse_jira_datetime(target_req.get("created"))
        if source_created is None or target_created is None:
            skipped["missing_created"] += 1
            continue
        available_at = max(source_created, target_created)
        normalized_type = str(raw_pair.get("dependency_type") or "")
        label = taxonomy.map_normalized_type(normalized_type)
        if label is None:
            skipped["unmapped_dependency_type"] += 1
            continue
        grouped_pairs[(source_key, target_key)].append(
            {
                "source_key": source_key,
                "target_key": target_key,
                "source_req": source_req,
                "target_req": target_req,
                "available_at": available_at,
                "label": label,
                "dependency_type": normalized_type,
                "raw_link_type": raw_pair.get("raw_link_type"),
            }
        )

    for (source_key, target_key), rows in sorted(grouped_pairs.items()):
        labels = {row["label"] for row in rows}
        if len(labels) > 1:
            skipped["conflicting_multi_label_pair"] += 1
            continue
        if len(rows) > 1:
            skipped["collapsed_same_label_multi_link_pair"] += len(rows) - 1
        first = sorted(rows, key=lambda row: (row["dependency_type"], str(row["raw_link_type"])))[0]
        source_req = first["source_req"]
        target_req = first["target_req"]
        pair_id = pair_identity(repository, source_key, target_key)
        dependency_types = sorted({str(row["dependency_type"]) for row in rows})
        raw_link_types = sorted({str(row["raw_link_type"]) for row in rows if row.get("raw_link_type") is not None})
        output_pairs.append(
            {
                "pair_id": pair_id,
                "repo": repository,
                "window": None,
                "split": None,
                "pair_available_at": format_datetime(first["available_at"]),
                "issue_i": {
                    "id": source_key,
                    "project": source_req.get("project_key"),
                    "created": source_req.get("created"),
                    "text": requirement_text(source_req),
                },
                "issue_j": {
                    "id": target_key,
                    "project": target_req.get("project_key"),
                    "created": target_req.get("created"),
                    "text": requirement_text(target_req),
                },
                "label": first["label"],
                "dependency_type": first["dependency_type"],
                "dependency_types": dependency_types,
                "raw_link_type": first["raw_link_type"],
                "raw_link_types": raw_link_types,
                "is_positive": True,
                "source": "v1_processed_public_jira",
            }
        )

    metadata = {
        "repository": repository,
        "requirements_in_repository": len(req_index),
        "positive_pairs_before_windowing": len(output_pairs),
        "skipped": dict(skipped),
    }
    return output_pairs, metadata


def assign_quantile_windows(
    pairs: list[dict[str, Any]],
    window_count: int,
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[WindowSpec]]:
    """Assign chronological quantile windows to pairs."""
    sorted_pairs = sorted(pairs, key=lambda row: (row["pair_available_at"], row["pair_id"]))
    if not sorted_pairs:
        return [], []
    windows: list[WindowSpec] = []
    total = len(sorted_pairs)
    for idx, row in enumerate(sorted_pairs):
        window_index = min((idx * window_count) // total, window_count - 1)
        window = f"W{window_index + 1}"
        row["window"] = window
        row["split"] = role_for_window(window, split_config)

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_pairs:
        by_window[row["window"]].append(row)

    for window_number in range(1, window_count + 1):
        window = f"W{window_number}"
        rows = by_window.get(window, [])
        if not rows:
            continue
        windows.append(
            WindowSpec(
                window=window,
                role=role_for_window(window, split_config),
                start_at=rows[0]["pair_available_at"],
                end_at=rows[-1]["pair_available_at"],
                count=len(rows),
            )
        )
    return sorted_pairs, windows


def build_fixed_sprints(
    pairs: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    sprint_days: int,
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[WindowSpec]]:
    """Assign pairs by availability to a complete half-open UTC calendar."""
    start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    if sprint_days <= 0 or end <= start:
        raise ValueError("Invalid sprint calendar")
    step = timedelta(days=sprint_days)
    count, remainder = divmod(end - start, step)
    if remainder or count != split_config.get("window_count", count):
        raise ValueError("Date range must contain the configured number of full sprints")
    rows = []
    counts = Counter()
    for row in pairs:
        when = parse_jira_datetime(row.get("pair_available_at"))
        if when is not None and start <= when < end:
            window = f"W{(when - start) // step + 1}"
            rows.append(dict(row, window=window, split=role_for_window(window, split_config)))
            counts[window] += 1
    windows = []
    for index in range(count):
        window = f"W{index + 1}"
        windows.append(WindowSpec(window, role_for_window(window, split_config),
                                  (start + index * step).isoformat(), (start + (index + 1) * step).isoformat(),
                                  counts[window]))
    return sorted(rows, key=lambda row: (row["pair_available_at"], row["pair_id"])), windows


def build_chronological_windows(
    pairs: list[dict[str, Any]],
    window_count: int,
    split_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign chronological 2-week sprint windows to pairs.

    Windows are determined by the min/max pair_available_at across all pairs,
    divided into equal 2-week intervals.
    """
    sorted_pairs = sorted(pairs, key=lambda r: (r["pair_available_at"], r["pair_id"]))
    if not sorted_pairs:
        return [], []

    min_dt = parse_jira_datetime(sorted_pairs[0]["pair_available_at"])
    max_dt = parse_jira_datetime(sorted_pairs[-1]["pair_available_at"])
    if min_dt is None or max_dt is None:
        return sorted_pairs, []

    sprint_days = 14

    def window_for_dt(dt: datetime) -> str:
        delta_days = (dt - min_dt).days
        idx = min(delta_days // sprint_days, window_count - 1)
        return f"W{idx + 1}"

    windows: list[dict[str, Any]] = []
    for row in sorted_pairs:
        dt = parse_jira_datetime(row["pair_available_at"])
        if dt is None:
            row["window"] = "W1"
        else:
            row["window"] = window_for_dt(dt)
        row["split"] = role_for_window(row["window"], split_config)

    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_pairs:
        by_window[row["window"]].append(row)

    for wn in range(1, window_count + 1):
        w = f"W{wn}"
        rows = by_window.get(w, [])
        if not rows:
            continue
        windows.append({
            "window": w,
            "role": role_for_window(w, split_config),
            "start_at": rows[0]["pair_available_at"],
            "end_at": rows[-1]["pair_available_at"],
            "count": len(rows),
        })

    return sorted_pairs, windows


def build_multi_repo_positive_pairs(
    payload: dict[str, Any],
    repositories: list[str],
    taxonomy: Taxonomy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build positive pairs across multiple repositories.

    Wraps build_positive_pairs() for each repository and aggregates results.

    Args:
        payload: The full processed JSON payload.
        repositories: List of repository names to process.
        taxonomy: The dependency taxonomy mapper.

    Returns:
        (all_pairs, aggregate_metadata) tuple.
    """
    all_pairs: list[dict[str, Any]] = []
    repo_metadata: dict[str, dict[str, Any]] = {}
    total_skipped: Counter = Counter()

    for repository in repositories:
        pairs, metadata = build_positive_pairs(payload, repository, taxonomy)
        all_pairs.extend(pairs)
        repo_metadata[repository] = metadata
        total_skipped.update(metadata.get("skipped", {}))

    aggregate_metadata = {
        "repositories": repositories,
        "total_positive_pairs": len(all_pairs),
        "repository_details": repo_metadata,
        "total_skipped": dict(total_skipped),
    }
    return all_pairs, aggregate_metadata


def window_distribution(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pair counts by window, split, and label."""
    summary: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[row["window"]].append(row)
    for window, rows in sorted(grouped.items()):
        summary[window] = {
            "split": rows[0]["split"] if rows else None,
            "count": len(rows),
            "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "dependency_type_counts": dict(sorted(Counter(row["dependency_type"] for row in rows).items())),
            "start_at": rows[0]["pair_available_at"] if rows else None,
            "end_at": rows[-1]["pair_available_at"] if rows else None,
        }
    return summary


def render_window_distribution_markdown(report: dict[str, Any]) -> str:
    """Render minimal window split report as Markdown."""
    lines = [
        "# Minimal Window Distribution",
        "",
        "## Assumptions",
        "",
        f"- Repository: `{report['repository']}`",
        f"- Window strategy: `{report['window_strategy']}`",
        f"- Pair timestamp policy: `{report['pair_available_at_policy']}`",
        "- Pair timestamp is `max(source_issue_created, target_issue_created)` because v1 processed pairs do not include link creation time.",
        "",
        "## Summary",
        "",
        f"- Positive pairs: `{report['positive_pairs']}`",
        f"- Requirements in repository: `{report['metadata']['requirements_in_repository']}`",
        f"- Skipped pairs: `{report['metadata']['skipped']}`",
        "",
        "## Windows",
        "",
        "| Window | Split | Count | Start | End | Label counts |",
        "|---|---|---:|---|---|---|",
    ]
    for window, details in report["distribution"].items():
        lines.append(
            f"| {window} | {details['split']} | {details['count']} | "
            f"{details['start_at']} | {details['end_at']} | `{details['label_counts']}` |"
        )
    lines.append("")
    return "\n".join(lines)
