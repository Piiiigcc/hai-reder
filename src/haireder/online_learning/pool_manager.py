"""Example-pool candidate records and bounded pool snapshots."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from haireder.taxonomy.labels import LABELS


def _accepted_label(row: dict[str, Any]) -> str:
    label = str(row.get("accepted_label") or row.get("label") or row.get("gold_label") or "")
    if label not in LABELS:
        raise ValueError(f"Unknown label for pool record: {label}")
    return label


def _difficulty_score(row: dict[str, Any]) -> float:
    confidence = float(row.get("confidence") or row.get("calibrated_confidence") or 0.0)
    corrected = 1.0 if row.get("analyst_status") == "corrected" else 0.0
    low_region = 1.0 if row.get("confidence_region") == "low" else 0.0
    return corrected + low_region + (1.0 - confidence)


def build_pool_record(
    row: dict[str, Any],
    *,
    pool_sequence: int,
    role_intended: str | None = None,
    has_transcript: bool = False,
) -> dict[str, Any]:
    """Build an append-only candidate record for the retrieval/example pool.

    Args:
        row: Accepted decision row.
        pool_sequence: Monotonic sequence number for FIFO ordering.
        role_intended: Target agent role ("proposer", "critic", "judge", or None).
        has_transcript: Whether this record carries a full debate transcript.
    """
    accepted_label = _accepted_label(row)
    review_source = str(row.get("review_source") or "unknown")
    partition = "difficulty" if review_source in {"low_review", "medium_review"} or row.get("analyst_status") == "corrected" else "recency"
    transcript = row.get("transcript")
    record = {
        "pair_id": row["pair_id"],
        "window": row.get("window"),
        "repo": row.get("repo"),
        "issue_i": row.get("issue_i"),
        "issue_j": row.get("issue_j"),
        "original_predicted_label": row.get("predicted_label"),
        "accepted_label": accepted_label,
        "label": accepted_label,
        "original_confidence": row.get("confidence"),
        "confidence": row.get("confidence"),
        "confidence_region": row.get("confidence_region"),
        "review_source": review_source,
        "human_reviewed": row.get("human_reviewed") is True,
        "review_detail": row.get("review_detail"),
        "analyst_status": row.get("analyst_status"),
        "decision_path": row.get("decision_path"),
        "s1": row.get("s1"),
        "s2": row.get("s2"),
        "s3": row.get("s3"),
        "calibration_mode": row.get("calibration_mode"),
        "partition": partition,
        "difficulty_score": _difficulty_score(row),
        "pool_sequence": int(pool_sequence),
        "role_intended": role_intended,
        "has_transcript": bool(has_transcript or (transcript or {}).get("rounds")),
    }
    if transcript:
        record["transcript"] = transcript
    return record


def compute_smoothed_label_quotas(
    label_counts: dict[str, int],
    *,
    capacity: int,
    alpha: float = 1.0,
) -> dict[str, int]:
    """Allocate fixed capacity proportionally to smoothed accepted-label counts."""
    if capacity <= 0:
        return {label: 0 for label in LABELS}
    smoothed = {label: float(label_counts.get(label, 0)) + alpha for label in LABELS}
    total = sum(smoothed.values())
    raw = {label: smoothed[label] * capacity / total for label in LABELS}
    quotas = {label: int(raw[label]) for label in LABELS}
    remaining = capacity - sum(quotas.values())
    order = sorted(LABELS, key=lambda label: (raw[label] - quotas[label], label), reverse=True)
    for label in order[:remaining]:
        quotas[label] += 1
    return quotas


def _dedupe_latest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["pair_id"])] = record
    return list(latest.values())


def _select_partition_records(
    records: list[dict[str, Any]],
    *,
    capacity: int,
    quotas: dict[str, int],
    partition: str,
) -> list[dict[str, Any]]:
    if capacity <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("accepted_label") or record.get("label"))].append(record)

    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = grouped.get(label, [])
        if partition == "difficulty":
            candidates = sorted(
                candidates,
                key=lambda row: (float(row.get("difficulty_score") or 0.0), int(row.get("pool_sequence") or 0), str(row.get("pair_id"))),
                reverse=True,
            )
        else:
            candidates = sorted(
                candidates,
                key=lambda row: (int(row.get("pool_sequence") or 0), str(row.get("pair_id"))),
                reverse=True,
            )
        selected.extend(candidates[: quotas.get(label, 0)])

    return selected[:capacity]


def update_pool_snapshot(
    existing_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    capacity: int,
    recency_partition_ratio: float = 0.6,
    difficulty_partition_ratio: float = 0.4,
    alpha: float = 1.0,
    quota_label_counts: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge candidate records and keep a bounded recency+difficulty snapshot."""
    all_records = _dedupe_latest([*existing_records, *candidate_records])
    label_counts = Counter(str(record.get("accepted_label") or record.get("label")) for record in all_records)
    quota_counts = dict(quota_label_counts) if quota_label_counts is not None else dict(label_counts)
    recency_capacity = int(round(capacity * recency_partition_ratio))
    difficulty_capacity = max(0, capacity - recency_capacity)
    if difficulty_partition_ratio > 0 and recency_partition_ratio <= 0:
        difficulty_capacity = capacity
    quotas_recency = compute_smoothed_label_quotas(quota_counts, capacity=recency_capacity, alpha=alpha)
    quotas_difficulty = compute_smoothed_label_quotas(quota_counts, capacity=difficulty_capacity, alpha=alpha)

    recency_records = [record for record in all_records if record.get("partition") == "recency"]
    difficulty_records = [record for record in all_records if record.get("partition") == "difficulty"]
    selected = [
        *_select_partition_records(
            recency_records,
            capacity=recency_capacity,
            quotas=quotas_recency,
            partition="recency",
        ),
        *_select_partition_records(
            difficulty_records,
            capacity=difficulty_capacity,
            quotas=quotas_difficulty,
            partition="difficulty",
        ),
    ]
    selected = _dedupe_latest(selected)
    if len(selected) > capacity:
        selected = sorted(
            selected,
            key=lambda row: (int(row.get("pool_sequence") or 0), str(row.get("pair_id"))),
            reverse=True,
        )[:capacity]
    selected = sorted(selected, key=lambda row: (str(row.get("window") or ""), int(row.get("pool_sequence") or 0), str(row.get("pair_id"))))
    summary = {
        "capacity": capacity,
        "input_existing_records": len(existing_records),
        "input_candidate_records": len(candidate_records),
        "snapshot_records": len(selected),
        "label_counts": dict(sorted(Counter(str(row.get("accepted_label") or row.get("label")) for row in selected).items())),
        "partition_counts": dict(sorted(Counter(str(row.get("partition")) for row in selected).items())),
        "recency_capacity": recency_capacity,
        "difficulty_capacity": difficulty_capacity,
        "quota_label_counts_source": "provided" if quota_label_counts is not None else "pool_snapshot",
        "quota_label_counts": dict(sorted(quota_counts.items())),
        "recency_quotas": quotas_recency,
        "difficulty_quotas": quotas_difficulty,
    }
    return selected, summary


def retrieve_for_role(
    pool: list[dict[str, Any]],
    role: str,
    k: int = 5,
    *,
    role_filter_enabled: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve top-k pool records filtered by agent role.

    Implements the role-specific retrieval described in Section 3.5.2:
    - Proposer: high-confidence canonical examples from recency partition.
    - Critic: difficulty-partition pairs (originally misclassified or low-confidence).
    - Judge: pairs that carry full debate transcripts.

    Args:
        pool: The full pool snapshot (list of pool records).
        role: Agent role — "proposer", "critic", or "judge".
        k: Maximum number of records to return.
        role_filter_enabled: When False, returns the pool unfiltered (default).

    Returns:
        Filtered and ranked list of pool records, up to k items.
    """
    if not pool or k <= 0:
        return []

    if not role_filter_enabled:
        return list(pool)[:k]

    candidates = list(pool)

    if role == "proposer":
        # Prefer recency partition with high confidence canonical examples
        candidates.sort(
            key=lambda row: (
                0 if row.get("partition") == "recency" else 1,
                0 if row.get("confidence_region") == "high" else 1,
                -float(row.get("confidence") or row.get("original_confidence") or 0.0),
                -int(row.get("pool_sequence") or 0),
            ),
        )
    elif role == "critic":
        # Prefer difficulty partition with low-confidence / corrected pairs
        candidates.sort(
            key=lambda row: (
                0 if row.get("partition") == "difficulty" else 1,
                -float(row.get("difficulty_score") or 0.0),
                0 if row.get("analyst_status") == "corrected" else 1,
                -int(row.get("pool_sequence") or 0),
            ),
        )
    elif role == "judge":
        # Prefer pairs that carry full debate transcripts
        candidates.sort(
            key=lambda row: (
                0 if row.get("has_transcript") else 1,
                -int(row.get("pool_sequence") or 0),
            ),
        )
    else:
        # Unknown role: no filtering
        pass

    return candidates[:k]
