"""Build LoRA refresh corpora from accepted decisions and pool replay."""

from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from typing import Any

from haireder.augmentation import augment_minority_classes
from haireder.taxonomy.labels import LABELS


def pool_record_to_training_row(record: dict[str, Any], *, split: str = "online") -> dict[str, Any]:
    label = str(record.get("accepted_label") or record.get("label") or "")
    if label not in LABELS:
        raise ValueError(f"Unknown label in pool record: {label}")
    row = {
        "pair_id": record["pair_id"],
        "repo": record.get("repo"),
        "window": record.get("window"),
        "split": split,
        "issue_i": record.get("issue_i"),
        "issue_j": record.get("issue_j"),
        "label": label,
        "gold_label": label,
        "accepted_label": label,
        "source": "online_learning_pool_replay",
        "transcript": record.get("transcript"),
        "original_predicted_label": record.get("original_predicted_label"),
        "confidence": record.get("confidence"),
        "review_source": record.get("review_source"),
        "human_reviewed": record.get("human_reviewed") is True,
        "review_detail": record.get("review_detail"),
        "decision_path": record.get("decision_path"),
        "analyst_status": record.get("analyst_status"),
    }
    return row


def _accepted_to_training_row(row: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(row)
    label = str(output.get("accepted_label") or output.get("gold_label") or output.get("label") or "")
    if label not in LABELS:
        raise ValueError(f"Unknown accepted label: {label}")
    output["label"] = label
    output["gold_label"] = label
    output["accepted_label"] = label
    output["source"] = output.get("source") or "online_learning_accepted_decision"
    return output


def _sample_rows(rows: list[dict[str, Any]], *, target_count: int, seed: int) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if len(rows) <= target_count:
        return list(rows)
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("label") or ""), []).append(row)
    selected: list[dict[str, Any]] = []
    labels = sorted(grouped)
    while len(selected) < target_count:
        added = False
        for label in labels:
            bucket = grouped[label]
            if not bucket:
                continue
            index = rng.randrange(len(bucket))
            selected.append(bucket.pop(index))
            added = True
            if len(selected) == target_count:
                break
        if not added:
            break
    return selected


def build_refresh_corpus(
    accepted_rows: list[dict[str, Any]],
    pool_records: list[dict[str, Any]],
    *,
    max_rows: int | None = None,
    new_data_ratio: float = 0.7,
    replay_ratio: float = 0.3,
    augment_to_balance: bool = True,
    max_augmented_variants_per_seed: int = 2,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a single unified refresh corpus from new accepted rows plus pool replay.

    Produces ONE corpus for all three roles (proposer, critic, judge). If role-
    specific corpora are needed, call this function with role-filtered inputs.
    The corpus_role field is set to 'unified_all_roles' to make this intent
    explicit in the output.
    """
    if new_data_ratio < 0 or replay_ratio < 0:
        raise ValueError("new_data_ratio and replay_ratio must be non-negative")
    new_rows = [_accepted_to_training_row(row) for row in accepted_rows]
    new_ids = {r["pair_id"] for r in new_rows}
    if len(new_ids) != len(new_rows):
        raise ValueError("Duplicate new training pair")
    historical = {record["pair_id"]: record for record in pool_records if record["pair_id"] not in new_ids}
    replay_rows = [pool_record_to_training_row(record) for record in historical.values()]

    if max_rows is None:
        selected_new = new_rows
        replay_target = int(round(len(new_rows) * replay_ratio / max(new_data_ratio, 0.0001)))
        selected_replay = _sample_rows(replay_rows, target_count=replay_target, seed=seed)
    else:
        selected_new_target = min(len(new_rows), int(round(max_rows * new_data_ratio)))
        selected_new = _sample_rows(new_rows, target_count=selected_new_target, seed=seed)
        selected_replay = _sample_rows(
            replay_rows,
            target_count=max(0, max_rows - len(selected_new)),
            seed=seed + 1,
        )

    corpus = [*selected_new, *selected_replay]
    augmented_rows: list[dict[str, Any]] = []
    augmentation_summary: dict[str, Any] | None = None
    if augment_to_balance and corpus:
        augmented_rows, augmentation_summary = augment_minority_classes(
            corpus,
            max_variants_per_seed=max_augmented_variants_per_seed,
            include_none=False,
            seed=seed,
        )
        corpus = [*corpus, *augmented_rows]

    summary = {
        "corpus_role": "unified_all_roles",
        "accepted_input_rows": len(accepted_rows),
        "pool_input_rows": len(pool_records),
        "selected_new_rows": len(selected_new),
        "selected_new_sources": dict(Counter(r.get("review_source") or "unspecified" for r in selected_new)),
        "selected_replay_rows": len(selected_replay),
        "distinct_historical_candidates": len(historical),
        "achieved_replay_fraction": len(selected_replay) / (len(selected_new) + len(selected_replay)) if selected_new or selected_replay else None,
        "augmented_rows": len(augmented_rows),
        "output_rows": len(corpus),
        "label_counts": dict(sorted(Counter(str(row.get("label") or "") for row in corpus).items())),
        "augment_to_balance": augment_to_balance,
        "augmentation": augmentation_summary,
    }
    return corpus, summary


def build_role_specific_refresh_corpus(
    accepted_rows: list[dict[str, Any]],
    pool_records: list[dict[str, Any]],
    *,
    max_rows: int | None = None,
    new_data_ratio: float = 0.7,
    replay_ratio: float = 0.3,
    augment_to_balance: bool = True,
    max_augmented_variants_per_seed: int = 2,
    seed: int = 42,
    role_filter_enabled: bool = False,
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Build separate refresh corpora for each agent role.

    Produces THREE corpora aligned with each agent's epistemic responsibility:
    - Proposer: high-confidence accepted rows + recency pool records.
    - Critic: low-confidence / corrected rows + difficulty pool records.
    - Judge: rows that carry debate transcripts + transcript pool records.

    Args:
        accepted_rows: New accepted decision rows.
        pool_records: Historical pool records.
        max_rows: Cap per role corpus.
        new_data_ratio: Fraction of accepted rows to include.
        replay_ratio: Replay fraction relative to new data.
        augment_to_balance: Whether to apply minority-class augmentation.
        max_augmented_variants_per_seed: Max variants per seed.
        seed: Random seed.
        role_filter_enabled: When False, all roles get the same unfiltered corpus.

    Returns:
        Dict mapping role name → (corpus_rows, summary).
    """
    if not role_filter_enabled:
        # Fallback: all roles get the same unified corpus
        corpus, summary = build_refresh_corpus(
            accepted_rows, pool_records,
            max_rows=max_rows, new_data_ratio=new_data_ratio,
            replay_ratio=replay_ratio, augment_to_balance=augment_to_balance,
            max_augmented_variants_per_seed=max_augmented_variants_per_seed,
            seed=seed,
        )
        shared = (corpus, summary)
        return {"proposer": shared, "critic": shared, "judge": shared}

    from haireder.online_learning.pool_manager import retrieve_for_role

    role_corpora: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}

    for role in ("proposer", "critic", "judge"):
        # Filter accepted rows by role
        role_accepted = [
            row for row in accepted_rows
            if row.get("role_intended") is None or row.get("role_intended") == role
        ]

        # Retrieve role-specific pool records
        role_pool = retrieve_for_role(pool_records, role, k=len(pool_records), role_filter_enabled=True)

        corpus, summary = build_refresh_corpus(
            role_accepted, role_pool,
            max_rows=max_rows, new_data_ratio=new_data_ratio,
            replay_ratio=replay_ratio, augment_to_balance=augment_to_balance,
            max_augmented_variants_per_seed=max_augmented_variants_per_seed,
            seed=seed + sum(map(ord, role)) % 10000,
        )
        summary["corpus_role"] = role
        role_corpora[role] = (corpus, summary)

    return role_corpora
