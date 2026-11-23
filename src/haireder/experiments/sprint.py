"""Chronological uncertainty-first review with two resumable human handoffs."""

import json
from collections import Counter
from pathlib import Path

from haireder.online_learning.pool_manager import (
    build_pool_record,
    update_pool_snapshot,
)
from haireder.utils.io import read_jsonl, write_json, write_jsonl
from haireder.taxonomy import LABELS
from .evaluation import reject_substituted_turns, review_workload
from .review import review_queue, read_decisions, fingerprint


def reviewed(row, decision, kind):
    accepted = decision["accepted_label"]
    return dict(
        row,
        accepted_label=accepted,
        label=accepted,
        review_detail=decision,
        decision_path=kind + "_review",
        review_source=kind + "_review",
        human_reviewed=True,
        analyst_simulation=False,
        analyst_status="confirmed"
        if accepted == row["predicted_label"]
        else "corrected",
    )


def auto(row, promoted=False):
    return dict(
        row,
        accepted_label=row["predicted_label"],
        label=row["predicted_label"],
        human_reviewed=False,
        analyst_simulation=False,
        decision_path="medium_promoted" if promoted else "high_auto",
        review_source="automatic",
        analyst_status="auto_accepted",
    )


def records(rows, sequence):
    return [
        build_pool_record(
            r,
            pool_sequence=sequence + i,
            has_transcript=bool((r.get("transcript") or {}).get("rounds")),
        )
        for i, r in enumerate(rows)
    ]


def process_sprint(
    predictions,
    pool,
    previous_counts,
    config,
    directory,
    batch_id,
    reinfer,
    *,
    sequence=0,
):
    directory = Path(directory)
    if (
        not predictions
        or len({r["pair_id"] for r in predictions}) != len(predictions)
        or len({r["window"] for r in predictions}) != 1
    ):
        raise ValueError("Each sprint requires one unique prediction per candidate")
    n = int(predictions[0]["window"][1:])
    if any(int(r["window"][1:]) >= n for r in pool):
        raise ValueError("Starting pool contains current or future evidence")
    low = [r for r in predictions if r["confidence_region"] == "low"]
    medium = [r for r in predictions if r["confidence_region"] == "medium"]
    high = [r for r in predictions if r["confidence_region"] == "high"]
    if len(low) + len(medium) + len(high) != len(predictions):
        raise ValueError("Unrecognized confidence region")
    decisions = read_decisions(review_queue(directory / "low", batch_id, "low", low))
    low_done = [reviewed(r, decisions[r["pair_id"]], "low") for r in low]
    context = pool + records(low_done, sequence)
    # Keep all current low decisions in working memory regardless of long-term class quotas.
    medium_path = directory / "medium_predictions.jsonl"
    receipt = directory / "reinference.json"
    key = fingerprint([batch_id, medium, context])
    committed = receipt.exists()
    if committed:
        state = json.loads(receipt.read_text())
        corrected = read_jsonl(medium_path)
        if state != {"inputs": key, "outputs": fingerprint(corrected)}:
            raise ValueError("Re-inference evidence or its low-review context changed")
    else:
        corrected = reinfer(medium, context) if medium else []

    if [r["pair_id"] for r in corrected] != [r["pair_id"] for r in medium]:
        raise ValueError("Re-inference coverage mismatch")
    for old, new in zip(medium, corrected, strict=True):
        reject_substituted_turns(new)
        if (
            any(
                new.get(k) != old.get(k)
                for k in ("repo", "window", "split", "issue_i", "issue_j")
            )
            or new.get("predicted_label") not in LABELS
            or new.get("confidence_region") not in {"low", "medium", "high"}
            or type(new.get("confidence")) not in (int, float)
            or not 0 <= new["confidence"] <= 1
            or new.get("error")
        ):
            raise ValueError(f"Invalid re-inference for {old['pair_id']}")
    if not committed:
        write_jsonl(medium_path, corrected)
        write_json(receipt, {"inputs": key, "outputs": fingerprint(corrected)})
    remaining = [r for r in corrected if r["confidence_region"] != "high"]
    decisions = read_decisions(
        review_queue(
            directory / "medium", fingerprint([batch_id, low_done]), "medium", remaining
        )
    )
    merged = (
        low_done
        + [auto(r) for r in high]
        + [auto(r, True) for r in corrected if r["confidence_region"] == "high"]
        + [reviewed(r, decisions[r["pair_id"]], "medium") for r in remaining]
    )
    accepted = sorted(merged, key=lambda r: r["pair_id"])
    counts = Counter(r["accepted_label"] for r in accepted)
    bounded, quota = update_pool_snapshot(
        pool,
        records(accepted, sequence),
        capacity=config["pool_capacity"],
        recency_partition_ratio=config["recency_partition_ratio"],
        difficulty_partition_ratio=config["difficulty_partition_ratio"],
        quota_label_counts=previous_counts,
    )
    write_jsonl(directory / "accepted.jsonl", accepted)
    write_jsonl(directory / "pool.jsonl", bounded)
    write_json(
        directory / "summary.json",
        dict(review_workload(accepted), accepted_counts=counts, quota=quota),
    )
    return accepted, bounded
