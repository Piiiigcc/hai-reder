"""Blinded human review with immutable batch identity and editable UTF-8 CSV."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

from haireder.taxonomy import LABELS
from haireder.utils.io import read_jsonl, write_json, write_jsonl


def fingerprint(value):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class PendingReviews(Exception):
    def __init__(self, path, pending, total):
        self.path, self.pending, self.total = Path(path), pending, total
        super().__init__(f"{pending} of {total} human decisions pending in {path}")


FIELDS = (
    "batch_id",
    "pair_id",
    "item_hash",
    "kind",
    "issue_i",
    "issue_j",
    "seed_i",
    "seed_j",
    "proposed_label",
    "model_rationale",
    "accepted_label",
    "preserves_relation",
    "reviewer",
    "reviewed_at",
    "rationale",
)


def low_review_rationales(row):
    transcript = row.get("transcript") or {}
    turns = []
    for number, exchange in enumerate(transcript.get("rounds", []), 1):
        for role in ("proposer", "critic"):
            turns.append((f"Round {number} {role}", exchange.get(role, {})))
    if transcript.get("proposer"):
        turns.append(("Proposer", transcript["proposer"]))
    if transcript.get("judge"):
        turns.append(("Judge", transcript["judge"]))
    notes = []
    for name, turn in turns:
        output = turn.get("parsed_output") or {}
        reason = output.get("rationale") or output.get("justification")
        if reason:
            notes.append(f"{name}: {reason}")
    if not notes and row.get("rationale"):
        notes.append(row["rationale"])
    if not notes:
        raise ValueError(f"Missing model reasoning for low review of {row['pair_id']}")
    return "\n".join(notes)


def review_queue(
    directory, batch_id, kind, rows, *, seeds=None, sample_size=None, seed=42
):
    """Create once; a resumed run can never overwrite an analyst's entries."""
    if kind not in ("augmentation", "low", "medium", "online_intake"):
        raise ValueError(f"Unknown review kind {kind}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["pair_id"])
    if len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review candidate")
    population = len(rows)
    if sample_size is not None:
        if population < sample_size:
            raise ValueError(
                f"Need {sample_size} complete review candidates, have {population}"
            )
        rows = random.Random(seed).sample(rows, sample_size)
    batch = []
    for row in rows:
        item = dict(
            batch_id=batch_id,
            kind=kind,
            pair_id=row["pair_id"],
            repo=row.get("repo"),
            window=row.get("window"),
            issue_i=row["issue_i"]["text"],
            issue_j=row["issue_j"]["text"],
        )
        if kind == "augmentation":
            parent = seeds[row["source_seed_pair_id"]]
            item.update(
                seed_id=parent["pair_id"],
                seed_i=parent["issue_i"]["text"],
                seed_j=parent["issue_j"]["text"],
            )
        if kind == "low":
            item["model_rationale"] = low_review_rationales(row)
        if kind == "medium":
            turn = row.get("transcript", {}).get("judge", {}).get("parsed_output") or {}
            item.update(
                proposed_label=row["predicted_label"],
                model_rationale=turn.get("justification") or row.get("rationale"),
                transcript=row.get("transcript"),
                confidence=row["confidence"],
            )
        item["item_hash"] = fingerprint(item)
        batch.append(item)
    path = directory / "review.csv"
    manifest = {
        "batch_id": batch_id,
        "kind": kind,
        "population": population,
        "sample_size": len(batch),
        "sampling_seed": seed if sample_size is not None else None,
        "content_hash": fingerprint(batch),
    }
    manifest_path = directory / "batch.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            f"Review batch changed at {directory}. Preserve the old reviews and select a new run."
        )
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    evidence = directory / "items.jsonl"
    if (
        evidence.exists()
        and fingerprint(read_jsonl(evidence)) != manifest["content_hash"]
    ):
        raise ValueError(f"Review evidence changed at {directory}")
    if not evidence.exists():
        write_jsonl(evidence, batch)
    if not path.exists():
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(batch)
        Path(handle.name).replace(path)
    return path


def read_decisions(path, *, require_all=True):
    path = Path(path)
    manifest = json.loads((path.parent / "batch.json").read_text())
    evidence = read_jsonl(path.parent / "items.jsonl")
    if fingerprint(evidence) != manifest["content_hash"]:
        raise ValueError("Review evidence fingerprint mismatch")
    items = {r["pair_id"]: r for r in evidence}
    decisions = {}
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Review CSV header mismatch")
        for row in reader:
            key = row["pair_id"]
            item = items.get(key)
            if (
                key in seen
                or item is None
                or any(
                    (row.get(k) or "") != str(item.get(k) or "") for k in FIELDS[:10]
                )
            ):
                raise ValueError(f"Duplicate, unbound or edited review input for {key}")
            seen.add(key)
            if not any(row[k].strip() for k in FIELDS[10:]):
                continue
            if (
                row["accepted_label"] not in LABELS
                or not row["reviewer"].strip()
                or not row["rationale"].strip()
            ):
                raise ValueError(f"Incomplete decision for {key}")
            stamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError(f"Review timestamp requires offset for {key}")
            if item["kind"] == "augmentation" and row[
                "preserves_relation"
            ].lower() not in {"yes", "no"}:
                raise ValueError(f"Enter yes or no for preserves_relation on {key}")
            decisions[key] = {k: row[k] for k in (*FIELDS[:4], *FIELDS[10:])}
    if seen != set(items):
        raise ValueError("Review CSV rows differ from the bound batch")
    if require_all and len(decisions) != len(items):
        raise PendingReviews(path, len(items) - len(decisions), len(items))
    return decisions


def import_independent_decisions(intake, low_queue):
    """Make matching intake notes available before a reviewer signs the aided low queue."""
    from .stages import run_lock

    intake, low_queue = Path(intake), Path(low_queue)
    source = read_decisions(intake, require_all=False)
    old_items = {i["pair_id"]: i for i in read_jsonl(intake.parent / "items.jsonl")}
    new_items = {i["pair_id"]: i for i in read_jsonl(low_queue.parent / "items.jsonl")}
    source_batch = json.loads((intake.parent / "batch.json").read_text())
    destination = json.loads((low_queue.parent / "batch.json").read_text())
    if source_batch["kind"] != "online_intake" or destination["kind"] != "low":
        raise ValueError(
            "Pre-triage judgments may be imported only into a matching independent low-review queue"
        )
    with run_lock(low_queue.parent):
        existing = read_decisions(low_queue, require_all=False)
        notes = []
        for key in sorted((source.keys() & new_items.keys()) - existing.keys()):
            if any(
                old_items[key][k] != new_items[key][k]
                for k in ("repo", "window", "issue_i", "issue_j")
            ):
                raise ValueError(f"Intake content differs for {key}")
            notes.append(
                dict(low_item_hash=new_items[key]["item_hash"], intake=source[key])
            )
        write_json(
            low_queue.parent / "intake_import.json",
            dict(
                source_batch=source_batch["batch_id"],
                target_batch=destination["batch_id"],
                source_decisions=fingerprint(source),
                notes=notes,
            ),
        )
    return len(notes)
