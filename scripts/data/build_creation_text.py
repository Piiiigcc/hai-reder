#!/usr/bin/env python3
"""Recover creation-time text and recorded links from the original MongoDB archive."""

import argparse
import gzip
import json
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

from bson import BSON

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from haireder.utils.config import load_yaml
from haireder.data.windowing import parse_jira_datetime
from datetime import datetime, timezone
from haireder.utils.io import write_jsonl
from extract_public_jira_requirements import REQUIREMENT_TYPES


def creation_text(issue):
    fields = issue["fields"]
    change = issue.get("changelog") or {}
    histories = change.get("histories", [])
    complete = change.get("startAt") == 0 and change.get("total") == len(histories)
    if not complete and fields.get("updated") != fields.get("created"):
        return None
    values = {key: fields.get(key) or "" for key in ("summary", "description")}
    for history in sorted(
        histories,
        key=lambda h: (parse_jira_datetime(h["created"]), int(h.get("id") or 0)),
        reverse=True,
    ):
        for item in reversed(history.get("items") or []):
            name = str(item.get("field", "")).lower()
            if name in values:
                if "fromString" not in item:
                    return None
                values[name] = item["fromString"] or ""
    if not all(isinstance(v, str) for v in values.values()) or not any(values.values()):
        return None
    return values


def source_rows(archive_path, repositories, end, counts):
    end_at = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    with zipfile.ZipFile(archive_path) as archive:
        fields = json.loads(
            archive.read(
                next(
                    n
                    for n in archive.namelist()
                    if n.endswith("/jira_field_information.json")
                )
            )
        )
        epic = {
            repo: [f["id"] for f in entries if f["name"] == "Epic Link"]
            for repo, entries in fields.items()
        }
        member = next(n for n in archive.namelist() if n.endswith(".archive"))
        with archive.open(member) as raw, gzip.GzipFile(fileobj=raw) as stream:
            stream.read(4)
            collection = None
            while prefix := stream.read(4):
                size = struct.unpack("<i", prefix)[0]
                if size == -1:
                    continue
                if not 5 <= size <= 100_000_000:
                    raise ValueError(f"Invalid BSON length {size}")
                body = stream.read(size - 4)
                if len(body) != size - 4:
                    raise EOFError("Incomplete BSON record")
                if collection not in repositories and body[:1] == b"\x07":
                    continue
                doc = BSON(prefix + body).decode()
                if isinstance(doc.get("collection"), str):
                    collection = doc["collection"]
                fields = doc.get("fields")
                if collection not in repositories or not fields or not doc.get("key"):
                    continue
                if (fields.get("issuetype") or {}).get("name") not in REQUIREMENT_TYPES[
                    collection
                ]:
                    continue
                created = parse_jira_datetime(fields.get("created"))
                if created is None or created >= end_at:
                    continue
                links = {
                    linked["key"]
                    for link in fields.get("issuelinks") or []
                    for side in ("outwardIssue", "inwardIssue")
                    if (linked := link.get(side)) and linked.get("key")
                }
                links.update(
                    s["key"] for s in fields.get("subtasks") or [] if s.get("key")
                )
                if (fields.get("parent") or {}).get("key"):
                    links.add(fields["parent"]["key"])
                links.update(
                    fields[name]
                    for name in epic.get(collection, [])
                    if isinstance(fields.get(name), str) and fields[name]
                )
                values = creation_text(doc)
                counts[
                    "text_reconstructed"
                    if values is not None
                    else "incomplete_history_or_text"
                ] += 1
                yield {
                    "source": collection,
                    "issue_key": doc["key"],
                    "created": fields["created"],
                    "creation_text": values,
                    "recorded_targets": sorted(links),
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config", default="configs/dataset/formal_6repos.yaml"
    )
    parser.add_argument(
        "--out", default="data/processed/public_jira/formal_creation_history.jsonl"
    )
    args = parser.parse_args()
    config = load_yaml(args.dataset_config)
    counts = Counter()
    out = ROOT / args.out
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out.parent, prefix=".history-", delete=False
    ) as fd:
        tmp = Path(fd.name)
    try:
        write_jsonl(
            tmp,
            source_rows(
                ROOT / config["raw_sources"]["public_jira_zip"],
                set(config["target_repositories"]),
                config["formal_run"]["end_date"],
                counts,
            ),
        )
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(counts))
