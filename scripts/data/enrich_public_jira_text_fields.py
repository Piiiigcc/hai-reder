#!/usr/bin/env python3
"""Enrich the current Public Jira requirement data with raw text fields.

It reads the extracted requirement pairs and scans the original
ThePublicJiraDataset mongodump archive, recovers fields.description and
fields.comments, and writes issue-level and pair-level JSONL companions.
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRENT_JSON = ROOT / "data/interim/public_jira/requirement_pairs_dependencies.json"
DEFAULT_SOURCE_ZIP = ROOT / "data/raw/public_jira/2025-06-23 ThePublicJiraDataset.zip"
DEFAULT_ARCHIVE_IN_ZIP = "ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"
DEFAULT_REQUIREMENTS_OUT = ROOT / "data/interim/public_jira/requirements_text_enriched.jsonl"
DEFAULT_PAIRS_OUT = ROOT / "data/interim/public_jira/dependency_pairs_text_enriched.jsonl"
DEFAULT_SUMMARY_OUT = ROOT / "outputs/logs/public_jira_text_enrichment_summary.json"


REQUIREMENT_TYPES: dict[str, set[str]] = {
    "Apache": {"Epic", "Story", "New Feature", "Improvement", "Request", "Wish", "Brainstorming"},
    "Hyperledger": {"Epic", "Story", "New Feature", "Improvement"},
    "IntelDAOS": {"Epic", "Story"},
    "JFrog": {"Epic", "Story", "New Feature", "Feature Request", "Improvement", "Change Request"},
    "Jira": {"New Feature", "Improvement", "Suggestion"},
    "JiraEcosystem": {"Epic", "Story", "New Feature", "Improvement", "Suggestion"},
    "MariaDB": {"Epic", "New Feature"},
    "MongoDB": {"Epic", "Story", "New Feature", "Improvement"},
    "Qt": {"Epic", "User Story", "Improvement", "Suggestion", "Change Request"},
    "RedHat": {
        "Epic",
        "Market Problem",
        "Initiative",
        "Story",
        "Requirement",
        "New Feature",
        "Feature",
        "Feature Request",
        "Improvement",
        "Enhancement",
    },
    "Sonatype": {"Epic", "Story", "Improvement"},
    "Spring": {"Epic", "Story", "New Feature", "Improvement"},
    "Sakai": {"Roadmap item", "Feature Request", "Improvement"},
    "SecondLife": {"Story", "New Feature"},
    "Mindville": {"New Feature"},
    "Mojang": set(),
}


BSON_FIXED = {
    0x01: 8,
    0x07: 12,
    0x08: 1,
    0x09: 8,
    0x0A: 0,
    0x10: 4,
    0x11: 8,
    0x12: 8,
    0x13: 16,
    0xFF: 0,
    0x7F: 0,
    0x06: 0,
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_bson_string(data: bytes, off: int) -> tuple[str, int]:
    n = struct.unpack_from("<i", data, off)[0]
    if n <= 0:
        return "", off + 4
    end = off + 4 + n
    try:
        return data[off + 4 : end - 1].decode("utf-8", "replace"), end
    except UnicodeDecodeError:
        return "", end


def read_cstring(data: bytes, off: int) -> tuple[str, int]:
    end = data.index(b"\x00", off)
    try:
        return data[off:end].decode("utf-8", "replace"), end + 1
    except UnicodeDecodeError:
        return "", end + 1


def skip_bson_value(data: bytes, off: int, element_type: int) -> int:
    if element_type in BSON_FIXED:
        return off + BSON_FIXED[element_type]
    if element_type in (0x03, 0x04):
        return off + struct.unpack_from("<i", data, off)[0]
    if element_type == 0x02:
        return off + 4 + max(0, struct.unpack_from("<i", data, off)[0])
    if element_type == 0x05:
        return off + 5 + struct.unpack_from("<i", data, off)[0]
    if element_type == 0x0B:
        _, off = read_cstring(data, off)
        _, off = read_cstring(data, off)
        return off
    if element_type == 0x0C:
        _, off = read_bson_string(data, off)
        return off + 12
    if element_type in (0x0D, 0x0E):
        _, off = read_bson_string(data, off)
        return off
    if element_type == 0x0F:
        return off + struct.unpack_from("<i", data, off)[0]
    return off + 4


def iter_bson_doc(data: bytes, off: int) -> Iterable[tuple[str, int, int]]:
    doc_size = struct.unpack_from("<i", data, off)[0]
    pos = off + 4
    end = off + doc_size - 1
    while pos < end:
        if data[pos] == 0:
            break
        element_type = data[pos]
        pos += 1
        name, pos = read_cstring(data, pos)
        value_pos = pos
        yield name, element_type, value_pos
        pos = skip_bson_value(data, value_pos, element_type)


def iter_issue_top(data: bytes) -> Iterable[tuple[str, int, int]]:
    """Iterate fields in type 0x07 records, where _id has no BSON type byte."""
    if len(data) < 16 or data[0:4] != b"_id\x00":
        return
    pos = 16
    end = len(data) - 1
    while pos < end:
        if data[pos] == 0:
            break
        element_type = data[pos]
        pos += 1
        name, pos = read_cstring(data, pos)
        value_pos = pos
        yield name, element_type, value_pos
        pos = skip_bson_value(data, value_pos, element_type)


def find_string_in_doc(data: bytes, off: int, field_name: str) -> str | None:
    for name, element_type, value_pos in iter_bson_doc(data, off):
        if name == field_name and element_type == 0x02:
            return read_bson_string(data, value_pos)[0]
    return None


def truncate_text(value: str | None, max_chars: int) -> tuple[str, bool]:
    if not value:
        return "", False
    text = value.replace("\x00", " ").strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars], True
    return text, False


def assign_source(issuetype: str | None, collection: str | None = None) -> str | None:
    return collection if collection and issuetype in REQUIREMENT_TYPES.get(collection, ()) else None


def extract_collection_name(data: bytes) -> str | None:
    idx = data.find(b"collection\x00")
    if idx < 0:
        return None
    start = idx + len(b"collection\x00")
    if start + 4 > len(data):
        return None
    string_len = struct.unpack_from("<i", data, start)[0]
    if string_len <= 0 or string_len >= 256 or start + 4 + string_len > len(data):
        return None
    try:
        return data[start + 4 : start + 4 + string_len - 1].decode("utf-8", "replace")
    except UnicodeDecodeError:
        return None


def parse_issue_header(data: bytes) -> tuple[str | None, int | None]:
    issue_key = None
    fields_off = None
    for name, element_type, value_pos in iter_issue_top(data):
        if name == "key" and element_type == 0x02:
            issue_key = read_bson_string(data, value_pos)[0]
        elif name == "fields" and element_type == 0x03:
            fields_off = value_pos
        if issue_key and fields_off is not None:
            break
    return issue_key, fields_off


def find_issuetype(data: bytes, fields_off: int) -> str | None:
    for name, element_type, value_pos in iter_bson_doc(data, fields_off):
        if name == "issuetype" and element_type == 0x03:
            return find_string_in_doc(data, value_pos, "name")
    return None


def parse_comment_doc(data: bytes, off: int, max_body_chars: int) -> dict[str, Any]:
    comment: dict[str, Any] = {
        "id": "",
        "created": "",
        "updated": "",
        "body": "",
        "body_truncated": False,
    }
    for name, element_type, value_pos in iter_bson_doc(data, off):
        if name in {"id", "created", "updated"} and element_type == 0x02:
            comment[name] = read_bson_string(data, value_pos)[0]
        elif name == "body" and element_type == 0x02:
            body, truncated = truncate_text(read_bson_string(data, value_pos)[0], max_body_chars)
            comment["body"] = body
            comment["body_truncated"] = truncated
    return comment


def parse_comments_array(
    data: bytes,
    off: int,
    max_comment_bodies: int,
    max_comment_body_chars: int,
) -> tuple[int, list[dict[str, Any]], int]:
    array_size = struct.unpack_from("<i", data, off)[0]
    pos = off + 4
    end = off + array_size - 1
    comment_count = 0
    excerpts: list[dict[str, Any]] = []
    truncated_bodies = 0

    while pos < end:
        if data[pos] == 0:
            break
        element_type = data[pos]
        pos += 1
        _, pos = read_cstring(data, pos)
        value_pos = pos
        if element_type == 0x03:
            comment_count += 1
            if len(excerpts) < max_comment_bodies:
                comment = parse_comment_doc(data, value_pos, max_comment_body_chars)
                if comment["body_truncated"]:
                    truncated_bodies += 1
                excerpts.append(comment)
        pos = skip_bson_value(data, value_pos, element_type)

    return comment_count, excerpts, truncated_bodies


def parse_comment_container(
    data: bytes,
    off: int,
    max_comment_bodies: int,
    max_comment_body_chars: int,
) -> tuple[int, list[dict[str, Any]], int]:
    total = None
    excerpts: list[dict[str, Any]] = []
    truncated_bodies = 0
    counted = 0

    for name, element_type, value_pos in iter_bson_doc(data, off):
        if name == "total" and element_type == 0x10:
            total = struct.unpack_from("<i", data, value_pos)[0]
        elif name == "total" and element_type == 0x12:
            total = struct.unpack_from("<q", data, value_pos)[0]
        elif name in {"comments", "comment"} and element_type == 0x04:
            counted, excerpts, truncated_bodies = parse_comments_array(
                data,
                value_pos,
                max_comment_bodies,
                max_comment_body_chars,
            )

    return int(total if total is not None else counted), excerpts, truncated_bodies


def parse_text_enrichment(
    data: bytes,
    fields_off: int,
    max_description_chars: int,
    max_comment_bodies: int,
    max_comment_body_chars: int,
) -> dict[str, Any]:
    enrichment: dict[str, Any] = {
        "raw_summary": "",
        "description": "",
        "description_truncated": False,
        "comment_count": 0,
        "comment_excerpt_count": 0,
        "comment_excerpts": [],
        "comment_body_truncated_count": 0,
    }

    for name, element_type, value_pos in iter_bson_doc(data, fields_off):
        if name == "summary" and element_type == 0x02:
            enrichment["raw_summary"] = read_bson_string(data, value_pos)[0]
        elif name == "description":
            if element_type == 0x02:
                description, truncated = truncate_text(
                    read_bson_string(data, value_pos)[0],
                    max_description_chars,
                )
                enrichment["description"] = description
                enrichment["description_truncated"] = truncated
            elif element_type == 0x0A:
                enrichment["description"] = ""
                enrichment["description_truncated"] = False
        elif name == "comments" and element_type == 0x04:
            count, excerpts, truncated_bodies = parse_comments_array(
                data,
                value_pos,
                max_comment_bodies,
                max_comment_body_chars,
            )
            enrichment["comment_count"] = count
            enrichment["comment_excerpts"] = excerpts
            enrichment["comment_excerpt_count"] = len(excerpts)
            enrichment["comment_body_truncated_count"] = truncated_bodies
        elif name == "comment" and element_type == 0x03:
            count, excerpts, truncated_bodies = parse_comment_container(
                data,
                value_pos,
                max_comment_bodies,
                max_comment_body_chars,
            )
            enrichment["comment_count"] = count
            enrichment["comment_excerpts"] = excerpts
            enrichment["comment_excerpt_count"] = len(excerpts)
            enrichment["comment_body_truncated_count"] = truncated_bodies

    return enrichment


def iter_archive_sections(
    zip_path: Path,
    archive_in_zip: str,
) -> Iterable[tuple[int, bytes, int]]:
    bytes_in = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(archive_in_zip) as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                gz.read(4)
                bytes_in = 4
                while True:
                    hdr = gz.read(5)
                    if len(hdr) < 5:
                        break
                    size = struct.unpack_from("<i", hdr, 0)[0]
                    section_type = hdr[4]
                    bytes_in += 5
                    if size == -1:
                        saved = bytes([section_type])
                        rest = gz.read(4)
                        if len(rest) < 4:
                            break
                        bytes_in += 4
                        hdr2 = saved + rest
                        size = struct.unpack_from("<i", hdr2, 0)[0]
                        section_type = hdr2[4]
                    if size <= 0 or size > 50_000_000:
                        break
                    data = gz.read(size - 5)
                    bytes_in += len(data)
                    yield section_type, data, bytes_in


def load_current_payload(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    log(f"Loading current JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    requirements = payload.get("requirements", [])
    dependency_pairs = payload.get("dependency_pairs", [])
    if not isinstance(requirements, list) or not isinstance(dependency_pairs, list):
        raise ValueError("Current JSON must contain list requirements and dependency_pairs.")
    metadata = payload.get("metadata", {})
    return requirements, dependency_pairs, metadata if isinstance(metadata, dict) else {}


def build_wanted_index(requirements: list[dict[str, Any]]) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = defaultdict(set)
    for req in requirements:
        issue_key = str(req.get("issue_key") or "")
        source = str(req.get("source") or "")
        if issue_key and source:
            wanted[issue_key].add(source)
    return wanted


def scan_archive_for_enrichments(
    args: argparse.Namespace,
    wanted_by_issue: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    enrichments: dict[tuple[str, str], dict[str, Any]] = {}
    unique_targets = sum(len(sources) for sources in wanted_by_issue.values())
    stats: dict[str, Any] = {
        "archive_sections_seen": 0,
        "archive_issue_docs_seen": 0,
        "archive_docs_with_wanted_issue_key": 0,
        "archive_target_matches": 0,
        "archive_last_collection": "",
        "archive_bytes_read": 0,
    }
    current_collection = ""
    last_report = 0
    started = time.time()

    for section_type, data, bytes_in in iter_archive_sections(args.source_zip, args.archive_in_zip):
        stats["archive_sections_seen"] += 1
        stats["archive_bytes_read"] = bytes_in

        if section_type in (0x02, 0x00) and b"collection" in data:
            collection = extract_collection_name(data)
            if collection:
                current_collection = collection
                stats["archive_last_collection"] = collection
        elif section_type == 0x07 and data[:4] == b"_id\x00":
            stats["archive_issue_docs_seen"] += 1
            issue_key, fields_off = parse_issue_header(data)
            if not issue_key or fields_off is None or issue_key not in wanted_by_issue:
                continue
            stats["archive_docs_with_wanted_issue_key"] += 1
            issuetype = find_issuetype(data, fields_off)
            source = assign_source(issuetype, current_collection)
            if not source or source not in wanted_by_issue[issue_key]:
                continue
            key = (source, issue_key)
            if key in enrichments:
                continue
            enrichment = parse_text_enrichment(
                data,
                fields_off,
                args.max_description_chars,
                args.max_comment_bodies_per_issue,
                args.max_comment_body_chars,
            )
            enrichment["enrichment_found"] = True
            enrichment["raw_issuetype"] = issuetype or ""
            enrichment["raw_collection"] = current_collection
            enrichments[key] = enrichment
            stats["archive_target_matches"] = len(enrichments)
            if len(enrichments) >= unique_targets:
                break

        if bytes_in - last_report >= args.progress_bytes:
            elapsed = max(time.time() - started, 0.001)
            log(
                "Scanned "
                f"{bytes_in / (1024**3):.2f} GiB, "
                f"issues={stats['archive_issue_docs_seen']}, "
                f"matched={len(enrichments)}/{unique_targets}, "
                f"collection={current_collection or '?'} "
                f"({bytes_in / (1024**2) / elapsed:.1f} MiB/s)"
            )
            last_report = bytes_in

    stats["archive_scan_seconds"] = round(time.time() - started, 3)
    return enrichments, stats


def build_requirement_text(
    req: dict[str, Any],
    enrichment: dict[str, Any],
    include_comments: bool = False,
) -> str:
    parts: list[str] = []
    summary = str(req.get("summary") or "").strip()
    description = str(enrichment.get("description") or "").strip()
    issue_type = str(req.get("issuetype") or "").strip()
    status = str(req.get("status") or "").strip()
    priority = str(req.get("priority") or "").strip()
    if summary:
        parts.append(f"Summary: {summary}")
    if description:
        parts.append(f"Description: {description}")
    if include_comments:
        comment_bodies = [
            str(comment.get("body") or "").strip()
            for comment in enrichment.get("comment_excerpts", [])
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


def empty_enrichment() -> dict[str, Any]:
    return {
        "enrichment_found": False,
        "raw_collection": "",
        "raw_issuetype": "",
        "raw_summary": "",
        "description": "",
        "description_truncated": False,
        "comment_count": 0,
        "comment_excerpt_count": 0,
        "comment_excerpts": [],
        "comment_body_truncated_count": 0,
    }


def open_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def write_requirements_jsonl(
    requirements: list[dict[str, Any]],
    enrichments: dict[tuple[str, str], dict[str, Any]],
    out_path: Path,
) -> dict[str, int]:
    stats = {
        "requirements_total": 0,
        "requirements_enrichment_found": 0,
        "requirements_with_description": 0,
        "requirements_with_comments": 0,
        "requirements_with_comment_excerpts": 0,
        "requirements_description_truncated": 0,
    }
    with open_output(out_path) as handle:
        for req in requirements:
            key = (str(req.get("source") or ""), str(req.get("issue_key") or ""))
            enrichment = enrichments.get(key, empty_enrichment())
            row = dict(req)
            row.update(enrichment)
            row["has_description"] = bool(row.get("description"))
            row["has_comments"] = int(row.get("comment_count") or 0) > 0
            row["requirement_text_with_description"] = build_requirement_text(
                req,
                enrichment,
                include_comments=False,
            )
            row["requirement_text_with_description_and_comments"] = build_requirement_text(
                req,
                enrichment,
                include_comments=True,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            stats["requirements_total"] += 1
            stats["requirements_enrichment_found"] += int(bool(row.get("enrichment_found")))
            stats["requirements_with_description"] += int(bool(row.get("description")))
            stats["requirements_with_comments"] += int(int(row.get("comment_count") or 0) > 0)
            stats["requirements_with_comment_excerpts"] += int(
                int(row.get("comment_excerpt_count") or 0) > 0
            )
            stats["requirements_description_truncated"] += int(
                bool(row.get("description_truncated"))
            )
    return stats


def write_pairs_jsonl(
    dependency_pairs: list[dict[str, Any]],
    enrichments: dict[tuple[str, str], dict[str, Any]],
    out_path: Path,
) -> dict[str, int]:
    stats = {
        "pairs_total": 0,
        "pairs_source_with_description": 0,
        "pairs_target_with_description": 0,
        "pairs_both_with_description": 0,
        "pairs_source_enrichment_found": 0,
        "pairs_target_enrichment_found": 0,
    }
    with open_output(out_path) as handle:
        for pair in dependency_pairs:
            source_key = (
                str(pair.get("source_source") or ""),
                str(pair.get("source_issue_key") or ""),
            )
            target_key = (
                str(pair.get("target_source") or ""),
                str(pair.get("target_issue_key") or ""),
            )
            source_enrichment = enrichments.get(source_key, empty_enrichment())
            target_enrichment = enrichments.get(target_key, empty_enrichment())
            source_req = {
                "summary": pair.get("source_summary"),
                "issuetype": pair.get("source_issuetype"),
                "status": "",
                "priority": "",
            }
            target_req = {
                "summary": pair.get("target_summary"),
                "issuetype": pair.get("target_issuetype"),
                "status": "",
                "priority": "",
            }
            row = dict(pair)
            row["source_enrichment_found"] = bool(source_enrichment.get("enrichment_found"))
            row["target_enrichment_found"] = bool(target_enrichment.get("enrichment_found"))
            row["source_has_description"] = bool(source_enrichment.get("description"))
            row["target_has_description"] = bool(target_enrichment.get("description"))
            row["source_comment_count"] = int(source_enrichment.get("comment_count") or 0)
            row["target_comment_count"] = int(target_enrichment.get("comment_count") or 0)
            row["source_text_with_description"] = build_requirement_text(
                source_req,
                source_enrichment,
                include_comments=False,
            )
            row["target_text_with_description"] = build_requirement_text(
                target_req,
                target_enrichment,
                include_comments=False,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            stats["pairs_total"] += 1
            stats["pairs_source_enrichment_found"] += int(row["source_enrichment_found"])
            stats["pairs_target_enrichment_found"] += int(row["target_enrichment_found"])
            stats["pairs_source_with_description"] += int(row["source_has_description"])
            stats["pairs_target_with_description"] += int(row["target_has_description"])
            stats["pairs_both_with_description"] += int(
                row["source_has_description"] and row["target_has_description"]
            )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-json", type=Path, default=DEFAULT_CURRENT_JSON)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--archive-in-zip", default=DEFAULT_ARCHIVE_IN_ZIP)
    parser.add_argument("--requirements-out", type=Path, default=DEFAULT_REQUIREMENTS_OUT)
    parser.add_argument("--pairs-out", type=Path, default=DEFAULT_PAIRS_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--max-description-chars", type=int, default=4000)
    parser.add_argument("--max-comment-bodies-per-issue", type=int, default=2)
    parser.add_argument("--max-comment-body-chars", type=int, default=1200)
    parser.add_argument("--progress-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--skip-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements, dependency_pairs, current_metadata = load_current_payload(args.current_json)
    wanted_by_issue = build_wanted_index(requirements)
    unique_targets = sum(len(sources) for sources in wanted_by_issue.values())
    log(f"Current requirements={len(requirements)}, unique source/key targets={unique_targets}")

    enrichments, archive_stats = scan_archive_for_enrichments(args, wanted_by_issue)
    log(f"Archive matched {len(enrichments)} / {unique_targets} targets")

    log(f"Writing requirements JSONL: {args.requirements_out}")
    requirement_stats = write_requirements_jsonl(requirements, enrichments, args.requirements_out)

    pair_stats: dict[str, int] = {}
    if not args.skip_pairs:
        log(f"Writing pairs JSONL: {args.pairs_out}")
        pair_stats = write_pairs_jsonl(dependency_pairs, enrichments, args.pairs_out)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_json": str(args.current_json),
        "source_zip": str(args.source_zip),
        "archive_in_zip": args.archive_in_zip,
        "requirements_out": str(args.requirements_out),
        "pairs_out": "" if args.skip_pairs else str(args.pairs_out),
        "max_description_chars": args.max_description_chars,
        "max_comment_bodies_per_issue": args.max_comment_bodies_per_issue,
        "max_comment_body_chars": args.max_comment_body_chars,
        "current_metadata": current_metadata,
        **archive_stats,
        **requirement_stats,
        **pair_stats,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    log(f"Wrote summary: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
