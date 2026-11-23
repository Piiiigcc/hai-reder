#!/usr/bin/env python3
"""Extract requirement pairs from MongoDB archive — handles type 0x07 sections with special _id encoding."""

import argparse
import gzip
import json
import struct
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ZIP = ROOT / "data/raw/public_jira/2025-06-23 ThePublicJiraDataset.zip"
DEFAULT_OUTPUT_JSON = ROOT / "data/interim/public_jira/requirement_pairs_dependencies.json"
ARCHIVE_IN_ZIP = "ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"

REQUIREMENT_TYPES = {
    "Apache": {"Epic", "Story", "New Feature", "Improvement", "Request", "Wish", "Brainstorming"},
    "Hyperledger": {"Epic", "Story", "New Feature", "Improvement"},
    "IntelDAOS": {"Epic", "Story"},
    "JFrog": {"Epic", "Story", "New Feature", "Feature Request", "Improvement", "Change Request"},
    "Jira": {"New Feature", "Improvement", "Suggestion"},
    "JiraEcosystem": {"Epic", "Story", "New Feature", "Improvement", "Suggestion"},
    "MariaDB": {"Epic", "New Feature"},
    "MongoDB": {"Epic", "Story", "New Feature", "Improvement"},
    "Qt": {"Epic", "User Story", "Improvement", "Suggestion", "Change Request"},
    "RedHat": {"Epic", "Market Problem", "Initiative", "Story", "Requirement", "New Feature", "Feature", "Feature Request", "Improvement", "Enhancement"},
    "Sonatype": {"Epic", "Story", "Improvement"},
    "Spring": {"Epic", "Story", "New Feature", "Improvement"},
    "Sakai": {"Roadmap item", "Feature Request", "Improvement"},
    "SecondLife": {"Story", "New Feature"},
    "Mindville": {"New Feature"},
    "Mojang": set(),
}

LINK_TYPE_MAPPING = {
    "Backports": "Backport", "Blocked": "Block", "Blocker": "Block", "Blocks": "Block",
    "Causality": "Cause", "Cause": "Cause", "Caused": "Cause", "Problem/Incident": "Cause",
    "Child-Issue": "Parent-Child", "Parent Feature": "Parent-Child", "Parent/Child": "Parent-Child",
    "multi-level hierarchy [GANTT]": "Parent-Child", "Parent-Relation": "Parent-Child",
    "Cloners": "Clone", "Cloners (old)": "Clone",
    "Collection": "Incorporate", "Container": "Incorporate", "Contains(WBSGantt)": "Incorporate",
    "Incorporate": "Incorporate", "Incorporates": "Incorporate", "Part": "Incorporate",
    "PartOf": "Incorporate", "Superset": "Incorporate",
    "Completes": "Fix", "Fixes": "Fix", "Resolve": "Fix",
    "Depend": "Depend", "Dependency": "Depend", "Dependent": "Depend", "Depends": "Depend",
    "Gantt Dependency": "Depend", "dependent": "Depend",
    "Derived": "Derive", "Detail": "Detail",
    "Duplicate": "Duplicate", "Epic": "Epic", "Epic-Relation": "Epic",
    "Follows": "Follow", "Sequence": "Follow",
    "Implement": "Implement", "Implements": "Implement",
    "Issue split": "Split", "Split": "Split", "Work Breakdown": "Split",
    "Preceded By": "Precede",
    "Reference": "Relate", "Relate": "Relate", "Related": "Relate", "Relates": "Relate",
    "Relationship": "Relate", "Regression": "Breaks", "Replacement": "Replace", "Required": "Require",
    "Supercedes": "Supercede", "Supersede": "Supercede", "Supersession": "Supercede",
    "Test": "Test", "Tested": "Test", "Trigger": "Trigger", "Subtask": "Subtask",
    "Bonfire Testing": "Bonfire Testing", "Bonfire testing": "Bonfire Testing",
    "Git Code Review": "Bonfire Testing", "Testing": "Bonfire Testing",
    "Documentation": "Documented", "Documented": "Documented",
    "Finish-to-Finish link (WBSGantt)": "finish-finish",
    "Gantt End to End": "finish-finish", "Gantt: finish-finish": "finish-finish",
    "finish-finish [GANTT]": "finish-finish",
    "Gantt End to Start": "finish-start", "Gantt: finish-start": "finish-start",
    "finish-start [GANTT]": "finish-start",
    "Gantt Start to Start": "start-start", "Gantt: start-finish": "start-finish",
}
DEPENDENCIES = {
    "Block", "Depend", "Require", "Cause", "Parent-Child", "Epic",
    "Incorporate", "Split", "Derive", "Precede", "Fix", "Implement",
    "Follow", "Clone", "Duplicate", "Replace", "Relate", "Breaks",
    "Detail", "Supercede", "Backport", "Bonfire Testing", "Documented",
    "Test", "Trigger", "finish-finish", "finish-start", "start-finish",
    "start-start", "Subtask",
}

BSON_FIXED = {0x01: 8, 0x07: 12, 0x08: 1, 0x09: 8, 0x0A: 0, 0x10: 4, 0x11: 8, 0x12: 8, 0x13: 16, 0xFF: 0, 0x7F: 0, 0x06: 0}

def bs(data, off):
    n = struct.unpack_from('<i', data, off)[0]
    if n <= 0:
        return "", off + 4
    e = off + 4 + n
    try:
        return data[off+4:e-1].decode('utf-8', 'replace'), e
    except:
        return "", e

def bc(data, off):
    e = data.index(b'\x00', off)
    try:
        return data[off:e].decode('utf-8', 'replace'), e + 1
    except:
        return "", e + 1

def bsk(data, off, et):
    if et in BSON_FIXED:
        return off + BSON_FIXED[et]
    if et in (0x03, 0x04):
        return off + struct.unpack_from('<i', data, off)[0]
    if et == 0x02:
        return off + 4 + max(0, struct.unpack_from('<i', data, off)[0])
    if et == 0x05:
        return off + 5 + struct.unpack_from('<i', data, off)[0]
    if et == 0x0B:
        _, o = bc(data, off); _, o = bc(data, o); return o
    if et == 0x0C:
        _, o = bs(data, off); return o + 12
    if et in (0x0D, 0x0E):
        _, o = bs(data, off); return o
    if et == 0x0F:
        return off + struct.unpack_from('<i', data, off)[0]
    return off + 4

def parse_issue_v7(data):
    """Parse a type 0x07 section data — special BSON variant where _id has no type byte."""
    pos = 0
    dsize = len(data)
    doc = {"links": []}

    if dsize < 4:
        return None

    if data[0:4] == b'_id\x00':
        doc["_id"] = data[4:16].hex()
        pos = 16
    else:
        return None

    while pos < dsize - 1:
        if data[pos] == 0x00:
            break
        et = data[pos]
        pos += 1
        nm, pos = bc(data, pos)

        if nm == "key" and et == 0x02:
            doc["issue_key"], pos = bs(data, pos)
        elif nm == "self" and et == 0x02:
            doc["self"], pos = bs(data, pos)
        elif nm == "id" and et == 0x02:
            doc["jira_id"], pos = bs(data, pos)
        elif nm == "fields" and et == 0x03:
            fsize = struct.unpack_from('<i', data, pos)[0]
            doc["fields"] = parse_fields(data, pos)
            pos += fsize
        elif nm == "changelog" and et == 0x03:
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "names" and et == 0x03:
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "expand" and et == 0x02:
            _, pos = bs(data, pos)
        elif nm == "renderedFields" and et == 0x03:
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "transitions" and et == 0x04:
            pos += struct.unpack_from('<i', data, pos)[0]
        else:
            pos = bsk(data, pos, et)

    return doc if "issue_key" in doc and "fields" in doc else None

def parse_fields(data, off):
    """Parse the fields embedded document — extract relevant fields."""
    fsize = struct.unpack_from('<i', data, off)[0]
    pos = off + 4
    end = off + fsize - 1
    result = {}

    while pos < end:
        if data[pos] == 0x00:
            break
        et = data[pos]
        pos += 1
        nm, pos = bc(data, pos)

        if nm == "issuetype" and et == 0x03:
            result["issuetype"] = find_field_in_doc(data, pos, "name")
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "project" and et == 0x03:
            result["project_key"] = find_field_in_doc(data, pos, "key")
            result["project_name"] = find_field_in_doc(data, pos, "name")
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "summary" and et == 0x02:
            result["summary"], pos = bs(data, pos)
        elif nm == "status" and et == 0x03:
            result["status"] = find_field_in_doc(data, pos, "name")
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "priority" and et == 0x03:
            result["priority"] = find_field_in_doc(data, pos, "name")
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "created" and et == 0x02:
            result["created"], pos = bs(data, pos)
        elif nm == "resolutiondate" and et in (0x02, 0x0A):
            if et == 0x02:
                result["resolutiondate"], pos = bs(data, pos)
            else:
                result["resolutiondate"] = None
        elif nm == "issuelinks" and et == 0x04:
            result["links"] = parse_issuelinks(data, pos)
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm == "description" and et in (0x02, 0x0A):
            if et == 0x02:
                result["description"], pos = bs(data, pos)
            else:
                result["description"] = ""
                pos = bsk(data, pos, et)
        else:
            pos = bsk(data, pos, et)

    return result

def find_field_in_doc(data, off, field_name):
    """Find a string field value in an embedded document."""
    dsize = struct.unpack_from('<i', data, off)[0]
    pos = off + 4
    end = off + dsize - 1
    while pos < end:
        if data[pos] == 0x00:
            break
        et = data[pos]
        pos += 1
        nm, pos = bc(data, pos)
        if nm == field_name and et == 0x02:
            return bs(data, pos)[0]
        pos = bsk(data, pos, et)
    return None

def parse_issuelinks(data, off):
    """Parse issuelinks array."""
    links = []
    asize = struct.unpack_from('<i', data, off)[0]
    pos = off + 4
    end = off + asize - 1

    while pos < end:
        if data[pos] == 0x00:
            break
        et = data[pos]
        pos += 1
        _, pos = bc(data, pos)  # skip numeric key

        if et == 0x03:
            lsize = struct.unpack_from('<i', data, pos)[0]
            link = parse_one_issuelink(data, pos)
            if link:
                links.append(link)
            pos += lsize
        else:
            pos = bsk(data, pos, et)

    return links

def parse_one_issuelink(data, off):
    dsize = struct.unpack_from('<i', data, off)[0]
    pos = off + 4
    end = off + dsize - 1
    li = {}

    while pos < end:
        if data[pos] == 0x00:
            break
        et = data[pos]
        pos += 1
        nm, pos = bc(data, pos)

        if nm == "type" and et == 0x03:
            li["link_type_name"] = find_field_in_doc(data, pos, "name") or ""
            li["inward"] = find_field_in_doc(data, pos, "inward") or ""
            li["outward"] = find_field_in_doc(data, pos, "outward") or ""
            pos += struct.unpack_from('<i', data, pos)[0]
        elif nm in ("inwardIssue", "outwardIssue") and et == 0x03:
            li["linked_issue_key"] = find_field_in_doc(data, pos, "key") or ""
            li["direction"] = "inward" if nm == "inwardIssue" else "outward"
            pos += struct.unpack_from('<i', data, pos)[0]
        else:
            pos = bsk(data, pos, et)

    if "link_type_name" in li and "linked_issue_key" in li:
        return li
    return None

def assign_source(issuetype, collection=None):
    return collection if issuetype in REQUIREMENT_TYPES.get(collection, ()) else None

def extract_collection_name(data):
    idx = data.find(b'collection\x00')
    if idx >= 0:
        start = idx + len(b'collection\x00')
        if start + 4 <= len(data):
            slen = struct.unpack_from('<i', data, start)[0]
            if slen > 0 and slen < 256 and start + 4 + slen <= len(data):
                try:
                    return data[start+4:start+4+slen-1].decode('utf-8', 'replace')
                except:
                    pass
    return None

def log(msg):
    print(msg, file=sys.stderr, flush=True)

def process(source_zip: Path, output_json: Path):
    t0 = time.time()
    log(f"Output: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    stats = {"sec": 0, "req": 0, "pairs": 0,
             "by_src": defaultdict(lambda: {"r": 0, "p": 0}),
             "by_dep": defaultdict(int)}
    all_reqs, lookup = [], {}
    bytes_in = last_rpt = 0
    current_coll = "?"

    with zipfile.ZipFile(source_zip, 'r') as zf:
        with zf.open(ARCHIVE_IN_ZIP) as raw:
            with gzip.GzipFile(fileobj=raw) as gz:
                gz.read(4)
                bytes_in = 4

                while True:
                    hdr = gz.read(5)
                    if len(hdr) < 5:
                        break
                    sz = struct.unpack_from('<i', hdr, 0)[0]
                    tp = hdr[4]
                    bytes_in += 5

                    if sz == -1:
                        saved = bytes([tp])
                        rest = gz.read(4)
                        if len(rest) < 4:
                            break
                        bytes_in += 4
                        hdr2 = saved + rest
                        sz = struct.unpack_from('<i', hdr2, 0)[0]
                        tp = hdr2[4]

                    if sz <= 0 or sz > 50000000:
                        break
                    dsz = sz - 5
                    data = gz.read(dsz)
                    bytes_in += len(data)
                    stats["sec"] += 1

                    if tp in (0x02, 0x00) and b'collection' in data:
                        coll_name = extract_collection_name(data)
                        if coll_name:
                            current_coll = coll_name
                    elif tp == 0x07 and data[:4] == b'_id\x00':
                        doc = parse_issue_v7(data)
                        if doc and doc.get("fields"):
                            f = doc["fields"]
                            ik = doc.get("issue_key")
                            it = f.get("issuetype")
                            if ik and it:
                                src = assign_source(it, current_coll)
                                if src and it in REQUIREMENT_TYPES.get(src, set()):
                                    ri = {
                                        "issue_key": ik, "source": src,
                                        "project_name": f.get("project_name", ""),
                                        "project_key": f.get("project_key", ""),
                                        "issuetype": it,
                                        "summary": (f.get("summary") or "")[:500],
                                        "status": f.get("status", ""),
                                        "priority": f.get("priority", ""),
                                        "created": f.get("created"),
                                        "resolutiondate": f.get("resolutiondate"),
                                        "links": f.get("links", []),
                                    }
                                    lookup[src, ik] = ri
                                    all_reqs.append(ri)
                                    stats["req"] += 1
                                    stats["by_src"][src]["r"] += 1

                    if bytes_in - last_rpt >= 1 * 1024**3:
                        elapsed = time.time() - t0
                        log(f"  {bytes_in/(1024**3):.1f}GB [{elapsed:.0f}s] sec:{stats['sec']} req:{stats['req']} coll:{current_coll}")
                        last_rpt = bytes_in

    elapsed = time.time() - t0
    log(f"\nDone [{elapsed:.0f}s]. Sections: {stats['sec']}, Req: {stats['req']}")
    for s, d in sorted(stats["by_src"].items()):
        if d["r"] > 0:
            log(f"  {s}: {d['r']} req")

    log("Building dependency pairs...")
    pairs = []
    for req in all_reqs:
        for lk in req.get("links", []):
            n = LINK_TYPE_MAPPING.get(lk.get("link_type_name", ""), lk.get("link_type_name", ""))
            lkey = lk.get("linked_issue_key", "")
            if n in DEPENDENCIES and lkey and (req["source"], lkey) in lookup:
                lr = lookup[req["source"], lkey]
                pairs.append({
                    "source_issue_key": req["issue_key"],
                    "target_issue_key": lkey,
                    "dependency_type": n,
                    "raw_link_type": lk.get("link_type_name", ""),
                    "direction": lk.get("direction", "unknown"),
                    "source_source": req["source"],
                    "target_source": lr["source"],
                    "source_issuetype": req["issuetype"],
                    "target_issuetype": lr["issuetype"],
                    "source_project": req["project_key"],
                    "target_project": lr["project_key"],
                    "source_summary": req["summary"][:200],
                    "target_summary": lr["summary"][:200],
                })
                stats["pairs"] += 1
                stats["by_dep"][n] += 1
                stats["by_src"][req["source"]]["p"] += 1

    log(f"Pairs: {stats['pairs']}")

    out = {
        "metadata": {
            "description": "Requirement pairs and dependencies from ThePublicJiraDataset",
            "total_requirement_issues": stats["req"],
            "total_dependency_pairs": stats["pairs"],
            "dependency_categories": sorted(DEPENDENCIES),
            "extraction_time_seconds": int(elapsed),
        },
        "statistics": {
            "sections_processed": stats["sec"],
            "requirement_issues": stats["req"],
            "dependency_pairs": stats["pairs"],
            "by_source": {s: {"requirement_issues": d["r"], "dependency_pairs": d["p"]}
                         for s, d in sorted(stats["by_src"].items()) if d["r"] > 0},
            "by_dependency_type": dict(sorted(stats["by_dep"].items(), key=lambda x: -x[1])),
        },
        "requirements": all_reqs,
        "dependency_pairs": pairs,
    }

    log(f"Writing {output_json}...")
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"Done! {output_json.stat().st_size / (1024**2):.1f} MB")
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=DEFAULT_SOURCE_ZIP)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()
    process(args.source_zip, args.output_json)
