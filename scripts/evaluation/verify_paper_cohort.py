"""Independently verify every published entity, edge, window, class and negative quota."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import argparse
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from haireder.data.formal import file_hash
from haireder.taxonomy import LABELS, load_taxonomy
from haireder.utils.config import load_yaml
from haireder.utils.io import read_jsonl, write_json, write_text


def timestamp(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def verify(directory, random_dir):
    cfg = load_yaml("configs/dataset/formal_6repos.yaml")
    taxonomy = load_taxonomy()
    manifest = json.loads((directory / "formal_manifest.json").read_text())
    for path, value in manifest["inputs"].items():
        assert file_hash(ROOT / path) == value, path
    for name, value in manifest["artifacts"].items():
        assert file_hash(directory / name) == value, name
    rows = read_jsonl(directory / "formal_pairs.jsonl")
    random_rows = read_jsonl(random_dir / "formal_pairs.jsonl")
    random_manifest = json.loads((random_dir / "formal_manifest.json").read_text())
    for name, value in random_manifest["artifacts"].items():
        assert file_hash(random_dir / name) == value, name
    assert len({r["pair_id"] for r in rows}) == len(rows)
    cohort = {(r["repo"], r[k]["id"]) for r in rows for k in ("issue_i", "issue_j")}
    all_endpoints = cohort | {
        (r["repo"], r[k]["id"]) for r in random_rows for k in ("issue_i", "issue_j")
    }
    issues = {}
    recorded = defaultdict(set)
    with (ROOT / cfg["raw_sources"]["creation_history"]).open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["source"], r["issue_key"])
            if key in all_endpoints:
                issues[key] = r
                recorded[key].update(r["recorded_targets"])
    assert set(issues) == all_endpoints
    raw = json.loads((ROOT / cfg["raw_sources"]["processed_pairs_json"]).read_text())
    edge_labels = defaultdict(set)
    known = set()
    for e in raw["dependency_pairs"]:
        key = (e["source_source"], e["source_issue_key"], e["target_issue_key"])
        if e["source_source"] != e["target_source"]:
            continue
        known.add(key)
        edge_labels[key].add(
            taxonomy.normalized_dependency_type_to_label[e["dependency_type"]]
        )
    del raw
    grouped = defaultdict(Counter)
    by_split = defaultdict(Counter)
    by_repo = defaultdict(Counter)
    cross = {}
    historical_negative = Counter()
    independent_positive = set()
    start = datetime(2018, 1, 11, tzinfo=timezone.utc)
    for i, w in enumerate(manifest["windows"]):
        assert w["window"] == f"W{i + 1}" and timestamp(
            w["start_at"]
        ) == start + timedelta(days=14 * i)
        assert timestamp(w["end_at"]) == start + timedelta(days=14 * (i + 1))
    assert len(manifest["windows"]) == 32
    for r in rows:
        repo = r["repo"]
        a, b = (r[k]["id"] for k in ("issue_i", "issue_j"))
        assert a != b
        available = max(timestamp(r[k]["created"]) for k in ("issue_i", "issue_j"))
        index = (available - start).days // 14 + 1
        assert (
            1 <= index <= 32
            and timestamp(r["pair_available_at"]) == available
            and r["window"] == f"W{index}"
        )
        assert r["split"] == (
            "train"
            if index <= 3
            else "validation"
            if index == 4
            else "online"
            if index <= 28
            else "test"
        )
        for k in ("issue_i", "issue_j"):
            snap = issues[repo, r[k]["id"]]
            expected = "\n".join(
                f"{field.title()}: {snap['creation_text'][field].strip()}"
                for field in ("summary", "description")
                if snap["creation_text"][field].strip()
            )
            assert r[k]["text"] == expected and timestamp(r[k]["created"]) == timestamp(
                snap["created"]
            )
        unordered = (repo, *sorted((a, b)))
        assert cross.setdefault(unordered, r["split"]) == r["split"]
        raw_link = b in recorded[repo, a] or a in recorded[repo, b]
        if r["label"] == "None":
            assert (
                not raw_link and (repo, a, b) not in known and (repo, b, a) not in known
            )
            assert (
                r["negative_mining"]["backend"] == "sbert"
                and -1.00001 <= r["negative_mining"]["similarity"] <= 1.00001
            )
            if min(
                timestamp(r[k]["created"]) for k in ("issue_i", "issue_j")
            ) < start + timedelta(days=14 * (index - 1)):
                historical_negative[repo + "/" + r["window"]] += 1
                assert (
                    r["negative_mining"]["candidate_source"]
                    == "current_and_prior_repository_requirements"
                )
        else:
            assert raw_link and edge_labels[repo, a, b] == {r["label"]}
            independent_positive.add(unordered)
        grouped[repo, r["window"]][r["label"]] += 1
        by_repo[repo][r["label"]] += 1
        by_split[r["split"]][r["label"]] += 1
    for key, counts in grouped.items():
        assert counts["None"] * 2 == sum(counts.values()), key
    for split in by_split:
        assert read_jsonl(directory / f"formal_{split}.jsonl") == [
            r for r in rows if r["split"] == split
        ]
    assert file_hash(directory / "formal_test.jsonl") == file_hash(
        random_dir / "formal_test.jsonl"
    )
    assert len(random_rows) == len(rows) and len(
        {r["pair_id"] for r in random_rows}
    ) == len(random_rows)
    assert Counter(
        (r["repo"], r["window"], r["label"]) for r in random_rows
    ) == Counter((r["repo"], r["window"], r["label"]) for r in rows)
    assert {r["pair_id"] for r in random_rows if r["is_positive"]} == {
        r["pair_id"] for r in rows if r["is_positive"]
    }
    for r in random_rows:
        if r["label"] == "None" and r["split"] != "test":
            a, b = (r[k]["id"] for k in ("issue_i", "issue_j"))
            assert (
                r["negative_mining"]["backend"] == "random"
                and (r["repo"], a, b) not in known
                and (r["repo"], b, a) not in known
                and b not in recorded[r["repo"], a]
                and a not in recorded[r["repo"], b]
            )
    return dict(
        pairs=len(rows),
        unique_ids=len(rows),
        unique_endpoints=len(cohort),
        repository_windows=len(grouped),
        windows=32,
        positive=len(rows) // 2,
        distinct_unordered_positive_links=len(independent_positive),
        negative=len(rows) // 2,
        historical_endpoint_negatives=dict(historical_negative),
        split_classes=dict(by_split),
        repo_classes=dict(by_repo),
        raw_and_clean_link_violations=0,
        creation_text_mismatches=0,
        cross_split_pairs=0,
        shared_test_hash=file_hash(directory / "formal_test.jsonl"),
        sbert_revision=manifest.get("sbert_revision"),
        phase_manifest_hash=file_hash(directory / "formal_manifest.json"),
    )


def similarity_probe(directory, sample_size, seed=42):
    """Re-encode a reproducible simple random sample on CPU, independently of sklearn's mining index."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch
    from haireder.utils.model_versions import revision_kwargs

    torch.set_num_threads(2)
    rows = [
        r for r in read_jsonl(directory / "formal_pairs.jsonl") if r["label"] == "None"
    ]
    sample = random.Random(seed).sample(
        sorted(rows, key=lambda r: r["pair_id"]), sample_size
    )
    model_name = load_yaml("configs/dataset/formal_6repos.yaml")["pair_construction"][
        "sbert_model_name"
    ]
    model = SentenceTransformer(model_name, device="cpu", **revision_kwargs(model_name))
    embeddings = model.encode(
        [r[key]["text"] for r in sample for key in ("issue_i", "issue_j")],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=8,
    )
    measured = np.sum(embeddings[::2] * embeddings[1::2], axis=1)
    reference = np.array([r["negative_mining"]["similarity"] for r in sample])
    error = np.abs(measured - reference)
    if float(error.max()) > 1e-4:
        raise ValueError(f"Independent SBERT cosine discrepancy: {error.max()}")
    return dict(
        sample_size=sample_size,
        seed=seed,
        model=model_name,
        **revision_kwargs(model_name),
        max_absolute_error=float(error.max()),
        median_cosine=float(np.median(measured)),
        samples=[
            dict(pair_id=r["pair_id"], computed=float(v), stored=float(ref))
            for r, v, ref in zip(sample, measured, reference, strict=True)
        ],
    )


def table(report):
    parts = [
        "# Cohort counts",
        "",
        "Counts refer to ordered issue pairs in the frozen paper cohort. The None column counts sampled pairs without a recorded link.",
        "",
    ]
    for name, key in (
        ("Temporal splits", "split_classes"),
        ("Source repositories", "repo_classes"),
    ):
        parts.extend(
            [
                f"## {name}",
                "",
                "| Cohort | " + " | ".join(LABELS) + " | Total |",
                "| --- |" + " ---: |" * (len(LABELS) + 1),
            ]
        )
        for group, counts in report[key].items():
            parts.append(
                "| "
                + group
                + " | "
                + " | ".join(str(counts.get(k, 0)) for k in LABELS)
                + f" | {sum(counts.values())} |"
            )
        parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/splits")
    p.add_argument(
        "--random-data", type=Path, default=ROOT / "data/processed/splits/formal_random"
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--table", type=Path)
    p.add_argument("--similarity-out", type=Path)
    p.add_argument("--similarity-sample", type=int, default=64)
    args = p.parse_args()
    report = verify(args.data, args.random_data)
    write_json(args.out, report)
    if args.table:
        write_text(args.table, table(report))
    if args.similarity_out:
        write_json(
            args.similarity_out, similarity_probe(args.data, args.similarity_sample)
        )
    print(json.dumps(report, ensure_ascii=False))
