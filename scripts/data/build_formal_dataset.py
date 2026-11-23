#!/usr/bin/env python
"""Build formal experiment dataset for HAI-ReDeR.

Generates train/validation/test splits across 6 repositories with 32 chronological
time windows, covering all 6 dependency labels (None, General Relation, Duplication,
Composition, Temporal/Causal, Workflow).

Usage (on server):
    python scripts/data/build_formal_dataset.py --mode formal --dataset-config configs/dataset/formal_6repos.yaml
    python scripts/data/build_formal_dataset.py --mode minimal --output-dir data/processed/splits/minimal_build

Pipeline:
    1. Load fixed requirement_pairs_dependencies.json (all 30 link types)
    2. For each repository, build positive pairs via taxonomy mapping
    3. Assign 32 chronological windows (2-week sprints)
    4. Mine hard-negative pairs (None class) via SBERT similarity by default
    5. Output per-split JSONL files matching the existing schema
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.taxonomy import load_taxonomy
from haireder.data.windowing import (
    assign_quantile_windows,
    build_fixed_sprints,
    build_positive_pairs,
    parse_jira_datetime,
    requirement_text,
)
from haireder.data.pair_construction import (
    mine_hard_negatives_for_window,
)

# ---------------------------------------------------------------------------
# Per-window issue pool (unique to this script — takes windows list from
# build_chronological_windows instead of a window_report dict)
# ---------------------------------------------------------------------------
def build_window_issue_pools(
    requirements: list[dict[str, Any]],
    repository: str,
    windows: list[dict[str, Any]],
    rng: random.Random,
    max_candidates_per_window: int = 5000,
) -> dict[str, dict[str, dict[str, Any]]]:
    repo_reqs = [r for r in requirements if r.get("source") == repository]
    parsed_windows: list[tuple[str, datetime, datetime]] = []
    for w in windows:
        start = parse_jira_datetime(w.start_at)
        end = parse_jira_datetime(w.end_at)
        if start and end:
            parsed_windows.append((w.window, start, end))

    candidates: dict[str, list[dict[str, Any]]] = {w: [] for w, _, _ in parsed_windows}
    for req in repo_reqs:
        created = parse_jira_datetime(req.get("created"))
        if created is None:
            continue
        issue_key = str(req.get("issue_key", ""))
        if not issue_key:
            continue
        for window, start, end in parsed_windows:
            if start <= created <= end:
                candidates[window].append({
                    "id": issue_key,
                    "project": req.get("project_key"),
                    "created": req.get("created"),
                    "text": requirement_text(req),
                })
                break

    issue_pools: dict[str, dict[str, dict[str, Any]]] = {}
    for window, rows in candidates.items():
        rows = sorted(rows, key=lambda r: (r.get("created") or "", r["id"]))
        if len(rows) > max_candidates_per_window:
            sampled = rng.sample(rows, max_candidates_per_window)
            rows = sorted(sampled, key=lambda r: (r.get("created") or "", r["id"]))
        issue_pools[window] = {r["id"]: r for r in rows}
    return issue_pools


def stratified_sample(
    rows: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
    key_fields: tuple[str, ...] = ("label",),
) -> list[dict[str, Any]]:
    if target_count is None or len(rows) <= target_count:
        return list(rows)
    if target_count <= 0:
        return []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    key_order: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in grouped:
            grouped[key] = []
            key_order.append(key)
        grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count:
        added = False
        for key in key_order:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            added = True
            if len(selected) == target_count:
                break
        if not added:
            break
    return selected


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["minimal", "formal"], default="formal")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--taxonomy", default="data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml")
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--negative-backend", choices=["sbert", "tfidf", "random"])
    parser.add_argument("--shared-test-dir", help="Canonical split directory for the random ablation")
    args = parser.parse_args()
    args.dataset_config = args.dataset_config or ("configs/dataset/formal_6repos.yaml" if args.mode == "formal" else "configs/dataset/minimal_local.yaml")

    config_path = PROJECT_ROOT / args.dataset_config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    taxonomy = load_taxonomy(args.taxonomy)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "formal":
        from haireder.data.formal import build_formal
        build_formal(
            config, taxonomy, PROJECT_ROOT, output_dir, config_path,
            PROJECT_ROOT / args.taxonomy, args.negative_backend,
            shared_test_dir=PROJECT_ROOT / args.shared_test_dir if args.shared_test_dir else None,
        )
        return 0

    minimal = config.get("minimal_run", {})
    window_strategy = minimal.get("window_strategy", "quantile_by_pair_count")
    start_date = minimal.get("start_date")
    end_date = minimal.get("end_date")
    sprint_days = int(minimal.get("sprint_days", 14)) if start_date and end_date else None
    selected = minimal.get("selected_repository", "Apache")
    repositories = [selected]
    window_count = int(minimal.get("window_count", 8))
    split_config = {
        "train_windows": minimal.get("train_windows", ["W1", "W2"]),
        "validation_windows": minimal.get("validation_windows", ["W3"]),
        "online_windows": minimal.get("online_windows", ["W4", "W5", "W6"]),
        "test_windows": minimal.get("test_windows", ["W7", "W8"]),
    }
    max_pairs_per_window = int(minimal.get("max_pairs_per_window", 300))
    negative_ratio = float(minimal.get("negative_ratio", 1.0))
    max_negative_candidates = int(minimal.get("max_negative_candidate_issues_per_window", 2500))
    seed = int(minimal.get("seed", 42))
    prefix = "minimal"

    rng = random.Random(seed)

    # Load raw data
    raw_json_path = PROJECT_ROOT / config["raw_sources"]["processed_pairs_json"]
    print(f"Loading {raw_json_path} ...", file=sys.stderr)
    with raw_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    all_requirements = payload.get("requirements", [])
    print(f"Total requirements in payload: {len(all_requirements):,}", file=sys.stderr)

    # Process each repository
    all_positive_pairs: list[dict[str, Any]] = []
    repo_reports: dict[str, dict[str, Any]] = {}

    for repo in repositories:
        print(f"\nProcessing repository: {repo}", file=sys.stderr)
        pairs, metadata = build_positive_pairs(payload, repo, taxonomy)
        print(f"  Positive pairs: {len(pairs):,}", file=sys.stderr)
        if metadata["skipped"]:
            print(f"  Skipped: {metadata['skipped']}", file=sys.stderr)
        all_positive_pairs.extend(pairs)
        repo_reports[repo] = metadata

    print(f"\nTotal positive pairs across all repos: {len(all_positive_pairs):,}", file=sys.stderr)

    # Assign windows based on strategy
    if window_strategy == "fixed_sprints" and start_date and end_date:
        print(f"\nAssigning fixed {sprint_days}-day sprints from {start_date} to {end_date}...", file=sys.stderr)
        windowed_pairs, windows = build_fixed_sprints(
            all_positive_pairs, start_date, end_date, sprint_days, split_config
        )
    else:
        print(f"\nAssigning {window_count} quantile windows...", file=sys.stderr)
        windowed_pairs, windows = assign_quantile_windows(
            all_positive_pairs, window_count, split_config
        )
    print(f"Windows created: {len(windows)}", file=sys.stderr)

    # Build per-repository, per-window issue pools for negative mining
    print("\nBuilding per-window issue pools...", file=sys.stderr)
    repo_window_pools: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for repo in repositories:
        repo_window_pools[repo] = build_window_issue_pools(
            all_requirements, repo, windows, rng, max_negative_candidates
        )

    # Record the backend used for each repository's hard negatives in the manifest.
    negative_mining_backend = "tfidf"
    print("\nMining hard negatives...", file=sys.stderr)
    all_negative_pairs: list[dict[str, Any]] = []

    # Group positive pairs by (repo, window)
    pos_by_repo_window: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in windowed_pairs:
        pos_by_repo_window[(row["repo"], row["window"])].append(row)

    for repo in repositories:
        for w in windows:
            window_id = w.window
            split = w.role
            pos_rows = pos_by_repo_window.get((repo, window_id), [])
            if not pos_rows:
                continue

            # Determine target negative count
            if max_pairs_per_window is not None:
                neg_target = max(0, max_pairs_per_window - len(pos_rows))
            else:
                neg_target = int(len(pos_rows) * negative_ratio)

            # Build known link pairs set
            known_links: set[tuple[str, str]] = set()
            for row in pos_rows:
                known_links.add((row["issue_i"]["id"], row["issue_j"]["id"]))

            # Get candidate issues for this window
            issue_pool = repo_window_pools.get(repo, {}).get(window_id, {})

            negatives = mine_hard_negatives_for_window(
                window=window_id,
                split=split,
                positive_rows_all_window=pos_rows,
                candidate_issues=issue_pool,
                known_link_pairs=known_links,
                target_count=neg_target,
                rng=rng,
                backend=negative_mining_backend,
            )
            all_negative_pairs.extend(negatives)

    print(f"Total negative pairs mined: {len(all_negative_pairs):,}", file=sys.stderr)

    # Combine positives and negatives
    all_pairs = windowed_pairs + all_negative_pairs

    # Sort by (pair_available_at, pair_id)
    all_pairs.sort(key=lambda r: (r.get("pair_available_at") or "", r.get("pair_id") or ""))

    # Write combined file
    combined_path = output_dir / f"{prefix}_pairs.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in all_pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_pairs):,} pairs to {combined_path}", file=sys.stderr)

    # Write per-split files
    split_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_pairs:
        split = row.get("split", "unused")
        if split in ("train", "validation", "online", "test"):
            split_to_rows[split].append(row)

    for split, rows in sorted(split_to_rows.items()):
        split_path = output_dir / f"{prefix}_{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(rows):,} rows -> {split_path}", file=sys.stderr)

    # Write window manifest
    manifest = {
        "mode": args.mode,
        "dataset_config": args.dataset_config,
        "data_version": config.get("data_version"),
        "processed_pairs_json": config["raw_sources"]["processed_pairs_json"],
        "negative_mining_backend": negative_mining_backend,
        "repositories": repositories,
        "window_strategy": window_strategy,
        "window_count": len(windows),
        "window_count_config": window_count,
        "start_date": start_date,
        "end_date": end_date,
        "sprint_days": sprint_days,
        "split_config": split_config,
        "seed": seed,
        "total_positive_pairs": len(windowed_pairs),
        "total_negative_pairs": len(all_negative_pairs),
        "total_pairs": len(all_pairs),
        "windows": [
            {"window": w.window, "role": w.role, "start_at": w.start_at, "end_at": w.end_at, "count": w.count}
            for w in windows
        ],
        "repository_reports": repo_reports,
        "split_counts": {s: len(r) for s, r in sorted(split_to_rows.items())},
    }
    manifest_path = output_dir / f"{prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nManifest written to {manifest_path}", file=sys.stderr)

    # Print summary
    print("\n" + "=" * 60)
    print(f"DATASET BUILD COMPLETE (mode={args.mode})")
    print("=" * 60)
    print(f"Repositories: {', '.join(repositories)}")
    print(f"Windows: {window_count}")
    print(f"Positive pairs: {len(windowed_pairs):,}")
    print(f"Negative pairs: {len(all_negative_pairs):,}")
    print(f"Total pairs: {len(all_pairs):,}")
    print()
    print("Split distribution:")
    for split in ["train", "validation", "online", "test"]:
        rows = split_to_rows.get(split, [])
        if rows:
            label_counts = Counter(r["label"] for r in rows)
            print(f"  {split}: {len(rows):,} pairs | labels: {dict(sorted(label_counts.items()))}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
