"""Pair sampling and hard-negative mining for minimal runs."""

from __future__ import annotations

import json
import random
from functools import lru_cache
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from haireder.data.windowing import format_datetime, parse_jira_datetime, requirement_text, pair_identity
from haireder.taxonomy.labels import LABEL_NONE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grouped_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    """Group rows by a top-level key."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return groups


def stratified_sample_positive(
    rows: list[dict[str, Any]],
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Deterministically sample positives while preserving labels where possible."""
    if target_count <= 0:
        return []
    if len(rows) <= target_count:
        return list(rows)
    by_label = grouped_by(rows, "label")
    selected: list[dict[str, Any]] = []
    total = len(rows)
    labels = sorted(by_label)

    allocations: dict[str, int] = {}
    remaining = target_count
    if target_count < len(labels):
        chosen = sorted(labels, key=lambda label: (-len(by_label[label]), label))[:target_count]
        return [rng.choice(by_label[label]) for label in chosen]
    for label in labels:
        allocation = max(1, round(target_count * len(by_label[label]) / total))
        allocation = min(allocation, len(by_label[label]))
        allocations[label] = allocation
        remaining -= allocation

    while remaining < 0:
        reducible = [label for label in labels if allocations[label] > 1]
        if not reducible:
            break
        label = max(reducible, key=lambda name: allocations[name])
        allocations[label] -= 1
        remaining += 1

    while remaining > 0:
        expandable = [label for label in labels if allocations[label] < len(by_label[label])]
        if not expandable:
            break
        label = max(expandable, key=lambda name: len(by_label[name]) - allocations[name])
        allocations[label] += 1
        remaining -= 1

    for label in labels:
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocations[label]])
    return sorted(selected, key=lambda row: (row["pair_available_at"], row["pair_id"]))


def issue_key_pair(row: dict[str, Any]) -> tuple[str, str]:
    """Return ordered issue IDs for a pair row."""
    return str(row["issue_i"]["id"]), str(row["issue_j"]["id"])


def collect_issue_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collect issue texts from pair rows."""
    issues: dict[str, dict[str, Any]] = {}
    for row in rows:
        for side in ["issue_i", "issue_j"]:
            issue = row[side]
            issue_id = str(issue["id"])
            issues[issue_id] = {
                "id": issue_id,
                "project": issue.get("project"),
                "created": issue.get("created"),
                "text": issue.get("text", ""),
            }
    return issues


def build_window_issue_pools_from_requirements(
    requirements: list[dict[str, Any]],
    repository: str,
    window_report: dict[str, Any],
    max_candidates_per_window: int,
    rng: random.Random,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build per-window issue pools from all requirements in the selected repository."""
    windows = window_report["distribution"]
    parsed_windows: list[tuple[str, datetime, datetime]] = []
    for window, details in sorted(windows.items()):
        start_at = datetime.fromisoformat(details["start_at"])
        end_at = datetime.fromisoformat(details["end_at"])
        parsed_windows.append((window, start_at, end_at))

    candidates: dict[str, list[dict[str, Any]]] = {window: [] for window, _, _ in parsed_windows}
    for requirement in requirements:
        if requirement.get("source") != repository:
            continue
        issue_id = requirement.get("issue_key")
        created = parse_jira_datetime(requirement.get("created"))
        if not issue_id or created is None:
            continue
        for window, start_at, end_at in parsed_windows:
            if start_at <= created <= end_at:
                candidates[window].append(
                    {
                        "id": str(issue_id),
                        "project": requirement.get("project_key"),
                        "created": requirement.get("created"),
                        "created_dt": created,
                        "text": requirement_text(requirement),
                    }
                )
                break

    issue_pools: dict[str, dict[str, dict[str, Any]]] = {}
    for window, rows in candidates.items():
        rows = sorted(rows, key=lambda row: (row["created_dt"], row["id"]))
        if len(rows) > max_candidates_per_window:
            # Preserve chronological coverage while keeping the TF-IDF step local-machine friendly.
            sampled = rng.sample(rows, max_candidates_per_window)
            rows = sorted(sampled, key=lambda row: (row["created_dt"], row["id"]))
        issue_pools[window] = {
            row["id"]: {
                "id": row["id"],
                "project": row.get("project"),
                "created": row.get("created"),
                "text": row.get("text", ""),
            }
            for row in rows
        }
    return issue_pools


@lru_cache(maxsize=2)
def sentence_encoder(name: str):
    from sentence_transformers import SentenceTransformer
    from haireder.utils.model_versions import revision_kwargs
    return SentenceTransformer(name, **revision_kwargs(name))


def _encode_texts(texts: list[str], backend: str, sbert_model_name: str) -> Any:
    if backend == "sbert":
        return sentence_encoder(sbert_model_name).encode(texts, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    if backend == "tfidf":
        return TfidfVectorizer(min_df=1, max_features=20000, ngram_range=(1, 2)).fit_transform(texts)
    raise ValueError(f"Unknown embedding backend {backend}")


def mine_hard_negatives_for_window(
    window: str,
    split: str,
    positive_rows_all_window: list[dict[str, Any]],
    candidate_issues: dict[str, dict[str, Any]],
    known_link_pairs: set[tuple[str, str]],
    target_count: int,
    rng: random.Random,
    *,
    backend: str = "tfidf",
    sbert_model_name: str = "all-MiniLM-L6-v2",
    anchor_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select distinct unlinked neighbours. At least one endpoint is in the current sprint."""
    issues = candidate_issues
    issue_ids = sorted(issues)
    anchors = [i for i, key in enumerate(issue_ids) if anchor_ids is None or key in anchor_ids]
    if len(issue_ids) < 2 or target_count <= 0 or not anchors:
        return []
    repo = positive_rows_all_window[0]["repo"]
    candidates = {}
    def eligible(i, j):
        left, right = issue_ids[i], issue_ids[j]
        return left != right and (left, right) not in known_link_pairs and (right, left) not in known_link_pairs

    if backend == "random":
        # Sample unordered pair indices exactly uniformly without materializing O(n²) pairs.
        from math import isqrt
        size = len(issue_ids) * (len(issue_ids)-1) // 2
        swaps = {}
        anchor_set = set(anchors)
        while size and len(candidates) < target_count:
            position = rng.randrange(size)
            key = swaps.get(position, position)
            size -= 1
            swaps[position] = swaps.get(size, size)
            j = (1 + isqrt(1 + 8 * key)) // 2
            i = key - j * (j - 1) // 2
            if not (i in anchor_set or j in anchor_set) or not eligible(i,j):
                continue
            if i not in anchor_set or (j in anchor_set and rng.randrange(2)):
                i,j = j,i
            candidates[key] = (None,issue_ids[i],issue_ids[j])
        selected = list(candidates.values())
    else:
        matrix = _encode_texts([issues[key]["text"] for key in issue_ids], backend, sbert_model_name)
        model = NearestNeighbors(metric="cosine").fit(matrix)
        k = min(20, len(issue_ids))
        while True:
            distances, indices = model.kneighbors(matrix[anchors], n_neighbors=k)
            for i, ds, js in zip(anchors, distances, indices):
                for distance, j in zip(ds, js):
                    if eligible(i, j):
                        left, right = issue_ids[i], issue_ids[j]
                        candidates.setdefault(tuple(sorted((left, right))), (1 - float(distance), left, right))
            if len(candidates) >= target_count or k == len(issue_ids):
                break
            k = min(k * 2, len(issue_ids))
        selected = sorted(candidates.values(), key=lambda item: (-item[0], item[1], item[2]))[:target_count]

    method_name = "uniform_random" if backend == "random" else f"{backend}_nearest_neighbors"
    source_name = "random_negative" if backend == "random" else f"{backend}_hard_negative"
    negatives: list[dict[str, Any]] = []
    for similarity, issue_i, issue_j in selected:
        source_issue = issues[issue_i]
        target_issue = issues[issue_j]
        source_created = parse_jira_datetime(source_issue.get("created"))
        target_created = parse_jira_datetime(target_issue.get("created"))
        available_at = max(
            value for value in [source_created, target_created] if value is not None
        )
        negatives.append(
            {
                "pair_id": pair_identity(repo, issue_i, issue_j),
                "repo": repo,
                "window": window,
                "split": split,
                "pair_available_at": format_datetime(available_at),
                "issue_i": source_issue,
                "issue_j": target_issue,
                "label": LABEL_NONE,
                "dependency_type": None,
                "dependency_types": [],
                "raw_link_type": None,
                "raw_link_types": [],
                "is_positive": False,
                "source": source_name,
                "negative_mining": {
                    "method": method_name,
                    "backend": backend,
                    "similarity": similarity,
                    "candidate_source": "all_repository_requirements_in_window",
                },
            }
        )
    return negatives


def construct_minimal_pairs(
    positive_rows: list[dict[str, Any]],
    issue_pools_by_window: dict[str, dict[str, dict[str, Any]]],
    max_pairs_per_window: int,
    negative_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct capped positives plus mined negatives for the minimal run."""
    rng = random.Random(seed)
    by_window = grouped_by(positive_rows, "window")
    known_link_pairs = {issue_key_pair(row) for row in positive_rows}
    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"windows": {}, "seed": seed}

    positive_target = max(1, int(round(max_pairs_per_window / (1.0 + negative_ratio))))
    negative_target = max(0, max_pairs_per_window - positive_target)

    for window in sorted(by_window):
        window_rows = sorted(by_window[window], key=lambda row: (row["pair_available_at"], row["pair_id"]))
        split = window_rows[0]["split"]
        selected_positives = stratified_sample_positive(window_rows, positive_target, rng)
        negatives = mine_hard_negatives_for_window(
            window=window,
            split=split,
            positive_rows_all_window=window_rows,
            candidate_issues=issue_pools_by_window.get(window, {}),
            known_link_pairs=known_link_pairs,
            target_count=negative_target,
            rng=rng,
        )
        combined = selected_positives + negatives
        combined.sort(key=lambda row: (row["pair_available_at"], row["pair_id"]))
        all_rows.extend(combined)
        report["windows"][window] = {
            "split": split,
            "available_positive_pairs": len(window_rows),
            "negative_candidate_issues": len(issue_pools_by_window.get(window, {})),
            "selected_positive_pairs": len(selected_positives),
            "selected_negative_pairs": len(negatives),
            "total_pairs": len(combined),
            "label_counts": dict(sorted(Counter(row["label"] for row in combined).items())),
        }

    report["total_pairs"] = len(all_rows)
    report["label_counts"] = dict(sorted(Counter(row["label"] for row in all_rows).items()))
    report["split_counts"] = dict(sorted(Counter(row["split"] for row in all_rows).items()))
    return all_rows, report


def construct_multi_repo_pairs(
    payload: dict[str, Any],
    repositories: list[str],
    window_report: dict[str, Any],
    max_pairs_per_window: int | None,
    negative_ratio: float,
    max_negative_candidates: int,
    seed: int,
    *,
    backend: str = "tfidf",
    sbert_model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct positive + negative pairs across multiple repositories.

    Wraps construct_minimal_pairs() logic for each repository and aggregates.

    Args:
        payload: The full processed JSON payload.
        repositories: List of repository names.
        window_report: Window distribution report (from windowing module).
        max_pairs_per_window: Cap per window (None for no cap).
        negative_ratio: Ratio of negatives to positives.
        max_negative_candidates: Max candidate issues per window.
        seed: Random seed.
        backend: Negative mining backend ("tfidf" or "sbert").
        sbert_model_name: SBERT model name (if backend="sbert").

    Returns:
        (all_pairs, aggregate_report) tuple.
    """

    rng = random.Random(seed)
    all_rows = []
    repo_reports = {}
    total_positives = total_negatives = 0
    from haireder.data.windowing import build_positive_pairs
    from haireder.taxonomy import load_taxonomy
    taxonomy=load_taxonomy()
    calendar=window_report["distribution"]
    for repo in repositories:
        positives,_=build_positive_pairs(payload,repo,taxonomy)
        windows=defaultdict(list)
        for r in positives:
            at=parse_jira_datetime(r["pair_available_at"])
            for w,info in calendar.items():
                if parse_jira_datetime(info["start_at"]) <= at <= parse_jira_datetime(info["end_at"]):
                    windows[w].append(dict(r,window=w,split=info.get("split") or info.get("role") or "unused"))
                    break
        pools=build_window_issue_pools_from_requirements(payload["requirements"],repo,window_report,max_negative_candidates,rng)
        known={(r["source_issue_key"],r["target_issue_key"]) for r in payload["dependency_pairs"]
               if r["source_source"]==repo and r["target_source"]==repo}
        pcount=ncount=0
        for w,all_positive in windows.items():
            target=len(all_positive) if max_pairs_per_window is None else int(max_pairs_per_window/(1+negative_ratio))
            selected=stratified_sample_positive(all_positive,target,rng)
            neg=mine_hard_negatives_for_window(w,selected[0]["split"] if selected else "unused",all_positive,
                    pools[w],known,round(len(selected)*negative_ratio),rng,backend=backend,sbert_model_name=sbert_model_name)
            all_rows.extend(selected+neg)
            pcount+=len(selected)
            ncount+=len(neg)
        repo_reports[repo]=dict(positive_pairs=pcount,negative_pairs=ncount,total=pcount+ncount)
        total_positives+=pcount
        total_negatives+=ncount
    aggregate_report = {
        "repositories": repositories,
        "total_pairs": len(all_rows),
        "total_positives": total_positives,
        "total_negatives": total_negatives,
        "repository_reports": repo_reports,
        "backend": backend,
        "seed": seed,
    }
    return all_rows, aggregate_report


def render_pair_distribution_markdown(report: dict[str, Any]) -> str:
    """Render pair construction report."""
    lines = [
        "# Minimal Pair Distribution",
        "",
        "## Summary",
        "",
        f"- Total pairs: `{report['total_pairs']}`",
        f"- Seed: `{report['seed']}`",
        f"- Label counts: `{report['label_counts']}`",
        f"- Split counts: `{report['split_counts']}`",
        "",
        "## Windows",
        "",
        "| Window | Split | Positive | Negative | Total | Label counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for window, details in report["windows"].items():
        lines.append(
            f"| {window} | {details['split']} | {details['selected_positive_pairs']} | "
            f"{details['selected_negative_pairs']} | {details['total_pairs']} | `{details['label_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Negative Candidate Issue Pools",
            "",
            "| Window | Candidate issues |",
            "|---|---:|",
        ]
    )
    for window, details in report["windows"].items():
        lines.append(f"| {window} | {details['negative_candidate_issues']} |")
    lines.append("")
    return "\n".join(lines)
