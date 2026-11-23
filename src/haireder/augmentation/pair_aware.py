"""Pair-aware augmentation helpers.

The default generator is deterministic, rule-based, and uses TF-IDF similarity
(NOT the LLM+SBERT pipeline described in the paper). It preserves the row schema
and records enough metadata for a stronger LLM generator to replace it later.
Every augmented row is tagged with `generator='rule_based_v1'` and
`pair_sim_backend='tfidf_cosine'`.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from haireder.augmentation.embedding_filter import SBERTEmbeddingFilter
    from haireder.augmentation.llm_generator import LLMPairAugmenter
    from haireder.augmentation.llm_validator import LLMConsistencyValidator

from haireder.taxonomy.labels import LABEL_NONE, LABELS


def _issue_text(row: dict[str, Any], side: str) -> str:
    issue = row.get(side) or {}
    return str(issue.get("text") or "").strip()


def _pair_text(row: dict[str, Any]) -> str:
    return "\n\n".join(
        part for part in [_issue_text(row, "issue_i"), _issue_text(row, "issue_j")] if part
    )


def _paraphrase_text(text: str, *, side_name: str, variant_index: int) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return f"Paraphrased {side_name} requirement variant {variant_index}."

    lines = stripped.splitlines()
    output: list[str] = []
    summary_rewritten = False
    for line in lines:
        if line.lower().startswith("summary:") and not summary_rewritten:
            summary = line.split(":", 1)[1].strip()
            output.append(f"Summary: Rephrased {side_name} request v{variant_index}: {summary}")
            summary_rewritten = True
        else:
            output.append(line)
    if not summary_rewritten:
        output.insert(0, f"Summary: Rephrased {side_name} request v{variant_index}")
    return "\n".join(output)


def compute_pair_similarity(seed_row: dict[str, Any], augmented_row: dict[str, Any]) -> float:
    """Compute a bounded TF-IDF cosine similarity between two pair texts."""
    seed_text = _pair_text(seed_row)
    augmented_text = _pair_text(augmented_row)
    if not seed_text or not augmented_text:
        return 0.0
    try:
        matrix = TfidfVectorizer(min_df=1).fit_transform([seed_text, augmented_text])
    except ValueError:
        return 0.0
    return min(1.0, max(0.0, float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])))


def build_rule_based_augmented_pair(
    seed_row: dict[str, Any],
    *,
    variant_index: int,
    generator_id: str = "rule_based_v1",
) -> dict[str, Any]:
    """Create one schema-compatible augmented row from a seed requirement pair."""
    if variant_index <= 0:
        raise ValueError("variant_index must be positive")
    label = str(seed_row.get("label") or "")
    if label not in LABELS:
        raise ValueError(f"Unknown label for augmentation: {label}")

    augmented = deepcopy(seed_row)
    for field in ("transcript","review_detail","accepted_label","gold_label","review_source","analyst_status",
                  "human_reviewed","analyst_simulation","confidence","confidence_region","decision_path",
                  "predicted_label","rationale","s1","s2","s3","s3_neighbors","calibration_mode"):
        augmented.pop(field,None)
    seed_pair_id = str(seed_row["pair_id"])
    augmented["pair_id"] = f"AUG-{seed_pair_id}-{variant_index:03d}"
    augmented["label"] = label
    augmented["source"] = "pair_aware_rule_based_augmentation"
    augmented["is_augmented"] = True
    augmented["source_seed_pair_id"] = seed_pair_id
    augmented["augmentation"] = {
        "generator": generator_id,
        "variant_index": variant_index,
        "target_label": label,
        "source_seed_pair_id": seed_pair_id,
        "pair_sim_backend": "tfidf_cosine",
        "preservation_note": "Requirement texts were lightly rephrased as a pair while keeping the gold dependency label.",
    }

    for side, side_name in (("issue_i", "A"), ("issue_j", "B")):
        issue = dict(augmented.get(side) or {})
        issue["id"] = f"{issue.get('id', side)}-AUG{variant_index:03d}"
        issue["text"] = _paraphrase_text(issue.get("text", ""), side_name=side_name, variant_index=variant_index)
        augmented[side] = issue
    return augmented


def augmentation_cache_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    """Return the stable cache key for an augmented row."""
    metadata = row.get("augmentation") or {}
    seed_pair_id = str(row.get("source_seed_pair_id") or metadata.get("source_seed_pair_id") or "")
    target_label = str(metadata.get("target_label") or row.get("label") or "")
    variant_index = int(metadata.get("variant_index") or 0)
    generator = str(metadata.get("generator") or "")
    return seed_pair_id, target_label, variant_index, generator


def validate_augmented_pair(
    seed_row: dict[str, Any],
    augmented_row: dict[str, Any],
    *,
    target_label: str,
    min_similarity: float = 0.05,
    max_similarity: float = 0.995,
) -> dict[str, Any]:
    """Validate target-label preservation and simple lexical distance bounds."""
    if target_label not in LABELS:
        raise ValueError(f"Unknown target label: {target_label}")
    observed_label = str(augmented_row.get("label") or "")
    similarity = compute_pair_similarity(seed_row, augmented_row)
    label_matches = observed_label == target_label
    far_enough = similarity >= min_similarity
    not_duplicate = similarity <= max_similarity
    is_valid = bool(label_matches and far_enough and not_duplicate)
    reasons = []
    if not label_matches:
        reasons.append("label_mismatch")
    if not far_enough:
        reasons.append("similarity_too_low")
    if not not_duplicate:
        reasons.append("similarity_too_high")
    return {
        "is_valid": is_valid,
        "label_matches": label_matches,
        "similarity": similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "reasons": reasons,
    }


def _generate_llm_augmented_pair(
    seed_row: dict[str, Any],
    variant_index: int,
    llm_augmenter: LLMPairAugmenter,
    reference_pool: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    target_label = str(seed_row.get("label") or "")
    return llm_augmenter.generate(
        seed_row,
        target_label=target_label,
        variant_index=variant_index,
        reference_pool=reference_pool,
    )


def _validate_llm_augmented_pair(
    seed_row: dict[str, Any],
    augmented_row: dict[str, Any],
    target_label: str,
    llm_validator: LLMConsistencyValidator | None,
    embedding_filter: SBERTEmbeddingFilter | None,
) -> dict[str, Any]:
    result = {
        "is_valid": True,
        "label_matches": True,
        "similarity": None,
        "llm_validation": None,
        "embedding_filter_result": None,
        "reasons": [],
    }

    if llm_validator is not None:
        llm_result = llm_validator.validate(seed_row, augmented_row, target_label)
        result["llm_validation"] = llm_result
        result["label_matches"] = llm_result.get("label_matches") is True
        if llm_result.get("is_valid") is not True or not result["label_matches"]:
            result["is_valid"] = False
            result["reasons"].append("llm_consistency_rejected")

    if embedding_filter is not None:
        emb_result = embedding_filter.filter(seed_row, augmented_row)
        result["embedding_filter_result"] = emb_result
        if not emb_result.get("is_valid", False):
            result["is_valid"] = False
            result["reasons"].append("embedding_filter_rejected")

    return result


def _target_labels(rows: list[dict[str, Any]], *, include_none: bool) -> list[str]:
    present = {str(row.get("label") or "") for row in rows}
    labels = [label for label in LABELS if label in present]
    if not include_none:
        labels = [label for label in labels if label != LABEL_NONE]
    return labels


def _build_target_counts(
    rows: list[dict[str, Any]],
    *,
    include_none: bool,
    target_count_per_label: int | None,
) -> dict[str, int]:
    counts = Counter(str(row.get("label") or "") for row in rows)
    labels = _target_labels(rows, include_none=include_none)
    if not labels:
        return {}
    target = target_count_per_label if target_count_per_label is not None else max(counts[label] for label in labels)
    return {label: max(counts[label], int(target)) for label in labels}


def augment_minority_classes(
    rows: list[dict[str, Any]],
    *,
    existing_augmented_rows: list[dict[str, Any]] | None = None,
    max_variants_per_seed: int | None = 2,
    include_none: bool = False,
    target_count_per_label: int | None = None,
    min_similarity: float = 0.05,
    max_similarity: float = 0.995,
    seed: int = 42,
    generator_backend: str = "rule_based",
    llm_augmenter: LLMPairAugmenter | None = None,
    llm_validator: LLMConsistencyValidator | None = None,
    embedding_filter: SBERTEmbeddingFilter | None = None,
    reference_pool: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate validated augmented rows until selected labels reach target counts."""
    if max_variants_per_seed is not None and max_variants_per_seed <= 0:
        return [], {
            "generated_rows": 0,
            "rejected_rows": 0,
            "reason": "max_variants_per_seed <= 0",
            "generator_backend": generator_backend,
            "counts_before": dict(sorted(Counter(str(row.get("label") or "") for row in rows).items())),
        }

    rng = random.Random(seed)
    use_llm_backend = generator_backend == "llm"
    if use_llm_backend and any(x is None for x in (llm_augmenter, llm_validator, embedding_filter)):
        raise ValueError("LLM augmentation requires a generator, independent validator and embedding filter")
    counts_before = Counter(str(row.get("label") or "") for row in rows)
    target_counts = _build_target_counts(
        rows,
        include_none=include_none,
        target_count_per_label=target_count_per_label,
    )
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get("label") or "")
        if label in target_counts:
            by_label[label].append(row)

    seed_pair_ids = {str(row.get("pair_id") or "") for row in rows}
    reusable_existing: list[dict[str, Any]] = []
    existing_keys: set[tuple[str, str, int, str]] = set()
    for existing in existing_augmented_rows or []:
        key = augmentation_cache_key(existing)
        seed_pair_id, label, variant_index, generator = key
        if not seed_pair_id or seed_pair_id not in seed_pair_ids:
            continue
        if label not in target_counts or variant_index <= 0 or not generator:
            continue
        validation = (existing.get("augmentation") or {}).get("validation") or {}
        if use_llm_backend and (generator != "llm_v1" or not validation.get("label_matches") or not validation.get("embedding_filter_result")):
            continue
        if not validation.get("is_valid", False):
            continue
        if key in existing_keys:
            continue
        reusable_existing.append(existing)
        existing_keys.add(key)

    reusable_counts = Counter(str(row.get("label") or "") for row in reusable_existing)
    available_counts = counts_before + reusable_counts
    new_augmented_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    generated_by_label: Counter[str] = Counter()
    for label, target in target_counts.items():
        needed = max(0, target - available_counts[label])
        if needed == 0:
            continue
        seeds = sorted(by_label[label], key=lambda row: str(row.get("pair_id") or ""))
        rng.shuffle(seeds)
        if not seeds:
            continue
        generated_for_label = 0
        attempts = max_variants_per_seed or max(4, 4 * ((needed + len(seeds) - 1) // len(seeds)))
        for variant_index in range(1, attempts + 1):
            for seed_row in seeds:
                if use_llm_backend:
                    candidate = _generate_llm_augmented_pair(
                        seed_row,
                        variant_index,
                        llm_augmenter,
                        reference_pool,
                    )
                    if candidate is None:
                        continue
                else:
                    candidate = build_rule_based_augmented_pair(seed_row, variant_index=variant_index)
                if augmentation_cache_key(candidate) in existing_keys:
                    continue
                if use_llm_backend:
                    validation = _validate_llm_augmented_pair(
                        seed_row,
                        candidate,
                        label,
                        llm_validator,
                        embedding_filter,
                    )
                else:
                    validation = validate_augmented_pair(
                        seed_row,
                        candidate,
                        target_label=label,
                        min_similarity=min_similarity,
                        max_similarity=max_similarity,
                    )
                candidate["augmentation"]["validation"] = validation
                if validation["is_valid"]:
                    new_augmented_rows.append(candidate)
                    existing_keys.add(augmentation_cache_key(candidate))
                    generated_by_label[label] += 1
                    generated_for_label += 1
                    if generated_for_label >= needed:
                        break
                else:
                    rejected_rows.append(candidate)
            if generated_for_label >= needed:
                break

    augmented_rows = [*reusable_existing, *new_augmented_rows]
    counts_after = counts_before + Counter(str(row.get("label") or "") for row in augmented_rows)
    unmet_by_label = {
        label: max(0, target - counts_after.get(label, 0))
        for label, target in target_counts.items()
        if max(0, target - counts_after.get(label, 0)) > 0
    }
    summary = {
        "seed": seed,
        "generator_backend": generator_backend,
        "include_none": include_none,
        "max_variants_per_seed": max_variants_per_seed,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "counts_before": dict(sorted(counts_before.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "reused_by_label": dict(sorted(reusable_counts.items())),
        "generated_by_label": dict(sorted(generated_by_label.items())),
        "counts_after": dict(sorted(counts_after.items())),
        "unmet_by_label": dict(sorted(unmet_by_label.items())),
        "generated_rows": len(augmented_rows),
        "new_generated_rows": len(new_augmented_rows),
        "reused_rows": len(reusable_existing),
        "rejected_rows": len(rejected_rows),
    }
    return augmented_rows, summary


def render_augmentation_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Pair-Aware Augmentation Report",
        "",
        f"- Generated rows: `{summary.get('generated_rows', 0)}`",
        f"- Reused rows: `{summary.get('reused_rows', 0)}`",
        f"- Newly generated rows: `{summary.get('new_generated_rows', 0)}`",
        f"- Rejected rows: `{summary.get('rejected_rows', 0)}`",
        f"- Include None label: `{summary.get('include_none')}`",
        "",
        "| Label | Before | Target | Generated | After |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = sorted(
        set(summary.get("counts_before", {}))
        | set(summary.get("target_counts", {}))
        | set(summary.get("generated_by_label", {}))
        | set(summary.get("counts_after", {}))
    )
    for label in labels:
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                label,
                summary.get("counts_before", {}).get(label, 0),
                summary.get("target_counts", {}).get(label, 0),
                summary.get("generated_by_label", {}).get(label, 0),
                summary.get("counts_after", {}).get(label, 0),
            )
        )
    lines.append("")
    return "\n".join(lines)
