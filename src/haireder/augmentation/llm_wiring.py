"""Shared helpers for wiring LLM components into augmentation scripts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from haireder.taxonomy.labels import LABEL_NONE, LABELS


def resolve_augmentation_config(experiment_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve augmentation config across legacy and formal experiment layouts.

    The minimal pipeline stores augmentation at the top level, while the formal
    paper pipeline stores it under training.augmentation. Prefer the training
    scoped config when present so formal LLM+SBERT settings are not silently
    replaced by the rule-based defaults.
    """
    training_config = experiment_config.get("training") or {}
    scoped_config = training_config.get("augmentation") or {}
    top_level_config = experiment_config.get("augmentation") or {}
    resolved = dict(scoped_config or top_level_config)

    llm_config = dict(resolved.get("llm") or {})
    for key in (
        "prompt_template",
        "validation_prompt_template",
        "reference_pairs_per_prompt",
        "max_new_tokens",
        "temperature",
        "validation_temperature",
        "sbert_model",
        "embedding_min_distance",
        "embedding_max_distance",
    ):
        if key in resolved and key not in llm_config:
            llm_config[key] = resolved[key]
    if llm_config:
        resolved["llm"] = llm_config
    return resolved


def build_llm_components(
    llm_config: dict[str, Any],
    prompt_template: str,
    validation_prompt_template: str,
    taxonomy: Any,
) -> tuple[Any, Any, Any]:
    """Create LLM augmenter, validator, and embedding filter from config."""
    from haireder.augmentation import (
        LLMPairAugmenter,
        LLMConsistencyValidator,
        SBERTEmbeddingFilter,
    )
    from haireder.utils.llm_client import create_llm_client

    backend = llm_config.get("backend", "local")
    model_name = llm_config.get("model_name", "HuggingFaceTB/SmolLM2-135M-Instruct")
    validator_name = llm_config.get("validator_model_name", "Qwen/Qwen2.5-3B-Instruct")
    if model_name == validator_name:
        raise ValueError("Augmentation requires an independent validation model")
    gen_client = create_llm_client(backend=backend, model_name=model_name)
    val_client = create_llm_client(backend=backend, model_name=validator_name)
    augmenter = LLMPairAugmenter(
        gen_client,
        prompt_template,
        taxonomy,
        reference_pairs_per_prompt=int(llm_config.get("reference_pairs_per_prompt", 3)),
        max_new_tokens=int(llm_config.get("max_new_tokens", 512)),
        temperature=float(llm_config.get("temperature", 0.7)),
    )
    validator = LLMConsistencyValidator(
        val_client,
        validation_prompt_template,
        max_new_tokens=256,
        temperature=float(llm_config.get("validation_temperature", 0.3)),
    )
    emb_filter = SBERTEmbeddingFilter(
        model_name=llm_config.get("sbert_model", "all-MiniLM-L6-v2"),
        min_distance=float(llm_config.get("embedding_min_distance", 0.1)),
        max_distance=float(llm_config.get("embedding_max_distance", 0.9)),
    )
    return augmenter, validator, emb_filter


def check_all_targets_met(
    rows: list[dict[str, Any]],
    existing_augmented_rows: list[dict[str, Any]],
    include_none: bool,
    target_count_per_label: int | None,
) -> bool:
    """Return True if all label targets are already met by existing data."""
    counts_before = Counter(str(row.get("label") or "") for row in rows)
    reusable_counts = Counter(str(row.get("label") or "") for row in existing_augmented_rows)
    available = counts_before + reusable_counts
    present = {str(row.get("label") or "") for row in rows}
    labels = [label for label in LABELS if label in present]
    if not include_none:
        labels = [label for label in labels if label != LABEL_NONE]
    if not labels:
        return True
    target = target_count_per_label if target_count_per_label is not None else max(counts_before[label] for label in labels)
    return all(available[label] >= target for label in labels)
