from __future__ import annotations

import sys
from pathlib import Path
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import haireder.augmentation.pair_aware as pair_aware  # noqa: E402
from haireder.augmentation.llm_generator import LLMPairAugmenter  # noqa: E402
from haireder.augmentation.llm_wiring import resolve_augmentation_config  # noqa: E402
from haireder.augmentation.llm_validator import LLMConsistencyValidator  # noqa: E402
from haireder.taxonomy import Taxonomy  # noqa: E402

PROMPT_TEMPLATE = (PROJECT_ROOT / "prompts" / "augmentation" / "pair_augmentation_prompt.md").read_text(
    encoding="utf-8"
)
VALIDATION_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "prompts" / "validation" / "augmentation_validation_prompt.md"
).read_text(encoding="utf-8")


class FakeLLMClient:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )
        return self.output


class CaptureAugmenter:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        seed_row: dict[str, object],
        target_label: str,
        variant_index: int,
        reference_pool: list[dict[str, object]] | None = None,
    ) -> dict[str, object] | None:
        self.calls.append(
            {
                "seed_row": seed_row,
                "target_label": target_label,
                "variant_index": variant_index,
                "reference_pool": reference_pool,
            }
        )
        return deepcopy(self.result)


class RecordingEmbeddingFilter:
    def __init__(self, result: dict[str, object]):
        self.result = result
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def filter(self, seed_row: dict[str, object], augmented_row: dict[str, object]) -> dict[str, object]:
        self.calls.append((seed_row, augmented_row))
        return dict(self.result)


def make_taxonomy() -> Taxonomy:
    return Taxonomy(
        labels=[
            "None",
            "General Relation",
            "Duplication",
            "Composition",
            "Temporal/Causal",
            "Workflow",
        ],
        label_definitions={
            "None": "No dependency exists between the pair.",
            "General Relation": "A weak or unspecified dependency exists.",
            "Duplication": "The two requirements describe the same underlying need.",
            "Composition": "One requirement is part of or nested within the other.",
            "Temporal/Causal": "One requirement must happen before or cause the other.",
            "Workflow": "The requirements are linked by process or operational flow.",
        },
        normalized_dependency_type_to_label={},
    )


def make_row(
    pair_id: str,
    label: str,
    issue_i_text: str | None = None,
    issue_j_text: str | None = None,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": "train",
        "issue_i": {
            "id": f"{pair_id}-A",
            "text": issue_i_text or f"Summary: Requirement A for {label}",
        },
        "issue_j": {
            "id": f"{pair_id}-B",
            "text": issue_j_text or f"Summary: Requirement B for {label}",
        },
        "label": label,
        "is_positive": label != "None",
    }


def test_generate_llm_augmented_pair_forwards_reference_pool_to_augmenter() -> None:
    seed_row = make_row("seed-1", "Composition")
    reference_pool = [
        make_row("seed-2", "Composition", "Summary: Reference A", "Summary: Reference B"),
        make_row("seed-3", "Temporal/Causal", "Summary: Other A", "Summary: Other B"),
    ]
    expected_candidate = {
        "pair_id": "AUG-seed-1-004",
        "label": "Composition",
        "source": "pair_aware_llm_augmentation",
        "augmentation": {"generator": "llm_v1"},
    }
    augmenter = CaptureAugmenter(expected_candidate)

    result = pair_aware._generate_llm_augmented_pair(seed_row, 4, augmenter, reference_pool)

    assert result == expected_candidate
    assert augmenter.calls == [
        {
            "seed_row": seed_row,
            "target_label": "Composition",
            "variant_index": 4,
            "reference_pool": reference_pool,
        }
    ]


def test_validate_llm_augmented_pair_uses_llm_validator_and_embedding_filter() -> None:
    seed_row = make_row("seed-1", "Composition")
    augmented_row = make_row(
        "AUG-seed-1-001",
        "Composition",
        "Summary: LLM Augmented A",
        "Summary: LLM Augmented B",
    )
    augmented_row.update(
        {
            "source": "pair_aware_llm_augmentation",
            "is_augmented": True,
            "source_seed_pair_id": "seed-1",
            "augmentation": {
                "generator": "llm_v1",
                "variant_index": 1,
                "target_label": "Composition",
                "source_seed_pair_id": "seed-1",
                "pair_sim_backend": "llm_generated",
                "preservation_note": "Preserves the dependency relation.",
            },
        }
    )
    validation_client = FakeLLMClient(
        '{"is_valid": true, "label_matches": true, "rationale": "validated"}'
    )
    validator = LLMConsistencyValidator(
        validation_client,
        VALIDATION_PROMPT_TEMPLATE,
        max_new_tokens=64,
        temperature=0.25,
    )
    embedding_filter = RecordingEmbeddingFilter(
        {"is_valid": True, "distance": 0.23, "min_distance": 0.1, "max_distance": 0.9}
    )

    result = pair_aware._validate_llm_augmented_pair(
        seed_row,
        augmented_row,
        "Composition",
        validator,
        embedding_filter,
    )

    assert result == {
        "is_valid": True,
        "label_matches": True,
        "similarity": None,
        "llm_validation": {
            "is_valid": True,
            "label_matches": True,
            "rationale": "validated",
        },
        "embedding_filter_result": {
            "is_valid": True,
            "distance": 0.23,
            "min_distance": 0.1,
            "max_distance": 0.9,
        },
        "reasons": [],
    }
    assert embedding_filter.calls == [(seed_row, augmented_row)]
    assert len(validation_client.calls) == 1
    validation_prompt = str(validation_client.calls[0]["prompt"])
    assert validation_client.calls[0]["max_new_tokens"] == 64
    assert validation_client.calls[0]["temperature"] == 0.25
    assert "Target relation is Composition." in validation_prompt
    assert "LLM Augmented A" in validation_prompt
    assert "LLM Augmented B" in validation_prompt


def test_augment_minority_classes_uses_llm_backend_when_requested() -> None:
    rows = [
        make_row("composition-1", "Composition"),
        make_row("temporal-1", "Temporal/Causal"),
        make_row("temporal-2", "Temporal/Causal"),
    ]
    reference_pool = [
        make_row("composition-1", "Composition"),
        make_row("composition-ref", "Composition", "Summary: Reference A", "Summary: Reference B"),
        make_row("temporal-ref", "Temporal/Causal", "Summary: Temporal A", "Summary: Temporal B"),
    ]
    generation_client = FakeLLMClient(
        '{"augmented_issue_i_text": "Summary: LLM Rewritten A", "augmented_issue_j_text": "Summary: LLM Rewritten B", "target_label": "Composition", "preservation_note": "The part-of relation remains intact."}'
    )
    augmenter = LLMPairAugmenter(
        generation_client,
        PROMPT_TEMPLATE,
        make_taxonomy(),
        reference_pairs_per_prompt=1,
        max_new_tokens=128,
        temperature=0.2,
    )
    validation_client = FakeLLMClient(
        '{"is_valid": true, "label_matches": true, "rationale": "validated"}'
    )
    validator = LLMConsistencyValidator(
        validation_client,
        VALIDATION_PROMPT_TEMPLATE,
        max_new_tokens=48,
        temperature=0.15,
    )
    embedding_filter = RecordingEmbeddingFilter(
        {"is_valid": True, "distance": 0.35, "min_distance": 0.1, "max_distance": 0.9}
    )

    augmented_rows, summary = pair_aware.augment_minority_classes(
        rows,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
        generator_backend="llm",
        llm_augmenter=augmenter,
        llm_validator=validator,
        embedding_filter=embedding_filter,
        reference_pool=reference_pool,
    )

    assert len(augmented_rows) == 1
    assert summary["generator_backend"] == "llm"
    assert summary["generated_by_label"] == {"Composition": 1}
    assert summary["new_generated_rows"] == 1
    assert summary["rejected_rows"] == 0
    candidate = augmented_rows[0]
    assert candidate["pair_id"] == "AUG-composition-1-001"
    assert candidate["label"] == "Composition"
    assert candidate["source"] == "pair_aware_llm_augmentation"
    assert candidate["is_augmented"] is True
    assert candidate["source_seed_pair_id"] == "composition-1"
    assert candidate["augmentation"]["generator"] == "llm_v1"
    assert candidate["augmentation"]["validation"] == {
        "is_valid": True,
        "label_matches": True,
        "similarity": None,
        "llm_validation": {
            "is_valid": True,
            "label_matches": True,
            "rationale": "validated",
        },
        "embedding_filter_result": {
            "is_valid": True,
            "distance": 0.35,
            "min_distance": 0.1,
            "max_distance": 0.9,
        },
        "reasons": [],
    }
    assert len(generation_client.calls) == 1
    generation_prompt = str(generation_client.calls[0]["prompt"])
    assert generation_client.calls[0]["max_new_tokens"] == 128
    assert generation_client.calls[0]["temperature"] == 0.2
    assert "Reference pair 1:" in generation_prompt
    assert "Summary: Reference A" in generation_prompt
    assert "Summary: Reference B" in generation_prompt
    assert "Summary: Temporal A" not in generation_prompt
    assert len(validation_client.calls) == 1
    validation_prompt = str(validation_client.calls[0]["prompt"])
    assert "LLM Rewritten A" in validation_prompt
    assert "LLM Rewritten B" in validation_prompt
    assert embedding_filter.calls == [(rows[0], candidate)]


def test_augment_minority_classes_rule_based_backend_matches_default_behavior() -> None:
    rows = [
        make_row("temporal-1", "Temporal/Causal"),
        make_row("temporal-2", "Temporal/Causal"),
        make_row("composition-1", "Composition"),
        make_row("none-1", "None"),
        make_row("none-2", "None"),
        make_row("none-3", "None"),
    ]

    default_augmented, default_summary = pair_aware.augment_minority_classes(
        rows,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
    )
    rule_based_augmented, rule_based_summary = pair_aware.augment_minority_classes(
        rows,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
        generator_backend="rule_based",
    )

    assert default_augmented == rule_based_augmented
    assert default_summary == rule_based_summary
    assert default_summary["generator_backend"] == "rule_based"


def test_resolve_augmentation_config_prefers_training_scope_for_formal_config() -> None:
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "formal_full_pipeline.yaml").read_text(
        encoding="utf-8"
    ))

    augmentation_config = resolve_augmentation_config(config)

    assert augmentation_config["enabled"] is True
    assert augmentation_config["generator_backend"] == "llm"
    assert augmentation_config["llm"]["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert augmentation_config["llm"]["sbert_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_resolve_augmentation_config_keeps_top_level_minimal_config() -> None:
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "experiments" / "minimal_full_pipeline.yaml").read_text(
        encoding="utf-8"
    ))

    augmentation_config = resolve_augmentation_config(config)

    assert augmentation_config["enabled"] is True
    assert augmentation_config["generator_backend"] == "rule_based"
    assert augmentation_config["llm"]["model_name"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
