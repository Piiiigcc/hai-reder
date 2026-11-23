from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.augmentation.llm_generator import LLMPairAugmenter
from haireder.taxonomy import Taxonomy


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


PROMPT_TEMPLATE = (PROJECT_ROOT / "prompts" / "augmentation" / "pair_augmentation_prompt.md").read_text(
    encoding="utf-8"
)


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


def make_row(pair_id: str, label: str, issue_i_text: str, issue_j_text: str) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": "train",
        "issue_i": {"id": f"{pair_id}-A", "text": issue_i_text},
        "issue_j": {"id": f"{pair_id}-B", "text": issue_j_text},
        "label": label,
        "is_positive": label != "None",
    }


def test_llm_pair_augmenter_builds_schema_compatible_augmented_row() -> None:
    seed_row = make_row(
        "seed-1",
        "Composition",
        "Summary: Requirement A for composition",
        "Summary: Requirement B for composition",
    )
    reference_row_1 = make_row(
        "seed-2",
        "Composition",
        "Summary: Another composition example A",
        "Summary: Another composition example B",
    )
    reference_row_2 = make_row(
        "seed-3",
        "Composition",
        "Summary: Third composition example A",
        "Summary: Third composition example B",
    )
    irrelevant_row = make_row(
        "seed-4",
        "Temporal/Causal",
        "Summary: Temporal example A",
        "Summary: Temporal example B",
    )
    fake_client = FakeLLMClient(
        '{"augmented_issue_i_text": "Summary: Rewritten A", "augmented_issue_j_text": "Summary: Rewritten B", "target_label": "Composition", "preservation_note": "The paraphrase still expresses a part-of relation."}'
    )
    augmenter = LLMPairAugmenter(
        fake_client,
        PROMPT_TEMPLATE,
        make_taxonomy(),
        reference_pairs_per_prompt=2,
        max_new_tokens=128,
        temperature=0.2,
    )
    augmented = augmenter.generate(
        seed_row,
        "Composition",
        3,
        reference_pool=[seed_row, reference_row_1, reference_row_2, irrelevant_row],
    )

    assert augmented is not None
    assert augmented["pair_id"] == "AUG-seed-1-003"
    assert augmented["label"] == "Composition"
    assert augmented["source"] == "pair_aware_llm_augmentation"
    assert augmented["is_augmented"] is True
    assert augmented["source_seed_pair_id"] == "seed-1"
    assert augmented["augmentation"]["generator"] == "llm_v1"
    assert augmented["augmentation"]["variant_index"] == 3
    assert augmented["augmentation"]["target_label"] == "Composition"
    assert augmented["augmentation"]["source_seed_pair_id"] == "seed-1"
    assert augmented["augmentation"]["pair_sim_backend"] == "llm_generated"
    assert augmented["augmentation"]["preservation_note"] == "The paraphrase still expresses a part-of relation."
    assert augmented["issue_i"]["id"] == "seed-1-A-AUG003"
    assert augmented["issue_j"]["id"] == "seed-1-B-AUG003"
    assert augmented["issue_i"]["text"] == "Summary: Rewritten A"
    assert augmented["issue_j"]["text"] == "Summary: Rewritten B"

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    prompt = str(call["prompt"])
    assert call["max_new_tokens"] == 128
    assert call["temperature"] == 0.2
    assert "Composition" in prompt
    assert "One requirement is part of or nested within the other." in prompt
    assert "Summary: Requirement A for composition" in prompt
    assert "Summary: Requirement B for composition" in prompt
    assert "Summary: Another composition example A" in prompt
    assert "Summary: Third composition example A" in prompt
    assert "Summary: Temporal example A" not in prompt
    assert "{{target_label}}" not in prompt
    assert "{{issue_i_text}}" not in prompt
    assert "{{issue_j_text}}" not in prompt


def test_llm_pair_augmenter_returns_none_on_malformed_json() -> None:
    seed_row = make_row(
        "seed-1",
        "Composition",
        "Summary: Requirement A for composition",
        "Summary: Requirement B for composition",
    )
    augmenter = LLMPairAugmenter(
        FakeLLMClient("not json"),
        PROMPT_TEMPLATE,
        make_taxonomy(),
    )

    assert augmenter.generate(seed_row, "Composition", 1) is None


def test_llm_pair_augmenter_returns_none_on_target_label_mismatch() -> None:
    seed_row = make_row(
        "seed-1",
        "Composition",
        "Summary: Requirement A for composition",
        "Summary: Requirement B for composition",
    )
    fake_client = FakeLLMClient(
        '{"augmented_issue_i_text": "Summary: Rewritten A", "augmented_issue_j_text": "Summary: Rewritten B", "target_label": "Temporal/Causal", "preservation_note": "Wrong label"}'
    )
    augmenter = LLMPairAugmenter(
        fake_client,
        PROMPT_TEMPLATE,
        make_taxonomy(),
    )

    assert augmenter.generate(seed_row, "Composition", 1) is None
def test_literal_code_brace_in_rewrite_survives_json_extraction():
    import json
    from haireder.augmentation.llm_generator import _first_json_object
    payload={"augmented_issue_i_text":"Accept a literal } in an escaped value.",
             "augmented_issue_j_text":"Handle unmatched { in the parser."}
    assert json.loads(_first_json_object("Output: "+json.dumps(payload)+"\nDone.")) == payload
