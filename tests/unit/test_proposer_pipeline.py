import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

training_data_builder = importlib.import_module("haireder.training.data_builder")
build_proposer_training_examples = training_data_builder.build_proposer_training_examples
build_critic_training_examples = training_data_builder.build_critic_training_examples
build_judge_training_examples = training_data_builder.build_judge_training_examples
build_judge_training_example_from_real_transcript = (
    training_data_builder.build_judge_training_example_from_real_transcript
)
select_stratified_training_examples = training_data_builder.select_stratified_training_examples
parse_proposer_output = importlib.import_module(
    "haireder.agents.proposer"
).parse_proposer_output
build_proposer_target_payload = importlib.import_module(
    "haireder.agents.proposer"
).build_proposer_target_payload
lora_trainer = importlib.import_module("haireder.training.lora_trainer")
merge_prompt_and_target_ids = lora_trainer.merge_prompt_and_target_ids
require_valid_parsed_output = lora_trainer.require_valid_parsed_output
select_inference_device = lora_trainer.select_inference_device
resolve_scoped_path = importlib.import_module("haireder.utils.config").resolve_scoped_path
run_proposer_lora = importlib.import_module("scripts.training.run_proposer_lora")
run_critic_lora = importlib.import_module("scripts.training.run_critic_lora")
run_judge_lora = importlib.import_module("scripts.training.run_judge_lora")


def make_row(pair_id: str, label: str, split: str) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": split,
        "issue_i": {
            "id": f"{pair_id}-A",
            "text": "Summary: Requirement A\nIssue type: Story\nStatus: Open",
        },
        "issue_j": {
            "id": f"{pair_id}-B",
            "text": "Summary: Requirement B\nIssue type: Bug\nStatus: Closed",
        },
        "label": label,
    }


def test_build_proposer_training_examples_uses_only_train_rows_and_renders_json_targets() -> None:
    rows = [
        make_row("train-1", "Temporal/Causal", "train"),
        make_row("validation-1", "Composition", "validation"),
    ]
    prompt_template = "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}"

    examples = build_proposer_training_examples(rows, prompt_template)

    assert len(examples) == 1
    example = examples[0]
    assert example["pair_id"] == "train-1"
    assert "Requirement A:" in example["prompt_text"]
    assert "Summary: Requirement A" in example["prompt_text"]
    assert example["prompt_text"].endswith("JSON Output:\n")
    target_payload = json.loads(example["target_text"])
    assert target_payload["label"] == "Temporal/Causal"
    assert target_payload["verbal_confidence"] in {
        "very_low",
        "low",
        "medium",
        "high",
        "very_high",
    }


def test_payload_rationale_override() -> None:
    p = build_proposer_target_payload({"label": "Temporal/Causal"}, rationale="Block link detected")

    assert "Block link detected" in p["rationale"]
    assert "labeled Temporal/Causal based on" not in p["rationale"]


def test_critic_stance_follows_proposer_label_correctness(monkeypatch) -> None:
    rows = [
        {
            "pair_id": "pair-1",
            "repo": "Apache",
            "window": "W1",
            "split": "train",
            "label": "Temporal/Causal",
            "issue_i": {"id": "A", "text": "Summary: Add feature X\nIssue type: Story"},
            "issue_j": {"id": "B", "text": "Summary: Feature X depends on Y\nIssue type: Bug"},
        },
        {
            "pair_id": "pair-2",
            "repo": "Apache",
            "window": "W1",
            "split": "train",
            "label": "Composition",
            "issue_i": {"id": "C", "text": "Summary: Part of feature\nIssue type: Story"},
            "issue_j": {"id": "D", "text": "Summary: Parent feature\nIssue type: Epic"},
        },
    ]
    prompt_template = (
        "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}\n\n"
        "Proposer predicts:\n{{proposer_prediction}}\n\nJSON Output:\n"
    )
    original_build_critic_target_payload = training_data_builder.build_critic_target_payload

    def flipped_build_critic_target_payload(row: dict, proposer_label: str, rationale: str | None = None) -> dict:
        payload = original_build_critic_target_payload(row, proposer_label, rationale=rationale)
        payload["stance"] = "challenge" if payload["stance"] == "endorse" else "endorse"
        return payload

    monkeypatch.setattr(training_data_builder, "build_critic_target_payload", flipped_build_critic_target_payload)

    examples = build_critic_training_examples(rows, prompt_template, allowed_splits=("train",))

    assert examples[0]["critic_scenario"] == "endorse"
    assert examples[1]["critic_scenario"] == "challenge"


def test_judge_example_from_real_transcript_uses_critic_stance_for_termination() -> None:
    row = {
        "pair_id": "pair-1",
        "repo": "Apache",
        "window": "W1",
        "split": "train",
        "label": "Temporal/Causal",
        "issue_i": {"id": "A", "text": "Summary: Add feature X\nIssue type: Story"},
        "issue_j": {"id": "B", "text": "Summary: Feature X depends on Y\nIssue type: Bug"},
    }
    prompt = "Judge prompt {{issue_i_text}} {{issue_j_text}} {{debate_transcript_json}}\n\nJSON Output:\n"
    example = build_judge_training_example_from_real_transcript(
        row,
        {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"},
        {"stance": "endorse", "label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"},
        prompt,
    )
    assert example["termination_reason"] == "agreement", f"got {example['termination_reason']}"

    example2 = build_judge_training_example_from_real_transcript(
        row,
        {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"},
        {"stance": "challenge", "label": "Composition", "rationale": "r2", "verbal_confidence": "medium"},
        prompt,
    )
    assert example2["termination_reason"] == "round_budget", f"got {example2['termination_reason']}"


def test_parse_proposer_output_maps_confidence_and_preserves_raw_output() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = json.dumps(
        {
            "label": "Composition",
            "rationale": "Requirement A contains Requirement B.",
            "verbal_confidence": "high",
        }
    )

    parsed = parse_proposer_output(raw_output, confidence_map)

    assert parsed["predicted_label"] == "Composition"
    assert parsed["confidence"] == 0.7
    assert parsed["parsed_output"]["verbal_confidence"] == "high"
    assert parsed["error"] is None


def test_parse_proposer_output_uses_first_json_object_when_extra_text_follows() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = "```json\n{\n  \"label\": \"Temporal/Causal\",\n  \"rationale\": \"Evidence\",\n  \"verbal_confidence\": \"medium\"\n}\n```\n\nNote: keep JSON in the format {label: value}."

    parsed = parse_proposer_output(raw_output, confidence_map)

    assert parsed["predicted_label"] == "Temporal/Causal"
    assert parsed["confidence"] == 0.5
    assert parsed["error"] is None


def test_parse_proposer_output_returns_invalid_state_on_malformed_json() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = "```json\n{\n  \"label\": \"Temporal/Causal\",\n  \"rationale\": \"broken\"\n"

    parsed = parse_proposer_output(raw_output, confidence_map)

    assert parsed["predicted_label"] is None
    assert parsed["confidence"] is None
    assert parsed["error"] is not None


def test_require_valid_parsed_output_raises_on_parse_error() -> None:
    parsed = {
        "predicted_label": None,
        "confidence": None,
        "parsed_output": None,
        "error": "JSONDecodeError(...)",
        "raw_output": "bad output",
    }

    try:
        require_valid_parsed_output("pair-1", parsed)
    except ValueError as exc:
        assert "pair-1" in str(exc)
        assert "JSONDecodeError" in str(exc)
    else:
        raise AssertionError("expected parse failure to raise")


def test_select_inference_device_prefers_current_runtime_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        lora_trainer,
        "detect_torch_backend",
        lambda: {"selected_backend": "cpu"},
    )

    device = select_inference_device("mps")

    assert device == "cpu"


def test_resolve_scoped_path_rejects_escape_outside_allowed_roots() -> None:
    try:
        resolve_scoped_path("../outside.txt", allowed_roots=("outputs",))
    except ValueError as exc:
        assert "allowed roots" in str(exc)
    else:
        raise AssertionError("expected path escape to be rejected")


def test_merge_prompt_and_target_ids_masks_prompt_tokens() -> None:
    merged = merge_prompt_and_target_ids([10, 11], [20, 21, 22], max_length=8, pad_token_id=0)

    assert merged["input_ids"] == [10, 11, 20, 21, 22, 0, 0, 0]
    assert merged["attention_mask"] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert merged["labels"] == [-100, -100, 20, 21, 22, -100, -100, -100]


def test_merge_prompt_and_target_ids_preserves_target_when_prompt_is_too_long() -> None:
    merged = merge_prompt_and_target_ids([1, 2, 3, 4, 5], [8, 9], max_length=4, pad_token_id=0)

    assert merged["input_ids"] == [1, 2, 8, 9]
    assert merged["labels"] == [-100, -100, 8, 9]
    assert merged["attention_mask"] == [1, 1, 1, 1]


def test_build_proposer_training_examples_uses_rationales_when_provided() -> None:
    rows = [make_row("pair-1", "Temporal/Causal", "train")]
    prompt_template = "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}"
    rationales = {"pair-1": "Custom rationale for training"}

    examples = build_proposer_training_examples(rows, prompt_template, rationales_by_pair_id=rationales)

    target_payload = json.loads(examples[0]["target_text"])
    assert target_payload["rationale"] == "Custom rationale for training"
    assert target_payload["_provenance"] == "generated_rationale"
    assert examples[0]["rationale_provenance"] == "generated_rationale"


def test_build_proposer_training_examples_falls_back_to_bootstrap_when_rationale_missing() -> None:
    rows = [make_row("pair-1", "Temporal/Causal", "train")]
    prompt_template = "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}"
    rationales = {"pair-other": "Unused"}

    examples = build_proposer_training_examples(rows, prompt_template, rationales_by_pair_id=rationales)

    target_payload = json.loads(examples[0]["target_text"])
    assert target_payload["_provenance"] == "bootstrap_rationale"
    assert examples[0]["rationale_provenance"] == "bootstrap_rationale"


def test_build_judge_example_from_real_transcript_tags_transcript_source() -> None:
    row = make_row("pair-1", "Temporal/Causal", "train")
    prompt = "Judge prompt {{issue_i_text}} {{issue_j_text}} {{debate_transcript_json}}\n\nJSON Output:\n"

    example = build_judge_training_example_from_real_transcript(
        row,
        {"label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"},
        {"stance": "endorse", "label": "Temporal/Causal", "rationale": "r", "verbal_confidence": "high"},
        prompt,
    )

    assert example["transcript_source"] == "real_model_inference"
    assert example["rationale_provenance"] == "real_transcript"
    target_payload = json.loads(example["target_text"])
    assert target_payload["_provenance"] == "real_transcript"


def test_build_judge_training_examples_uses_rationale_as_justification() -> None:
    rows = [make_row("pair-1", "Temporal/Causal", "train")]
    prompt = "Judge prompt {{issue_i_text}} {{issue_j_text}} {{debate_transcript_json}}\n\nJSON Output:\n"

    examples = build_judge_training_examples(
        rows,
        prompt,
        rationales_by_pair_id={"pair-1": "Evidence-specific judge justification"},
    )

    target_payload = json.loads(examples[0]["target_text"])
    assert target_payload["justification"] == "Evidence-specific judge justification"
    assert target_payload["_provenance"] == "generated_rationale"
