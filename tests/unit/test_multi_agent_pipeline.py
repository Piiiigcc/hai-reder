import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

training_builder = importlib.import_module("haireder.training.data_builder")
critic_module = importlib.import_module("haireder.agents.critic")
judge_module = importlib.import_module("haireder.agents.judge")
debate_engine = importlib.import_module("haireder.debate.engine")

build_critic_training_examples = training_builder.build_critic_training_examples
build_judge_training_examples = training_builder.build_judge_training_examples
parse_critic_output = critic_module.parse_critic_output
parse_judge_output = judge_module.parse_judge_output
compact_transcript_for_judge = judge_module.compact_transcript_for_judge
build_fallback_judge_decision = judge_module.build_fallback_judge_decision
run_debate = debate_engine.run_debate


def make_row(pair_id: str, label: str, split: str = "train") -> dict:
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


def test_build_critic_training_examples_contains_endorse_and_challenge_cases() -> None:
    rows = [
        make_row("pair-1", "Temporal/Causal"),
        make_row("pair-2", "Composition"),
    ]
    prompt_template = "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}\n\nProposer output:\n{{proposer_output_json}}"

    examples = build_critic_training_examples(rows, prompt_template)

    assert len(examples) == 2
    stances = {json.loads(example["target_text"])["stance"] for example in examples}
    assert stances == {"endorse", "challenge"}


def test_build_judge_training_examples_contains_agreement_and_disagreement_cases() -> None:
    rows = [
        make_row("pair-1", "Temporal/Causal"),
        make_row("pair-2", "Composition"),
    ]
    prompt_template = "Requirement A:\n{{issue_i_text}}\n\nRequirement B:\n{{issue_j_text}}\n\nDebate transcript:\n{{debate_transcript_json}}"

    examples = build_judge_training_examples(rows, prompt_template)

    assert len(examples) == 2
    reasons = {example["termination_reason"] for example in examples}
    assert reasons == {"agreement", "round_budget"}
    assert '"raw_output"' not in examples[0]["debate_transcript_json"]


def test_compact_transcript_for_judge_strips_raw_outputs() -> None:
    transcript = {
        "pair_id": "pair-1",
        "repo": "Apache",
        "window": "W1",
        "split": "validation",
        "rounds": [
            {
                "round_index": 0,
                "proposer": {"raw_output": "raw proposer", "parsed_output": {"label": "Composition"}},
                "critic": {"raw_output": "raw critic", "parsed_output": {"stance": "challenge", "label": "Temporal/Causal"}},
            }
        ],
        "judge": {"raw_output": "raw judge", "parsed_output": {"final_label": "Temporal/Causal"}},
        "termination_reason": "round_budget",
    }

    compact_json = compact_transcript_for_judge(transcript)

    assert '"raw_output"' not in compact_json
    assert '"stance": "challenge"' in compact_json
    assert '"termination_reason": "round_budget"' in compact_json


def test_parse_critic_output_extracts_stance_and_confidence() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = json.dumps(
        {
            "stance": "challenge",
            "label": "Composition",
            "rationale": "The proposer missed a structural relation.",
            "verbal_confidence": "high",
        }
    )

    parsed = parse_critic_output(raw_output, confidence_map)

    assert parsed["parsed_output"]["stance"] == "challenge"
    assert parsed["predicted_label"] == "Composition"
    assert parsed["confidence"] == 0.7
    assert parsed["error"] is None


def test_parse_judge_output_extracts_final_label() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = json.dumps(
        {
            "final_label": "Workflow",
            "justification": "The critic's workflow argument is stronger.",
            "verbal_confidence": "medium",
        }
    )

    parsed = parse_judge_output(raw_output, confidence_map)

    assert parsed["predicted_label"] == "Workflow"
    assert parsed["confidence"] == 0.5
    assert parsed["error"] is None


def test_parse_judge_output_rejects_truncated_output_even_with_final_label() -> None:
    confidence_map = {
        "very_low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "very_high": 0.9,
    }
    raw_output = "```json\n{\n  \"final_label\": \"Composition\",\n  \"judge_prompt\": \"Temporal/Causal\",\n  \"judge_label\": \"Composition\""

    parsed = parse_judge_output(raw_output, confidence_map)

    assert parsed["predicted_label"] is None
    assert parsed["parsed_output"] is None
    assert parsed["confidence"] is None
    assert parsed["error"]


def test_build_fallback_judge_decision_prefers_last_critic_label() -> None:
    transcript = {
        "rounds": [
            {
                "proposer": {"parsed_output": {"label": "Temporal/Causal", "verbal_confidence": "medium"}},
                "critic": {"parsed_output": {"stance": "challenge", "label": "Composition", "verbal_confidence": "medium"}},
            }
        ]
    }

    parsed_output = build_fallback_judge_decision(transcript)

    assert parsed_output["final_label"] == "Composition"
    assert parsed_output["verbal_confidence"] == "very_low"
    assert parsed_output["repair_applied"] is True


def test_run_debate_records_rounds_and_termination() -> None:
    row = make_row("pair-1", "Temporal/Causal", split="validation")

    proposer_outputs = [
        {
            "raw_output": "p0",
            "parsed_output": {"label": "Composition", "rationale": "r0", "verbal_confidence": "medium"},
        },
        {
            "raw_output": "p1",
            "parsed_output": {"label": "Temporal/Causal", "rationale": "r1", "verbal_confidence": "high"},
        },
    ]
    critic_outputs = [
        {
            "raw_output": "c0",
            "parsed_output": {"stance": "challenge", "label": "Temporal/Causal", "rationale": "fix", "verbal_confidence": "high"},
        },
        {
            "raw_output": "c1",
            "parsed_output": {"stance": "endorse", "label": "Temporal/Causal", "rationale": "agree", "verbal_confidence": "high"},
        },
    ]

    def proposer_step(current_row: dict, round_index: int, transcript: dict) -> dict:
        return proposer_outputs[round_index]

    def critic_step(current_row: dict, proposer_turn: dict, round_index: int, transcript: dict) -> dict:
        return critic_outputs[round_index]

    def judge_step(current_row: dict, transcript: dict) -> dict:
        return {
            "raw_output": "j0",
            "parsed_output": {"final_label": "Temporal/Causal", "justification": transcript["termination_reason"], "verbal_confidence": "high"},
        }

    transcript = run_debate(row, proposer_step, critic_step, judge_step, max_rounds=3)

    assert transcript["termination_reason"] == "agreement"
    assert len(transcript["rounds"]) == 2
    assert transcript["judge"]["parsed_output"]["final_label"] == "Temporal/Causal"
    assert transcript["judge"]["parsed_output"]["justification"] == "agreement"


def test_run_debate_stops_on_same_committed_label() -> None:
    row = make_row("pair-2", "Composition", split="validation")

    def proposer_step(current_row: dict, round_index: int, transcript: dict) -> dict:
        return {
            "raw_output": "p0",
            "parsed_output": {"label": "Composition", "rationale": "r0", "verbal_confidence": "medium"},
        }

    def critic_step(current_row: dict, proposer_turn: dict, round_index: int, transcript: dict) -> dict:
        return {
            "raw_output": "c0",
            "parsed_output": {"stance": "challenge", "label": "Composition", "rationale": "contradiction", "verbal_confidence": "high"},
        }

    def judge_step(current_row: dict, transcript: dict) -> dict:
        return {
            "raw_output": "j0",
            "parsed_output": {"final_label": "Composition", "justification": transcript["termination_reason"], "verbal_confidence": "high"},
        }

    transcript = run_debate(row, proposer_step, critic_step, judge_step, max_rounds=2)

    assert transcript["termination_reason"] == "agreement"
    assert len(transcript["rounds"]) == 1
    assert transcript["judge"]["parsed_output"]["justification"] == "agreement"
