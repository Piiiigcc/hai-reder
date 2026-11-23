from __future__ import annotations

from typing import Any, Callable


def run_arguments(
    row: dict[str, Any],
    proposer_step: Callable[[dict[str, Any], int, dict[str, Any]], dict[str, Any]],
    critic_step: Callable[[dict[str, Any], dict[str, Any], int, dict[str, Any]], dict[str, Any]],
    max_rounds: int = 3,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    transcript: dict[str, Any] = {
        "pair_id": row["pair_id"],
        "repo": row["repo"],
        "window": row["window"],
        "split": row["split"],
        "rounds": [],
    }
    termination_reason = "round_budget"
    for round_index in range(max_rounds):
        proposer_turn = proposer_step(row, round_index, transcript)
        critic_turn = critic_step(row, proposer_turn, round_index, transcript)
        transcript["rounds"].append(
            {
                "round_index": round_index,
                "proposer": proposer_turn,
                "critic": critic_turn,
            }
        )
        critic_parsed = critic_turn.get("parsed_output") or {}
        proposer_parsed = proposer_turn.get("parsed_output") or {}
        critic_stance = critic_parsed.get("stance")
        critic_label = critic_parsed.get("label")
        proposer_label = proposer_parsed.get("label")
        if critic_label and critic_label == proposer_label:
            termination_reason = "critic_concede" if round_index and critic_stance == "concede" else "agreement"
            break
        if critic_stance in {"endorse", "concede"}:
            transcript["rounds"][-1]["critic"]["parsed_output"]["edge_case"] = "endorse_mismatch"
    transcript["termination_reason"] = termination_reason
    return transcript


def run_debate(row, proposer_step, critic_step, judge_step, max_rounds=3):
    transcript = run_arguments(row, proposer_step, critic_step, max_rounds)
    transcript["judge"] = judge_step(row, transcript)
    return transcript
