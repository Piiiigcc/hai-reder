#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-agent debate inference.")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/minimal_lora.yaml")
    parser.add_argument("--input", default="data/processed/splits/minimal_validation.jsonl")
    parser.add_argument("--proposer-prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--critic-prompt", default="prompts/agents/critic_prompt.md")
    parser.add_argument("--judge-prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--proposer-adapter", default="models/adapters/proposer")
    parser.add_argument("--critic-adapter", default="models/adapters/critic")
    parser.add_argument("--judge-adapter", default="models/adapters/judge")
    parser.add_argument("--predictions-out", default="outputs/predictions/minimal_3agent_validation.jsonl")
    parser.add_argument("--transcripts-out", default="outputs/debate_transcripts/minimal_validation.jsonl")
    parser.add_argument("--metadata-out", default="outputs/metrics/minimal_3agent_validation_inference.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Randomly sample N rows from input (for speed-up)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Override max_rounds from config")
    parser.add_argument("--pool-path", type=str, default=None,
                        help="Path to example pool for RAG retrieval")
    parser.add_argument("--retrieval-k", type=int, default=3,
                        help="Number of examples to retrieve per role")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--diagnostic-fallback", action="store_true", help="Retain explicitly marked invalid generations for debugging")
    return parser.parse_args()


def _proposer_round_prompt(
    row: dict[str, Any],
    prompt_template: str,
    round_index: int,
    transcript: dict[str, Any],
    retrieval_examples: str = "",
) -> str:
    from haireder.agents.proposer import render_proposer_prompt

    prompt = render_proposer_prompt(row, prompt_template, retrieval_examples=retrieval_examples)
    if round_index == 0 or not transcript["rounds"]:
        return prompt
    last_critic = transcript["rounds"][-1]["critic"]["parsed_output"]
    return (
        prompt.rstrip() +
        "\nPrevious Critic Feedback:\n```json\n" +
        json.dumps(last_critic, ensure_ascii=False) +
        "\n```\n\nRevise your answer if needed.\n\nJSON Output:\n"
    )


def _generate_valid_turn(
    *,
    pair_id: str,
    bundle: Any,
    prompt_text: str,
    max_new_tokens: int,
    confidence_map: dict[str, float],
    parse_fn,
    generate_output_fn,
    validate_fn,
    fallback_label_key: str,
    diagnostic_fallback: bool = False,
) -> dict[str, Any]:
    token_budgets = [max_new_tokens, max_new_tokens * 2]
    last_raw_output = ""
    validation_errors: list[str] = []
    for token_budget in token_budgets:
        last_raw_output = generate_output_fn(bundle, prompt_text, max_new_tokens=token_budget)
        parsed = parse_fn(last_raw_output, confidence_map)
        try:
            parsed = validate_fn(pair_id, parsed)
            return {"raw_output": last_raw_output, "parsed_output": parsed["parsed_output"]}
        except ValueError as exc:
            validation_errors.append(f"max_new_tokens={token_budget}: {exc}")
            continue
    if not diagnostic_fallback:
        raise ValueError(f"{pair_id}: invalid role generation: {validation_errors}")
    return {
        "raw_output": last_raw_output,
        "parsed_output": {
            fallback_label_key: None,
            "final_label": None,
            "rationale": "",
            "verbal_confidence": "very_low",
            "is_fallback_generation": True,
            "fallback_reason": "all_parse_attempts_exhausted",
            "fallback_errors": validation_errors,
        },
    }


def _generate_judge_turn(
    *,
    pair_id: str,
    bundle: Any,
    prompt_text: str,
    max_new_tokens: int,
    confidence_map: dict[str, float],
    transcript: dict[str, Any],
    parse_fn,
    generate_output_fn,
    validate_fn,
    fallback_fn,
    diagnostic_fallback: bool = False,
) -> dict[str, Any]:
    token_budgets = [max_new_tokens, max_new_tokens * 2]
    last_raw_output = ""
    validation_errors: list[str] = []
    for token_budget in token_budgets:
        last_raw_output = generate_output_fn(bundle, prompt_text, max_new_tokens=token_budget)
        parsed = parse_fn(last_raw_output, confidence_map)
        try:
            parsed = validate_fn(pair_id, parsed)
            return {"raw_output": last_raw_output, "parsed_output": parsed["parsed_output"]}
        except ValueError as exc:
            validation_errors.append(f"max_new_tokens={token_budget}: {exc}")
            continue
    if not diagnostic_fallback:
        raise ValueError(f"{pair_id}: invalid Judge generation: {validation_errors}")
    fallback_output = fallback_fn(transcript)
    fallback_output["repair_errors"] = validation_errors
    fallback_output.setdefault("label", fallback_output.get("final_label"))
    return {"raw_output": last_raw_output, "parsed_output": fallback_output}


def main() -> int:
    from haireder.agents.critic import parse_critic_output, render_critic_prompt
    from haireder.agents.judge import (
        build_fallback_judge_decision,
        compact_transcript_for_judge,
        parse_judge_output,
        render_judge_prompt,
    )
    from haireder.agents.proposer import parse_proposer_output
    from haireder.debate import run_debate
    from haireder.training import load_prompt_template
    from haireder.training.lora_trainer import generate_role_output, load_multi_role_bundle, require_valid_parsed_output, select_role_bundle
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    model_config = load_yaml(resolve_path(args.model_config))
    rows = read_jsonl(resolve_path(args.input))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if args.sample_rows is not None and len(rows) > args.sample_rows:
        import random
        random.seed(args.seed)
        rows = random.sample(rows, args.sample_rows)
        print(f"Sampled {len(rows)} rows from {args.input} (seed={args.seed})")

    proposer_prompt = load_prompt_template(resolve_path(args.proposer_prompt))
    critic_prompt = load_prompt_template(resolve_path(args.critic_prompt))
    judge_prompt = load_prompt_template(resolve_path(args.judge_prompt))
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]
    max_rounds = args.max_rounds if args.max_rounds is not None else int(experiment_config["debate"]["max_rounds"])

    # Load pool and SBERT for RAG retrieval
    pool = []
    sbert_model = None
    if args.pool_path:
        from haireder.utils.io import read_jsonl as _read_jsonl
        pool_path = resolve_path(args.pool_path)
        if pool_path.exists():
            pool = _read_jsonl(pool_path)
            print(f"Loaded pool: {len(pool)} examples from {pool_path}")
            # Load SBERT model for similarity retrieval
            sbert_config = experiment_config.get("confidence", {}).get("sbert", {})
            sbert_model_name = sbert_config.get("model_name", "models/calibration/all-MiniLM-L6-v2")
            try:
                from sentence_transformers import SentenceTransformer
                sbert_model = SentenceTransformer(str(resolve_path(sbert_model_name)))
                print(f"Loaded SBERT model: {sbert_model_name}")
            except Exception as exc:
                raise RuntimeError(
                    f"SBERT model is required for --pool-path retrieval but could not be loaded: {sbert_model_name}"
                ) from exc

    predictions_out = resolve_scoped_path(args.predictions_out, allowed_roots=("outputs",))
    transcripts_out = resolve_scoped_path(args.transcripts_out, allowed_roots=("outputs",))
    metadata_out = resolve_scoped_path(args.metadata_out, allowed_roots=("outputs",))
    existing_predictions = [] if args.force_rerun or not predictions_out.exists() else read_jsonl(predictions_out)
    existing_transcripts = [] if args.force_rerun or not transcripts_out.exists() else read_jsonl(transcripts_out)
    existing_prediction_by_pair_id = {str(row.get("pair_id")): row for row in existing_predictions}
    existing_transcript_by_pair_id = {str(row.get("pair_id")): row for row in existing_transcripts}
    completed_pair_ids = set(existing_prediction_by_pair_id) & set(existing_transcript_by_pair_id)
    pending_rows = [row for row in rows if str(row["pair_id"]) not in completed_pair_ids]

    proposer_bundle = critic_bundle = judge_bundle = None
    if pending_rows:
        shared_bundle = load_multi_role_bundle(
            {
                "proposer": resolve_scoped_path(args.proposer_adapter, allowed_roots=("models",)),
                "critic": resolve_scoped_path(args.critic_adapter, allowed_roots=("models",)),
                "judge": resolve_scoped_path(args.judge_adapter, allowed_roots=("models",)),
            },
            model_config=model_config,
        )
        proposer_bundle = select_role_bundle(shared_bundle, "proposer")
        critic_bundle = select_role_bundle(shared_bundle, "critic")
        judge_bundle = select_role_bundle(shared_bundle, "judge")

    new_transcripts = []
    new_predictions = []
    total = len(pending_rows)

    for idx, row in enumerate(pending_rows):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx+1}/{total}] Processing {row['pair_id']}...", flush=True)

        # RAG: retrieve similar examples for each role
        proposer_examples = ""
        critic_examples = ""
        judge_examples = ""
        if pool:
            from haireder.online_learning.retrieval import retrieve_similar_examples, format_examples_for_role
            k = args.retrieval_k
            proposer_exs = retrieve_similar_examples(row, pool, "proposer", k=k, sbert_model=sbert_model)
            critic_exs = retrieve_similar_examples(row, pool, "critic", k=k, sbert_model=sbert_model)
            judge_exs = retrieve_similar_examples(row, pool, "judge", k=k, sbert_model=sbert_model)
            proposer_examples = format_examples_for_role(proposer_exs, "proposer")
            critic_examples = format_examples_for_role(critic_exs, "critic")
            judge_examples = format_examples_for_role(judge_exs, "judge")

        def proposer_step(current_row: dict[str, Any], round_index: int, transcript: dict[str, Any]) -> dict[str, Any]:
            prompt_text = _proposer_round_prompt(current_row, proposer_prompt, round_index, transcript, retrieval_examples=proposer_examples)
            return _generate_valid_turn(
                pair_id=current_row["pair_id"],
                bundle=proposer_bundle,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                confidence_map=confidence_map,
                parse_fn=parse_proposer_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
                fallback_label_key="label",
                diagnostic_fallback=args.diagnostic_fallback,
            )

        def critic_step(current_row: dict[str, Any], proposer_turn: dict[str, Any], round_index: int, transcript: dict[str, Any]) -> dict[str, Any]:
            proposer_output_json = json.dumps(proposer_turn["parsed_output"], ensure_ascii=False)
            prompt_text = render_critic_prompt(current_row, proposer_output_json, critic_prompt, retrieval_examples=critic_examples)
            return _generate_valid_turn(
                pair_id=current_row["pair_id"],
                bundle=critic_bundle,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                confidence_map=confidence_map,
                parse_fn=parse_critic_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
                fallback_label_key="label",
                diagnostic_fallback=args.diagnostic_fallback,
            )

        def judge_step(current_row: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
            compact = compact_transcript_for_judge(transcript)
            prompt_text = render_judge_prompt(current_row, compact, judge_prompt, retrieval_examples=judge_examples)
            return _generate_judge_turn(
                pair_id=current_row["pair_id"],
                bundle=judge_bundle,
                prompt_text=prompt_text,
                max_new_tokens=args.max_new_tokens,
                confidence_map=confidence_map,
                transcript=transcript,
                parse_fn=parse_judge_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
                fallback_fn=build_fallback_judge_decision,
                diagnostic_fallback=args.diagnostic_fallback,
            )

        transcript = run_debate(row, proposer_step, critic_step, judge_step, max_rounds=max_rounds)
        new_transcripts.append(transcript)
        judge_parsed = transcript["judge"]["parsed_output"]
        confidence = float(confidence_map[judge_parsed["verbal_confidence"]])
        new_predictions.append(
            {
                "pair_id": row["pair_id"],
                "repo": row["repo"],
                "window": row["window"],
                "split": row["split"],
                "gold_label": row["label"],
                "predicted_label": judge_parsed["final_label"],
                "confidence": confidence,
                "method": "3agent_debate",
                "issue_i": row["issue_i"],
                "issue_j": row["issue_j"],
                "raw_output": transcript["judge"]["raw_output"],
                "parsed_output": judge_parsed,
                "error": None,
            }
        )

    prediction_by_pair_id = {
        **existing_prediction_by_pair_id,
        **{str(row["pair_id"]): row for row in new_predictions},
    }
    transcript_by_pair_id = {
        **existing_transcript_by_pair_id,
        **{str(row["pair_id"]): row for row in new_transcripts},
    }
    requested_pair_ids = {str(row["pair_id"]) for row in rows}
    predictions = [
        prediction_by_pair_id[str(row["pair_id"])]
        for row in rows
        if str(row["pair_id"]) in prediction_by_pair_id and str(row["pair_id"]) in requested_pair_ids
    ]
    transcripts = [
        transcript_by_pair_id[str(row["pair_id"])]
        for row in rows
        if str(row["pair_id"]) in transcript_by_pair_id and str(row["pair_id"]) in requested_pair_ids
    ]
    write_jsonl(predictions_out, predictions)
    write_jsonl(transcripts_out, transcripts)
    write_json(
        metadata_out,
        {
            "prediction_rows": len(predictions),
            "transcript_rows": len(transcripts),
            "skipped_existing_rows": len(completed_pair_ids & requested_pair_ids),
            "newly_processed_rows": len(new_predictions),
            "force_rerun": args.force_rerun,
            "max_rounds": max_rounds,
            "proposer_adapter": str(resolve_scoped_path(args.proposer_adapter, allowed_roots=("models",))),
            "critic_adapter": str(resolve_scoped_path(args.critic_adapter, allowed_roots=("models",))),
            "judge_adapter": str(resolve_scoped_path(args.judge_adapter, allowed_roots=("models",))),
        },
    )
    print(f"Predictions written: {args.predictions_out}")
    print(f"Transcripts written: {args.transcripts_out}")
    print(f"Metadata written: {args.metadata_out}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Skipped existing rows: {len(completed_pair_ids & requested_pair_ids)}")
    print(f"Newly processed rows: {len(new_predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
