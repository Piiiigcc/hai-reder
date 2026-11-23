#!/usr/bin/env python
"""Run the legacy Phase 15 held-out diagnostics. Use run_formal_pipeline.py for paper results."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/minimal_lora.yaml")
    parser.add_argument("--train", default="data/processed/splits/minimal_train.jsonl")
    parser.add_argument("--online", default="data/processed/splits/minimal_online.jsonl")
    parser.add_argument("--test", default="data/processed/splits/minimal_test.jsonl")
    parser.add_argument("--calibrator-in", default="models/calibration/minimal_platt_scaler.pkl")
    parser.add_argument("--allow-missing-calibrator", action="store_true")
    parser.add_argument("--calibration-summary", default="outputs/metrics/minimal_confidence_calibration_summary.json")
    parser.add_argument(
        "--triage-summaries",
        nargs="+",
        default=["outputs/metrics/minimal_online_W4_triage_summary.json"],
    )
    parser.add_argument("--predictions-out", default="outputs/predictions/minimal_test.jsonl")
    parser.add_argument("--transcripts-out", default="outputs/debate_transcripts/minimal_test.jsonl")
    parser.add_argument("--metrics-out", default="outputs/metrics/minimal_metrics.json")
    parser.add_argument("--table-out", default="outputs/tables/minimal_results.md")
    parser.add_argument("--auxiliary-predictions", default=None)
    parser.add_argument("--auxiliary-none-threshold", type=float, default=0.4)
    parser.add_argument("--proposer-prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--critic-prompt", default="prompts/agents/critic_prompt.md")
    parser.add_argument("--judge-prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--proposer-adapter", default="models/adapters/proposer")
    parser.add_argument("--critic-adapter", default="models/adapters/critic")
    parser.add_argument("--judge-adapter", default="models/adapters/judge")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Randomly sample N rows from test set (for speed-up)")
    parser.add_argument("--pool-path", type=str, default=None,
                        help="Path to example pool for RAG retrieval")
    parser.add_argument("--retrieval-k", type=int, default=3,
                        help="Number of examples to retrieve per role")
    parser.add_argument("--evaluation-mode", choices=["full", "adaptive"], default="full",
                        help="Evaluation strategy: full 3-agent debate or adaptive batched judge-first cascade")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Same-role generation batch size for adaptive evaluation")
    parser.add_argument("--adaptive-high-threshold", type=float, default=0.8,
                        help="Judge confidence threshold for accepting adaptive judge-only predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing-inference", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_phase15_fusion_weights(experiment_config: dict[str, Any]) -> dict[str, float]:
    confidence_config = experiment_config["confidence"]
    fusion_weights = {
        str(label): float(value)
        for label, value in confidence_config["fusion_weights"].items()
    }

    if not bool(confidence_config.get("weight_search_enabled", False)):
        return fusion_weights

    weight_search_config = confidence_config.get("weight_search", {})
    output_path = weight_search_config.get("output_path")
    if not output_path:
        print("[weight_search] No output_path configured, using config defaults", file=sys.stderr)
        return fusion_weights

    weight_file = _resolve(output_path)
    if not weight_file.exists():
        print(f"[weight_search] No pre-computed weights found at {weight_file}, using config defaults", file=sys.stderr)
        return fusion_weights

    try:
        loaded_weights = json.loads(weight_file.read_text(encoding="utf-8"))
        resolved_weights = {
            "self_evaluation": float(loaded_weights.get("self_evaluation", fusion_weights["self_evaluation"])),
            "inter_agent_agreement": float(
                loaded_weights.get("inter_agent_agreement", fusion_weights["inter_agent_agreement"])
            ),
            "retrieval_similarity": float(
                loaded_weights.get("retrieval_similarity", fusion_weights["retrieval_similarity"])
            ),
        }
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        print(
            f"[weight_search] Failed to load pre-computed weights from {weight_file} ({exc}), using config defaults",
            file=sys.stderr,
        )
        return fusion_weights

    print(f"[weight_search] Loaded pre-computed weights from {weight_file}: {resolved_weights}", file=sys.stderr)
    return resolved_weights


def _pair_text(row: dict[str, Any]) -> str:
    issue_i = row.get("issue_i") or {}
    issue_j = row.get("issue_j") or {}
    parts = [str(issue_i.get("text") or "").strip(), str(issue_j.get("text") or "").strip()]
    return "\n\n".join(part for part in parts if part)


def _load_bootstrap_text_pool(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [text for row in rows if (text := _pair_text(row))]


def _index_transcripts_by_pair_id(transcripts: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for transcript in transcripts:
        pair_id = str(transcript.get("pair_id") or "")
        if not pair_id:
            raise ValueError("transcript missing pair_id")
        if pair_id in indexed:
            raise ValueError(f"duplicate transcript pair_id: {pair_id}")
        indexed[pair_id] = transcript
    return indexed


def _index_rows_by_pair_id(rows: Sequence[dict[str, Any]], *, row_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError(f"{row_name} missing pair_id")
        if pair_id in indexed:
            raise ValueError(f"duplicate {row_name} pair_id: {pair_id}")
        indexed[pair_id] = row
    return indexed


def _apply_auxiliary_none_prior(
    row: dict[str, Any],
    *,
    auxiliary_by_pair_id: dict[str, dict[str, Any]],
    none_threshold: float,
) -> dict[str, Any]:
    auxiliary = auxiliary_by_pair_id.get(str(row.get("pair_id")))
    if auxiliary is None:
        return row
    probabilities = auxiliary.get("probabilities") or {}
    none_prob_raw = probabilities.get("None")
    none_probability = float(none_prob_raw if none_prob_raw is not None else auxiliary.get("confidence", 0.0))
    if str(auxiliary.get("predicted_label")) != "None" or none_probability < none_threshold:
        return row

    fused = dict(row)
    fused["predicted_label"] = "None"
    fused["method"] = "3agent_debate+aux_none_prior"
    fused["auxiliary_override"] = {
        "source_method": auxiliary.get("method"),
        "from_label": row.get("predicted_label"),
        "to_label": "None",
        "none_probability": none_probability,
        "threshold": none_threshold,
    }
    return fused


def _tfidf_baseline_hint(auxiliary_by_pair_id: dict[str, dict[str, Any]], row: dict[str, Any]) -> str:
    """Return an optional TF-IDF baseline hint string for proposer/judge prompts.

    This is an auxiliary external signal, NOT part of the core three-agent debate
    logic. When a hint is present, the prediction output carries
    provenance.tfidf_hint_used=True.
    """
    tfidf_row = auxiliary_by_pair_id.get(str(row.get("pair_id")))
    tfidf_hint = ""
    if tfidf_row:
        tfidf_hint = f"Hint: a simple classifier predicts {tfidf_row['predicted_label']}."
    return tfidf_hint


def _proposer_round_prompt(
    row: dict[str, Any],
    prompt_template: str,
    round_index: int,
    transcript: dict[str, Any],
    tfidf_baseline_hint: str = "",
    retrieval_examples: str = "",
) -> str:
    from haireder.agents.proposer import render_proposer_prompt

    prompt = render_proposer_prompt(row, prompt_template, tfidf_baseline_hint=tfidf_baseline_hint, retrieval_examples=retrieval_examples)
    if round_index == 0 or not transcript["rounds"]:
        return prompt
    last_critic = transcript["rounds"][-1]["critic"]["parsed_output"]
    return (
        prompt.rstrip() +
        "\nPrevious Critic Feedback:\n```json\n" +
        json.dumps(last_critic, ensure_ascii=False) +
        "\n```\n\nRevise your answer if needed.\n\nJSON Output:\n"
    )


def _run_test_debate_inference(
    *,
    test_rows: list[dict[str, Any]],
    auxiliary_by_pair_id: dict[str, dict[str, Any]],
    experiment_config_path: Path,
    predictions_out_path: Path,
    transcripts_out_path: Path,
    proposer_prompt_path: Path,
    critic_prompt_path: Path,
    judge_prompt_path: Path,
    proposer_adapter_path: Path,
    critic_adapter_path: Path,
    judge_adapter_path: Path,
    max_new_tokens: int,
    model_config_path: Path | None = None,
    force_rerun: bool = False,
    pool: list[dict[str, Any]] | None = None,
    sbert_model: Any = None,
    retrieval_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    from haireder.training.lora_trainer import (
        generate_role_output,
        load_multi_role_bundle,
        require_valid_parsed_output,
        select_role_bundle,
    )
    from haireder.utils.config import load_yaml
    from haireder.utils.io import read_jsonl, write_jsonl

    experiment_config = load_yaml(experiment_config_path)
    model_config = load_yaml(model_config_path) if model_config_path is not None else None
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]
    max_rounds = int(experiment_config["debate"]["max_rounds"])

    proposer_prompt = load_prompt_template(proposer_prompt_path)
    critic_prompt = load_prompt_template(critic_prompt_path)
    judge_prompt = load_prompt_template(judge_prompt_path)

    existing_predictions = [] if force_rerun or not predictions_out_path.exists() else read_jsonl(predictions_out_path)
    existing_transcripts = [] if force_rerun or not transcripts_out_path.exists() else read_jsonl(transcripts_out_path)
    existing_prediction_by_pair_id = {str(row.get("pair_id")): row for row in existing_predictions}
    existing_transcript_by_pair_id = {str(row.get("pair_id")): row for row in existing_transcripts}
    completed_pair_ids = set(existing_prediction_by_pair_id) & set(existing_transcript_by_pair_id)
    pending_rows = [row for row in test_rows if str(row["pair_id"]) not in completed_pair_ids]

    proposer_bundle = critic_bundle = judge_bundle = None
    if pending_rows:
        shared_bundle = load_multi_role_bundle(
            {
                "proposer": proposer_adapter_path,
                "critic": critic_adapter_path,
                "judge": judge_adapter_path,
            },
            model_config=model_config,
        )
        proposer_bundle = select_role_bundle(shared_bundle, "proposer")
        critic_bundle = select_role_bundle(shared_bundle, "critic")
        judge_bundle = select_role_bundle(shared_bundle, "judge")

    new_transcripts: list[dict[str, Any]] = []
    new_predictions: list[dict[str, Any]] = []
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
            k = retrieval_k
            proposer_exs = retrieve_similar_examples(row, pool, "proposer", k=k, sbert_model=sbert_model)
            critic_exs = retrieve_similar_examples(row, pool, "critic", k=k, sbert_model=sbert_model)
            judge_exs = retrieve_similar_examples(row, pool, "judge", k=k, sbert_model=sbert_model)
            proposer_examples = format_examples_for_role(proposer_exs, "proposer")
            critic_examples = format_examples_for_role(critic_exs, "critic")
            judge_examples = format_examples_for_role(judge_exs, "judge")

        def proposer_step(current_row: dict[str, Any], round_index: int, transcript: dict[str, Any]) -> dict[str, Any]:
            tfidf_hint = _tfidf_baseline_hint(auxiliary_by_pair_id, current_row)
            prompt_text = _proposer_round_prompt(
                current_row,
                proposer_prompt,
                round_index,
                transcript,
                tfidf_baseline_hint=tfidf_hint,
                retrieval_examples=proposer_examples,
            )
            return _generate_valid_turn(
                pair_id=current_row["pair_id"],
                bundle=proposer_bundle,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                confidence_map=confidence_map,
                parse_fn=parse_proposer_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
            )

        def critic_step(current_row: dict[str, Any], proposer_turn: dict[str, Any], round_index: int, transcript: dict[str, Any]) -> dict[str, Any]:
            proposer_output_json = json.dumps(proposer_turn["parsed_output"], ensure_ascii=False)
            prompt_text = render_critic_prompt(current_row, proposer_output_json, critic_prompt, retrieval_examples=critic_examples)
            return _generate_valid_turn(
                pair_id=current_row["pair_id"],
                bundle=critic_bundle,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                confidence_map=confidence_map,
                parse_fn=parse_critic_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
            )

        def judge_step(current_row: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
            compact = compact_transcript_for_judge(transcript)
            tfidf_hint = _tfidf_baseline_hint(auxiliary_by_pair_id, current_row)
            prompt_text = render_judge_prompt(
                current_row,
                compact,
                judge_prompt,
                tfidf_baseline_hint=tfidf_hint,
                retrieval_examples=judge_examples,
            )
            return _generate_judge_turn(
                pair_id=current_row["pair_id"],
                bundle=judge_bundle,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                confidence_map=confidence_map,
                transcript=transcript,
                parse_fn=parse_judge_output,
                generate_output_fn=generate_role_output,
                validate_fn=require_valid_parsed_output,
                fallback_fn=build_fallback_judge_decision,
            )

        transcript = run_debate(row, proposer_step, critic_step, judge_step, max_rounds=max_rounds)
        new_transcripts.append(transcript)
        judge_parsed = transcript["judge"]["parsed_output"]
        confidence = float(confidence_map[judge_parsed["verbal_confidence"]])

        proposer_is_fallback = any(
            round_data["proposer"]["parsed_output"].get("is_fallback_generation")
            for round_data in transcript["rounds"]
        )
        critic_is_fallback = any(
            round_data["critic"]["parsed_output"].get("is_fallback_generation")
            for round_data in transcript["rounds"]
        )
        judge_is_repaired = judge_parsed.get("repair_applied", False)
        row_tfidf_hint = _tfidf_baseline_hint(auxiliary_by_pair_id, row)
        tfidf_hint_used = bool(row_tfidf_hint)

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
                "provenance": {
                    "proposer_is_fallback": proposer_is_fallback,
                    "critic_is_fallback": critic_is_fallback,
                    "judge_is_repaired": judge_is_repaired,
                    "tfidf_hint_used": tfidf_hint_used,
                },
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
    predictions = [
        prediction_by_pair_id[str(row["pair_id"])]
        for row in test_rows
        if str(row["pair_id"]) in prediction_by_pair_id
    ]
    transcripts = [
        transcript_by_pair_id[str(row["pair_id"])]
        for row in test_rows
        if str(row["pair_id"]) in transcript_by_pair_id
    ]
    write_jsonl(predictions_out_path, predictions)
    write_jsonl(transcripts_out_path, transcripts)
    return predictions, transcripts


def _adaptive_candidate_transcript(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "pair_id": row["pair_id"],
            "repo": row["repo"],
            "window": row["window"],
            "split": row["split"],
            "rounds": [],
            "termination_reason": "adaptive_judge_candidate",
        },
        ensure_ascii=False,
    )


def _run_adaptive_debate_inference(
    *,
    test_rows: list[dict[str, Any]],
    auxiliary_by_pair_id: dict[str, dict[str, Any]],
    experiment_config_path: Path,
    predictions_out_path: Path,
    transcripts_out_path: Path,
    proposer_prompt_path: Path,
    critic_prompt_path: Path,
    judge_prompt_path: Path,
    proposer_adapter_path: Path,
    critic_adapter_path: Path,
    judge_adapter_path: Path,
    max_new_tokens: int,
    batch_size: int,
    adaptive_high_threshold: float,
    model_config_path: Path | None = None,
    force_rerun: bool = False,
    pool: list[dict[str, Any]] | None = None,
    sbert_model: Any = None,
    retrieval_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from haireder.agents.judge import parse_judge_output, render_judge_prompt
    from haireder.training import load_prompt_template
    from haireder.training.lora_trainer import generate_role_outputs, load_role_bundle, require_valid_parsed_output
    from haireder.utils.config import load_yaml
    from haireder.utils.io import read_jsonl, write_jsonl

    experiment_config = load_yaml(experiment_config_path)
    model_config = load_yaml(model_config_path) if model_config_path is not None else None
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]
    judge_prompt = load_prompt_template(judge_prompt_path)

    existing_predictions = [] if force_rerun or not predictions_out_path.exists() else read_jsonl(predictions_out_path)
    existing_transcripts = [] if force_rerun or not transcripts_out_path.exists() else read_jsonl(transcripts_out_path)
    existing_prediction_by_pair_id = {str(row.get("pair_id")): row for row in existing_predictions}
    existing_transcript_by_pair_id = {str(row.get("pair_id")): row for row in existing_transcripts}
    completed_pair_ids = set(existing_prediction_by_pair_id) & set(existing_transcript_by_pair_id)
    pending_rows = [row for row in test_rows if str(row["pair_id"]) not in completed_pair_ids]

    accepted_predictions: list[dict[str, Any]] = []
    accepted_transcripts: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    if pending_rows:
        if model_config is None:
            judge_bundle = load_role_bundle(judge_adapter_path)
        else:
            judge_bundle = load_role_bundle(judge_adapter_path, model_config=model_config)
        effective_batch_size = max(1, int(batch_size))
        for start in range(0, len(pending_rows), effective_batch_size):
            batch_rows = pending_rows[start:start + effective_batch_size]
            prompts: list[str] = []
            retrieval_by_pair_id: dict[str, str] = {}
            if pool:
                from haireder.online_learning.retrieval import format_examples_for_role, retrieve_similar_examples
            for row in batch_rows:
                judge_examples = ""
                if pool:
                    examples = retrieve_similar_examples(row, pool, "judge", k=retrieval_k, sbert_model=sbert_model)
                    judge_examples = format_examples_for_role(examples, "judge")
                retrieval_by_pair_id[str(row["pair_id"])] = judge_examples
                prompts.append(
                    render_judge_prompt(
                        row,
                        _adaptive_candidate_transcript(row),
                        judge_prompt,
                        tfidf_baseline_hint=_tfidf_baseline_hint(auxiliary_by_pair_id, row),
                        retrieval_examples=judge_examples,
                    )
                )

            raw_outputs = generate_role_outputs(
                judge_bundle,
                prompts,
                max_new_tokens=max_new_tokens,
                batch_size=effective_batch_size,
            )

            for row, raw_output in zip(batch_rows, raw_outputs, strict=True):
                parsed = parse_judge_output(raw_output, confidence_map)
                try:
                    parsed = require_valid_parsed_output(row["pair_id"], parsed)
                except ValueError:
                    fallback_rows.append(row)
                    continue
                judge_parsed = parsed["parsed_output"]
                confidence = float(confidence_map[judge_parsed["verbal_confidence"]])
                if confidence < adaptive_high_threshold:
                    fallback_rows.append(row)
                    continue

                judge_turn = {"raw_output": raw_output, "parsed_output": judge_parsed}
                transcript = {
                    "pair_id": row["pair_id"],
                    "repo": row["repo"],
                    "window": row["window"],
                    "split": row["split"],
                    "rounds": [],
                    "termination_reason": "adaptive_judge_only",
                    "judge": judge_turn,
                    "adaptive_metadata": {
                        "mode": "adaptive",
                        "stage": "judge_only",
                        "full_debate_used": False,
                        "batch_size": effective_batch_size,
                        "threshold": adaptive_high_threshold,
                    },
                }
                row_tfidf_hint = _tfidf_baseline_hint(auxiliary_by_pair_id, row)
                accepted_transcripts.append(transcript)
                accepted_predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "repo": row["repo"],
                        "window": row["window"],
                        "split": row["split"],
                        "gold_label": row["label"],
                        "predicted_label": judge_parsed["final_label"],
                        "confidence": confidence,
                        "method": "adaptive_judge",
                        "issue_i": row["issue_i"],
                        "issue_j": row["issue_j"],
                        "raw_output": raw_output,
                        "parsed_output": judge_parsed,
                        "error": None,
                        "provenance": {
                            "proposer_is_fallback": False,
                            "critic_is_fallback": False,
                            "judge_is_repaired": judge_parsed.get("repair_applied", False),
                            "tfidf_hint_used": bool(row_tfidf_hint),
                            "adaptive_mode": True,
                            "adaptive_stage": "judge_only",
                            "full_debate_used": False,
                            "batch_size": effective_batch_size,
                            "adaptive_high_threshold": adaptive_high_threshold,
                            "retrieval_examples_used": bool(retrieval_by_pair_id[str(row["pair_id"])]),
                        },
                    }
                )

    fallback_predictions: list[dict[str, Any]] = []
    fallback_transcripts: list[dict[str, Any]] = []
    if fallback_rows:
        if "judge_bundle" in locals():
            del judge_bundle
        fallback_predictions_out_path = predictions_out_path.with_name(
            f"{predictions_out_path.stem}.adaptive_fallback{predictions_out_path.suffix}"
        )
        fallback_transcripts_out_path = transcripts_out_path.with_name(
            f"{transcripts_out_path.stem}.adaptive_fallback{transcripts_out_path.suffix}"
        )
        fallback_predictions, fallback_transcripts = _run_test_debate_inference(
            test_rows=fallback_rows,
            auxiliary_by_pair_id=auxiliary_by_pair_id,
            experiment_config_path=experiment_config_path,
            predictions_out_path=fallback_predictions_out_path,
            transcripts_out_path=fallback_transcripts_out_path,
            proposer_prompt_path=proposer_prompt_path,
            critic_prompt_path=critic_prompt_path,
            judge_prompt_path=judge_prompt_path,
            proposer_adapter_path=proposer_adapter_path,
            critic_adapter_path=critic_adapter_path,
            judge_adapter_path=judge_adapter_path,
            model_config_path=model_config_path,
            max_new_tokens=max_new_tokens,
            force_rerun=True,
            pool=pool,
            sbert_model=sbert_model,
            retrieval_k=retrieval_k,
        )
        fallback_ids = {str(row["pair_id"]) for row in fallback_rows}
        fallback_predictions = [row for row in fallback_predictions if str(row.get("pair_id")) in fallback_ids]
        fallback_transcripts = [row for row in fallback_transcripts if str(row.get("pair_id")) in fallback_ids]
        for prediction in fallback_predictions:
            prediction["method"] = "adaptive_full_debate"
            provenance = dict(prediction.get("provenance") or {})
            provenance.update(
                {
                    "adaptive_mode": True,
                    "adaptive_stage": "full_debate",
                    "full_debate_used": True,
                    "adaptive_high_threshold": adaptive_high_threshold,
                }
            )
            prediction["provenance"] = provenance
        for transcript in fallback_transcripts:
            transcript["adaptive_metadata"] = {
                "mode": "adaptive",
                "stage": "full_debate",
                "full_debate_used": True,
                "threshold": adaptive_high_threshold,
            }

    prediction_by_pair_id = {
        **existing_prediction_by_pair_id,
        **{str(row["pair_id"]): row for row in accepted_predictions},
        **{str(row["pair_id"]): row for row in fallback_predictions},
    }
    transcript_by_pair_id = {
        **existing_transcript_by_pair_id,
        **{str(row["pair_id"]): row for row in accepted_transcripts},
        **{str(row["pair_id"]): row for row in fallback_transcripts},
    }
    predictions = [
        prediction_by_pair_id[str(row["pair_id"])]
        for row in test_rows
        if str(row["pair_id"]) in prediction_by_pair_id
    ]
    transcripts = [
        transcript_by_pair_id[str(row["pair_id"])]
        for row in test_rows
        if str(row["pair_id"]) in transcript_by_pair_id
    ]
    write_jsonl(predictions_out_path, predictions)
    write_jsonl(transcripts_out_path, transcripts)
    if fallback_rows:
        fallback_predictions_out_path.unlink(missing_ok=True)
        fallback_transcripts_out_path.unlink(missing_ok=True)
    return predictions, transcripts


def _compute_raw_confidence_without_inter_agent_agreement(
    *,
    s1: float,
    s3: float,
    weights: dict[str, float],
) -> float:
    active_weight = float(weights["self_evaluation"]) + float(weights["retrieval_similarity"])
    if active_weight <= 0.0:
        return 0.0
    confidence = (
        float(weights["self_evaluation"]) * s1
        + float(weights["retrieval_similarity"]) * s3
    ) / active_weight
    return min(1.0, max(0.0, confidence))


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
) -> dict[str, Any]:
    token_budgets = [max_new_tokens, max_new_tokens * 2]
    last_raw = ""
    validation_errors: list[str] = []
    for token_budget in token_budgets:
        last_raw = generate_output_fn(bundle, prompt_text, max_new_tokens=token_budget)
        parsed = parse_fn(last_raw, confidence_map)
        try:
            parsed = validate_fn(pair_id, parsed)
            return {"raw_output": last_raw, "parsed_output": parsed["parsed_output"]}
        except ValueError as exc:
            validation_errors.append(f"max_new_tokens={token_budget}: {exc}")
            continue
    return {
        "raw_output": last_raw,
        "parsed_output": {
            "label": None,
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
    fallback_output = fallback_fn(transcript)
    fallback_output["repair_errors"] = validation_errors
    fallback_output.setdefault("label", fallback_output.get("final_label"))
    return {"raw_output": last_raw_output, "parsed_output": fallback_output}


def _load_calibrator(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _fallback_calibrator(reason: str):
    from haireder.confidence.calibration import CalibrationModel

    return CalibrationModel(mode="fallback", reason=reason)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _select_phase15_calibrator(
    calibrator,
    *,
    calibration_summary: dict[str, Any] | None,
    min_rows: int = 30,
    min_confidence_regions: int = 2,
):
    from haireder.confidence.calibration import CalibrationModel

    if calibrator.mode != "platt":
        return calibrator, {
            "status": "not_platt_no_reliability_check",
            "mode": calibrator.mode,
            "reason": calibrator.reason,
        }

    if calibration_summary is None:
        return calibrator, {
            "status": "unchecked_missing_summary",
            "mode": calibrator.mode,
            "reason": calibrator.reason,
        }

    row_count = int(calibration_summary.get("row_count") or 0)
    confidence_regions = calibration_summary.get("confidence_regions") or {}
    confidence_region_count = len(confidence_regions)
    reliable = row_count >= min_rows and confidence_region_count >= min_confidence_regions
    if reliable:
        return calibrator, {
            "status": "trusted",
            "mode": calibrator.mode,
            "reason": calibrator.reason,
            "row_count": row_count,
            "confidence_region_count": confidence_region_count,
        }

    return CalibrationModel(mode="fallback", reason="unreliable_validation_calibrator"), {
        "status": "fallback_to_raw_confidence",
        "mode": calibrator.mode,
        "reason": calibrator.reason,
        "fallback_reason": "unreliable_validation_calibrator",
        "row_count": row_count,
        "min_rows": min_rows,
        "confidence_region_count": confidence_region_count,
        "min_confidence_regions": min_confidence_regions,
    }


def _aggregate_triage_summaries(paths: Sequence[Path]) -> dict[str, Any]:
    if not paths:
        return {"status":"unobserved","low_review_count":None,"medium_review_count":None,"total_review_count":None,"promoted_to_high_count":None,"total_rows":None,"triage_saving":None,"source_paths":[],"windows":[]}
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError(f"Missing requested triage summaries: {[str(p) for p in paths if not p.is_file()]}")
    measures = set()
    source_paths = [str(path.resolve()) for path in paths]
    low_review_count = 0
    medium_review_count = 0
    promoted_to_high_count = 0
    total_review_count = 0
    total_rows = 0
    windows: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        triage_payload = payload.get("triage") if isinstance(payload.get("triage"), dict) else payload
        required=("low_review_count","medium_review_count","promoted_to_high_count","total_review_count","total_rows")
        if any(type(triage_payload.get(k)) is not int or triage_payload[k]<0 for k in required):
            raise ValueError(f"Incomplete workload summary at {path}")
        measures.add(triage_payload.get("review_measurement","historical_unverified"))
        low_review_count += triage_payload["low_review_count"]
        medium_review_count += int(triage_payload.get("medium_review_count") or 0)
        promoted_to_high_count += int(triage_payload.get("promoted_to_high_count") or 0)
        total_review_count += int(triage_payload.get("total_review_count") or 0)
        total_rows += int(triage_payload.get("total_rows") or 0)
        windows.append(str(payload.get("window") or ""))
    initial_medium = medium_review_count + promoted_to_high_count
    triage_saving = promoted_to_high_count / initial_medium if initial_medium else None
    return {
        "review_measurement":sorted(measures),
        "low_review_count": low_review_count,
        "medium_review_count": medium_review_count,
        "promoted_to_high_count": promoted_to_high_count,
        "total_review_count": total_review_count,
        "total_rows": total_rows,
        "triage_saving": triage_saving,
        "windows": [window for window in windows if window],
        "source_paths": source_paths,
    }


def _confidence_region_accuracy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for region in ("low", "medium", "high"):
        region_rows = [row for row in rows if str(row.get("confidence_region")) == region]
        count = len(region_rows)
        accuracy = None
        if count:
            accuracy = sum(row["gold_label"] == row["predicted_label"] for row in region_rows) / count
        report[region] = {"count": count, "accuracy": accuracy}
    return report


def _prediction_health_report(
    rows: Sequence[dict[str, Any]],
    *,
    dominant_threshold: float = 0.9,
) -> dict[str, Any]:
    prediction_counts = Counter(str(row.get("predicted_label") or "") for row in rows if row.get("predicted_label") is not None)
    total = sum(prediction_counts.values())
    if total == 0:
        return {"status": "empty", "prediction_counts": {}, "dominant_label": None, "dominant_ratio": None}
    dominant_label, dominant_count = prediction_counts.most_common(1)[0]
    dominant_ratio = dominant_count / total
    status = "collapsed" if dominant_ratio >= dominant_threshold else "ok"
    return {
        "status": status,
        "prediction_counts": dict(prediction_counts),
        "dominant_label": dominant_label,
        "dominant_ratio": dominant_ratio,
    }


def _confidence_health_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    region_counts = Counter(str(row.get("confidence_region") or "") for row in rows if row.get("confidence_region") is not None)
    status = "collapsed_single_region" if len(region_counts) == 1 and sum(region_counts.values()) > 0 else "ok"
    return {
        "status": status,
        "region_counts": dict(region_counts),
    }


def _build_s3_index(experiment_config: dict[str, Any], bootstrap_pool: Sequence[str]):
    confidence_config = experiment_config.get("confidence", {})
    if str(confidence_config.get("s3_backend") or "tfidf").lower() != "sbert":
        return None
    sbert_config = confidence_config.get("sbert") or {}
    from haireder.confidence import SBERTIndex
    from haireder.utils.config import resolve_path

    model_name = sbert_config.get("model_name", "models/calibration/all-MiniLM-L6-v2")
    persist_path = sbert_config.get("persist_path")
    index = SBERTIndex(
        model_name=model_name,
        k=int(sbert_config.get("k") or 5),
        persist_path=resolve_path(persist_path) if persist_path else None,
    )
    index.update([text for text in bootstrap_pool if str(text).strip()])
    if persist_path:
        index.save()
    return index


def _blocked_report(status: str, blocker: str, inputs: dict[str, str]) -> dict[str, Any]:
    return {
        "phase": "phase_15_final_test_evaluation",
        "data_status": status,
        "blocker": blocker,
        "inputs": inputs,
    }


def _render_results_markdown(report: dict[str, Any]) -> str:
    lines = ["# Phase 15 Held-Out Diagnostics", ""]
    lines.append(f"- Data status: `{report['data_status']}`")
    if report["data_status"] != "complete":
        lines.append(f"- Blocker: {report['blocker']}")
        return "\n".join(lines) + "\n"

    classification = report["classification"]
    confidence = report["confidence"]
    human_review = report["human_review"]
    coverage = report.get("coverage")
    if coverage:
        lines.append(
            f"- Evaluated rows: `{coverage['evaluated']}/{coverage['input_rows']}` "
            f"(`{coverage['scope']}`)"
        )
    lines.extend(
        [
            f"- Test rows: `{classification['count']}`",
            f"- Accuracy: `{classification['accuracy']}`",
            f"- Macro-F1: `{classification['macro_f1']}`",
            f"- ECE: `{confidence['ece']}`",
            f"- Calibration mode: `{confidence['calibration_mode']}`",
            f"- Calibration reliability: `{confidence['calibration_reliability']['status']}`",
            f"- Prediction health: `{report['prediction_health']['status']}`",
            f"- Confidence health: `{report['confidence_health']['status']}`",
            f"- Low review count: `{human_review['low_review_count']}`",
            f"- Medium review count: `{human_review['medium_review_count']}`",
            f"- Triage saving: `{human_review['triage_saving']}`",
            "",
            "## Confidence Region Accuracy",
            "",
            "| Region | Count | Accuracy |",
            "|---|---:|---:|",
        ]
    )
    for region, values in confidence["confidence_region_accuracy"].items():
        lines.append(f"| `{region}` | {values['count']} | {values['accuracy']} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    from haireder.confidence import (
        apply_confidence_calibration,
        compute_bootstrap_retrieval_similarity_signal,
        compute_self_evaluation_signal,
        enrich_prediction_row,
        enrich_predictions_from_transcripts,
    )
    from haireder.evaluation.metrics import compute_metrics
    from haireder.utils.config import load_yaml
    from haireder.utils.io import read_jsonl, write_json, write_jsonl, write_text

    args = parse_args()
    experiment_config_path = _resolve(args.experiment_config)
    model_config_path = _resolve(args.model_config)
    train_path = _resolve(args.train)
    test_path = _resolve(args.test)
    calibrator_path = _resolve(args.calibrator_in)
    calibration_summary_path = _resolve(args.calibration_summary)
    predictions_out_path = _resolve(args.predictions_out)
    transcripts_out_path = _resolve(args.transcripts_out)
    metrics_out_path = _resolve(args.metrics_out)
    table_out_path = _resolve(args.table_out)
    auxiliary_predictions_path = _resolve(args.auxiliary_predictions) if args.auxiliary_predictions else None
    triage_summary_paths = [_resolve(path) for path in args.triage_summaries]

    required_paths = [train_path, test_path]
    missing_paths = [str(path.resolve()) for path in required_paths if not path.exists()]
    if missing_paths:
        report = _blocked_report(
            "blocked_missing_inputs",
            f"Required input files are missing: {missing_paths}",
            {
                "train": str(train_path),
                "test": str(test_path),
            },
        )
        write_json(metrics_out_path, report)
        write_text(table_out_path, _render_results_markdown(report))
        return 1

    if not calibrator_path.exists():
        if args.allow_missing_calibrator:
            calibrator = _fallback_calibrator("missing_calibrator_allowed")
        else:
            report = _blocked_report(
                "blocked_missing_calibrator",
                f"Missing calibration artifact: {calibrator_path}",
                {
                    "calibrator": str(calibrator_path),
                    "experiment_config": str(experiment_config_path),
                },
            )
            write_json(metrics_out_path, report)
            write_text(table_out_path, _render_results_markdown(report))
            return 1
    else:
        calibrator = _load_calibrator(calibrator_path)

    experiment_config = load_yaml(experiment_config_path)
    confidence_map = {
        str(label): float(value)
        for label, value in experiment_config["confidence"]["verbal_confidence_map"].items()
    }
    s2_round_weighted = bool(experiment_config["confidence"].get("s2_round_weighted", False))
    tau_low = float(experiment_config["confidence"]["tau_low"])
    tau_high = float(experiment_config["confidence"]["tau_high"])

    fusion_weights = _resolve_phase15_fusion_weights(experiment_config)

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)
    input_test_count = len(test_rows)
    if args.max_rows is not None:
        test_rows = test_rows[: args.max_rows]
    if args.sample_rows is not None and len(test_rows) > args.sample_rows:
        import random
        random.seed(args.seed)
        test_rows = random.sample(test_rows, args.sample_rows)
        print(f"Sampled {len(test_rows)} test rows (seed={args.seed})")

    auxiliary_by_pair_id: dict[str, dict[str, Any]] = {}
    if auxiliary_predictions_path is not None:
        auxiliary_by_pair_id = _index_rows_by_pair_id(
            read_jsonl(auxiliary_predictions_path),
            row_name="auxiliary prediction",
        )

    if args.force_rerun and args.reuse_existing_inference:
        raise ValueError("--force-rerun cannot be combined with --reuse-existing-inference")

    # Load pool and SBERT for RAG retrieval
    pool = []
    sbert_model = None
    if args.pool_path:
        pool_path = _resolve(args.pool_path)
        if pool_path.exists():
            pool = read_jsonl(pool_path)
            print(f"Loaded pool: {len(pool)} examples from {pool_path}")
            sbert_config = experiment_config.get("confidence", {}).get("sbert", {})
            sbert_model_name = sbert_config.get("model_name", "models/calibration/all-MiniLM-L6-v2")
            try:
                from sentence_transformers import SentenceTransformer
                sbert_model = SentenceTransformer(sbert_model_name)
                print(f"Loaded SBERT model: {sbert_model_name}")
            except Exception as exc:
                raise RuntimeError(
                    f"SBERT model is required for --pool-path retrieval but could not be loaded: {sbert_model_name}"
                ) from exc

    from haireder.data.formal import file_hash
    from haireder.experiments.review import fingerprint
    receipt_path = predictions_out_path.with_suffix(".inference.json")
    model_inputs = [experiment_config_path, model_config_path,
                    *(_resolve(getattr(args, role+"_prompt")) for role in ("proposer","critic","judge"))]
    for role in ("proposer","critic","judge"):
        model_inputs.extend(sorted(_resolve(getattr(args, role+"_adapter")).glob("adapter*")))
    inference_key = fingerprint([test_rows,args.evaluation_mode,args.max_new_tokens,args.adaptive_high_threshold,
                                 args.retrieval_k,pool,auxiliary_by_pair_id,
                                 {str(p):file_hash(p) for p in model_inputs if p.is_file()}])
    if (args.reuse_existing_inference or not args.force_rerun and
            (predictions_out_path.exists() or transcripts_out_path.exists())):
        if not receipt_path.exists():
            raise ValueError("Inference provenance is missing; run fresh inference with --force-rerun")
        receipt=json.loads(receipt_path.read_text())
        if receipt.get("identity")!=inference_key or receipt.get("predictions")!=file_hash(predictions_out_path) or receipt.get("transcripts")!=file_hash(transcripts_out_path):
            raise ValueError("Inference provenance or artifact mismatch")
    if args.reuse_existing_inference:
        raw_predictions = read_jsonl(predictions_out_path)
        transcripts = read_jsonl(transcripts_out_path)
    elif args.evaluation_mode == "adaptive":
        raw_predictions, transcripts = _run_adaptive_debate_inference(
            test_rows=test_rows,
            auxiliary_by_pair_id=auxiliary_by_pair_id,
            experiment_config_path=experiment_config_path,
            predictions_out_path=predictions_out_path,
            transcripts_out_path=transcripts_out_path,
            proposer_prompt_path=_resolve(args.proposer_prompt),
            critic_prompt_path=_resolve(args.critic_prompt),
            judge_prompt_path=_resolve(args.judge_prompt),
            proposer_adapter_path=_resolve(args.proposer_adapter),
            critic_adapter_path=_resolve(args.critic_adapter),
            judge_adapter_path=_resolve(args.judge_adapter),
            max_new_tokens=args.max_new_tokens,
            model_config_path=model_config_path,
            batch_size=args.batch_size,
            adaptive_high_threshold=args.adaptive_high_threshold,
            force_rerun=args.force_rerun,
            pool=pool if pool else None,
            sbert_model=sbert_model,
            retrieval_k=args.retrieval_k,
        )
    else:
        raw_predictions, transcripts = _run_test_debate_inference(
            test_rows=test_rows,
            auxiliary_by_pair_id=auxiliary_by_pair_id,
            experiment_config_path=experiment_config_path,
            predictions_out_path=predictions_out_path,
            transcripts_out_path=transcripts_out_path,
            proposer_prompt_path=_resolve(args.proposer_prompt),
            critic_prompt_path=_resolve(args.critic_prompt),
            judge_prompt_path=_resolve(args.judge_prompt),
            proposer_adapter_path=_resolve(args.proposer_adapter),
            critic_adapter_path=_resolve(args.critic_adapter),
            judge_adapter_path=_resolve(args.judge_adapter),
            max_new_tokens=args.max_new_tokens,
            model_config_path=model_config_path,
            force_rerun=args.force_rerun,
            pool=pool if pool else None,
            sbert_model=sbert_model,
            retrieval_k=args.retrieval_k,
        )

    from haireder.experiments.evaluation import validate_predictions, reject_substituted_turns
    validate_predictions(test_rows, raw_predictions)
    for transcript in transcripts:
        reject_substituted_turns(transcript)
    if len(transcripts) != len(test_rows) or {t["pair_id"] for t in transcripts} != {r["pair_id"] for r in test_rows}:
        raise ValueError("Debate transcript coverage differs from the input cohort")
    earliest = min(int(r["window"][1:]) for r in test_rows)
    if any(int(r["window"][1:]) >= earliest for r in pool):
        raise ValueError("Retrieval pool contains current or future window data")
    bootstrap_pool = _load_bootstrap_text_pool(pool or train_rows)
    s3_backend = str(experiment_config.get("confidence", {}).get("s3_backend") or "tfidf")
    sbert_index = _build_s3_index(experiment_config, bootstrap_pool)
    transcript_by_pair_id = _index_transcripts_by_pair_id(transcripts)
    calibration_summary = _load_optional_json(calibration_summary_path)
    if calibration_summary and calibration_summary.get("bundle"):
        fusion_weights=calibration_summary["bundle"]["weights"]
    effective_calibrator, calibration_reliability = _select_phase15_calibrator(
        calibrator,
        calibration_summary=calibration_summary,
    )

    for row in raw_predictions:
        if str(row["pair_id"]) not in transcript_by_pair_id:
            raise KeyError(f"missing transcript for pair_id={row['pair_id']}")

    fused_predictions = [
        _apply_auxiliary_none_prior(
            row,
            auxiliary_by_pair_id=auxiliary_by_pair_id,
            none_threshold=float(args.auxiliary_none_threshold),
        )
        for row in raw_predictions
    ]
    non_adaptive_rows = [
        row for row in fused_predictions
        if (transcript_by_pair_id[str(row["pair_id"])].get("adaptive_metadata") or {}).get("stage") != "judge_only"
    ]
    adaptive_rows = [
        row for row in fused_predictions
        if (transcript_by_pair_id[str(row["pair_id"])].get("adaptive_metadata") or {}).get("stage") == "judge_only"
    ]
    enriched_by_pair_id = {
        str(row["pair_id"]): row
        for row in enrich_predictions_from_transcripts(
            non_adaptive_rows,
            transcripts_by_pair_id=transcript_by_pair_id,
            text_pool=bootstrap_pool,
            confidence_map=confidence_map,
            fusion_weights=fusion_weights,
            calibrator=effective_calibrator,
            tau_low=tau_low,
            tau_high=tau_high,
            s2_round_weighted=s2_round_weighted,
            s3_backend=s3_backend,
            sbert_index=sbert_index,
        )
    }

    for row in adaptive_rows:
        transcript = transcript_by_pair_id[str(row["pair_id"])]
        judge_output = transcript["judge"]["parsed_output"]
        s1 = compute_self_evaluation_signal(judge_output, confidence_map)
        s2 = 0.0
        s3, resolved_s3_backend = compute_bootstrap_retrieval_similarity_signal(
            _pair_text(row),
            bootstrap_pool,
            backend=s3_backend,
            sbert_index=sbert_index,
        )
        raw_confidence = _compute_raw_confidence_without_inter_agent_agreement(
            s1=s1,
            s3=s3,
            weights=fusion_weights,
        )
        calibrated_confidence = apply_confidence_calibration(raw_confidence, effective_calibrator)
        enriched_by_pair_id[str(row["pair_id"])] = enrich_prediction_row(
            row,
            s1=s1,
            s2=s2,
            s3=s3,
            s3_backend=resolved_s3_backend,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            calibration_mode=effective_calibrator.mode,
            tau_low=tau_low,
            tau_high=tau_high,
        )

    enriched_rows = [enriched_by_pair_id[str(row["pair_id"])] for row in raw_predictions]

    write_jsonl(predictions_out_path, enriched_rows)
    write_json(receipt_path,{"identity":inference_key,"predictions":file_hash(predictions_out_path),"transcripts":file_hash(transcripts_out_path)})
    classification = compute_metrics(enriched_rows, compute_ece=True)
    human_review = _aggregate_triage_summaries(triage_summary_paths)
    prediction_health = _prediction_health_report(enriched_rows)
    confidence_health = _confidence_health_report(enriched_rows)
    report = {
        "phase": "phase_15_final_test_evaluation",
        "evidence_scope": "legacy_diagnostic",
        "data_status": "complete",
        "coverage": {
            "input_rows": input_test_count,
            "evaluated": len(enriched_rows),
            "scope": "full_input" if len(enriched_rows) == input_test_count else "diagnostic_subset",
        },
        "classification": classification,
        "confidence": {
            "ece": classification.get("ece"),
            "calibration_mode": effective_calibrator.mode,
            "calibration_reason": effective_calibrator.reason,
            "calibration_reliability": calibration_reliability,
            "confidence_region_accuracy": _confidence_region_accuracy(enriched_rows),
        },
        "prediction_health": prediction_health,
        "confidence_health": confidence_health,
        "human_review": human_review,
        "artifacts": {
            "predictions": str(predictions_out_path.resolve()),
            "transcripts": str(transcripts_out_path.resolve()),
            "calibrator": str(calibrator_path.resolve()),
            "calibration_summary": str(calibration_summary_path.resolve()) if calibration_summary_path.exists() else None,
            "auxiliary_predictions": str(auxiliary_predictions_path.resolve()) if auxiliary_predictions_path is not None else None,
            "triage_summaries": [str(path.resolve()) for path in triage_summary_paths if path.exists()],
        },
        "run_control": {
            "force_rerun": args.force_rerun,
            "reuse_existing_inference": args.reuse_existing_inference,
            "evaluation_mode": args.evaluation_mode,
            "batch_size": args.batch_size,
            "adaptive_high_threshold": args.adaptive_high_threshold,
            "raw_prediction_rows": len(raw_predictions),
            "transcript_rows": len(transcripts),
        },
        "errors": [
            {
                "pair_id": row["pair_id"],
                "gold_label": row["gold_label"],
                "predicted_label": row["predicted_label"],
                "transcript_path": str(transcripts_out_path.resolve()),
            }
            for row in enriched_rows
            if row["gold_label"] != row["predicted_label"]
        ],
        "inputs": {
            "experiment_config": str(experiment_config_path.resolve()),
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
    }
    write_json(metrics_out_path, report)
    write_text(table_out_path, _render_results_markdown(report))
    print(f"Predictions written: {args.predictions_out}")
    print(f"Transcripts written: {args.transcripts_out}")
    print(f"Metrics written: {args.metrics_out}")
    print(f"Results table written: {args.table_out}")
    print("Test macro-F1:", classification["macro_f1"])
    print("Test accuracy:", classification["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
