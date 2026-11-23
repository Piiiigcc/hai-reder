#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real Proposer-Critic transcript examples for Judge training.")
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/formal_qwen_qlora.yaml")
    parser.add_argument("--train", default="data/processed/splits/formal_train.jsonl")
    parser.add_argument("--proposer-prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--critic-prompt", default="prompts/agents/critic_prompt.md")
    parser.add_argument("--judge-prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--proposer-adapter", default="models/adapters/formal/proposer")
    parser.add_argument("--critic-adapter", default="models/adapters/formal/critic")
    parser.add_argument("--transcripts-out", default="data/processed/splits/debate_transcripts.jsonl")
    parser.add_argument("--raw-transcripts-out", default="outputs/debate_transcripts/formal_train_proposer_critic_transcripts.jsonl")
    parser.add_argument("--summary-out", default="outputs/metrics/formal_debate_transcript_collection.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    from haireder.agents.critic import parse_critic_output, render_critic_prompt
    from haireder.agents.proposer import parse_proposer_output, render_proposer_prompt
    from haireder.training import build_judge_training_example_from_real_transcript, load_prompt_template
    from haireder.training.lora_trainer import generate_role_output, load_role_bundle
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    rows = read_jsonl(resolve_path(args.train))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    confidence_map = {
        str(label): float(value)
        for label, value in experiment_config["confidence"]["verbal_confidence_map"].items()
    }
    proposer_prompt = load_prompt_template(resolve_path(args.proposer_prompt))
    critic_prompt = load_prompt_template(resolve_path(args.critic_prompt))
    judge_prompt = load_prompt_template(resolve_path(args.judge_prompt))
    proposer_bundle = load_role_bundle(resolve_path(args.proposer_adapter), model_config=load_yaml(resolve_path(args.model_config)))
    critic_bundle = load_role_bundle(resolve_path(args.critic_adapter), model_config=load_yaml(resolve_path(args.model_config)))

    judge_examples: list[dict] = []
    raw_transcripts: list[dict] = []
    skipped = 0
    for row in rows:
        proposer_raw = generate_role_output(
            proposer_bundle,
            render_proposer_prompt(row, proposer_prompt),
            max_new_tokens=args.max_new_tokens,
        )
        proposer_parsed = parse_proposer_output(proposer_raw, confidence_map)
        if not proposer_parsed.get("parsed_output"):
            skipped += 1
            continue
        critic_raw = generate_role_output(
            critic_bundle,
            render_critic_prompt(row, json.dumps(proposer_parsed["parsed_output"], ensure_ascii=False), critic_prompt),
            max_new_tokens=args.max_new_tokens,
        )
        critic_parsed = parse_critic_output(critic_raw, confidence_map)
        if not critic_parsed.get("parsed_output"):
            skipped += 1
            continue
        raw_transcripts.append({
            "pair_id": row["pair_id"],
            "proposer_parsed": proposer_parsed["parsed_output"],
            "critic_parsed": critic_parsed["parsed_output"],
            "proposer_raw_output": proposer_raw,
            "critic_raw_output": critic_raw,
        })
        example = build_judge_training_example_from_real_transcript(
            row,
            proposer_parsed["parsed_output"],
            critic_parsed["parsed_output"],
            judge_prompt,
        )
        example["transcript_source"] = "real_model_inference"
        judge_examples.append(example)

    write_jsonl(resolve_scoped_path(args.raw_transcripts_out, allowed_roots=("outputs",)), raw_transcripts)
    write_jsonl(resolve_scoped_path(args.transcripts_out, allowed_roots=("data",)), judge_examples)
    write_json(resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), {
        "train_rows": len(rows),
        "judge_examples": len(judge_examples),
        "raw_transcripts": len(raw_transcripts),
        "skipped_rows": skipped,
        "proposer_adapter": args.proposer_adapter,
        "critic_adapter": args.critic_adapter,
    })
    print(f"Collected real debate judge examples: {len(judge_examples)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
