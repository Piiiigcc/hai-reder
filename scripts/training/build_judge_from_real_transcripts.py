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
    parser = argparse.ArgumentParser(description="Build judge training data from real proposer/critic outputs.")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/minimal_qwen_qlora.yaml")
    parser.add_argument("--train", default="data/processed/splits/minimal_train_plus_online.jsonl")
    parser.add_argument("--proposer-adapter", default="models/adapters/proposer")
    parser.add_argument("--critic-adapter", default="models/adapters/critic")
    parser.add_argument("--proposer-prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--critic-prompt", default="prompts/agents/critic_prompt.md")
    parser.add_argument("--prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default="data/processed/splits/train_judge_real.jsonl")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> int:
    from haireder.agents.critic import parse_critic_output, render_critic_prompt
    from haireder.agents.proposer import parse_proposer_output, render_proposer_prompt
    from haireder.training import build_judge_training_example_from_real_transcript, load_prompt_template
    from haireder.training.lora_trainer import generate_role_output, load_multi_role_bundle, require_valid_parsed_output, select_role_bundle
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_jsonl

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    model_config = load_yaml(resolve_path(args.model_config))
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]
    rows = read_jsonl(resolve_path(args.train))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out_path = resolve_scoped_path(args.out, allowed_roots=("data",))
    existing_examples = [] if args.force_rerun or not out_path.exists() else read_jsonl(out_path)
    existing_by_pair_id = {str(row.get("pair_id")): row for row in existing_examples}
    pending_rows = [row for row in rows if str(row["pair_id"]) not in existing_by_pair_id]

    proposer_bundle = critic_bundle = None
    if pending_rows:
        shared_bundle = load_multi_role_bundle(
            {
                "proposer": resolve_path(args.proposer_adapter),
                "critic": resolve_path(args.critic_adapter),
            },
            model_config=model_config,
        )
        proposer_bundle = select_role_bundle(shared_bundle, "proposer")
        critic_bundle = select_role_bundle(shared_bundle, "critic")
    proposer_prompt = load_prompt_template(resolve_path(args.proposer_prompt))
    critic_prompt = load_prompt_template(resolve_path(args.critic_prompt))
    judge_prompt = load_prompt_template(resolve_path(args.prompt))

    new_examples: list[dict] = []
    skipped_rows = 0
    for index, row in enumerate(pending_rows):
        if index % 50 == 0:
            print(f"Processing {index}/{len(pending_rows)}...")
        try:
            proposer_prompt_text = render_proposer_prompt(row, proposer_prompt)
            proposer_raw_output = generate_role_output(
                proposer_bundle,
                proposer_prompt_text,
                max_new_tokens=args.max_new_tokens,
            )
            proposer_parsed = parse_proposer_output(proposer_raw_output, confidence_map)
            proposer_valid = require_valid_parsed_output(row["pair_id"], proposer_parsed)

            critic_prompt_text = render_critic_prompt(
                row,
                json.dumps(proposer_valid["parsed_output"], ensure_ascii=False),
                critic_prompt,
            )
            critic_raw_output = generate_role_output(
                critic_bundle,
                critic_prompt_text,
                max_new_tokens=args.max_new_tokens,
            )
            critic_parsed = parse_critic_output(critic_raw_output, confidence_map)
            critic_valid = require_valid_parsed_output(row["pair_id"], critic_parsed)
        except Exception as exc:
            skipped_rows += 1
            print(f"Skipping pair_id={row.get('pair_id')}: {exc!r}", file=sys.stderr)
            continue

        new_examples.append(
            build_judge_training_example_from_real_transcript(
                row,
                proposer_valid["parsed_output"],
                critic_valid["parsed_output"],
                judge_prompt,
            )
        )

    merged_by_pair_id = {
        **existing_by_pair_id,
        **{str(row["pair_id"]): row for row in new_examples},
    }
    examples = [
        merged_by_pair_id[str(row["pair_id"])]
        for row in rows
        if str(row["pair_id"]) in merged_by_pair_id
    ]
    write_jsonl(out_path, examples)
    print(f"Wrote {len(examples)} judge training examples to {out_path}")
    print(f"Skipped existing rows: {len(existing_by_pair_id)}")
    print(f"Newly processed rows: {len(new_examples)}")
    print(f"Skipped rows: {skipped_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
