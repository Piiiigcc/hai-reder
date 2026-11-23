#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real Proposer failure rows for Critic training.")
    parser.add_argument("--experiment-config", default="configs/experiments/formal_full_pipeline.yaml")
    parser.add_argument("--model-config", default="configs/models/formal_qwen_qlora.yaml")
    parser.add_argument("--train", default="data/processed/splits/formal_train.jsonl")
    parser.add_argument("--prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--proposer-adapter", default="models/adapters/formal/proposer")
    parser.add_argument("--failures-out", default="data/processed/splits/proposer_failures.jsonl")
    parser.add_argument("--predictions-out", default="outputs/predictions/formal_train_proposer_predictions.jsonl")
    parser.add_argument("--summary-out", default="outputs/metrics/formal_proposer_failures.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    from haireder.training import load_prompt_template
    from haireder.training.lora_trainer import run_proposer_inference
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    model_config = load_yaml(resolve_path(args.model_config))
    rows = read_jsonl(resolve_path(args.train))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    prompt_template = load_prompt_template(resolve_path(args.prompt))
    confidence_map = {
        str(label): float(value)
        for label, value in experiment_config["confidence"]["verbal_confidence_map"].items()
    }

    predictions, metadata = run_proposer_inference(
        rows,
        prompt_template,
        confidence_map,
        resolve_path(args.proposer_adapter),
        max_new_tokens=args.max_new_tokens,
        method="formal_proposer_failure_collection",
        model_config=model_config,
    )
    failures = [
        {
            **row,
            "failure_source": "real_proposer_inference",
        }
        for row in predictions
        if str(row.get("predicted_label")) != str(row.get("gold_label"))
    ]

    write_jsonl(resolve_scoped_path(args.predictions_out, allowed_roots=("outputs",)), predictions)
    write_jsonl(resolve_scoped_path(args.failures_out, allowed_roots=("data",)), failures)
    write_json(resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), {
        "train_rows": len(rows),
        "prediction_rows": len(predictions),
        "failure_rows": len(failures),
        "proposer_adapter": args.proposer_adapter,
        "metadata": metadata,
    })
    print(f"Collected proposer failures: {len(failures)} / {len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
