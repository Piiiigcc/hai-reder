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
    parser = argparse.ArgumentParser(description="Run proposer-only inference with a trained LoRA adapter.")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--input", default="data/processed/splits/minimal_validation.jsonl")
    parser.add_argument("--prompt", default="prompts/agents/proposer_prompt.md")
    parser.add_argument("--adapter", default="models/adapters/proposer")
    parser.add_argument("--predictions-out", default="outputs/predictions/minimal_proposer_validation.jsonl")
    parser.add_argument("--metadata-out", default="outputs/metrics/minimal_proposer_validation_inference.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    from haireder.training import load_prompt_template
    from haireder.training.lora_trainer import run_proposer_inference
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    rows = read_jsonl(resolve_path(args.input))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    prompt_template = load_prompt_template(resolve_path(args.prompt))
    confidence_map = experiment_config["confidence"]["verbal_confidence_map"]
    predictions, metadata = run_proposer_inference(
        rows,
        prompt_template,
        confidence_map,
        resolve_scoped_path(args.adapter, allowed_roots=("models",)),
        max_new_tokens=args.max_new_tokens,
    )
    predictions_out = resolve_scoped_path(args.predictions_out, allowed_roots=("outputs",))
    metadata_out = resolve_scoped_path(args.metadata_out, allowed_roots=("outputs",))
    write_jsonl(predictions_out, predictions)
    write_json(metadata_out, metadata)
    print(f"Predictions written: {args.predictions_out}")
    print(f"Inference metadata written: {args.metadata_out}")
    print(f"Prediction rows: {len(predictions)}")
    print(f"Parse failures: {metadata['parse_failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
