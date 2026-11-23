#!/usr/bin/env python3
"""Assemble a reproducible pilot human-review batch from genuine paired rewrites."""

import argparse
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from haireder.experiments.augmentation import generate_pairs
from haireder.utils.io import read_jsonl
from haireder.utils.config import load_yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/formal_full_pipeline.yaml")
    p.add_argument("--train", default="data/processed/splits/formal_train.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--apple-mlx", action="store_true")
    p.add_argument(
        "--attempts",
        type=int,
        default=144,
        help="Predeclared generation attempts for blinded quality review",
    )
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    rows = read_jsonl(args.train)
    if any(
        r["split"] != "train" or r["window"] not in {"W1", "W2", "W3"} for r in rows
    ):
        raise ValueError("Pilot seeds must come from W1–W3")
    if args.apple_mlx:
        llm = cfg["training"]["augmentation"]["llm"]
        llm.update(
            backend="mlx",
            model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",
            validator_model_name="mlx-community/Qwen2.5-3B-Instruct-4bit",
        )
    generate_pairs(
        rows,
        Path(args.out),
        cfg,
        proposal_attempts=args.attempts,
        batch_size=args.batch_size,
        human_sample=100,
    )
