#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the judge LoRA adapter.")
    parser.add_argument("--dataset-config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument("--model-config", default="configs/models/minimal_lora.yaml")
    parser.add_argument("--train", default="data/processed/splits/minimal_train.jsonl")
    parser.add_argument("--prompt", default="prompts/agents/judge_prompt.md")
    parser.add_argument("--training-data-out", default="data/processed/splits/train_judge.jsonl")
    parser.add_argument("--mixture-table-out", default="outputs/tables/role_training_mixture_judge.md")
    parser.add_argument("--adapter-out", default="models/adapters/judge")
    parser.add_argument("--summary-out", default="outputs/metrics/minimal_judge_training_summary.json")
    parser.add_argument("--max-train-rows", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--allowed-splits", nargs="+", default=["train"])
    parser.add_argument("--resume-from-checkpoint", default="auto")
    parser.add_argument("--checkpoint-save-steps", type=int, default=25)
    parser.add_argument("--checkpoint-save-total-limit", type=int, default=3)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--rationales", default=None, help="Path to generated rationales JSONL (from generate_rationales.py)")
    parser.add_argument("--rationales-required", action="store_true", help="Fail if rationales are missing for any training row")
    parser.add_argument("--real-transcripts", default=None, help="Path to real-transcript judge training data (from build_judge_from_real_transcripts.py)")
    parser.add_argument("--real-transcripts-only", action="store_true", help="Use ONLY real transcripts, skip bootstrap judge examples entirely")
    return parser.parse_args()


def render_role_training_mixture(examples: list[dict]) -> str:
    label_counts = dict(sorted(Counter(example["gold_label"] for example in examples).items()))
    reason_counts = dict(sorted(Counter(example["termination_reason"] for example in examples).items()))
    provenance_counts = dict(sorted(Counter(example.get("rationale_provenance", "unknown") for example in examples).items()))
    source_counts = dict(sorted(Counter(example.get("transcript_source", "bootstrap") for example in examples).items()))
    return "\n".join(
        [
            "# Role Training Mixture",
            "",
            "## Judge",
            "",
            f"- Training examples: `{len(examples)}`",
            f"- Label counts: `{label_counts}`",
            f"- Termination reasons: `{reason_counts}`",
            f"- Rationale provenance: `{provenance_counts}`",
            f"- Transcript sources: `{source_counts}`",
            f"- Source splits: `{sorted({example['split'] for example in examples})}`",
            "",
        ]
    )


def main() -> int:
    from haireder.training import build_judge_training_examples, load_prompt_template, select_stratified_training_examples
    from haireder.training.lora_trainer import train_role_adapter
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import (
        read_jsonl,
        write_json,
        write_jsonl,
        write_text,
    )

    args = parse_args()
    dataset_config = load_yaml(resolve_path(args.dataset_config))
    model_config = load_yaml(resolve_path(args.model_config))
    train_rows = read_jsonl(resolve_path(args.train))
    prompt_template = load_prompt_template(resolve_path(args.prompt))
    allowed_splits = tuple(args.allowed_splits)

    rationales_by_pair_id: dict[str, str] | None = None
    rationale_missing_count = 0
    if args.rationales is not None:
        rationale_rows = read_jsonl(resolve_path(args.rationales))
        rationales_by_pair_id = {
            str(row["pair_id"]): str(row["rationale"])
            for row in rationale_rows
            if row.get("pair_id") and row.get("rationale")
        }
        for row in train_rows:
            if str(row["pair_id"]) not in rationales_by_pair_id:
                rationale_missing_count += 1
        if args.rationales_required and rationale_missing_count > 0:
            raise SystemExit(f"Rationales required but missing for {rationale_missing_count} rows")
        if rationale_missing_count > 0:
            print(f"Warning: {rationale_missing_count} rows lack generated rationales, falling back to bootstrap.")

    real_examples: list[dict] = []
    if args.real_transcripts is not None:
        real_examples = read_jsonl(resolve_path(args.real_transcripts))
        print(f"Loaded {len(real_examples)} real-transcript judge examples from {args.real_transcripts}")

    if args.real_transcripts_only:
        if not real_examples:
            raise SystemExit("--real-transcripts-only was set but no real transcript examples were loaded")
        examples = real_examples
    else:
        bootstrap_examples = build_judge_training_examples(
            train_rows, prompt_template,
            allowed_splits=allowed_splits,
            rationales_by_pair_id=rationales_by_pair_id,
        )
        seen_pair_ids: set[str] = set()
        examples = []
        for ex in real_examples:
            pair_id = str(ex["pair_id"])
            seen_pair_ids.add(pair_id)
            if ex.get("transcript_source") is None:
                ex["transcript_source"] = "real_model_inference"
            examples.append(ex)
        for ex in bootstrap_examples:
            pair_id = str(ex["pair_id"])
            if pair_id not in seen_pair_ids:
                seen_pair_ids.add(pair_id)
                examples.append(ex)

    examples = select_stratified_training_examples(
        examples,
        max_examples=args.max_train_rows,
        key_fields=("gold_label", "termination_reason"),
    )
    training_data_out = resolve_scoped_path(args.training_data_out, allowed_roots=("data",))
    mixture_table_out = resolve_scoped_path(args.mixture_table_out, allowed_roots=("outputs",))
    adapter_out = resolve_scoped_path(args.adapter_out, allowed_roots=("models",))
    summary_out = resolve_scoped_path(args.summary_out, allowed_roots=("outputs",))
    if args.force_retrain and adapter_out.exists():
        shutil.rmtree(adapter_out)
    write_jsonl(training_data_out, examples)
    write_text(mixture_table_out, render_role_training_mixture(examples))
    training_metadata = train_role_adapter(
        examples,
        model_config,
        adapter_out,
        max_steps=args.max_steps,
        resume_from_checkpoint=None if args.force_retrain else args.resume_from_checkpoint,
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
    )
    bootstrap_count = sum(1 for ex in examples if ex.get("transcript_source", "bootstrap") == "bootstrap")
    real_transcript_count = sum(1 for ex in examples if ex.get("transcript_source") == "real_model_inference")
    summary = {
        "dataset_run_id": dataset_config.get("run_id"),
        "model_run_id": model_config.get("run_id"),
        "train_input_rows": len(train_rows),
        "train_examples": len(examples),
        "bootstrap_examples": bootstrap_count,
        "real_transcript_examples": real_transcript_count,
        "training_data_path": str(training_data_out.resolve()),
        "mixture_table_path": str(mixture_table_out.resolve()),
        "adapter_path": str(adapter_out.resolve()),
        "force_retrain": args.force_retrain,
        "rationales_input_path": args.rationales,
        "rationales_missing_count": rationale_missing_count,
        "rationales_required": args.rationales_required,
        "real_transcripts_input_path": args.real_transcripts,
        "real_transcripts_only": args.real_transcripts_only,
        "metadata": training_metadata,
    }
    write_json(summary_out, summary)
    print(f"Training data written: {args.training_data_out}")
    print(f"Mixture table written: {args.mixture_table_out}")
    print(f"Adapter written: {args.adapter_out}")
    print(f"Summary written: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
