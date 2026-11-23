#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline pair-aware augmentation.")
    parser.add_argument(
        "--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml"
    )
    parser.add_argument("--input")
    parser.add_argument("--out")
    parser.add_argument("--merged-out")
    parser.add_argument("--summary-out")
    parser.add_argument("--report-out")
    parser.add_argument("--max-variants-per-seed", type=int, default=None)
    parser.add_argument("--target-count-per-label", type=int, default=None)
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--max-similarity", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generator-backend",
        choices=["rule_based", "llm"],
        default=None,
        help="Override augmentation generator backend from config.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore existing augmentation output and rebuild augmented rows.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Alias for --force-regenerate.",
    )
    return parser.parse_args()


def main() -> int:
    from haireder.augmentation import (
        augment_minority_classes,
        build_llm_components,
        render_augmentation_report,
        resolve_augmentation_config,
    )
    from haireder.taxonomy import load_taxonomy
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl, write_text

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    augmentation_config = resolve_augmentation_config(experiment_config)
    formal = "augmentation" in experiment_config.get("training", {})
    scope = "formal" if formal else "minimal"
    args.input = args.input or f"data/processed/splits/{scope}_train.jsonl"
    args.out = args.out or f"data/processed/augmented/{scope}_augmented.jsonl"
    args.merged_out = (
        args.merged_out or f"data/processed/splits/{scope}_train_augmented.jsonl"
    )
    args.summary_out = (
        args.summary_out or f"outputs/metrics/{scope}_augmentation_summary.json"
    )
    args.report_out = (
        args.report_out or f"outputs/tables/{scope}_augmentation_report.md"
    )
    generator_backend = args.generator_backend or augmentation_config.get(
        "generator_backend", "rule_based"
    )
    max_variants = (
        args.max_variants_per_seed
        if args.max_variants_per_seed is not None
        else augmentation_config.get("max_variants_per_seed", 2)
    )

    input_path = resolve_scoped_path(args.input, allowed_roots=("data",))
    out_path = resolve_scoped_path(args.out, allowed_roots=("data",))
    rows = read_jsonl(input_path)
    existing_augmented_rows = []
    force_regenerate = bool(args.force_regenerate or args.force_rerun)
    if formal:
        from haireder.experiments.augmentation import generate_pairs
        from haireder.experiments.stages import run_lock

        if generator_backend != "llm" or any(
            value is not None
            for value in (
                args.max_variants_per_seed,
                args.target_count_per_label,
                args.min_similarity,
                args.max_similarity,
            )
        ):
            raise ValueError(
                "The paper experiment uses validated LLM generation, configured SBERT distances and the observed majority count. Local overrides require a diagnostic configuration."
            )
        if not rows or any(
            r.get("split") != "train" or r.get("window") not in {"W1", "W2", "W3"}
            for r in rows
        ):
            raise ValueError(
                "Initial paper augmentation requires the W1–W3 training split"
            )
        batches = out_path.with_suffix(".batches")
        if force_regenerate:
            batches = batches / datetime.now(timezone.utc).strftime(
                "run_%Y%m%dT%H%M%S%fZ"
            )
        with run_lock(batches):
            augmented = generate_pairs(
                rows,
                batches,
                experiment_config,
                seed=args.seed,
                human_sample=augmentation_config.get("manual_review_sample_size", 100),
            )
            balance = json.loads((batches / "balance.json").read_text())
            summary = dict(
                balance,
                generated_rows=len(augmented),
                generated_by_label=dict(Counter(r["label"] for r in augmented)),
                target_counts=dict.fromkeys(
                    balance["counts_before"], balance["target_per_class"]
                ),
                rejected_rows=balance["attempts"] - len(augmented),
                include_none=True,
                generation_directory=str(batches),
                input_path=str(input_path),
            )
            write_jsonl(out_path, augmented)
            write_jsonl(
                resolve_scoped_path(args.merged_out, allowed_roots=("data",)),
                [*rows, *augmented],
            )
            write_json(
                resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)),
                summary,
            )
            write_text(
                resolve_scoped_path(args.report_out, allowed_roots=("outputs",)),
                render_augmentation_report(summary),
            )
        return 0
    if out_path.exists() and not force_regenerate:
        existing_augmented_rows = read_jsonl(out_path)

    llm_augmenter = None
    llm_validator = None
    embedding_filter = None
    reference_pool = None
    if generator_backend == "llm":
        llm_config = augmentation_config.get("llm", {})
        prompt_template_path = resolve_path(
            llm_config.get(
                "prompt_template", "prompts/augmentation/pair_augmentation_prompt.md"
            )
        )
        validation_prompt_template_path = resolve_path(
            llm_config.get(
                "validation_prompt_template",
                "prompts/validation/augmentation_validation_prompt.md",
            )
        )
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
        validation_prompt_template = validation_prompt_template_path.read_text(
            encoding="utf-8"
        )
        taxonomy = load_taxonomy(
            resolve_path(
                "data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml"
            )
        )
        llm_augmenter, llm_validator, embedding_filter = build_llm_components(
            llm_config, prompt_template, validation_prompt_template, taxonomy
        )
        reference_pool = rows

    augmented_rows, summary = augment_minority_classes(
        rows,
        existing_augmented_rows=existing_augmented_rows,
        max_variants_per_seed=max_variants,
        include_none=args.include_none,
        target_count_per_label=args.target_count_per_label,
        min_similarity=0.05 if args.min_similarity is None else args.min_similarity,
        max_similarity=0.995 if args.max_similarity is None else args.max_similarity,
        seed=args.seed,
        generator_backend=generator_backend,
        llm_augmenter=llm_augmenter,
        llm_validator=llm_validator,
        embedding_filter=embedding_filter,
        reference_pool=reference_pool,
    )
    summary.update(
        {
            "input_path": str(input_path.resolve()),
            "output_path": str(out_path.resolve()),
            "merged_output_path": str(
                resolve_scoped_path(args.merged_out, allowed_roots=("data",)).resolve()
            ),
            "force_regenerate": force_regenerate,
            "existing_augmented_rows_read": len(existing_augmented_rows),
        }
    )

    write_jsonl(out_path, augmented_rows)
    write_jsonl(
        resolve_scoped_path(args.merged_out, allowed_roots=("data",)),
        [*rows, *augmented_rows],
    )
    write_json(
        resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), summary
    )
    write_text(
        resolve_scoped_path(args.report_out, allowed_roots=("outputs",)),
        render_augmentation_report(summary),
    )

    print(f"Input rows: {len(rows)}")
    print(f"Generator backend: {generator_backend}")
    print(f"Augmented rows: {len(augmented_rows)}")
    print(f"Reused rows: {summary.get('reused_rows', 0)}")
    print(f"Newly generated rows: {summary.get('new_generated_rows', 0)}")
    print(f"Augmented data written: {args.out}")
    print(f"Merged data written: {args.merged_out}")
    print(f"Summary written: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
