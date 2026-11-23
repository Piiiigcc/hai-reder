#!/usr/bin/env python
"""Build a reusable local augmented dataset bundle.

This script is intended for expensive local augmentation runs whose artifacts
will later be copied to a server/Colab runtime and reused without regenerating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EVAL_SPLITS = {"validation", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/processed/splits/minimal_train_plus_online.jsonl"],
        help="Input pair JSONL files. Multiple files are concatenated and deduplicated by pair_id.",
    )
    parser.add_argument(
        "--allowed-splits",
        nargs="+",
        default=["train", "online"],
        help="Splits eligible for augmentation. Defaults exclude validation/test.",
    )
    parser.add_argument(
        "--allow-eval-splits",
        action="store_true",
        help="Allow validation/test augmentation. Use only for debugging, not formal experiments.",
    )
    parser.add_argument("--augmented-out", default="data/processed/augmented/local_augmented.jsonl")
    parser.add_argument("--merged-out", default="data/processed/splits/local_train_online_augmented.jsonl")
    parser.add_argument("--manifest-out", default="data/processed/augmented/local_augmented_manifest.json")
    parser.add_argument("--report-out", default="outputs/tables/local_augmentation_report.md")
    parser.add_argument("--summary-out", default="outputs/metrics/local_augmentation_summary.json")
    parser.add_argument("--max-variants-per-seed", type=int, default=2)
    parser.add_argument("--target-count-per-label", type=int, default=None)
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument("--min-similarity", type=float, default=0.05)
    parser.add_argument("--max-similarity", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--generator-backend",
        choices=["rule_based", "llm"],
        default=None,
        help="Override augmentation generator backend from config.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dedupe_by_pair_id(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError("Input row missing pair_id")
        if pair_id in deduped:
            duplicate_count += 1
        deduped[pair_id] = row
    return list(deduped.values()), duplicate_count


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    allowed_splits: set[str],
    allow_eval_splits: bool,
) -> list[dict[str, Any]]:
    if not allow_eval_splits and allowed_splits & EVAL_SPLITS:
        raise ValueError(
            "Refusing to augment validation/test splits. Pass --allow-eval-splits only for debugging."
        )
    return [row for row in rows if str(row.get("split") or "") in allowed_splits]


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "labels": dict(sorted(Counter(str(row.get("label") or "") for row in rows).items())),
        "splits": dict(sorted(Counter(str(row.get("split") or "") for row in rows).items())),
        "windows": dict(sorted(Counter(str(row.get("window") or "") for row in rows).items())),
        "sources": dict(sorted(Counter(str(row.get("source") or "") for row in rows).items())),
    }


def _render_report(manifest: dict[str, Any], augmentation_summary: dict[str, Any]) -> str:
    lines = [
        "# Local Augmented Dataset Report",
        "",
        f"- Created at: `{manifest['created_at']}`",
        f"- Eligible input rows: `{manifest['eligible_input']['rows']}`",
        f"- Augmented rows: `{augmentation_summary.get('generated_rows', 0)}`",
        f"- Reused rows: `{augmentation_summary.get('reused_rows', 0)}`",
        f"- Newly generated rows: `{augmentation_summary.get('new_generated_rows', 0)}`",
        f"- Merged rows: `{manifest['outputs']['merged_rows']}`",
        f"- Force rerun: `{manifest['run_parameters']['force_rerun']}`",
        "",
        "## Label Distribution",
        "",
        "| Label | Before | Target | Reused | New | After | Unmet |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = sorted(
        set(augmentation_summary.get("counts_before", {}))
        | set(augmentation_summary.get("target_counts", {}))
        | set(augmentation_summary.get("reused_by_label", {}))
        | set(augmentation_summary.get("generated_by_label", {}))
        | set(augmentation_summary.get("counts_after", {}))
        | set(augmentation_summary.get("unmet_by_label", {}))
    )
    for label in labels:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                label,
                augmentation_summary.get("counts_before", {}).get(label, 0),
                augmentation_summary.get("target_counts", {}).get(label, 0),
                augmentation_summary.get("reused_by_label", {}).get(label, 0),
                augmentation_summary.get("generated_by_label", {}).get(label, 0),
                augmentation_summary.get("counts_after", {}).get(label, 0),
                augmentation_summary.get("unmet_by_label", {}).get(label, 0),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The generated artifact is reusable; rerunning without `--force-rerun` reuses existing augmented rows.",
            "- Do not train initial models on online-window augmented rows before those windows are accepted in the online loop.",
            "- Validation/test augmentation is blocked by default to avoid evaluation leakage.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from haireder.augmentation import (
        augment_minority_classes,
        build_llm_components,
        check_all_targets_met,
    )
    from haireder.taxonomy import load_taxonomy
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import write_json, write_jsonl, write_text

    args = parse_args()
    input_paths = [resolve_path(path) for path in args.inputs]
    augmented_out = resolve_scoped_path(args.augmented_out, allowed_roots=("data",))
    merged_out = resolve_scoped_path(args.merged_out, allowed_roots=("data",))
    manifest_out = resolve_scoped_path(args.manifest_out, allowed_roots=("data",))
    report_out = resolve_scoped_path(args.report_out, allowed_roots=("outputs",))
    summary_out = resolve_scoped_path(args.summary_out, allowed_roots=("outputs",))

    input_rows: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _read_jsonl(path)
        input_rows.extend(rows)
        input_files.append(
            {
                "path": str(path.resolve()),
                "rows": len(rows),
                "sha256": _sha256(path),
            }
        )

    deduped_rows, duplicate_count = _dedupe_by_pair_id(input_rows)
    eligible_rows = _filter_rows(
        deduped_rows,
        allowed_splits={str(split) for split in args.allowed_splits},
        allow_eval_splits=args.allow_eval_splits,
    )

    existing_augmented_rows = []
    if augmented_out.exists() and not args.force_rerun:
        existing_augmented_rows = _read_jsonl(augmented_out)

    from haireder.augmentation import resolve_augmentation_config

    experiment_config = load_yaml(resolve_path("configs/experiments/minimal_full_pipeline.yaml"))
    augmentation_config = resolve_augmentation_config(experiment_config)
    generator_backend = args.generator_backend or augmentation_config.get("generator_backend", "rule_based")

    if not args.force_rerun and check_all_targets_met(
        eligible_rows, existing_augmented_rows, args.include_none, args.target_count_per_label
    ):
        print("All augmentation targets already met. Skipping.")
        print(f"Eligible input rows: {len(eligible_rows)}")
        print(f"Existing augmented rows: {len(existing_augmented_rows)}")
        return 0

    llm_augmenter = None
    llm_validator = None
    embedding_filter = None
    reference_pool = None
    if generator_backend == "llm":
        llm_config = augmentation_config.get("llm", {})
        prompt_template_path = resolve_path(llm_config.get("prompt_template", "prompts/augmentation/pair_augmentation_prompt.md"))
        validation_prompt_template_path = resolve_path(llm_config.get("validation_prompt_template", "prompts/validation/augmentation_validation_prompt.md"))
        prompt_template = prompt_template_path.read_text(encoding="utf-8")
        validation_prompt_template = validation_prompt_template_path.read_text(encoding="utf-8")
        taxonomy = load_taxonomy(resolve_path("data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml"))
        llm_augmenter, llm_validator, embedding_filter = build_llm_components(
            llm_config, prompt_template, validation_prompt_template, taxonomy
        )
        reference_pool = eligible_rows

    augmented_rows, augmentation_summary = augment_minority_classes(
        eligible_rows,
        existing_augmented_rows=existing_augmented_rows,
        max_variants_per_seed=args.max_variants_per_seed,
        include_none=args.include_none,
        target_count_per_label=args.target_count_per_label,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        seed=args.seed,
        generator_backend=generator_backend,
        llm_augmenter=llm_augmenter,
        llm_validator=llm_validator,
        embedding_filter=embedding_filter,
        reference_pool=reference_pool,
    )
    merged_rows = [*eligible_rows, *augmented_rows]

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "local_precomputed_pair_augmentation",
        "input_files": input_files,
        "input_rows": len(input_rows),
        "deduplicated_input_rows": len(deduped_rows),
        "duplicate_pair_ids_dropped": duplicate_count,
        "eligible_input": _distribution(eligible_rows),
        "existing_augmented_rows_read": len(existing_augmented_rows),
        "augmentation_summary": augmentation_summary,
        "run_parameters": {
            "allowed_splits": list(args.allowed_splits),
            "allow_eval_splits": args.allow_eval_splits,
            "include_none": args.include_none,
            "max_variants_per_seed": args.max_variants_per_seed,
            "target_count_per_label": args.target_count_per_label,
            "min_similarity": args.min_similarity,
            "max_similarity": args.max_similarity,
            "seed": args.seed,
            "generator_backend": generator_backend,
            "force_rerun": args.force_rerun,
            "dry_run": args.dry_run,
        },
        "outputs": {
            "augmented_out": str(augmented_out.resolve()),
            "augmented_rows": len(augmented_rows),
            "merged_out": str(merged_out.resolve()),
            "merged_rows": len(merged_rows),
            "manifest_out": str(manifest_out.resolve()),
            "summary_out": str(summary_out.resolve()),
            "report_out": str(report_out.resolve()),
        },
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    write_jsonl(augmented_out, augmented_rows)
    write_jsonl(merged_out, merged_rows)
    write_json(manifest_out, manifest)
    write_json(summary_out, augmentation_summary)
    write_text(report_out, _render_report(manifest, augmentation_summary))

    print(f"Eligible input rows: {len(eligible_rows)}")
    print(f"Generator backend: {generator_backend}")
    print(f"Augmented rows: {len(augmented_rows)}")
    print(f"Reused rows: {augmentation_summary.get('reused_rows', 0)}")
    print(f"Newly generated rows: {augmentation_summary.get('new_generated_rows', 0)}")
    print(f"Merged rows: {len(merged_rows)}")
    print(f"Augmented data written: {args.augmented_out}")
    print(f"Merged data written: {args.merged_out}")
    print(f"Manifest written: {args.manifest_out}")
    if augmentation_summary.get("unmet_by_label"):
        print(f"Unmet label targets: {augmentation_summary['unmet_by_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
