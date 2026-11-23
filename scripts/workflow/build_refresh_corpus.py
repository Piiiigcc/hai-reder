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
    parser = argparse.ArgumentParser(description="Build a LoRA refresh corpus from online accepted data and pool replay.")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--accepted", nargs="+", default=["outputs/predictions/minimal_online_accepted.jsonl"])
    parser.add_argument("--pool", default="data/processed/retrieval_pools/minimal_pool_after_online.jsonl")
    parser.add_argument("--out", default="data/processed/splits/minimal_refresh_corpus.jsonl")
    parser.add_argument("--summary-out", default="outputs/metrics/minimal_refresh_corpus_summary.json")
    parser.add_argument("--report-out", default="outputs/tables/minimal_refresh_corpus_report.md")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--role-specific", action="store_true",
                        help="Build role-specific refresh corpora (Proposer/Critic/Judge)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _render_refresh_report(summary: dict) -> str:
    if summary.get("role_specific"):
        lines = ["# Refresh Corpus Report", "", "- Role-specific: `true`", "", "| Role | Output Rows | Accepted Input Rows | Pool Input Rows |", "|---|---:|---:|---:|"]
        for role, role_summary in sorted((summary.get("roles") or {}).items()):
            lines.append(
                f"| `{role}` | {role_summary.get('output_rows', 0)} | "
                f"{role_summary.get('accepted_input_rows', 0)} | {role_summary.get('pool_input_rows', 0)} |"
            )
        lines.append("")
        return "\n".join(lines)

    lines = [
        "# Refresh Corpus Report",
        "",
        f"- Accepted input rows: `{summary.get('accepted_input_rows', 0)}`",
        f"- Pool input rows: `{summary.get('pool_input_rows', 0)}`",
        f"- Selected new rows: `{summary.get('selected_new_rows', 0)}`",
        f"- Selected replay rows: `{summary.get('selected_replay_rows', 0)}`",
        f"- Augmented rows: `{summary.get('augmented_rows', 0)}`",
        f"- Output rows: `{summary.get('output_rows', 0)}`",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted((summary.get("label_counts") or {}).items()):
        lines.append(f"| `{label}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    from haireder.online_learning import (
        build_refresh_corpus,
        build_role_specific_refresh_corpus,
    )
    from haireder.augmentation import resolve_augmentation_config
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl, write_text

    args = parse_args()
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    online_config = experiment_config.get("online_learning") or {}
    augmentation_config = resolve_augmentation_config(experiment_config)
    accepted_rows = []
    for accepted_path in args.accepted:
        accepted_rows.extend(read_jsonl(resolve_scoped_path(accepted_path, allowed_roots=("outputs",))))
    pool_rows = read_jsonl(resolve_scoped_path(args.pool, allowed_roots=("data",)))

    role_specific = args.role_specific or bool(online_config.get("role_specific_refresh_corpus", False))

    if role_specific:
        # Build separate corpora for each agent role (paper §3.2.2/§3.5.3)
        role_corpora = build_role_specific_refresh_corpus(
            accepted_rows,
            pool_rows,
            max_rows=args.max_rows,
            new_data_ratio=float(online_config.get("new_data_ratio") or 0.7),
            replay_ratio=float(online_config.get("replay_ratio") or 0.3),
            augment_to_balance=not args.disable_augmentation and bool(augmentation_config.get("enabled", True)),
            max_augmented_variants_per_seed=int(augmentation_config.get("max_variants_per_seed") or 2),
            seed=args.seed,
            role_filter_enabled=True,
        )
        total_rows = 0
        for role, (corpus, summary) in role_corpora.items():
            role_out = resolve_scoped_path(
                args.out.replace(".jsonl", f"_{role}.jsonl"),
                allowed_roots=("data",),
            )
            write_jsonl(role_out, corpus)
            total_rows += len(corpus)
            print(f"[role_specific] {role}: {len(corpus)} rows -> {role_out}", file=sys.stderr)
        output_summary = {
            "role_specific": True,
            "config": online_config,
            "roles": {role: summary for role, (_, summary) in sorted(role_corpora.items())},
            "total_output_rows": total_rows,
        }
        print(f"Role-specific refresh corpora written: {total_rows} total rows")
    else:
        corpus, summary = build_refresh_corpus(
            accepted_rows,
            pool_rows,
            max_rows=args.max_rows,
            new_data_ratio=float(online_config.get("new_data_ratio") or 0.7),
            replay_ratio=float(online_config.get("replay_ratio") or 0.3),
            augment_to_balance=not args.disable_augmentation and bool(augmentation_config.get("enabled", True)),
            max_augmented_variants_per_seed=int(augmentation_config.get("max_variants_per_seed") or 2),
            seed=args.seed,
        )
        write_jsonl(resolve_scoped_path(args.out, allowed_roots=("data",)), corpus)
        output_summary = {"role_specific": False, "config": online_config, **summary}
        print(f"Refresh corpus rows: {len(corpus)}")
        print(f"Refresh corpus written: {args.out}")

    write_json(resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), output_summary)
    write_text(resolve_scoped_path(args.report_out, allowed_roots=("outputs",)), _render_refresh_report(output_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
