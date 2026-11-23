#!/usr/bin/env python
"""Train and evaluate the minimal TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/minimal_local.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--train", default="data/processed/splits/minimal_train.jsonl")
    parser.add_argument("--online", default="data/processed/splits/minimal_online.jsonl")
    parser.add_argument("--validation", default="data/processed/splits/minimal_validation.jsonl")
    parser.add_argument("--test", default="data/processed/splits/minimal_test.jsonl")
    parser.add_argument("--model-out", default="models/checkpoints/minimal_tfidf_logreg.joblib")
    parser.add_argument("--pred-prefix", default="outputs/predictions/minimal_tfidf_logreg")
    parser.add_argument("--metrics-out", default="outputs/metrics/minimal_tfidf_logreg_metrics.json")
    parser.add_argument("--table-out", default="outputs/tables/minimal_tfidf_logreg_results.md")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def unseen_labels(training_rows: list[dict], eval_rows: list[dict]) -> list[str]:
    training_labels = {row["label"] for row in training_rows}
    eval_labels = {row["label"] for row in eval_rows}
    return sorted(eval_labels - training_labels)


def main() -> int:
    from haireder.baselines import TfidfLogRegBaseline
    from haireder.evaluation.metrics import compute_metrics_by_split, render_metrics_markdown
    from haireder.taxonomy import LABELS
    from haireder.utils.config import load_yaml, resolve_path
    from haireder.utils.io import ensure_parent, read_jsonl, write_json, write_jsonl, write_text

    args = parse_args()
    dataset_config = load_yaml(resolve_path(args.dataset_config))
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    train_rows = read_jsonl(resolve_path(args.train))
    online_rows = read_jsonl(resolve_path(args.online))
    if args.experiment_config.endswith("formal_full_pipeline.yaml") or any(r.get("accepted_label") for r in online_rows):
        from haireder.experiments.baselines import cumulative_supervision
        training_rows = cumulative_supervision(train_rows, online_rows, read_jsonl(resolve_path(args.test)))
        online_rows = [dict(r,label=r["accepted_label"]) for r in online_rows]
    else:
        training_rows = train_rows + online_rows
    split_rows = {
        "train": train_rows,
        "online": online_rows,
        "validation": read_jsonl(resolve_path(args.validation)),
        "test": read_jsonl(resolve_path(args.test)),
    }

    baseline = TfidfLogRegBaseline(max_features=args.max_features, random_state=args.seed)
    baseline.fit(training_rows)
    model_path = ensure_parent(PROJECT_ROOT / args.model_out)
    joblib.dump(baseline, model_path)

    all_predictions = []
    split_reports = {}
    for split, rows in split_rows.items():
        predictions = baseline.predict_rows(rows)
        split_path = PROJECT_ROOT / f"{args.pred_prefix}_{split}.jsonl"
        write_jsonl(split_path, predictions)
        all_predictions.extend(predictions)
        split_reports[split] = {
            "rows": len(rows),
            "gold_label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
            "unseen_gold_labels_from_training_data": unseen_labels(training_rows, rows),
            "prediction_path": str(split_path.resolve()),
        }

    training_label_counts = dict(sorted(Counter(row["label"] for row in training_rows).items()))
    all_predictions_path = PROJECT_ROOT / f"{args.pred_prefix}_all.jsonl"
    write_jsonl(all_predictions_path, all_predictions)
    metrics = compute_metrics_by_split(all_predictions, compute_ece=False)
    report = {
        "method": "tfidf_logreg",
        "dataset_config": str(resolve_path(args.dataset_config)),
        "experiment_config": str(resolve_path(args.experiment_config)),
        "model_path": str(model_path),
        "all_predictions_path": str(all_predictions_path.resolve()),
        "canonical_labels": LABELS,
        "trained_labels": baseline.trained_labels_,
        "training_data_splits": ["train", "online"],
        "training_row_count": len(training_rows),
        "training_label_counts": training_label_counts,
        "baseline_scope": "static_supervised_train_plus_online_no_test_leakage",
        "ece_policy": "not_computed_for_baseline_no_confidence_calibration_module",
        "dataset_run_id": dataset_config.get("run_id"),
        "experiment_run_id": experiment_config.get("run_id"),
        "split_reports": split_reports,
        "metrics": metrics,
    }
    write_json(PROJECT_ROOT / args.metrics_out, report)
    markdown = render_metrics_markdown(metrics, title="Minimal TF-IDF Logistic Regression Results")
    preface = [
        "# Minimal TF-IDF Logistic Regression Results",
        "",
        "## Run Summary",
        "",
        f"- Model path: `{model_path}`",
        f"- All predictions: `{all_predictions_path.resolve()}`",
        f"- Trained labels: `{baseline.trained_labels_}`",
        f"- Canonical labels: `{LABELS}`",
        f"- Training data splits: `{report['training_data_splits']}`",
        f"- Training row count: `{report['training_row_count']}`",
        f"- Training label counts: `{report['training_label_counts']}`",
        "",
        "## Split Notes",
        "",
        "| Split | Rows | Gold labels | Unseen gold labels from training data |",
        "|---|---:|---|---|",
    ]
    for split, details in split_reports.items():
        preface.append(
            f"| {split} | {details['rows']} | `{details['gold_label_counts']}` | "
            f"`{details['unseen_gold_labels_from_training_data']}` |"
        )
    preface.extend(["", markdown])
    write_text(PROJECT_ROOT / args.table_out, "\n".join(preface))

    print(f"Model written: {args.model_out}")
    print(f"Predictions written: {args.pred_prefix}_*.jsonl")
    print(f"Metrics written: {args.metrics_out}")
    print(f"Results table written: {args.table_out}")
    print("Test macro-F1:", metrics["by_split"]["test"]["macro_f1"])
    print("Test accuracy:", metrics["by_split"]["test"]["accuracy"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
