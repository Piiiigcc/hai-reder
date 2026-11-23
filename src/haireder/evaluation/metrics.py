"""Classification and confidence metrics for HAI-ReDeR predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from haireder.taxonomy import LABELS


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        if np.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def expected_calibration_error(
    gold_labels: list[str],
    predicted_labels: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute ECE with equal-width confidence bins."""
    if not (len(gold_labels) == len(predicted_labels) == len(confidences)) or n_bins < 1 or any(not 0 <= c <= 1 for c in confidences):
        raise ValueError("Invalid calibration inputs")
    if not gold_labels:
        return {"ece": None, "n_bins": n_bins, "bins": []}
    bins = []
    ece = 0.0
    total = len(gold_labels)
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            mask = [low <= c <= high for c in confidences]
        else:
            mask = [low <= c < high for c in confidences]
        count = sum(mask)
        if count == 0:
            bins.append(
                {
                    "bin": index,
                    "low": low,
                    "high": high,
                    "count": 0,
                    "accuracy": None,
                    "avg_confidence": None,
                    "gap": None,
                }
            )
            continue
        indices = [i for i, flag in enumerate(mask) if flag]
        accuracy = sum(gold_labels[i] == predicted_labels[i] for i in indices) / count
        avg_confidence = sum(confidences[i] for i in indices) / count
        gap = abs(accuracy - avg_confidence)
        ece += (count / total) * gap
        bins.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "count": count,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "gap": gap,
            }
        )
    return {"ece": ece, "n_bins": n_bins, "bins": bins}


def compute_metrics(
    predictions: list[dict[str, Any]],
    labels: list[str] | None = None,
    compute_ece: bool = False,
) -> dict[str, Any]:
    """Compute classification and confidence metrics from prediction rows."""
    labels = labels or LABELS
    gold = [row["gold_label"] for row in predictions]
    pred = [row["predicted_label"] for row in predictions]
    if any(label not in labels for label in [*gold, *pred]):
        raise ValueError("Prediction contains a label outside the configured taxonomy")
    confidences = [_safe_float(row.get("confidence")) for row in predictions]
    confidence_available = all(value is not None for value in confidences)
    confidence_values = [float(value) if value is not None else 0.0 for value in confidences]

    if not predictions:
        return {
            "count": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "label_counts": {},
            "prediction_counts": {},
            "calibration_status": "not_requested" if not compute_ece else "confidence_unavailable",
        }

    present_gold_labels = [label for label in labels if label in set(gold)]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        pred,
        labels=labels,
        zero_division=0,
    )
    per_class: dict[str, Any] = {}
    for index, label in enumerate(labels):
        label_support = int(support[index])
        if label_support == 0:
            per_class[label] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "support": 0,
                "status": "not_applicable_no_gold_support",
            }
        else:
            per_class[label] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": label_support,
                "status": "ok",
            }

    macro_f1 = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    result = {
        "count": len(predictions),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": float(macro_f1),
        "observed_macro_f1": float(f1_score(gold, pred, labels=present_gold_labels, average="macro", zero_division=0)),
        "labels_evaluated_for_macro_f1": list(labels),
        "per_class": per_class,
        "label_counts": dict(sorted(Counter(gold).items())),
        "prediction_counts": dict(sorted(Counter(pred).items())),
        "confidence_available": confidence_available,
    }
    if compute_ece and confidence_available:
        calibration = expected_calibration_error(gold, pred, confidence_values)
        result["ece"] = calibration["ece"]
        result["calibration"] = calibration
    else:
        result["calibration_status"] = (
            "not_requested" if not compute_ece else "confidence_unavailable"
        )
    return result


def compute_metrics_by_split(
    predictions: list[dict[str, Any]],
    compute_ece: bool = False,
) -> dict[str, Any]:
    """Compute metrics for all rows and each split."""
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_split[str(row.get("split", "unknown"))].append(row)
    return {
        "overall": compute_metrics(predictions, compute_ece=compute_ece),
        "by_split": {
            split: compute_metrics(rows, compute_ece=compute_ece)
            for split, rows in sorted(by_split.items())
        },
    }


def render_metrics_markdown(report: dict[str, Any], title: str = "Evaluation Metrics") -> str:
    """Render metrics as Markdown."""
    lines = [f"# {title}", ""]

    def append_section(name: str, metrics: dict[str, Any]) -> None:
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Count: `{metrics['count']}`",
                f"- Accuracy: `{metrics['accuracy']}`",
                f"- Macro-F1: `{metrics['macro_f1']}`",
                f"- Gold label counts: `{metrics['label_counts']}`",
                f"- Prediction counts: `{metrics['prediction_counts']}`",
                "",
                "| Label | Precision | Recall | F1 | Support | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        if metrics.get("ece") is not None:
            insert_at = len(lines) - 4
            lines.insert(insert_at, f"- ECE: `{metrics.get('ece')}`")
        for label, values in metrics["per_class"].items():
            lines.append(
                f"| `{label}` | {values['precision']} | {values['recall']} | "
                f"{values['f1']} | {values['support']} | {values['status']} |"
            )
        lines.append("")

    append_section("Overall", report["overall"])
    for split, metrics in report.get("by_split", {}).items():
        append_section(f"Split: {split}", metrics)
    return "\n".join(lines)


def compute_metrics_by_window(
    predictions: list[dict[str, Any]],
    compute_ece: bool = False,
) -> dict[str, Any]:
    """Compute metrics grouped by time window (sprint).

    Enables tracking performance progression across sprints for RQ1.

    Args:
        predictions: Prediction rows with 'window' field.
        compute_ece: Whether to compute Expected Calibration Error.

    Returns:
        Dict with 'overall', 'by_window', and 'window_order' keys.
    """
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    window_order: list[str] = []
    for row in predictions:
        window = str(row.get("window") or "unknown")
        if window not in by_window:
            window_order.append(window)
        by_window[window].append(row)

    return {
        "overall": compute_metrics(predictions, compute_ece=compute_ece),
        "by_window": {
            window: compute_metrics(rows, compute_ece=compute_ece)
            for window, rows in sorted(by_window.items())
        },
        "window_order": window_order,
    }


def compute_refresh_round_metrics(
    predictions_before: list[dict[str, Any]],
    predictions_after: list[dict[str, Any]],
    compute_ece: bool = False,
) -> dict[str, Any]:
    """Compare metrics before and after a LoRA refresh round.

    Enables measuring the impact of each refresh for RQ3.

    Args:
        predictions_after: Predictions after the refresh.
        predictions_before: Predictions before the refresh.
        compute_ece: Whether to compute Expected Calibration Error.

    Returns:
        Dict with 'before', 'after', and 'delta' keys.
    """
    before = compute_metrics(predictions_before, compute_ece=compute_ece)
    after = compute_metrics(predictions_after, compute_ece=compute_ece)

    delta: dict[str, Any] = {}
    if before.get("macro_f1") is not None and after.get("macro_f1") is not None:
        delta["macro_f1_change"] = after["macro_f1"] - before["macro_f1"]
    if before.get("accuracy") is not None and after.get("accuracy") is not None:
        delta["accuracy_change"] = after["accuracy"] - before["accuracy"]
    if before.get("ece") is not None and after.get("ece") is not None:
        delta["ece_change"] = after["ece"] - before["ece"]

    return {
        "before": before,
        "after": after,
        "delta": delta,
    }


def compute_cumulative_metrics(
    predictions_by_sprint: dict[str, list[dict[str, Any]]],
    compute_ece: bool = False,
) -> dict[str, Any]:
    """Compute cumulative performance metrics across sprints.

    Tracks progressive improvement as the online learning loop processes
    more sprints for RQ1.

    Args:
        predictions_by_sprint: Dict mapping sprint/window ID to prediction rows.
        compute_ece: Whether to compute Expected Calibration Error.

    Returns:
        Dict with 'per_sprint', 'cumulative', and 'sprint_order' keys.
    """
    per_sprint: dict[str, dict[str, Any]] = {}
    cumulative_rows: list[dict[str, Any]] = []
    sprint_order: list[str] = []

    for sprint_id in sorted(predictions_by_sprint.keys()):
        rows = predictions_by_sprint[sprint_id]
        sprint_order.append(sprint_id)
        per_sprint[sprint_id] = compute_metrics(rows, compute_ece=compute_ece)
        cumulative_rows.extend(rows)

    return {
        "per_sprint": per_sprint,
        "cumulative": compute_metrics(cumulative_rows, compute_ece=compute_ece),
        "sprint_order": sprint_order,
    }
