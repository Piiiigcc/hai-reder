"""Coverage, calibration, selective risk, and review workload for formal runs."""

from collections import Counter
from itertools import groupby
import math

from haireder.confidence.calibration import fit_confidence_calibrator
from haireder.confidence.signals import assign_confidence_region, compute_raw_confidence
from haireder.confidence.weight_search import search_confidence_weights
from haireder.evaluation.metrics import compute_metrics
from haireder.taxonomy import LABELS
from .review import fingerprint


INFERENCE_FIELDS = ("pair_id", "repo", "window", "split", "issue_i", "issue_j")


def reject_substituted_turns(row):
    trace = row.get("transcript") or row
    turns = [row, trace.get("judge") or {}, trace.get("proposer") or {}]
    for round_ in trace.get("rounds") or []:
        turns.extend([round_.get("proposer") or {}, round_.get("critic") or {}])
    for turn in turns:
        parsed = turn.get("parsed_output") or {}
        if (
            turn.get("error")
            or turn.get("is_fallback_generation")
            or turn.get("repair_applied")
            or turn.get("proposer_is_fallback")
            or turn.get("critic_is_fallback")
            or turn.get("analyst_simulation")
            or turn.get("re_infer_source") == "placeholder_deterministic"
            or parsed.get("is_fallback_generation")
            or parsed.get("repair_applied")
        ):
            raise ValueError(
                f"Unobserved or substituted role output for {row.get('pair_id')}"
            )


def validate_predictions(inputs, predictions, *, require_entities=True, labelled=True):
    expected = {r["pair_id"]: r for r in inputs}
    keys = [r["pair_id"] for r in predictions]
    if (
        not inputs
        or len(expected) != len(inputs)
        or len(set(keys)) != len(keys)
        or set(keys) != set(expected)
    ):
        raise ValueError(
            f"Prediction coverage mismatch: {len(predictions)} predictions for {len(inputs)} input rows"
        )
    for p in predictions:
        reject_substituted_turns(p)
        r = expected[p["pair_id"]]
        if (
            r.get("label") not in LABELS
            or p.get("predicted_label") not in LABELS
            or (labelled and p.get("gold_label") != r["label"])
            or (not labelled and ("gold_label" in p or "label" in p))
            or p.get("error")
        ):
            raise ValueError(f"Invalid prediction {p['pair_id']}")
        if (
            type(p.get("confidence")) not in (float, int)
            or not 0 <= p["confidence"] <= 1
            or any(r.get(key) != p.get(key) for key in ("repo", "window", "split"))
        ):
            raise ValueError(f"Prediction metadata mismatch for {p['pair_id']}")
        if require_entities and any(r[k] != p.get(k) for k in ("issue_i", "issue_j")):
            raise ValueError(f"Prediction endpoint mismatch for {p['pair_id']}")


def fit_bundle(rows, *, step=0.1, n_bins=10):
    if not rows or any(r["split"] != "validation" or r["window"] != "W4" for r in rows):
        raise ValueError("Calibration requires W4 validation predictions")
    for row in rows:
        if (
            row["gold_label"] not in LABELS
            or row["predicted_label"] not in LABELS
            or any(not 0 <= row[k] <= 1 for k in ("s1", "s2", "s3"))
        ):
            raise ValueError("Invalid W4 labels or signals")
    search = search_confidence_weights(rows, step=step, n_bins=n_bins)
    weights = search["best_weights"]
    scores = [raw_score(r, weights) for r in rows]
    model = fit_confidence_calibrator(
        scores, [r["gold_label"] for r in rows], [r["predicted_label"] for r in rows]
    )
    if model.mode != "platt" or max(scores) == min(scores):
        raise ValueError(
            "Platt requires varying scores and both correct and incorrect W4 predictions"
        )
    return dict(
        weights=weights,
        slope=float(model.estimator.coef_[0, 0]),
        intercept=float(model.estimator.intercept_[0]),
        w4_fingerprint=fingerprint(rows),
        n=len(rows),
        search=search,
        mode="platt",
    )


def raw_score(row, weights):
    signals = {k: row.get(k) for k in ("s1", "s2", "s3")}
    if any(type(v) not in (float, int) or not 0 <= v <= 1 for v in signals.values()):
        raise ValueError(f"Invalid confidence signals for {row.get('pair_id')}")
    return compute_raw_confidence(**signals, weights=weights)


def calibrate(row, bundle, low, high):
    score = raw_score(row, bundle["weights"])
    confidence = 1 / (1 + math.exp(-(bundle["slope"] * score + bundle["intercept"])))
    return dict(
        row,
        raw_confidence=score,
        confidence=confidence,
        calibration_mode="platt",
        confidence_region=assign_confidence_region(
            confidence, tau_low=low, tau_high=high
        ),
    )


def evaluation(rows):
    result = compute_metrics(rows, compute_ece=True)
    if not rows:
        return dict(result, aurc=None)
    count = errors = 0
    area = 0.0
    for _, group in groupby(
        sorted(rows, key=lambda r: -r["confidence"]), key=lambda r: r["confidence"]
    ):
        group = list(group)
        count += len(group)
        errors += sum(r["predicted_label"] != r["gold_label"] for r in group)
        area += len(group) / len(rows) * errors / count
    result["aurc"] = area
    result["regions"] = {
        name: compute_metrics(
            [r for r in rows if r.get("confidence_region") == name], compute_ece=True
        )
        for name in ("low", "medium", "high")
    }
    result["by_window"] = {
        w: compute_metrics([r for r in rows if r["window"] == w], compute_ece=True)
        for w in sorted({r["window"] for r in rows})
    }
    return result


def review_workload(accepted):
    paths = Counter(r["decision_path"] for r in accepted)
    medium = paths["medium_review"] + paths["medium_promoted"]
    return dict(
        review_measurement="verified_human_entries",
        total_rows=len(accepted),
        low_review_count=paths["low_review"],
        medium_review_count=paths["medium_review"],
        initial_medium_count=medium,
        promoted_to_high_count=paths["medium_promoted"],
        total_review_count=sum(r["human_reviewed"] is True for r in accepted),
        triage_saving=paths["medium_promoted"] / medium if medium else None,
    )
