from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from haireder.confidence.calibration import CalibrationModel, apply_confidence_calibration
from haireder.confidence.signals import (
    assign_confidence_region,
    compute_bootstrap_retrieval_similarity_signal,
    compute_inter_agent_agreement_signal,
    compute_raw_confidence,
    compute_self_evaluation_signal,
)



def enrich_prediction_row(
    row: dict[str, Any],
    *,
    s1: float,
    s2: float,
    s3: float,
    s3_backend: str = "unknown",
    raw_confidence: float,
    calibrated_confidence: float | None,
    calibration_mode: str,
    tau_low: float,
    tau_high: float,
) -> dict[str, Any]:
    final_confidence = float(calibrated_confidence if calibrated_confidence is not None else raw_confidence)
    enriched = dict(row)
    existing_provenance = dict(enriched.get("provenance") or {})
    enriched.update(
        {
            "s1": float(s1),
            "s2": float(s2),
            "s3": float(s3),
            "s3_backend": s3_backend,
            "raw_confidence": float(raw_confidence),
            "calibrated_confidence": final_confidence,
            "calibration_mode": calibration_mode,
            "confidence_region": assign_confidence_region(
                final_confidence,
                tau_low=tau_low,
                tau_high=tau_high,
            ),
            "confidence": final_confidence,
        }
    )
    if not enriched.get("provenance"):
        enriched["provenance"] = {}
    enriched["provenance"]["s3_backend"] = s3_backend
    enriched["provenance"]["_prior_provenance"] = existing_provenance
    return enriched


def _pair_text(row: dict[str, Any]) -> str:
    issue_i = row.get("issue_i") or {}
    issue_j = row.get("issue_j") or {}
    return "\n\n".join(
        part for part in (
            str(issue_i.get("text") or "").strip(),
            str(issue_j.get("text") or "").strip(),
        )
        if part
    )


def enrich_predictions_from_transcripts(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    transcripts_by_pair_id: dict[str, dict[str, Any]],
    text_pool: Sequence[str],
    confidence_map: dict[str, float],
    fusion_weights: dict[str, float],
    calibrator: CalibrationModel | None,
    tau_low: float,
    tau_high: float,
    s2_round_weighted: bool = False,
    s3_backend: str = "tfidf",
    sbert_index: Any = None,
) -> list[dict[str, Any]]:
    """Enrich prediction rows with the paper's multi-signal confidence fields.

    Rows without a matching transcript are returned unchanged so callers can keep
    the existing confidence-region fallback in the online triage state machine.
    """
    effective_calibrator = calibrator or CalibrationModel(mode="fallback", reason="missing_calibrator")
    enriched_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        transcript = transcripts_by_pair_id.get(str(row.get("pair_id")))
        if transcript is None or not transcript.get("judge"):
            enriched_rows.append(dict(row))
            continue

        judge_output = transcript["judge"]["parsed_output"]
        s1 = compute_self_evaluation_signal(judge_output, confidence_map)
        s2, s2_none_rounds = compute_inter_agent_agreement_signal(transcript, round_weighted=s2_round_weighted)
        s3, resolved_s3_backend = compute_bootstrap_retrieval_similarity_signal(
            _pair_text(row),
            text_pool,
            backend=s3_backend,
            sbert_index=sbert_index,
        )
        raw_confidence = compute_raw_confidence(s1=s1, s2=s2, s3=s3, weights=fusion_weights)
        calibrated_confidence = apply_confidence_calibration(raw_confidence, effective_calibrator)
        source_row = dict(row)
        if s2_none_rounds:
            source_row["s2_none_label_rounds"] = s2_none_rounds
        enriched_rows.append(
            enrich_prediction_row(
                source_row,
                s1=s1,
                s2=s2,
                s3=s3,
                s3_backend=resolved_s3_backend,
                raw_confidence=raw_confidence,
                calibrated_confidence=calibrated_confidence,
                calibration_mode=effective_calibrator.mode,
                tau_low=tau_low,
                tau_high=tau_high,
            )
        )
    return enriched_rows
