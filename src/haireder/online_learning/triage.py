"""Within-sprint uncertainty-first triage state machine."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from haireder.confidence import assign_confidence_region
from haireder.online_learning.analyst import build_accepted_decision, simulate_analyst_review
from haireder.online_learning.pool_manager import build_pool_record
from haireder.online_learning.retrieval import retrieve_similar_examples


MediumReinferFn = Callable[[dict[str, Any], Sequence[dict[str, Any]]], dict[str, Any]]


def _format_pool_examples(corrected_context: Sequence[dict[str, Any]], max_examples: int = 5) -> str:
    """Format pool examples as in-context text for re-inference."""
    examples = []
    for idx, record in enumerate(corrected_context[:max_examples], start=1):
        issue_i = record.get("issue_i", {})
        issue_j = record.get("issue_j", {})
        label = record.get("accepted_label", record.get("label", ""))
        examples.append(
            f"Example {idx}:\n"
            f"Requirement A: {issue_i.get('text', '(no text)')}\n"
            f"Requirement B: {issue_j.get('text', '(no text)')}\n"
            f"Label: {label}"
        )
    return "\n\n".join(examples)


def create_lora_reinfer_fn(
    adapter_dir: str | Path,
    prompt_template: str,
    confidence_map: dict[str, float],
    max_examples: int = 5,
    max_new_tokens: int = 128,
) -> MediumReinferFn:
    """Create a MediumReinferFn that uses a LoRA adapter for real re-inference.

    Args:
        adapter_dir: Path to the LoRA adapter directory.
        prompt_template: The proposer prompt template.
        confidence_map: Mapping from verbal confidence to numeric values.
        max_examples: Maximum number of pool examples to inject.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        A MediumReinferFn callable.
    """
    bundle_holder: dict[str, Any] = {}

    def _get_bundle() -> dict[str, Any]:
        if "bundle" not in bundle_holder:
            from haireder.training.lora_trainer import generate_role_output, load_role_bundle

            bundle_holder["bundle"] = load_role_bundle(adapter_dir)
            bundle_holder["generate_role_output"] = generate_role_output
        return bundle_holder["bundle"], bundle_holder["generate_role_output"]

    def _reinfer_fn(row: dict[str, Any], corrected_context: Sequence[dict[str, Any]]) -> dict[str, Any]:
        updated = dict(row)
        updated["re_infer_source"] = "lora_reinference"

        issue_i_text = row.get("issue_i", {}).get("text", "")
        issue_j_text = row.get("issue_j", {}).get("text", "")

        pool_block = _format_pool_examples(corrected_context, max_examples)
        retrieval_section = ""
        if pool_block:
            retrieval_section = (
                "\n\nReference examples from previously corrected pairs:\n\n"
                + pool_block
                + "\n\nNow classify the following pair:\n"
            )

        prompt = prompt_template.replace("{{issue_i_text}}", issue_i_text)
        prompt = prompt.replace("{{issue_j_text}}", issue_j_text)
        prompt = prompt.replace("{{tfidf_baseline_hint}}", "")
        if retrieval_section:
            prompt = prompt.rstrip() + retrieval_section

        try:
            bundle, generate_role_output = _get_bundle()
            raw_output = generate_role_output(bundle, prompt, max_new_tokens=max_new_tokens)
            parsed = _parse_reinfer_output(raw_output, confidence_map)
            if parsed.get("predicted_label"):
                updated["predicted_label"] = parsed["predicted_label"]
                updated["confidence"] = parsed.get("confidence", updated.get("confidence", 0.5))
                updated["re_infer_parsed_output"] = parsed.get("parsed_output")
                updated["re_infer_raw_output"] = raw_output
            else:
                updated["re_infer_source"] = "lora_reinference_parse_failed"
                updated["re_infer_note"] = parsed.get("error", "unknown_parse_error")
        except Exception as exc:
            updated["re_infer_source"] = "lora_reinference_error"
            updated["re_infer_note"] = repr(exc)

        return updated

    return _reinfer_fn


def _parse_reinfer_output(raw_output: str, confidence_map: dict[str, float]) -> dict[str, Any]:
    """Parse LoRA re-inference output."""
    from haireder.agents.proposer import parse_proposer_output

    result = parse_proposer_output(raw_output, confidence_map)
    return {
        "predicted_label": result.get("predicted_label"),
        "confidence": result.get("confidence"),
        "parsed_output": result.get("parsed_output"),
        "error": result.get("error"),
    }


def partition_by_confidence_region(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}
    for row in rows:
        region = str(row.get("confidence_region") or "")
        if region not in buckets:
            raise ValueError(f"invalid confidence_region: {region}")
        buckets[region].append(dict(row))
    return buckets


def finalize_accepted_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    finalized: list[dict[str, Any]] = []
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id:
            raise ValueError("accepted row missing pair_id")
        if pair_id in seen:
            raise ValueError(f"duplicate accepted pair_id: {pair_id}")
        if not row.get("accepted_label"):
            raise ValueError(f"accepted row missing accepted_label: {pair_id}")
        seen.add(pair_id)
        finalized.append(dict(row))
    return finalized


def _default_medium_reinfer(
    row: dict[str, Any],
    corrected_context: Sequence[dict[str, Any]],
    *,
    tau_high: float,
) -> dict[str, Any]:
    """A deterministic placeholder for sprint-local re-inference.

    Does NOT perform actual model re-inference. It only promotes medium rows
    that already agree with a corrected low-confidence label and sit close
    to the high threshold. Every row processed by this function is tagged
    with re_infer_source='placeholder_deterministic' so downstream consumers
    can distinguish placeholder outputs from real re-inference.
    """
    updated = dict(row)
    accepted_labels = {str(context.get("accepted_label")) for context in corrected_context}
    updated["re_infer_source"] = "placeholder_deterministic"
    updated["re_infer_note"] = "confidence_bump_only_no_model_reinference"
    if updated.get("predicted_label") in accepted_labels:
        updated["confidence"] = min(1.0, float(updated.get("confidence") or 0.0) + 0.1)
        if updated["confidence"] >= tau_high:
            updated["confidence_region"] = "high"
    return updated


def infer_role_intended(row: dict[str, Any]) -> str | None:
    if row.get("transcript"):
        return "judge"
    review_source = str(row.get("review_source") or "")
    analyst_status = str(row.get("analyst_status") or "")
    confidence_region = str(row.get("confidence_region") or "")
    if review_source in {"low_review", "medium_review"} or analyst_status == "corrected":
        return "critic"
    if review_source == "auto_accept_high" and confidence_region == "high":
        return "proposer"
    return row.get("role_intended")


def run_uncertainty_first_triage(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    tau_low: float,
    tau_high: float,
    medium_reinfer_fn: MediumReinferFn | None = None,
    allow_placeholder_reinfer: bool = True,
    existing_pool: Sequence[dict[str, Any]] | None = None,
    medium_retrieval_k: int = 5,
    medium_retrieval_model: Any = None,
    medium_role_filter_enabled: bool = True,
    starting_pool_sequence: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the within-sprint triage protocol and return accepted rows and pool records."""
    normalized_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        normalized = dict(row)
        if not normalized.get("confidence_region"):
            normalized["confidence_region"] = assign_confidence_region(
                float(normalized.get("confidence") or 0.0),
                tau_low=tau_low,
                tau_high=tau_high,
            )
        normalized_rows.append(normalized)

    initial_buckets = partition_by_confidence_region(normalized_rows)
    accepted_rows: list[dict[str, Any]] = []

    low_reviews = [
        simulate_analyst_review(
            row,
            decision_path="review_low_corrected",
            review_source="low_review",
        )
        for row in initial_buckets["low"]
    ]
    accepted_rows.extend(low_reviews)

    reinfer = medium_reinfer_fn
    medium_after_reinfer: list[dict[str, Any]] = []
    medium_context_pool = [*(dict(record) for record in (existing_pool or [])), *low_reviews]
    for row in initial_buckets["medium"]:
        corrected_context = retrieve_similar_examples(
            row,
            medium_context_pool,
            role="proposer",
            k=medium_retrieval_k,
            sbert_model=medium_retrieval_model,
            role_filter_enabled=medium_role_filter_enabled,
        )
        if reinfer is None:
            if not allow_placeholder_reinfer:
                raise ValueError(
                    "medium re-inference requires a real medium_reinfer_fn; "
                    "placeholder re-inference is disabled"
                )
            updated = _default_medium_reinfer(row, corrected_context, tau_high=tau_high)
        else:
            updated = reinfer(dict(row), corrected_context)
        updated["confidence_region"] = assign_confidence_region(
            float(updated["confidence"]), tau_low=tau_low, tau_high=tau_high
        )
        medium_after_reinfer.append(updated)

    promoted_medium = [row for row in medium_after_reinfer if row.get("confidence_region") == "high"]
    unresolved_medium = [row for row in medium_after_reinfer if row.get("confidence_region") != "high"]

    accepted_rows.extend(
        build_accepted_decision(
            row,
            accepted_label=str(row.get("predicted_label")),
            decision_path="medium_promoted_to_high",
            review_source="auto_accept_high",
            analyst_status="auto_accepted",
            promoted_to_high=True,
        )
        for row in promoted_medium
    )
    accepted_rows.extend(
        simulate_analyst_review(
            row,
            decision_path="medium_reviewed",
            review_source="medium_review",
        )
        for row in unresolved_medium
    )
    accepted_rows.extend(
        build_accepted_decision(
            row,
            accepted_label=str(row.get("predicted_label")),
            decision_path="auto_accept_high_initial",
            review_source="auto_accept_high",
            analyst_status="auto_accepted",
        )
        for row in initial_buckets["high"]
    )

    accepted_rows = finalize_accepted_rows(accepted_rows)
    pool_records = [
        build_pool_record(
            row,
            pool_sequence=starting_pool_sequence + index + 1,
            role_intended=infer_role_intended(row),
            has_transcript=bool(row.get("transcript")),
        )
        for index, row in enumerate(accepted_rows)
    ]

    final_region_counts = Counter(str(row.get("confidence_region")) for row in [*initial_buckets["low"], *medium_after_reinfer, *initial_buckets["high"]])
    summary = {
        "review_measurement": "oracle_simulation",
        "actual_human_review_count": 0,
        "total_rows": len(normalized_rows),
        "low_review_count": len(low_reviews),
        "medium_review_count": len(unresolved_medium),
        "total_review_count": len(low_reviews) + len(unresolved_medium),
        "promoted_to_high_count": len(promoted_medium),
        "triage_saving": (len(promoted_medium) / len(initial_buckets["medium"])) if initial_buckets["medium"] else 0.0,
        "initial_region_counts": {region: len(rows) for region, rows in initial_buckets.items()},
        "final_region_counts": dict(sorted(final_region_counts.items())),
        "accepted_path_counts": dict(sorted(Counter(str(row.get("decision_path")) for row in accepted_rows).items())),
        "accepted_label_counts": dict(sorted(Counter(str(row.get("accepted_label")) for row in accepted_rows).items())),
        "pool_candidate_rows": len(pool_records),
    }
    return accepted_rows, pool_records, summary


def render_triage_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Online Triage Report",
        "",
        f"- Total rows: `{summary.get('total_rows', 0)}`",
        f"- Low review count: `{summary.get('low_review_count', 0)}`",
        f"- Medium review count: `{summary.get('medium_review_count', 0)}`",
        f"- Promoted to high count: `{summary.get('promoted_to_high_count', 0)}`",
        f"- Triage saving: `{summary.get('triage_saving', 0.0)}`",
        "",
        "## Accepted Paths",
        "",
        "| Path | Count |",
        "|---|---:|",
    ]
    for path, count in sorted((summary.get("accepted_path_counts") or {}).items()):
        lines.append(f"| `{path}` | {count} |")
    lines.extend(["", "## Accepted Labels", "", "| Label | Count |", "|---|---:|"])
    for label, count in sorted((summary.get("accepted_label_counts") or {}).items()):
        lines.append(f"| `{label}` | {count} |")
    lines.append("")
    return "\n".join(lines)
