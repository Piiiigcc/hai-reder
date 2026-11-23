#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run within-sprint uncertainty-first online triage.")
    parser.add_argument("--experiment-config", default="configs/experiments/minimal_full_pipeline.yaml")
    parser.add_argument("--predictions", default="outputs/predictions/minimal_online_W4.jsonl")
    parser.add_argument("--transcripts", default=None)
    parser.add_argument("--window", default=None)
    parser.add_argument("--previous-pool", default=None)
    parser.add_argument("--reviews-out", default="data/annotations/analyst_reviews/minimal_online_reviews.jsonl")
    parser.add_argument("--accepted-out", default="outputs/predictions/minimal_online_accepted.jsonl")
    parser.add_argument("--pool-candidates-out", default="data/processed/retrieval_pools/minimal_pool_candidates.jsonl")
    parser.add_argument("--pool-out", default="data/processed/retrieval_pools/minimal_pool_after_online.jsonl")
    parser.add_argument("--summary-out", default="outputs/metrics/minimal_online_triage_summary.json")
    parser.add_argument("--report-out", default="outputs/tables/minimal_online_triage_report.md")
    parser.add_argument("--pool-capacity", type=int, default=None)
    parser.add_argument("--calibrator-in", default=None,
                        help="Optional Platt calibrator pickle for online triage confidence enrichment")
    parser.add_argument("--reinference-adapter", default=None,
                        help="Override online_learning.reinference.adapter_dir for this sprint")
    parser.add_argument("--require-reinference", action="store_true",
                        help="Fail instead of using placeholder medium-confidence re-inference")
    parser.add_argument("--role-specific-pool", action="store_true",
                        help="Enable role-specific pool retrieval (Proposer=high-confidence, Critic=difficulty, Judge=transcripts)")
    parser.add_argument("--simulate-analyst",action="store_true",help="Use retrospective labels for an explicitly tagged diagnostic simulation")
    return parser.parse_args()


def _max_pool_sequence(rows: list[dict]) -> int:
    values = []
    for row in rows:
        try:
            values.append(int(row.get("pool_sequence") or 0))
        except (TypeError, ValueError):
            values.append(0)
    return max(values) if values else 0


def _build_reinfer_fn(experiment_config: dict, online_config: dict, adapter_override: str | None = None):
    """Build LoraReinferFn if reinference is enabled, else None."""
    if not online_config.get("reinference_enabled", False):
        return None
    from haireder.online_learning import create_lora_reinfer_fn
    from haireder.utils.config import resolve_path

    reinference_config = online_config.get("reinference", {})
    adapter_dir = resolve_path(adapter_override or reinference_config.get("adapter_dir", "models/adapters/proposer"))
    prompt_template_path = resolve_path(
        reinference_config.get("prompt_template", "prompts/agents/proposer_prompt.md")
    )
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    confidence_map = experiment_config.get("confidence", {}).get("verbal_confidence_map", {})
    max_examples = int(reinference_config.get("max_examples", 5))
    max_new_tokens = int(reinference_config.get("max_new_tokens", 128))
    return create_lora_reinfer_fn(
        adapter_dir=adapter_dir,
        prompt_template=prompt_template,
        confidence_map=confidence_map,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
    )


def _pair_text(row: dict) -> str:
    issue_i = row.get("issue_i") or {}
    issue_j = row.get("issue_j") or {}
    return "\n\n".join(
        part for part in (
            str(issue_i.get("text") or "").strip(),
            str(issue_j.get("text") or "").strip(),
        )
        if part
    )


def _load_optional_calibrator(path: str | None):
    if not path:
        return None
    calibrator_path = Path(path)
    if not calibrator_path.is_absolute():
        calibrator_path = PROJECT_ROOT / calibrator_path
    if not calibrator_path.exists():
        return None
    with calibrator_path.open("rb") as handle:
        return pickle.load(handle)


def _load_medium_retrieval_model(experiment_config: dict, online_config: dict):
    retrieval_config = online_config.get("medium_retrieval") or {}
    if retrieval_config.get("enabled") is False:
        return None
    sbert_config = experiment_config.get("confidence", {}).get("sbert", {})
    model_name = retrieval_config.get("model_name") or sbert_config.get("model_name")
    if not model_name:
        return None
    from haireder.utils.config import resolve_path

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(str(resolve_path(model_name)))
    except Exception as exc:
        print(
            f"[triage] WARNING: failed to load medium retrieval SBERT model {model_name!r}: {exc!r}; falling back to non-semantic retrieval",
            file=sys.stderr,
        )
        return None


def _build_s3_index(experiment_config: dict, bootstrap_pool: list[str]):
    confidence_config = experiment_config.get("confidence", {})
    if str(confidence_config.get("s3_backend") or "tfidf").lower() != "sbert":
        return None
    sbert_config = confidence_config.get("sbert") or {}
    from haireder.confidence import SBERTIndex
    from haireder.utils.config import resolve_path

    model_name = sbert_config.get("model_name", "models/calibration/all-MiniLM-L6-v2")
    persist_path = sbert_config.get("persist_path")
    index = SBERTIndex(
        model_name=str(resolve_path(model_name)),
        k=int(sbert_config.get("k") or 5),
        persist_path=resolve_path(persist_path) if persist_path else None,
    )
    index.update([text for text in bootstrap_pool if str(text).strip()])
    if persist_path:
        index.save()
    return index


def main() -> int:
    from haireder.online_learning import (
        render_triage_report,
        retrieve_for_role,
        run_uncertainty_first_triage,
        update_pool_snapshot,
    )
    from haireder.utils.config import load_yaml, resolve_path, resolve_scoped_path
    from haireder.utils.io import read_jsonl, write_json, write_jsonl, write_text

    args = parse_args()
    if not args.simulate_analyst:
        raise ValueError("For real low/medium review run scripts/workflow/run_formal_pipeline.py. This diagnostic requires --simulate-analyst.")
    experiment_config = load_yaml(resolve_path(args.experiment_config))
    confidence_config = experiment_config["confidence"]
    online_config = experiment_config.get("online_learning") or {}
    tau_low = float(confidence_config["tau_low"])
    tau_high = float(confidence_config["tau_high"])
    pool_capacity = args.pool_capacity if args.pool_capacity is not None else int(online_config.get("pool_capacity") or 500)
    role_filter_enabled = args.role_specific_pool or bool(online_config.get("role_filter_enabled", False))

    predictions_path = resolve_scoped_path(args.predictions, allowed_roots=("outputs",))
    prediction_rows = read_jsonl(predictions_path)
    if args.transcripts:
        transcripts_path = resolve_scoped_path(args.transcripts, allowed_roots=("outputs",))
        if transcripts_path.exists():
            transcript_by_pair_id = {str(row.get("pair_id")): row for row in read_jsonl(transcripts_path)}
            prediction_rows = [
                {**row, "transcript": transcript_by_pair_id[str(row.get("pair_id"))]}
                if str(row.get("pair_id")) in transcript_by_pair_id else row
                for row in prediction_rows
            ]
    if args.window:
        prediction_rows = [row for row in prediction_rows if str(row.get("window")) == args.window]

    existing_pool: list[dict] = []
    if args.previous_pool:
        previous_pool_path = resolve_scoped_path(args.previous_pool, allowed_roots=("data",))
        if previous_pool_path.exists():
            existing_pool = read_jsonl(previous_pool_path)

    if args.transcripts:
        from haireder.confidence import enrich_predictions_from_transcripts, resolve_confidence_weights

        text_pool = [_pair_text(row) for row in existing_pool]
        s3_backend = str(confidence_config.get("s3_backend") or "tfidf")
        prediction_rows = enrich_predictions_from_transcripts(
            prediction_rows,
            transcripts_by_pair_id={
                str(row.get("pair_id")): row["transcript"]
                for row in prediction_rows
                if row.get("transcript")
            },
            text_pool=text_pool,
            confidence_map=confidence_config.get("verbal_confidence_map", {}),
            fusion_weights=resolve_confidence_weights(experiment_config),
            calibrator=_load_optional_calibrator(args.calibrator_in),
            tau_low=tau_low,
            tau_high=tau_high,
            s2_round_weighted=bool(confidence_config.get("s2_round_weighted", False)),
            s3_backend=s3_backend,
            sbert_index=_build_s3_index(experiment_config, text_pool),
        )

    medium_retrieval_model = _load_medium_retrieval_model(experiment_config, online_config)

    accepted_rows, pool_candidates, triage_summary = run_uncertainty_first_triage(
        prediction_rows,
        tau_low=tau_low,
        tau_high=tau_high,
        medium_reinfer_fn=_build_reinfer_fn(experiment_config, online_config, args.reinference_adapter),
        allow_placeholder_reinfer=not args.require_reinference,
        existing_pool=existing_pool,
        medium_retrieval_k=int(online_config.get("medium_retrieval_k") or online_config.get("rag_k") or 5),
        medium_retrieval_model=medium_retrieval_model,
        medium_role_filter_enabled=role_filter_enabled,
        starting_pool_sequence=_max_pool_sequence(existing_pool),
    )
    accepted_label_counts = Counter(str(row.get("accepted_label") or row.get("label") or "") for row in accepted_rows)
    pool_snapshot_kwargs = {
        "capacity": pool_capacity,
        "recency_partition_ratio": float(online_config.get("recency_partition_ratio") or 0.6),
        "difficulty_partition_ratio": float(online_config.get("difficulty_partition_ratio") or 0.4),
    }
    if "quota_label_counts" in update_pool_snapshot.__code__.co_varnames:
        pool_snapshot_kwargs["quota_label_counts"] = dict(accepted_label_counts)
    pool_snapshot, pool_summary = update_pool_snapshot(
        existing_pool,
        pool_candidates,
        **pool_snapshot_kwargs,
    )

    # Role-specific pool retrieval (paper §3.5.2)
    if role_filter_enabled:
        for role in ("proposer", "critic", "judge"):
            role_pool = retrieve_for_role(pool_snapshot, role, k=pool_capacity, role_filter_enabled=True)
            pool_out_path = resolve_scoped_path(args.pool_out, allowed_roots=("data",))
            role_pool_path = pool_out_path.with_name(f"{pool_out_path.stem}_{role}{pool_out_path.suffix}")
            write_jsonl(role_pool_path, role_pool)
            print(f"[role_filter] {role}: {len(role_pool)} examples -> {role_pool_path}", file=sys.stderr)

    reviewed_rows = [row for row in accepted_rows if row.get("human_reviewed")]
    summary = {
        "window": args.window,
        "predictions_path": str(predictions_path.resolve()),
        "triage": triage_summary,
        "pool": pool_summary,
        "accepted_rows": len(accepted_rows),
        "reviewed_rows": len(reviewed_rows),
    }

    write_jsonl(resolve_scoped_path(args.reviews_out, allowed_roots=("data",)), reviewed_rows)
    write_jsonl(resolve_scoped_path(args.accepted_out, allowed_roots=("outputs",)), accepted_rows)
    write_jsonl(resolve_scoped_path(args.pool_candidates_out, allowed_roots=("data",)), pool_candidates)
    write_jsonl(resolve_scoped_path(args.pool_out, allowed_roots=("data",)), pool_snapshot)
    write_json(resolve_scoped_path(args.summary_out, allowed_roots=("outputs",)), summary)
    write_text(resolve_scoped_path(args.report_out, allowed_roots=("outputs",)), render_triage_report(triage_summary))

    print(f"Accepted rows: {len(accepted_rows)}")
    print(f"Reviewed rows: {len(reviewed_rows)}")
    print(f"Pool snapshot rows: {len(pool_snapshot)}")
    print(f"Summary written: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
