from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BUILD_REFRESH_SCRIPT = PROJECT_ROOT / "scripts" / "workflow" / "build_refresh_corpus.py"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.online_learning import (  # noqa: E402
    build_refresh_corpus,
    compute_smoothed_label_quotas,
    finalize_accepted_rows,
    partition_by_confidence_region,
    run_uncertainty_first_triage,
    update_pool_snapshot,
)


def make_prediction(pair_id: str, region: str, predicted: str, gold: str, confidence: float) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W4",
        "split": "online",
        "gold_label": gold,
        "predicted_label": predicted,
        "confidence": confidence,
        "confidence_region": region,
        "method": "fixture",
        "issue_i": {"id": f"{pair_id}-A", "text": f"Summary: A {pair_id}"},
        "issue_j": {"id": f"{pair_id}-B", "text": f"Summary: B {pair_id}"},
        "s1": 0.5,
        "s2": 0.5,
        "s3": 0.5,
        "calibration_mode": "fallback",
    }


def test_partition_predictions_by_confidence_region() -> None:
    rows = [
        make_prediction("low-1", "low", "Composition", "Temporal/Causal", 0.2),
        make_prediction("med-1", "medium", "Composition", "Composition", 0.6),
        make_prediction("high-1", "high", "None", "None", 0.9),
    ]

    buckets = partition_by_confidence_region(rows)

    assert [row["pair_id"] for row in buckets["low"]] == ["low-1"]
    assert [row["pair_id"] for row in buckets["medium"]] == ["med-1"]
    assert [row["pair_id"] for row in buckets["high"]] == ["high-1"]


def test_finalize_accepted_rows_rejects_duplicate_pair_ids() -> None:
    with pytest.raises(ValueError, match="duplicate accepted pair_id"):
        finalize_accepted_rows(
            [
                {"pair_id": "p1", "accepted_label": "Composition"},
                {"pair_id": "p1", "accepted_label": "Temporal/Causal"},
            ]
        )


def test_run_uncertainty_first_triage_accepts_every_pair_and_writes_pool_records() -> None:
    rows = [
        make_prediction("low-1", "low", "Composition", "Temporal/Causal", 0.2),
        make_prediction("medium-1", "medium", "Temporal/Causal", "Temporal/Causal", 0.75),
        make_prediction("high-1", "high", "None", "None", 0.9),
    ]

    accepted, pool_records, summary = run_uncertainty_first_triage(
        rows,
        tau_low=0.4,
        tau_high=0.8,
    )

    assert len(accepted) == 3
    assert {row["pair_id"] for row in accepted} == {"low-1", "medium-1", "high-1"}
    assert all(row["accepted_label"] for row in accepted)
    assert len(pool_records) == 3
    assert summary["low_review_count"] == 1
    assert summary["medium_review_count"] == 0
    assert summary["promoted_to_high_count"] == 1


def test_triage_preserves_transcript_metadata_in_pool_records() -> None:
    rows = [make_prediction("high-1", "high", "None", "None", 0.9)]
    rows[0]["transcript"] = {
        "rounds": [
            {
                "round_index": 0,
                "proposer": {"parsed_output": {"label": "None"}},
                "critic": {"parsed_output": {"label": "None", "stance": "endorse"}},
            }
        ]
    }

    _, pool_records, _ = run_uncertainty_first_triage(rows, tau_low=0.4, tau_high=0.8)

    assert pool_records[0]["has_transcript"] is True
    assert pool_records[0]["transcript"] == rows[0]["transcript"]


def test_update_pool_snapshot_respects_capacity() -> None:
    rows = [
        make_prediction(f"p{i}", "low" if i % 2 else "high", "Composition", "Composition", 0.5)
        for i in range(6)
    ]
    accepted, pool_records, _ = run_uncertainty_first_triage(rows, tau_low=0.4, tau_high=0.8)
    assert len(accepted) == 6

    snapshot, summary = update_pool_snapshot([], pool_records, capacity=3)

    assert len(snapshot) == 3
    assert summary["snapshot_records"] == 3


def test_update_pool_snapshot_uses_current_accepted_distribution_for_quotas() -> None:
    previous = [
        {
            "pair_id": f"old-{i}",
            "accepted_label": "None",
            "label": "None",
            "partition": "recency",
            "pool_sequence": i,
        }
        for i in range(20)
    ]
    current = [
        {
            "pair_id": "new-1",
            "accepted_label": "Composition",
            "label": "Composition",
            "partition": "recency",
            "pool_sequence": 21,
        }
    ]

    _, summary = update_pool_snapshot(
        previous,
        current,
        capacity=12,
        recency_partition_ratio=1.0,
        difficulty_partition_ratio=0.0,
        quota_label_counts={"Composition": 1},
    )

    assert summary["quota_label_counts_source"] == "provided"
    assert summary["recency_quotas"]["Composition"] > summary["recency_quotas"]["None"]


def test_compute_smoothed_label_quotas_sums_to_capacity() -> None:
    quotas = compute_smoothed_label_quotas({"None": 10, "Workflow": 1}, capacity=7)

    assert sum(quotas.values()) == 7
    assert set(quotas) >= {"None", "Workflow"}


def test_build_refresh_corpus_uses_accepted_and_pool_replay() -> None:
    rows = [
        make_prediction("low-1", "low", "Composition", "Temporal/Causal", 0.2),
        make_prediction("high-1", "high", "None", "None", 0.9),
    ]
    accepted, pool_records, _ = run_uncertainty_first_triage(rows, tau_low=0.4, tau_high=0.8)

    corpus, summary = build_refresh_corpus(
        accepted,
        pool_records,
        max_rows=4,
        augment_to_balance=False,
        seed=3,
    )

    assert len(corpus) >= 2
    assert summary["selected_new_rows"] == 2
    assert summary["corpus_role"] == "unified_all_roles"
    assert all("label" in row for row in corpus)


def test_build_refresh_corpus_cli_writes_role_specific_summaries(tmp_path: Path) -> None:
    prefix = f"pytest_refresh_{tmp_path.name}"
    accepted_path = PROJECT_ROOT / "outputs" / "predictions" / f"{prefix}_accepted.jsonl"
    pool_path = PROJECT_ROOT / "data" / "processed" / "retrieval_pools" / f"{prefix}_pool.jsonl"
    out_path = PROJECT_ROOT / "data" / "processed" / "splits" / f"{prefix}_refresh.jsonl"
    summary_path = PROJECT_ROOT / "outputs" / "metrics" / f"{prefix}_summary.json"
    report_path = PROJECT_ROOT / "outputs" / "tables" / f"{prefix}_report.md"
    created_paths = [
        accepted_path,
        pool_path,
        out_path.with_name(f"{out_path.stem}_proposer.jsonl"),
        out_path.with_name(f"{out_path.stem}_critic.jsonl"),
        out_path.with_name(f"{out_path.stem}_judge.jsonl"),
        summary_path,
        report_path,
    ]
    for path in created_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    accepted_path.write_text(
        '{"pair_id":"a1","repo":"R","window":"W5","split":"online",'
        '"issue_i":{"text":"A"},"issue_j":{"text":"B"},'
        '"accepted_label":"Composition","gold_label":"Composition"}\n',
        encoding="utf-8",
    )
    pool_path.write_text(
        '{"pair_id":"p1","repo":"R","window":"W4","issue_i":{"text":"C"},'
        '"issue_j":{"text":"D"},"accepted_label":"Temporal/Causal",'
        '"label":"Temporal/Causal","partition":"difficulty","pool_sequence":1}\n',
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(BUILD_REFRESH_SCRIPT),
                "--experiment-config", "configs/experiments/minimal_full_pipeline.yaml",
                "--accepted", str(accepted_path.relative_to(PROJECT_ROOT)),
                "--pool", str(pool_path.relative_to(PROJECT_ROOT)),
                "--out", str(out_path.relative_to(PROJECT_ROOT)),
                "--summary-out", str(summary_path.relative_to(PROJECT_ROOT)),
                "--report-out", str(report_path.relative_to(PROJECT_ROOT)),
                "--role-specific",
                "--disable-augmentation",
            ],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        assert result.returncode == 0, result.stderr
        summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        assert summary["role_specific"] is True
        assert set(summary["roles"]) == {"proposer", "critic", "judge"}
        assert summary["roles"]["proposer"]["output_rows"] > 0
    finally:
        for path in created_paths:
            path.unlink(missing_ok=True)


def test_default_medium_reinfer_tags_rows_with_placeholder_source() -> None:
    from haireder.online_learning.triage import _default_medium_reinfer

    row = {
        "pair_id": "medium-1",
        "predicted_label": "Composition",
        "confidence": 0.75,
        "confidence_region": "medium",
    }
    corrected = [{"accepted_label": "Composition"}]

    result = _default_medium_reinfer(row, corrected, tau_high=0.8)

    assert result["re_infer_source"] == "placeholder_deterministic"
    assert result["re_infer_note"] == "confidence_bump_only_no_model_reinference"
    assert result["confidence"] == 0.85
    assert result["confidence_region"] == "high"


def test_triage_can_require_real_medium_reinfer() -> None:
    rows = [make_prediction("medium-1", "medium", "Composition", "Composition", 0.6)]

    with pytest.raises(ValueError, match="requires a real medium_reinfer_fn"):
        run_uncertainty_first_triage(
            rows,
            tau_low=0.4,
            tau_high=0.8,
            allow_placeholder_reinfer=False,
        )


def test_medium_reinfer_receives_retrieved_examples_from_existing_pool() -> None:
    rows = [
        make_prediction("low-1", "low", "Composition", "Temporal/Causal", 0.2),
        make_prediction("medium-1", "medium", "Temporal/Causal", "Temporal/Causal", 0.75),
    ]
    existing_pool = [
        {
            "pair_id": "pool-similar",
            "accepted_label": "Composition",
            "label": "Composition",
            "confidence_region": "high",
            "confidence": 0.95,
            "partition": "recency",
            "pool_sequence": 17,
            "issue_i": {"id": "pool-A", "text": "Summary: A medium-1"},
            "issue_j": {"id": "pool-B", "text": "Summary: B medium-1"},
        }
    ]
    seen_context_pair_ids: list[list[str]] = []

    def reinfer(row: dict, corrected_context: list[dict]) -> dict:
        seen_context_pair_ids.append([str(context.get("pair_id")) for context in corrected_context])
        updated = dict(row)
        updated["re_infer_source"] = "fixture_reinference"
        return updated

    run_uncertainty_first_triage(
        rows,
        tau_low=0.4,
        tau_high=0.8,
        medium_reinfer_fn=reinfer,
        existing_pool=existing_pool,
        medium_retrieval_k=2,
    )

    assert seen_context_pair_ids == [["pool-similar", "low-1"]]


def test_triage_assigns_role_intended_to_pool_records() -> None:
    rows = [
        make_prediction("low-1", "low", "Composition", "Temporal/Causal", 0.2),
        make_prediction("medium-1", "medium", "Temporal/Causal", "Temporal/Causal", 0.75),
        make_prediction("high-1", "high", "None", "None", 0.9),
    ]
    rows[2]["transcript"] = {"rounds": [], "judge": {"parsed_output": {"final_label": "None"}}}

    _, pool_records, _ = run_uncertainty_first_triage(rows, tau_low=0.4, tau_high=0.8)

    roles_by_pair_id = {record["pair_id"]: record["role_intended"] for record in pool_records}
    assert roles_by_pair_id["low-1"] == "critic"
    assert roles_by_pair_id["medium-1"] == "proposer"
    assert roles_by_pair_id["high-1"] == "judge"


def test_medium_retrieval_model_is_forwarded_to_retrieval(monkeypatch) -> None:
    from haireder.online_learning import triage as triage_module

    rows = [make_prediction("medium-1", "medium", "Composition", "Composition", 0.6)]
    sentinel_model = object()
    seen_models: list[object] = []

    def fake_retrieve(row, pool, role, k=5, sbert_model=None, role_filter_enabled=True):
        seen_models.append(sbert_model)
        return []

    monkeypatch.setattr(triage_module, "retrieve_similar_examples", fake_retrieve)

    run_uncertainty_first_triage(
        rows,
        tau_low=0.4,
        tau_high=0.8,
        medium_retrieval_model=sentinel_model,
    )

    assert seen_models == [sentinel_model]


def test_sbert_retrieval_reuses_pool_embeddings_across_roles() -> None:
    from haireder.online_learning.retrieval import clear_retrieval_embedding_cache, retrieve_similar_examples

    class CountingModel:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def encode(self, texts, show_progress_bar=False):
            self.calls.append(texts)
            if isinstance(texts, str):
                return [1.0, 0.0]
            return [[1.0, 0.0] for _ in texts]

    pool = [
        {"pair_id": "p1", "issue_i": {"text": "login"}, "issue_j": {"text": "auth"}},
        {"pair_id": "p2", "issue_i": {"text": "cache"}, "issue_j": {"text": "timeout"}},
    ]
    query = {"pair_id": "q1", "issue_i": {"text": "login"}, "issue_j": {"text": "failure"}}
    model = CountingModel()
    clear_retrieval_embedding_cache()

    retrieve_similar_examples(query, pool, "proposer", k=1, sbert_model=model)
    retrieve_similar_examples(query, pool, "critic", k=1, sbert_model=model)
    retrieve_similar_examples(query, pool, "judge", k=1, sbert_model=model)

    pool_encode_calls = [call for call in model.calls if isinstance(call, list)]
    query_encode_calls = [call for call in model.calls if isinstance(call, str)]
    assert len(pool_encode_calls) == 1
    assert len(query_encode_calls) == 1


def test_simulate_analyst_review_tags_as_simulation() -> None:
    from haireder.online_learning.analyst import simulate_analyst_review

    row = {
        "pair_id": "p1",
        "gold_label": "Temporal/Causal",
        "predicted_label": "Temporal/Causal",
        "confidence": 0.2,
    }

    result = simulate_analyst_review(row, decision_path="review_low", review_source="low_review")

    assert result["analyst_simulation"] is True
    assert result["analyst_method"] == "gold_label_oracle"
    assert result["analyst_status"] == "confirmed"
    assert result["decision_path"] == "review_low"


def test_compute_bootstrap_signal_returns_tfidf_backend_identifier() -> None:
    from haireder.confidence.signals import compute_bootstrap_retrieval_similarity_signal

    val, backend = compute_bootstrap_retrieval_similarity_signal("test query text here", ["sample target text here"])

    assert backend == "tfidf_max_similarity"
    assert 0.0 <= val <= 1.0


def test_compute_bootstrap_signal_empty_input_identifies_reason() -> None:
    from haireder.confidence.signals import compute_bootstrap_retrieval_similarity_signal

    val, backend = compute_bootstrap_retrieval_similarity_signal("", [])

    assert backend == "tfidf_max_similarity_empty_input"
    assert val == 0.0


def test_online_triage_confidence_enrichment_uses_configured_s3_backend() -> None:
    source = (PROJECT_ROOT / "scripts" / "workflow" / "run_online_triage.py").read_text(encoding="utf-8")
    enrichment_block = source.split("prediction_rows = enrich_predictions_from_transcripts(", maxsplit=1)[1].split(
        "medium_retrieval_model = _load_medium_retrieval_model",
        maxsplit=1,
    )[0]

    assert "s3_backend=" in enrichment_block
    assert "sbert_index=" in enrichment_block
    assert 'confidence_config.get("s3_backend")' in source or "confidence_config.get('s3_backend')" in source
    assert "SBERTIndex" in source


def test_online_triage_pool_snapshot_call_handles_older_pool_manager_signature() -> None:
    source = (PROJECT_ROOT / "scripts" / "workflow" / "run_online_triage.py").read_text(encoding="utf-8")

    assert '"quota_label_counts" in update_pool_snapshot.__code__.co_varnames' in source
    assert 'pool_snapshot_kwargs["quota_label_counts"] = dict(accepted_label_counts)' in source
    assert "**pool_snapshot_kwargs" in source
