"""Tests for GAP-09: Role-specific pool retrieval filtering.

Verifies that:
- build_pool_record() includes role_intended and has_transcript fields
- retrieve_for_role() filters correctly for each agent role
- Default behavior (role_filter_enabled=False) is unchanged
- Proposer gets high-confidence canonical examples
- Critic gets difficulty-partition / low-confidence examples
- Judge gets pairs with debate transcripts
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pool_manager = importlib.import_module("haireder.online_learning.pool_manager")
build_pool_record = pool_manager.build_pool_record
retrieve_for_role = pool_manager.retrieve_for_role


def _make_accepted_row(
    pair_id: str,
    label: str = "None",
    confidence: float = 0.5,
    region: str = "medium",
    review_source: str = "auto_accept_high",
    analyst_status: str = "auto_accepted",
) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W4",
        "issue_i": {"id": f"{pair_id}-A", "text": f"Issue A {pair_id}"},
        "issue_j": {"id": f"{pair_id}-B", "text": f"Issue B {pair_id}"},
        "predicted_label": label,
        "accepted_label": label,
        "label": label,
        "confidence": confidence,
        "confidence_region": region,
        "review_source": review_source,
        "analyst_status": analyst_status,
        "decision_path": "auto_accept_high",
    }


def test_build_pool_record_has_role_intended_field() -> None:
    row = _make_accepted_row("p1")
    record = build_pool_record(row, pool_sequence=1, role_intended="proposer")
    assert record["role_intended"] == "proposer"


def test_build_pool_record_has_transcript_field() -> None:
    row = _make_accepted_row("p1")
    record = build_pool_record(row, pool_sequence=1, has_transcript=True)
    assert record["has_transcript"] is True


def test_build_pool_record_persists_transcript_payload() -> None:
    row = _make_accepted_row("p1")
    row["transcript"] = {
        "rounds": [
            {
                "round_index": 0,
                "proposer": {"parsed_output": {"label": "None"}},
                "critic": {"parsed_output": {"label": "None", "stance": "endorse"}},
            }
        ]
    }
    record = build_pool_record(row, pool_sequence=1, has_transcript=True)
    assert record["has_transcript"] is True
    assert record["transcript"] == row["transcript"]


def test_build_pool_record_defaults_backward_compatible() -> None:
    """Default values for role_intended and has_transcript should be safe."""
    row = _make_accepted_row("p1")
    record = build_pool_record(row, pool_sequence=1)
    assert record["role_intended"] is None
    assert record["has_transcript"] is False


def test_retrieve_for_role_disabled_returns_unfiltered() -> None:
    """When role_filter_enabled=False, returns pool as-is."""
    pool = [
        {"pair_id": "p1", "pool_sequence": 1},
        {"pair_id": "p2", "pool_sequence": 2},
        {"pair_id": "p3", "pool_sequence": 3},
    ]
    result = retrieve_for_role(pool, "proposer", k=2, role_filter_enabled=False)
    assert len(result) == 2
    assert result[0]["pair_id"] == "p1"
    assert result[1]["pair_id"] == "p2"


def test_retrieve_for_role_empty_pool() -> None:
    assert retrieve_for_role([], "proposer", k=5) == []
    assert retrieve_for_role([], "critic", k=5) == []
    assert retrieve_for_role([], "judge", k=5) == []


def test_retrieve_for_role_k_zero() -> None:
    pool = [{"pair_id": "p1"}]
    assert retrieve_for_role(pool, "proposer", k=0) == []


def test_retrieve_for_role_proposer_prefers_high_confidence_recency() -> None:
    """Proposer should get recency + high-confidence records first."""
    pool = [
        {"pair_id": "low_conf", "partition": "recency", "confidence_region": "low", "confidence": 0.2, "pool_sequence": 1},
        {"pair_id": "high_conf_recency", "partition": "recency", "confidence_region": "high", "confidence": 0.95, "pool_sequence": 2},
        {"pair_id": "high_conf_difficulty", "partition": "difficulty", "confidence_region": "high", "confidence": 0.9, "pool_sequence": 3},
        {"pair_id": "medium_conf", "partition": "recency", "confidence_region": "medium", "confidence": 0.5, "pool_sequence": 4},
    ]
    result = retrieve_for_role(pool, "proposer", k=2, role_filter_enabled=True)
    assert len(result) == 2
    # First should be recency + high confidence
    assert result[0]["pair_id"] == "high_conf_recency"
    # Second should be recency + medium (better than difficulty + high)
    assert result[1]["pair_id"] == "medium_conf"


def test_retrieve_for_role_critic_prefers_difficulty_low_confidence() -> None:
    """Critic should get difficulty partition + low-confidence/corrected first."""
    pool = [
        {"pair_id": "recency_high", "partition": "recency", "confidence_region": "high", "confidence": 0.95, "difficulty_score": 0.1, "analyst_status": "auto_accepted", "pool_sequence": 1},
        {"pair_id": "difficulty_corrected", "partition": "difficulty", "confidence_region": "low", "confidence": 0.2, "difficulty_score": 3.0, "analyst_status": "corrected", "pool_sequence": 2},
        {"pair_id": "difficulty_low", "partition": "difficulty", "confidence_region": "low", "confidence": 0.15, "difficulty_score": 2.5, "analyst_status": "confirmed", "pool_sequence": 3},
        {"pair_id": "recency_medium", "partition": "recency", "confidence_region": "medium", "confidence": 0.5, "difficulty_score": 0.5, "analyst_status": "auto_accepted", "pool_sequence": 4},
    ]
    result = retrieve_for_role(pool, "critic", k=2, role_filter_enabled=True)
    assert len(result) == 2
    # First should be difficulty + corrected + highest difficulty_score
    assert result[0]["pair_id"] == "difficulty_corrected"
    # Second should be difficulty + low confidence
    assert result[1]["pair_id"] == "difficulty_low"


def test_retrieve_for_role_judge_prefers_transcripts() -> None:
    """Judge should get pairs with debate transcripts first."""
    pool = [
        {"pair_id": "no_transcript", "has_transcript": False, "pool_sequence": 1},
        {"pair_id": "has_transcript", "has_transcript": True, "pool_sequence": 2},
        {"pair_id": "also_no_transcript", "has_transcript": False, "pool_sequence": 3},
        {"pair_id": "also_has_transcript", "has_transcript": True, "pool_sequence": 4},
    ]
    result = retrieve_for_role(pool, "judge", k=3, role_filter_enabled=True)
    assert len(result) == 3
    # First two should have transcripts
    assert result[0]["has_transcript"] is True
    assert result[1]["has_transcript"] is True
    # Third is the best non-transcript
    assert result[2]["has_transcript"] is False


def test_retrieve_for_role_unknown_role_no_filtering() -> None:
    """Unknown role should return pool without filtering."""
    pool = [
        {"pair_id": "p3", "pool_sequence": 3},
        {"pair_id": "p1", "pool_sequence": 1},
        {"pair_id": "p2", "pool_sequence": 2},
    ]
    result = retrieve_for_role(pool, "unknown_role", k=3, role_filter_enabled=True)
    assert len(result) == 3
    # Order should be unchanged (no sorting applied)
    assert result[0]["pair_id"] == "p3"


def test_retrieve_for_role_returns_at_most_k() -> None:
    """Should never return more than k items."""
    pool = [{"pair_id": f"p{i}", "pool_sequence": i} for i in range(10)]
    for role in ("proposer", "critic", "judge"):
        result = retrieve_for_role(pool, role, k=3, role_filter_enabled=True)
        assert len(result) == 3


def test_three_roles_get_different_rankings() -> None:
    """With role_filter_enabled, the three roles should get different orderings."""
    pool = [
        {"pair_id": "recency_high", "partition": "recency", "confidence_region": "high", "confidence": 0.95, "difficulty_score": 0.1, "analyst_status": "auto_accepted", "has_transcript": False, "pool_sequence": 1},
        {"pair_id": "difficulty_corrected", "partition": "difficulty", "confidence_region": "low", "confidence": 0.2, "difficulty_score": 3.0, "analyst_status": "corrected", "has_transcript": False, "pool_sequence": 2},
        {"pair_id": "with_transcript", "partition": "recency", "confidence_region": "medium", "confidence": 0.5, "difficulty_score": 0.5, "analyst_status": "auto_accepted", "has_transcript": True, "pool_sequence": 3},
    ]

    proposer_result = retrieve_for_role(pool, "proposer", k=3, role_filter_enabled=True)
    critic_result = retrieve_for_role(pool, "critic", k=3, role_filter_enabled=True)
    judge_result = retrieve_for_role(pool, "judge", k=3, role_filter_enabled=True)

    # Proposer should rank recency_high first
    assert proposer_result[0]["pair_id"] == "recency_high"
    # Critic should rank difficulty_corrected first
    assert critic_result[0]["pair_id"] == "difficulty_corrected"
    # Judge should rank with_transcript first
    assert judge_result[0]["pair_id"] == "with_transcript"


def test_similarity_retrieval_fallback_still_applies_role_ranking() -> None:
    from haireder.online_learning.retrieval import retrieve_similar_examples

    query = {"pair_id": "query", "issue_i": {"text": "A"}, "issue_j": {"text": "B"}}
    pool = [
        {"pair_id": "low", "partition": "recency", "confidence_region": "low", "confidence": 0.2, "pool_sequence": 1},
        {"pair_id": "high", "partition": "recency", "confidence_region": "high", "confidence": 0.95, "pool_sequence": 2},
    ]

    result = retrieve_similar_examples(query, pool, "proposer", k=1, sbert_model=None)

    assert result[0]["pair_id"] == "high"


def test_similarity_retrieval_tie_break_prefers_role_specific_records() -> None:
    from haireder.online_learning.retrieval import retrieve_similar_examples

    class EqualEmbeddingModel:
        def encode(self, text, show_progress_bar=False):
            if isinstance(text, list):
                return [[1.0, 0.0] for _ in text]
            return [1.0, 0.0]

    query = {"pair_id": "query", "issue_i": {"text": "A"}, "issue_j": {"text": "B"}}
    pool = [
        {"pair_id": "no_transcript", "has_transcript": False, "pool_sequence": 2, "issue_i": {"text": "A"}, "issue_j": {"text": "B"}},
        {"pair_id": "with_transcript", "has_transcript": True, "pool_sequence": 1, "issue_i": {"text": "A"}, "issue_j": {"text": "B"}},
    ]

    result = retrieve_similar_examples(query, pool, "judge", k=1, sbert_model=EqualEmbeddingModel())

    assert result[0]["pair_id"] == "with_transcript"
