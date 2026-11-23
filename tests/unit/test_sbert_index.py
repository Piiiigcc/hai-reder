from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.confidence.sbert_index import SBERTIndex, _cosine_similarity
from haireder.confidence.signals import compute_bootstrap_retrieval_similarity_signal


def test_cosine_similarity_identical_vectors():
    assert abs(_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_sbert_index_empty_pool():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=5)
    assert idx.size == 0
    assert idx.knn_average_similarity("any text") == 0.0


def test_sbert_index_add_and_query():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=3)
    texts = [
        "The user login page displays an error message",
        "Authentication fails with invalid credentials",
        "The database connection timeout occurs during peak hours",
        "Button color does not match the style guide",
    ]
    idx.add_texts(texts)
    assert idx.size == 4
    score = idx.knn_average_similarity("Login error when password is wrong")
    assert 0.0 <= score <= 1.0


def test_sbert_index_knn_uses_top_k():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=2)
    texts = [
        "The user login page displays an error message",
        "Authentication fails with invalid credentials",
        "The database connection timeout occurs during peak hours",
        "Button color does not match the style guide",
    ]
    idx.add_texts(texts)
    score_k2 = idx.knn_average_similarity("Login error when password is wrong")
    idx2 = SBERTIndex(model_name="all-MiniLM-L6-v2", k=4)
    idx2.add_texts(texts)
    score_k4 = idx2.knn_average_similarity("Login error when password is wrong")
    assert 0.0 <= score_k2 <= 1.0
    assert 0.0 <= score_k4 <= 1.0


def test_sbert_index_update_replaces():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=3)
    idx.add_texts(["text one", "text two"])
    assert idx.size == 2
    idx.update(["new text A", "new text B", "new text C"])
    assert idx.size == 3


def test_sbert_index_invalid_k():
    import pytest
    with pytest.raises(ValueError):
        SBERTIndex(k=0)


def test_signal_tfidf_backend_default():
    val, backend = compute_bootstrap_retrieval_similarity_signal(
        "test query", ["test pool text"]
    )
    assert backend == "tfidf_max_similarity"
    assert 0.0 <= val <= 1.0


def test_signal_tfidf_backend_explicit():
    val, backend = compute_bootstrap_retrieval_similarity_signal(
        "test query", ["test pool text"], backend="tfidf"
    )
    assert backend == "tfidf_max_similarity"
    assert 0.0 <= val <= 1.0


def test_signal_tfidf_empty_input():
    val, backend = compute_bootstrap_retrieval_similarity_signal("", [])
    assert backend == "tfidf_max_similarity_empty_input"
    assert val == 0.0


def test_signal_sbert_backend():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=3)
    idx.add_texts([
        "The user login page displays an error message",
        "Authentication fails with invalid credentials",
    ])
    val, backend = compute_bootstrap_retrieval_similarity_signal(
        "Login error when password is wrong",
        [],
        backend="sbert",
        sbert_index=idx,
    )
    assert backend == "sbert_knn"
    assert 0.0 <= val <= 1.0


def test_signal_sbert_backend_empty_query():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=3)
    idx.add_texts(["some text"])
    val, backend = compute_bootstrap_retrieval_similarity_signal(
        "", [], backend="sbert", sbert_index=idx
    )
    assert backend == "sbert_knn_empty_input"
    assert val == 0.0


def test_signal_sbert_backend_no_index_falls_back_to_tfidf():
    val, backend = compute_bootstrap_retrieval_similarity_signal(
        "test query", ["test pool text"], backend="sbert", sbert_index=None
    )
    assert backend == "tfidf_max_similarity"
    assert 0.0 <= val <= 1.0


def test_signal_sbert_value_range():
    idx = SBERTIndex(model_name="all-MiniLM-L6-v2", k=3)
    idx.add_texts([
        "The user login page displays an error message",
        "Authentication fails with invalid credentials",
        "The database connection timeout occurs during peak hours",
    ])
    for query in [
        "Login error when password is wrong",
        "Server crashes on startup",
        "Button color does not match",
    ]:
        val, backend = compute_bootstrap_retrieval_similarity_signal(
            query, [], backend="sbert", sbert_index=idx
        )
        assert backend == "sbert_knn"
        assert 0.0 <= val <= 1.0, f"s3 out of range: {val}"
