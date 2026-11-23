from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.data.pair_construction import mine_hard_negatives_for_window


def _make_issue(issue_id: str, text: str, created: str = "2024-01-01T00:00:00.000+0000") -> dict:
    return {"id": issue_id, "text": text, "created": created}


def _make_positive_row(issue_i_id: str, issue_j_id: str, repo: str = "Apache") -> dict:
    return {
        "pair_id": f"POS-{issue_i_id}-{issue_j_id}",
        "repo": repo,
        "window": "W1",
        "split": "train",
        "issue_i": _make_issue(issue_i_id, f"Text for {issue_i_id}"),
        "issue_j": _make_issue(issue_j_id, f"Text for {issue_j_id}"),
        "label": "Composition",
        "is_positive": True,
    }


def test_mine_hard_negatives_tfidf_backend():
    import random

    positives = [
        _make_positive_row("A-1", "A-2"),
        _make_positive_row("A-3", "A-4"),
    ]
    candidate_issues = {
        "A-1": _make_issue("A-1", "Login page authentication error"),
        "A-2": _make_issue("A-2", "User password validation failed"),
        "A-3": _make_issue("A-3", "Database connection timeout"),
        "A-4": _make_issue("A-4", "SQL query performance issue"),
        "A-5": _make_issue("A-5", "User login session expired"),
    }
    known_links = {("A-1", "A-2"), ("A-3", "A-4")}
    rng = random.Random(42)
    negatives = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 3, rng, backend="tfidf"
    )
    assert len(negatives) > 0
    assert len(negatives) > 0
    for neg in negatives:
        assert neg["label"] == "None"
        assert neg["is_positive"] is False
        assert neg["source"] == "tfidf_hard_negative"
        assert neg["negative_mining"]["method"] == "tfidf_nearest_neighbors"
        assert neg["negative_mining"]["backend"] == "tfidf"


def test_mine_hard_negatives_sbert_backend():
    import random

    positives = [
        _make_positive_row("A-1", "A-2"),
        _make_positive_row("A-3", "A-4"),
    ]
    candidate_issues = {
        "A-1": _make_issue("A-1", "Login page authentication error"),
        "A-2": _make_issue("A-2", "User password validation failed"),
        "A-3": _make_issue("A-3", "Database connection timeout"),
        "A-4": _make_issue("A-4", "SQL query performance issue"),
        "A-5": _make_issue("A-5", "User login session expired"),
    }
    known_links = {("A-1", "A-2"), ("A-3", "A-4")}
    rng = random.Random(42)
    negatives = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 3, rng,
        backend="sbert", sbert_model_name="all-MiniLM-L6-v2",
    )
    assert len(negatives) > 0
    for neg in negatives:
        assert neg["label"] == "None"
        assert neg["is_positive"] is False
        assert neg["source"] == "sbert_hard_negative"
        assert neg["negative_mining"]["method"] == "sbert_nearest_neighbors"
        assert neg["negative_mining"]["backend"] == "sbert"


def test_mine_hard_negatives_sbert_finds_semantic_matches():
    import random

    positives = [_make_positive_row("A-1", "A-2")]
    candidate_issues = {
        "A-1": _make_issue("A-1", "Login page authentication error"),
        "A-2": _make_issue("A-2", "User password validation failed"),
        "A-3": _make_issue("A-3", "Database connection timeout"),
        "A-4": _make_issue("A-4", "Button color does not match style guide"),
    }
    known_links = {("A-1", "A-2")}
    rng = random.Random(42)
    negatives = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 2, rng,
        backend="sbert", sbert_model_name="all-MiniLM-L6-v2",
    )
    assert len(negatives) > 0
    for neg in negatives:
        assert neg["label"] == "None"
        assert neg["is_positive"] is False


def test_mine_hard_negatives_empty_issues():
    import random

    positives = [_make_positive_row("A-1", "A-2")]
    rng = random.Random(42)
    negatives = mine_hard_negatives_for_window(
        "W1", "train", positives, {}, set(), 3, rng, backend="sbert"
    )
    assert negatives == []


def test_mine_hard_negatives_default_backend_is_tfidf():
    import random

    positives = [
        _make_positive_row("A-1", "A-2"),
        _make_positive_row("A-3", "A-4"),
    ]
    candidate_issues = {
        "A-1": _make_issue("A-1", "Login page authentication error"),
        "A-2": _make_issue("A-2", "User password validation failed"),
        "A-3": _make_issue("A-3", "Database connection timeout"),
        "A-4": _make_issue("A-4", "SQL query performance issue"),
        "A-5": _make_issue("A-5", "User login session expired"),
    }
    known_links = {("A-1", "A-2"), ("A-3", "A-4")}
    rng = random.Random(42)
    negatives_default = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 3, rng
    )
    negatives_tfidf = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 3, rng, backend="tfidf"
    )
    assert len(negatives_default) == len(negatives_tfidf)
    for neg in negatives_default:
        assert neg["source"] == "tfidf_hard_negative"


def test_mine_hard_negatives_excludes_known_links():
    import random

    positives = [_make_positive_row("A-1", "A-2")]
    candidate_issues = {
        "A-1": _make_issue("A-1", "Login page authentication error"),
        "A-2": _make_issue("A-2", "User password validation failed"),
        "A-3": _make_issue("A-3", "User login session expired"),
    }
    known_links = {("A-1", "A-2"), ("A-1", "A-3")}
    rng = random.Random(42)
    negatives = mine_hard_negatives_for_window(
        "W1", "train", positives, candidate_issues, known_links, 5, rng, backend="tfidf"
    )
    for neg in negatives:
        pair = (neg["issue_i"]["id"], neg["issue_j"]["id"])
        rev_pair = (neg["issue_j"]["id"], neg["issue_i"]["id"])
        assert pair not in known_links
        assert rev_pair not in known_links
