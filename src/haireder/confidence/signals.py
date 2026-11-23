from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def compute_self_evaluation_signal(
    judge_output: dict[str, object],
    confidence_map: dict[str, float],
) -> float:
    verbal_confidence = str(judge_output.get("verbal_confidence") or "")
    if verbal_confidence not in confidence_map:
        raise ValueError(f"invalid verbal confidence: {verbal_confidence}")
    return float(confidence_map[verbal_confidence])



def compute_inter_agent_agreement_signal(
    transcript: dict[str, Any],
    *,
    round_weighted: bool = False,
) -> tuple[float, list[str]]:
    judge_label = str(transcript["judge"]["parsed_output"].get("final_label") or "")
    total = 0.0
    matches = 0.0

    none_label_rounds: list[str] = []
    for round_data in transcript.get("rounds", []):
        round_index = int(round_data.get("round_index", 0))
        weight = (round_index + 1) if round_weighted else 1.0
        labels = []
        total += weight
        for role in ("proposer", "critic"):
            label = ((round_data.get(role) or {}).get("parsed_output") or {}).get("label")
            labels.append(label)
            if label is None:
                none_label_rounds.append(f"round_{round_index}_{role}")
        if judge_label and labels == [judge_label, judge_label]:
            matches += weight

    if total == 0.0:
        return 0.0, none_label_rounds
    return matches / total, none_label_rounds



def compute_bootstrap_retrieval_similarity_signal(
    query_text: str,
    text_pool: Sequence[str],
    *,
    backend: str = "tfidf",
    sbert_index: Any = None,
) -> tuple[float, str]:
    """Compute retrieval similarity signal between query and example pool.

    Args:
        query_text: The input pair text to compare.
        text_pool: Sequence of texts from the example pool.
        backend: "tfidf" for TF-IDF max cosine, "sbert" for SBERT kNN average.
        sbert_index: Pre-built SBERTIndex instance (required when backend="sbert").

    Returns:
        (similarity_score, backend_identifier) tuple.
    """
    if backend == "sbert" and sbert_index is not None:
        normalized_query = str(query_text).strip()
        if not normalized_query or sbert_index.size == 0:
            return 0.0, "sbert_knn_empty_input"
        score = float(sbert_index.knn_average_similarity(normalized_query))
        return min(1.0, max(0.0, score)), "sbert_knn"

    normalized_query = str(query_text).strip()
    normalized_pool = [str(text).strip() for text in text_pool if str(text).strip()]
    if not normalized_query or not normalized_pool:
        return 0.0, "tfidf_max_similarity_empty_input"
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([normalized_query, *normalized_pool])
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    if similarities.size == 0:
        return 0.0, "tfidf_max_similarity_no_pool"
    return min(1.0, max(0.0, float(similarities.max()))), "tfidf_max_similarity"



def compute_raw_confidence(*, s1: float, s2: float, s3: float, weights: dict[str, float]) -> float:
    raw_confidence = (
        float(weights["self_evaluation"]) * s1
        + float(weights["inter_agent_agreement"]) * s2
        + float(weights["retrieval_similarity"]) * s3
    )
    return min(1.0, max(0.0, raw_confidence))



def assign_confidence_region(confidence: float, *, tau_low: float, tau_high: float) -> str:
    if not (0 <= tau_low < tau_high <= 1 and 0 <= confidence <= 1):
        raise ValueError("Invalid confidence or thresholds")
    if confidence < tau_low:
        return "low"
    if confidence < tau_high:
        return "medium"
    return "high"
