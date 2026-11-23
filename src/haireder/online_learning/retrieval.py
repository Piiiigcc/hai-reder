"""SBERT-based similarity retrieval from the example pool for RAG-enhanced debate inference.

Implements the retrieval mechanism described in Section 3.5 of the paper:
- Compute SBERT embeddings for query pair and pool examples
- Rank by cosine similarity
- Apply role-specific filtering (Proposer=high-confidence, Critic=difficulty, Judge=transcripts)
"""

from __future__ import annotations

from typing import Any


_POOL_EMBEDDING_CACHE: dict[tuple[int, tuple[str, ...]], Any] = {}
_QUERY_EMBEDDING_CACHE: dict[tuple[int, str], Any] = {}


def clear_retrieval_embedding_cache() -> None:
    """Clear process-local SBERT embedding caches used by retrieval."""
    _POOL_EMBEDDING_CACHE.clear()
    _QUERY_EMBEDDING_CACHE.clear()


def _pair_text(row: dict[str, Any]) -> str:
    """Extract comparable text from a pair record."""
    issue_i = row.get("issue_i") or {}
    issue_j = row.get("issue_j") or {}
    parts = [str(issue_i.get("text") or "").strip(), str(issue_j.get("text") or "").strip()]
    return "\n\n".join(part for part in parts if part)


def _apply_role_ranking(
    scored_records: list[tuple[dict[str, Any], float]],
    role: str,
) -> list[tuple[dict[str, Any], float]]:
    """Re-rank similarity-scored records by role-specific criteria.

    Implements the role-specific retrieval described in Section 3.5.2:
    - Proposer: prefer high-confidence canonical examples from recency partition
    - Critic: prefer difficulty-partition pairs (misclassified, low-confidence)
    - Judge: prefer pairs that carry full debate transcripts
    """
    if role == "proposer":
        scored_records.sort(
            key=lambda x: (
                1 if x[0].get("partition") == "recency" else 0,
                1 if x[0].get("confidence_region") == "high" else 0,
                float(x[0].get("confidence") or x[0].get("original_confidence") or 0.0),
                x[1],
                int(x[0].get("pool_sequence") or 0),
            ),
            reverse=True,
        )
    elif role == "critic":
        scored_records.sort(
            key=lambda x: (
                1 if x[0].get("partition") == "difficulty" else 0,
                float(x[0].get("difficulty_score") or 0.0),
                1 if x[0].get("analyst_status") == "corrected" else 0,
                x[1],
                int(x[0].get("pool_sequence") or 0),
            ),
            reverse=True,
        )
    elif role == "judge":
        scored_records.sort(
            key=lambda x: (
                1 if x[0].get("has_transcript") else 0,
                x[1],
                int(x[0].get("pool_sequence") or 0),
            ),
            reverse=True,
        )
    return scored_records


def retrieve_similar_examples(
    query_pair: dict[str, Any],
    pool: list[dict[str, Any]],
    role: str,
    k: int = 5,
    sbert_model: Any = None,
    role_filter_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve top-k similar examples from pool for a specific role.

    Args:
        query_pair: The input pair to classify.
        pool: The example pool (list of pool records).
        role: Agent role — "proposer", "critic", or "judge".
        k: Maximum number of examples to return.
        sbert_model: A SentenceTransformer model instance.
        role_filter_enabled: Whether to apply role-specific re-ranking.

    Returns:
        List of pool records, ranked by similarity and role relevance.
    """
    if not pool or k <= 0:
        return []

    if sbert_model is None:
        scored_records = [(record, 0.0) for record in pool]
        if role_filter_enabled:
            scored_records = _apply_role_ranking(scored_records, role)
        return [record for record, score in scored_records[:k]]

    pool_texts = [_pair_text(record) for record in pool]
    model_id = id(sbert_model)
    query_text = _pair_text(query_pair)
    query_key = (model_id, query_text)
    if query_key not in _QUERY_EMBEDDING_CACHE:
        _QUERY_EMBEDDING_CACHE[query_key] = sbert_model.encode(query_text, show_progress_bar=False)
    query_embedding = _QUERY_EMBEDDING_CACHE[query_key]

    pool_key = (model_id, tuple(pool_texts))
    if pool_key not in _POOL_EMBEDDING_CACHE:
        _POOL_EMBEDDING_CACHE[pool_key] = sbert_model.encode(pool_texts, show_progress_bar=False)
    pool_embeddings = _POOL_EMBEDDING_CACHE[pool_key]

    from numpy import dot
    from numpy.linalg import norm

    similarities = []
    for pool_emb in pool_embeddings:
        sim = dot(query_embedding, pool_emb) / (norm(query_embedding) * norm(pool_emb) + 1e-8)
        similarities.append(float(sim))

    scored_records = list(zip(pool, similarities))

    if role_filter_enabled:
        scored_records = _apply_role_ranking(scored_records, role)
    else:
        scored_records.sort(key=lambda x: x[1], reverse=True)

    return [record for record, score in scored_records[:k]]


def format_examples_for_role(
    examples: list[dict[str, Any]],
    role: str,
) -> str:
    """Format retrieval examples as in-context text for prompt injection.

    Args:
        examples: List of pool records to format.
        role: Agent role (affects formatting style).

    Returns:
        Formatted string ready for prompt injection, or empty string if no examples.
    """
    if not examples:
        return ""

    blocks = []
    for idx, ex in enumerate(examples, start=1):
        issue_i = ex.get("issue_i", {})
        issue_j = ex.get("issue_j", {})
        label = ex.get("accepted_label", ex.get("label", ""))

        if role == "judge" and ex.get("has_transcript"):
            transcript = ex.get("transcript", {})
            rounds_summary = []
            for r in transcript.get("rounds", []):
                p_label = (r.get("proposer", {}).get("parsed_output") or {}).get("label", "?")
                c_label = (r.get("critic", {}).get("parsed_output") or {}).get("label", "?")
                c_stance = (r.get("critic", {}).get("parsed_output") or {}).get("stance", "?")
                p_reason = (r.get("proposer", {}).get("parsed_output") or {}).get("rationale", "")
                c_reason = (r.get("critic", {}).get("parsed_output") or {}).get("rationale", "")
                rounds_summary.append(f"  Round {r.get('round_index', 0)}: Proposer={p_label}, Critic={c_label} (stance={c_stance}). Proposer rationale: {p_reason}. Critic rationale: {c_reason}")
            transcript_block = "\n".join(rounds_summary) if rounds_summary else "  (no rounds)"
            blocks.append(
                f"Example {idx}:\n"
                f"Requirement A: {issue_i.get('text', '(no text)')}\n"
                f"Requirement B: {issue_j.get('text', '(no text)')}\n"
                f"Label: {label}\n"
                f"Debate Transcript:\n{transcript_block}"
            )
        else:
            rationale = ""
            if role == "critic":
                orig = ex.get("original_predicted_label", "")
                if orig and orig != label:
                    rationale = f"\nOriginal prediction was '{orig}' (incorrect), corrected to '{label}'."

            blocks.append(
                f"Example {idx}:\n"
                f"Requirement A: {issue_i.get('text', '(no text)')}\n"
                f"Requirement B: {issue_j.get('text', '(no text)')}\n"
                f"Label: {label}"
                f"{rationale}"
            )

    header = "Here are some similar previously classified pairs:\n"
    footer = "\n\nNow classify the following pair based on these examples:\n"
    return header + "\n\n".join(blocks) + footer
