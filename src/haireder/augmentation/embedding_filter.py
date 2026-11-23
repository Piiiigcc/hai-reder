"""SBERT-based embedding distance filtering for augmented requirement pairs."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

__all__ = ["SBERTEmbeddingFilter"]


def _issue_text(row: dict[str, Any], side: str) -> str:
    issue = row.get(side) or {}
    return str(issue.get("text") or "").strip()


def _pair_text(row: dict[str, Any]) -> str:
    issue_i_text = _issue_text(row, "issue_i")
    issue_j_text = _issue_text(row, "issue_j")
    return "\n\n".join(part for part in (issue_i_text, issue_j_text) if part)


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vector_a, vector_b):
        fa = float(a)
        fb = float(b)
        dot_product += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot_product / math.sqrt(norm_a * norm_b)


class SBERTEmbeddingFilter:
    """SBERT-based embedding distance filter for augmented pairs.

    Filters out generated pairs whose embeddings are too close (duplicate) or
    too far (diverged).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_distance: float = 0.1,
        max_distance: float = 0.9,
    ):
        if min_distance < 0:
            raise ValueError("min_distance must be non-negative")
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        if min_distance > max_distance:
            raise ValueError("min_distance must be less than or equal to max_distance")

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - exercised when dependency is missing
            raise ImportError(
                "sentence-transformers is required. Install it with `pip install sentence-transformers`."
            ) from exc

        self.model_name = str(model_name)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        from haireder.utils.model_versions import revision_kwargs
        self.model = SentenceTransformer(self.model_name, **revision_kwargs(self.model_name))

    def compute_distance(self, seed_row: dict[str, Any], augmented_row: dict[str, Any]) -> float:
        """Compute cosine distance between seed and augmented pair embeddings.

        Returns: float in [0, 1] where 0 = identical, 1 = completely different.
        """
        seed_text = _pair_text(seed_row)
        augmented_text = _pair_text(augmented_row)
        if not seed_text or not augmented_text:
            return 1.0

        embeddings = self.model.encode(
            [seed_text, augmented_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        seed_vector = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else list(embeddings[0])
        augmented_vector = (
            embeddings[1].tolist() if hasattr(embeddings[1], "tolist") else list(embeddings[1])
        )

        similarity = _cosine_similarity(seed_vector, augmented_vector)
        distance = 1.0 - similarity
        if not math.isfinite(distance):
            return 1.0
        return float(min(1.0, max(0.0, distance)))

    def is_valid(self, distance: float) -> bool:
        """Check if distance is within acceptable range."""
        if not math.isfinite(distance):
            return False
        return self.min_distance <= float(distance) <= self.max_distance

    def filter(self, seed_row: dict[str, Any], augmented_row: dict[str, Any]) -> dict[str, Any]:
        """Compute distance and return filter result.

        Returns:
        {
            "is_valid": bool,
            "distance": float,
            "min_distance": float,
            "max_distance": float,
        }
        """
        distance = self.compute_distance(seed_row, augmented_row)
        return {
            "is_valid": self.is_valid(distance),
            "distance": distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
        }

    @staticmethod
    def is_available() -> bool:
        """Check if sentence-transformers is installed."""
        return importlib.util.find_spec("sentence_transformers") is not None
