"""SBERT-based embedding index for kNN retrieval similarity."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class SBERTIndex:
    """Manages SBERT embeddings for kNN average cosine similarity.

    Supports pre-computation, incremental updates, and optional persistence.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 5,
        persist_path: str | Path | None = None,
    ):
        self.model_name = str(model_name)
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        self.persist_path = Path(persist_path) if persist_path else None
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._model: Any = None

        if self.persist_path and self.persist_path.exists():
            self._load()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        from haireder.utils.model_versions import revision_kwargs
        self._model = SentenceTransformer(self.model_name, **revision_kwargs(self.model_name))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def add_texts(self, texts: list[str]) -> None:
        new_texts = [t for t in texts if t.strip()]
        if not new_texts:
            return
        new_embeddings = self._encode(new_texts)
        self._texts.extend(new_texts)
        self._embeddings.extend(new_embeddings)

    def update(self, texts: list[str]) -> None:
        self._texts = []
        self._embeddings = []
        self.add_texts(texts)

    def knn_average_similarity(self, query_text: str) -> float:
        if not self._embeddings:
            return 0.0
        query_embedding = self._encode([query_text])[0]
        similarities: list[float] = []
        for emb in self._embeddings:
            sim = _cosine_similarity(query_embedding, emb)
            if math.isfinite(sim):
                similarities.append(sim)
        if not similarities:
            return 0.0
        top_k = sorted(similarities, reverse=True)[: self.k]
        return sum(top_k) / len(top_k)

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.persist_path
        if not target:
            return
        import json

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "k": self.k,
            "texts": self._texts,
            "embeddings": self._embeddings,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        import json

        payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self._texts = [str(t) for t in payload.get("texts", [])]
        self._embeddings = [[float(x) for x in emb] for emb in payload.get("embeddings", [])]

    @property
    def size(self) -> int:
        return len(self._texts)

    def __len__(self) -> int:
        return self.size


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
