"""TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from haireder.taxonomy import LABELS


def pair_to_text(row: dict[str, Any]) -> str:
    """Convert a requirement pair row to a classifier input string."""
    issue_i = row["issue_i"]
    issue_j = row["issue_j"]
    return (
        "Requirement A:\n"
        f"{issue_i.get('text', '')}\n\n"
        "Requirement B:\n"
        f"{issue_j.get('text', '')}"
    )


class OrderedPairTfidf(TfidfVectorizer):
    """Fit one vocabulary to endpoint texts and preserve the two vector positions."""
    def fit_transform(self, pairs, y=None):
        sides = [text for pair in pairs for text in pair]
        super().fit_transform(sides)
        return self.transform(pairs)

    def transform(self, pairs):
        from scipy.sparse import hstack
        return hstack([super(OrderedPairTfidf, self).transform([p[index] for p in pairs])
                       for index in (0, 1)], format="csr")


@dataclass
class TfidfLogRegBaseline:
    """A small supervised baseline for requirement dependency classification."""

    max_features: int = 50000
    ngram_min: int = 1
    ngram_max: int = 2
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    random_state: int = 42

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    OrderedPairTfidf(
                        max_features=self.max_features,
                        ngram_range=(self.ngram_min, self.ngram_max),
                        min_df=1,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        self.trained_labels_: list[str] = []

    def fit(self, rows: list[dict[str, Any]]) -> "TfidfLogRegBaseline":
        """Fit the baseline from pair rows."""
        texts = [(row["issue_i"]["text"], row["issue_j"]["text"]) for row in rows]
        labels = [row["label"] for row in rows]
        self.pipeline.fit(texts, labels)
        self.trained_labels_ = [str(label) for label in self.pipeline.named_steps["logreg"].classes_]
        return self

    def predict_rows(self, rows: list[dict[str, Any]], method: str = "tfidf_logreg") -> list[dict[str, Any]]:
        """Predict labels and return HAI-ReDeR prediction rows."""
        texts = [(row["issue_i"]["text"], row["issue_j"]["text"]) for row in rows]
        predicted = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        trained_labels = [str(label) for label in self.pipeline.named_steps["logreg"].classes_]
        outputs: list[dict[str, Any]] = []
        for row, label, proba in zip(rows, predicted, probabilities):
            proba_by_label = {candidate: 0.0 for candidate in LABELS}
            for trained_label, value in zip(trained_labels, proba):
                proba_by_label[str(trained_label)] = float(value)
            confidence = max(proba_by_label.values()) if proba_by_label else 0.0
            outputs.append(
                {
                    "pair_id": row["pair_id"],
                    "repo": row["repo"],
                    "window": row["window"],
                    "split": row["split"],
                    "gold_label": row["label"],
                    "predicted_label": str(label),
                    "confidence": float(confidence),
                    "probabilities": proba_by_label,
                    "method": method,
                    "issue_i": row["issue_i"],
                    "issue_j": row["issue_j"],
                }
            )
        return outputs
