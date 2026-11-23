"""Canonical HAI-ReDeR dependency labels and mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haireder.utils.config import resolve_path
from haireder.utils.config import load_yaml


LABEL_NONE = "None"
LABEL_GENERAL = "General Relation"
LABEL_DUPLICATION = "Duplication"
LABEL_COMPOSITION = "Composition"
LABEL_TEMPORAL = "Temporal/Causal"
LABEL_WORKFLOW = "Workflow"

LABELS = [
    LABEL_NONE,
    LABEL_GENERAL,
    LABEL_DUPLICATION,
    LABEL_COMPOSITION,
    LABEL_TEMPORAL,
    LABEL_WORKFLOW,
]


@dataclass(frozen=True)
class Taxonomy:
    """In-memory taxonomy mapping."""

    labels: list[str]
    label_definitions: dict[str, str]
    normalized_dependency_type_to_label: dict[str, str]

    def map_normalized_type(self, dependency_type: str) -> str | None:
        """Map a normalized Public Jira dependency type to a HAI-ReDeR label."""
        return self.normalized_dependency_type_to_label.get(dependency_type)

    def require_label(self, label: str) -> str:
        """Validate and return a canonical label."""
        if label not in self.labels:
            raise ValueError(f"Unknown HAI-ReDeR label: {label}")
        return label


def load_taxonomy(path: str | Path = "data/external/taxonomy_mapping/haireder_dependency_taxonomy.yaml") -> Taxonomy:
    """Load the taxonomy YAML file."""
    payload: dict[str, Any] = load_yaml(resolve_path(path))
    labels = list(payload.get("labels", []))
    if labels != LABELS:
        raise ValueError(f"Taxonomy labels do not match canonical order: {labels}")
    mapping = dict(payload.get("normalized_dependency_type_to_label", {}))
    invalid_labels = sorted({label for label in mapping.values() if label not in labels})
    if invalid_labels:
        raise ValueError(f"Mapping contains invalid labels: {invalid_labels}")
    return Taxonomy(
        labels=labels,
        label_definitions=dict(payload.get("label_definitions", {})),
        normalized_dependency_type_to_label=mapping,
    )
