from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.augmentation import augment_minority_classes, build_rule_based_augmented_pair, validate_augmented_pair


def make_row(pair_id: str, label: str) -> dict:
    return {
        "pair_id": pair_id,
        "repo": "Apache",
        "window": "W1",
        "split": "train",
        "issue_i": {"id": f"{pair_id}-A", "text": f"Summary: Requirement A for {label}"},
        "issue_j": {"id": f"{pair_id}-B", "text": f"Summary: Requirement B for {label}"},
        "label": label,
        "is_positive": label != "None",
    }


def test_rule_based_augmented_pair_preserves_label_and_records_seed() -> None:
    seed = make_row("pair-1", "Composition")

    augmented = build_rule_based_augmented_pair(seed, variant_index=1)
    validation = validate_augmented_pair(seed, augmented, target_label="Composition")

    assert augmented["pair_id"] == "AUG-pair-1-001"
    assert augmented["label"] == "Composition"
    assert augmented["source_seed_pair_id"] == "pair-1"
    assert validation["label_matches"] is True
    assert validation["is_valid"] is True


def test_augment_minority_classes_balances_present_positive_labels() -> None:
    rows = [
        make_row("temporal-1", "Temporal/Causal"),
        make_row("temporal-2", "Temporal/Causal"),
        make_row("composition-1", "Composition"),
        make_row("none-1", "None"),
        make_row("none-2", "None"),
        make_row("none-3", "None"),
    ]

    augmented, summary = augment_minority_classes(
        rows,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
    )

    assert len(augmented) == 1
    assert summary["generated_by_label"] == {"Composition": 1}
    assert summary["counts_after"]["Composition"] == 2
    assert "None" not in summary["target_counts"]


def test_augment_minority_classes_reuses_existing_augmented_rows() -> None:
    rows = [
        make_row("temporal-1", "Temporal/Causal"),
        make_row("temporal-2", "Temporal/Causal"),
        make_row("composition-1", "Composition"),
    ]
    first_augmented, first_summary = augment_minority_classes(
        rows,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
    )

    second_augmented, second_summary = augment_minority_classes(
        rows,
        existing_augmented_rows=first_augmented,
        max_variants_per_seed=2,
        include_none=False,
        seed=7,
    )

    assert len(first_augmented) == 1
    assert first_summary["new_generated_rows"] == 1
    assert second_augmented == first_augmented
    assert second_summary["reused_rows"] == 1
    assert second_summary["new_generated_rows"] == 0
