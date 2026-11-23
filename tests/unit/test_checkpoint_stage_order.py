from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.utils.checkpoint import CheckpointManager  # noqa: E402


def test_checkpoint_stage_order_matches_formal_pipeline_stages() -> None:
    assert CheckpointManager.DEFAULT_STAGE_ORDER == (
        "data_prep",
        "initial_augmentation",
        "training_proposer",
        "collect_proposer_failures",
        "training_critic",
        "collect_debate_transcripts",
        "training_judge",
        "evaluation_validation",
        "confidence_weight_search",
        "calibration",
        "online_learning",
        "final_evaluation",
        "metrics",
    )
