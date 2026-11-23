from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from haireder.utils.checkpoint import CheckpointManager  # noqa: E402


CHECKPOINT_PATH = "outputs/pipeline_checkpoint.json"


def _base_checkpoint_data(stage_overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    stages = stage_overrides or {}
    return {
        "pipeline_id": "formal_full_pipeline",
        "started_at": "2026-07-07T10:00:00",
        "updated_at": "2026-07-07T14:30:00",
        "experiment_config": "configs/experiments/formal_full_pipeline.yaml",
        "stages": stages,
    }


def test_checkpoint_new_has_no_completed_stages(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)

    assert manager.load() == {}
    assert manager.list_completed_stages() == []
    assert manager.is_stage_complete("data_prep") is False
    assert manager.get_resume_point()["stage"] == "data_prep"


def test_checkpoint_mark_complete_and_verify(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)

    manager.mark_stage_started("data_prep")
    manager.mark_stage_complete("data_prep", {"train_rows": 50000, "test_rows": 10000})

    checkpoint = manager.load()
    stage = checkpoint["stages"]["data_prep"]

    assert stage["status"] == "completed"
    assert stage["metadata"] == {"train_rows": 50000, "test_rows": 10000}
    assert stage["completed_at"]
    assert stage["duration_seconds"] is not None
    assert stage["duration_seconds"] >= 0
    assert manager.is_stage_complete("data_prep") is True
    assert manager.list_completed_stages() == ["data_prep"]


def test_checkpoint_load_after_save(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)
    data = _base_checkpoint_data(
        {
            "data_prep": {
                "status": "completed",
                "started_at": "2026-07-07T10:00:00",
                "completed_at": "2026-07-07T10:02:00",
                "duration_seconds": 120.5,
                "metadata": {"train_rows": 50000, "test_rows": 10000},
            }
        }
    )

    manager.save(data)

    assert manager.load() == data


def test_checkpoint_stage_failed(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)

    manager.mark_stage_started("training")
    manager.mark_stage_failed("training", "trainer crashed")

    checkpoint = manager.load()
    stage = checkpoint["stages"]["training"]

    assert stage["status"] == "failed"
    assert stage["error"] == "trainer crashed"
    assert manager.is_stage_complete("training") is False
    assert manager.list_completed_stages() == []


def test_checkpoint_mark_stage_progress_keeps_stage_resumable(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)

    manager.mark_stage_started("online_learning")
    manager.mark_stage_progress(
        "online_learning",
        {
            "completed_windows": ["W5"],
            "current_window": "W5",
            "next_window": "W6",
            "sub_stage": "window",
        },
    )

    checkpoint = manager.load()
    stage = checkpoint["stages"]["online_learning"]
    resume_point = manager.get_resume_point()

    assert stage["status"] == "in_progress"
    assert manager.is_stage_complete("online_learning") is False
    assert resume_point["stage"] == "online_learning"
    assert resume_point["window"] == "W6"


def test_checkpoint_atomic_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / CHECKPOINT_PATH
    manager = CheckpointManager(checkpoint_path)
    calls: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def recording_replace(self: Path, target: Path) -> Path:
        calls.append((self, target))
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", recording_replace)

    manager.save(_base_checkpoint_data())

    assert calls, "expected atomic save to rename a temporary file"
    source, target = calls[0]
    assert source.name == "pipeline_checkpoint.json.tmp"
    assert target == checkpoint_path.resolve()
    assert checkpoint_path.exists()
    assert not source.exists()


def test_checkpoint_resume_point(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / CHECKPOINT_PATH)
    data = _base_checkpoint_data(
        {
            "data_prep": {
                "status": "completed",
                "started_at": "2026-07-07T10:00:00",
                "completed_at": "2026-07-07T10:02:00",
                "duration_seconds": 120.0,
                "metadata": {"train_rows": 50000},
            },
            "training": {
                "status": "completed",
                "started_at": "2026-07-07T10:02:00",
                "completed_at": "2026-07-07T11:30:00",
                "duration_seconds": 5280.0,
                "metadata": {"adapter_dir": "models/adapters/formal/proposer"},
            },
            "evaluation": {
                "status": "completed",
                "started_at": "2026-07-07T11:30:00",
                "completed_at": "2026-07-07T12:00:00",
                "duration_seconds": 1800.0,
                "metadata": {"validation_window": "W4"},
            },
            "online_learning": {
                "status": "in_progress",
                "started_at": "2026-07-07T12:00:00",
                "metadata": {
                    "completed_windows": ["W5", "W6"],
                    "next_window": "W7",
                    "sub_stage": "window",
                },
            },
        }
    )
    manager.save(data)

    resume_point = manager.get_resume_point()

    assert resume_point["stage"] == "online_learning"
    assert resume_point["sub_stage"] == "window"
    assert resume_point["window"] == "W7"
    assert resume_point["completed_stages"] == ["data_prep", "training", "evaluation"]
