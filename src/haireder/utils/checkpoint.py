from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class CheckpointManager:
    DEFAULT_STAGE_ORDER = (
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
    LEGACY_STAGE_ORDER = (
        "data_prep",
        "training",
        "evaluation",
        "online_learning",
        "final_evaluation",
        "metrics",
    )

    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    def load(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {}
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in checkpoint: {self.checkpoint_path}")
        return payload

    def save(self, data: dict[str, Any]) -> None:
        payload = deepcopy(data)
        if "stages" in payload:
            payload["stages"] = self._normalize_stages(payload["stages"])
        else:
            payload["stages"] = {}

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.checkpoint_path.with_name(f"{self.checkpoint_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(self.checkpoint_path)

    def is_stage_complete(self, stage: str) -> bool:
        return self._stage_record(stage).get("status") == "completed"

    def mark_stage_started(self, stage: str) -> None:
        data = self._current_checkpoint(create_if_missing=True)
        stages = data.setdefault("stages", {})
        stage_record = dict(stages.get(stage) or {})
        if stage_record.get("status") == "completed":
            return
        now = self._now_iso()
        stage_record.setdefault("started_at", now)
        stage_record["status"] = "in_progress"
        stage_record.pop("error", None)
        stage_record.pop("failed_at", None)
        stages[stage] = stage_record
        data.setdefault("started_at", now)
        data["updated_at"] = now
        self.save(data)

    def mark_stage_complete(self, stage: str, metadata: dict[str, Any] | None = None) -> None:
        data = self._current_checkpoint(create_if_missing=True)
        stages = data.setdefault("stages", {})
        stage_record = dict(stages.get(stage) or {})
        now = self._now_iso()
        started_at = str(stage_record.get("started_at") or data.get("started_at") or now)
        stage_record["started_at"] = started_at
        stage_record["completed_at"] = now
        stage_record["status"] = "completed"
        stage_record["duration_seconds"] = self._duration_seconds(started_at, now)
        if metadata is not None:
            merged_metadata = dict(stage_record.get("metadata") or {})
            merged_metadata.update(metadata)
            stage_record["metadata"] = merged_metadata
        elif "metadata" not in stage_record:
            stage_record["metadata"] = {}
        stage_record.pop("error", None)
        stage_record.pop("failed_at", None)
        stages[stage] = stage_record
        data.setdefault("started_at", now)
        data["updated_at"] = now
        self.save(data)

    def mark_stage_progress(self, stage: str, metadata: dict[str, Any] | None = None) -> None:
        data = self._current_checkpoint(create_if_missing=True)
        stages = data.setdefault("stages", {})
        stage_record = dict(stages.get(stage) or {})
        now = self._now_iso()
        started_at = str(stage_record.get("started_at") or data.get("started_at") or now)
        stage_record["started_at"] = started_at
        stage_record["status"] = "in_progress"
        stage_record["updated_at"] = now
        if metadata is not None:
            merged_metadata = dict(stage_record.get("metadata") or {})
            merged_metadata.update(metadata)
            stage_record["metadata"] = merged_metadata
        elif "metadata" not in stage_record:
            stage_record["metadata"] = {}
        stage_record.pop("completed_at", None)
        stage_record.pop("duration_seconds", None)
        stage_record.pop("error", None)
        stage_record.pop("failed_at", None)
        stages[stage] = stage_record
        data.setdefault("started_at", now)
        data["updated_at"] = now
        self.save(data)

    def mark_stage_failed(self, stage: str, error: str) -> None:
        data = self._current_checkpoint(create_if_missing=True)
        stages = data.setdefault("stages", {})
        stage_record = dict(stages.get(stage) or {})
        now = self._now_iso()
        started_at = str(stage_record.get("started_at") or data.get("started_at") or now)
        stage_record["started_at"] = started_at
        stage_record["failed_at"] = now
        stage_record["status"] = "failed"
        stage_record["error"] = str(error)
        stage_record["duration_seconds"] = self._duration_seconds(started_at, now)
        stage_record.setdefault("metadata", {})
        stages[stage] = stage_record
        data.setdefault("started_at", now)
        data["updated_at"] = now
        self.save(data)

    def get_resume_point(self) -> dict[str, Any]:
        data = self.load()
        if not data:
            return {
                "stage": self.DEFAULT_STAGE_ORDER[0],
                "sub_stage": "start",
                "window": None,
                "completed_stages": [],
                "status": "not_started",
            }

        stages = data.get("stages") or {}
        completed_stages = self.list_completed_stages(data)
        for stage in self._ordered_stage_names(stages):
            stage_record = dict(stages.get(stage) or {})
            status = str(stage_record.get("status") or "")
            if status == "completed":
                continue
            metadata = dict(stage_record.get("metadata") or {})
            sub_stage = (
                metadata.get("sub_stage")
                or stage_record.get("sub_stage")
                or ("start" if status not in {"in_progress", "failed"} else None)
            )
            window = (
                metadata.get("next_window")
                or metadata.get("window")
                or stage_record.get("window")
                or metadata.get("current_window")
            )
            result = {
                "stage": stage,
                "sub_stage": sub_stage,
                "window": window,
                "completed_stages": completed_stages,
                "status": status or "not_started",
            }
            if "error" in stage_record:
                result["error"] = stage_record["error"]
            return result

        return {
            "stage": None,
            "sub_stage": None,
            "window": None,
            "completed_stages": completed_stages,
            "status": "complete",
        }

    def list_completed_stages(self, data: dict[str, Any] | None = None) -> list[str]:
        checkpoint = data if data is not None else self.load()
        stages = checkpoint.get("stages") or {}
        ordered = []
        for stage in self._ordered_stage_names(stages):
            if str((stages.get(stage) or {}).get("status") or "") == "completed":
                ordered.append(stage)
        return ordered

    def _current_checkpoint(self, *, create_if_missing: bool = False) -> dict[str, Any]:
        data = self.load()
        if data or not create_if_missing:
            return data
        now = self._now_iso()
        return {
            "pipeline_id": self.checkpoint_path.stem,
            "started_at": now,
            "updated_at": now,
            "stages": {},
        }

    def _stage_record(self, stage: str) -> dict[str, Any]:
        data = self.load()
        stages = data.get("stages") or {}
        stage_record = stages.get(stage)
        return dict(stage_record) if isinstance(stage_record, dict) else {}

    def _normalize_stages(self, stages: Any) -> dict[str, Any]:
        if not isinstance(stages, dict):
            return {}
        normalized: dict[str, Any] = {}
        for stage, record in stages.items():
            if not isinstance(record, dict):
                continue
            normalized[str(stage)] = self._normalize_stage_record(record)
        return normalized

    def _normalize_stage_record(self, record: dict[str, Any]) -> dict[str, Any]:
        normalized = deepcopy(record)
        if "metadata" in normalized and normalized["metadata"] is None:
            normalized["metadata"] = {}
        elif "metadata" in normalized and not isinstance(normalized["metadata"], dict):
            normalized["metadata"] = dict(normalized["metadata"])
        return normalized

    def _ordered_stage_names(self, stages: dict[str, Any]) -> list[str]:
        order = self.LEGACY_STAGE_ORDER if any(stage in stages for stage in ("training", "evaluation")) else self.DEFAULT_STAGE_ORDER
        ordered = [stage for stage in order if stage in stages]
        for stage in stages:
            if stage not in ordered:
                ordered.append(stage)
        return ordered

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _duration_seconds(self, started_at: str, finished_at: str) -> float | None:
        try:
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(finished_at)
        except ValueError:
            return None
        return max(0.0, (end - start).total_seconds())
