"""Append-only lifecycle journal for retraining attempts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.execution.schemas import LifecycleEvent, LifecycleStatus
from ashare_quant.utils.manifest import atomic_write_json

_ORDER = {
    "CREATED": 0,
    "DATA_READY": 1,
    "TRAINING": 2,
    "ARTIFACT_VALIDATING": 3,
    "COMPLETED": 4,
    "FAILED": 4,
    "INTERRUPTED": 4,
}


class LifecycleJournal:
    def __init__(self, root: Path, training_run_id: str) -> None:
        self.root = root / training_run_id
        self.training_run_id = training_run_id

    def append(self, status: LifecycleStatus, message: str | None = None) -> LifecycleEvent:
        events = self.events()
        retry = bool(
            events and events[-1].status in {"FAILED", "INTERRUPTED"} and status == "CREATED"
        )
        if events and not retry and _ORDER[status] < _ORDER[events[-1].status]:
            raise DataValidationError("retraining lifecycle cannot move backwards")
        if events and not retry and events[-1].status in {"COMPLETED", "FAILED", "INTERRUPTED"}:
            if events[-1].status == status and events[-1].message == message:
                return events[-1]
            raise DataValidationError("retraining lifecycle is already terminal")
        event = LifecycleEvent(
            training_run_id=self.training_run_id,
            sequence=len(events),
            status=status,
            created_at=datetime.now(UTC).isoformat(),
            message=message,
        )
        path = self.root / "events" / f"{event.sequence:03d}_{status}.json"
        atomic_write_json(path, event.model_dump(mode="json"))
        return event

    def events(self) -> tuple[LifecycleEvent, ...]:
        paths = sorted((self.root / "events").glob("*.json"))
        events: list[LifecycleEvent] = []
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            events.append(LifecycleEvent.model_validate(payload))
        if [event.sequence for event in events] != list(range(len(events))):
            raise DataValidationError("retraining lifecycle sequence is invalid")
        return tuple(events)
