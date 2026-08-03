"""Append-only atomic storage for immutable human review events."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.approval_schema import (
    ApprovalEvent,
    ApprovalEventManifest,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class StoredApprovalEvent:
    """One complete event and its completion manifest."""

    event: ApprovalEvent
    manifest: ApprovalEventManifest
    event_path: Path
    manifest_path: Path


class ApprovalEventStorage:
    """Store at most one immutable terminal decision for each request."""

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root

    def event_root(self, request_id: str) -> Path:
        """Return the append-only event directory for one request."""

        return self.models_root / "promotion_requests" / request_id / "approval_events"

    def publish(
        self,
        *,
        event: ApprovalEvent,
        event_identity_hash: str,
        policy_hash: str,
    ) -> tuple[StoredApprovalEvent, bool]:
        """Write event first and its completion manifest last."""

        root = self.event_root(event.request_id)
        existing_events = self.list_events(event.request_id)
        if existing_events:
            existing = existing_events[0]
            if existing.manifest.event_identity_hash != event_identity_hash:
                raise DataValidationError(
                    "promotion request already has a different immutable review decision"
                )
            return existing, True
        root.mkdir(parents=True, exist_ok=True)
        incomplete = [
            path
            for path in root.glob("*.json")
            if not path.name.endswith(".manifest.json")
            and not path.with_name(f"{path.stem}.manifest.json").is_file()
        ]
        if incomplete:
            raise DataValidationError(
                f"incomplete approval event blocks append-only publication: {incomplete[0]}"
            )
        event_path = root / f"{event.event_id}.json"
        manifest_path = root / f"{event.event_id}.manifest.json"
        if event_path.exists() or manifest_path.exists():
            raise DataValidationError(
                "incomplete approval event cannot be overwritten or completed implicitly"
            )
        event_temp = _temporary_path(root, f".{event.event_id}.event.")
        manifest_temp = _temporary_path(root, f".{event.event_id}.manifest.")
        try:
            atomic_write_json(event_temp, event.model_dump(mode="json"))
            manifest = ApprovalEventManifest(
                event_id=event.event_id,
                request_id=event.request_id,
                event_identity_hash=event_identity_hash,
                event_file_sha256=file_sha256(event_temp),
                policy_hash=policy_hash,
                created_at=event.created_at,
            )
            atomic_write_json(manifest_temp, manifest.model_dump(mode="json"))
            os.replace(event_temp, event_path)
            os.replace(manifest_temp, manifest_path)
            stored = self.read(event.request_id, event.event_id)
            if stored is None:
                raise DataValidationError("published approval event is incomplete")
            return stored, False
        finally:
            event_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)

    def read(self, request_id: str, event_id: str) -> StoredApprovalEvent | None:
        """Read a complete event; a missing manifest means incomplete publication."""

        root = self.event_root(request_id)
        event_path = root / f"{event_id}.json"
        manifest_path = root / f"{event_id}.manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = ApprovalEventManifest.model_validate(_load_json(manifest_path))
            event = ApprovalEvent.model_validate(_load_json(event_path))
        except ValidationError as error:
            raise DataValidationError(f"invalid approval event schema: {error}") from error
        if manifest.request_id != request_id or event.request_id != request_id:
            raise DataValidationError("approval event request identity is inconsistent")
        if manifest.event_id != event_id or event.event_id != event_id:
            raise DataValidationError("approval event file identity is inconsistent")
        if file_sha256(event_path) != manifest.event_file_sha256:
            raise DataValidationError("approval event hash differs from completion manifest")
        if approval_event_identity(event, manifest.policy_hash) != manifest.event_identity_hash:
            raise DataValidationError("approval event logical identity hash is invalid")
        if event.event_id != f"review_{manifest.event_identity_hash[:24]}":
            raise DataValidationError("approval event_id is not derived from logical identity")
        return StoredApprovalEvent(event, manifest, event_path, manifest_path)

    def list_events(self, request_id: str) -> tuple[StoredApprovalEvent, ...]:
        """Return complete events in deterministic order."""

        root = self.event_root(request_id)
        if not root.exists():
            return ()
        events: list[StoredApprovalEvent] = []
        for path in sorted(root.glob("*.manifest.json")):
            event_id = path.name.removesuffix(".manifest.json")
            stored = self.read(request_id, event_id)
            if stored is not None:
                events.append(stored)
        return tuple(events)


def approval_event_identity(event: ApprovalEvent, policy_hash: str) -> str:
    """Return the logical identity excluding timestamps and self-derived ID."""

    return canonical_payload_hash(
        {
            "event_type": event.event_type,
            "request_id": event.request_id,
            "request_hash": event.request_hash,
            "gate_result_hash": event.gate_result_hash,
            "registry_hash_at_review": event.registry_hash_at_review,
            "reviewer": event.reviewer,
            "requester": event.requester,
            "decision": event.decision,
            "comments": event.comments,
            "policy_hash": policy_hash,
        }
    )


def _temporary_path(root: Path, prefix: str) -> Path:
    handle, value = tempfile.mkstemp(dir=root, prefix=prefix)
    os.close(handle)
    path = Path(value)
    path.unlink()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"approval event artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid approval event JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"approval event artifact must contain an object: {path}")
    return payload
