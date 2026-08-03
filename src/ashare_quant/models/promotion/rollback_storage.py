"""Append-only atomic storage for rollback governance artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.rollback_schema import (
    RollbackApprovalEvent,
    RollbackApprovalManifest,
    RollbackRequest,
    RollbackRequestManifest,
    RollbackValidationManifest,
    RollbackValidationResult,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class RollbackBundle:
    request: RollbackRequest
    manifest: RollbackRequestManifest
    output_dir: Path


@dataclass(frozen=True, slots=True)
class StoredRollbackApproval:
    event: RollbackApprovalEvent
    manifest: RollbackApprovalManifest
    event_path: Path
    manifest_path: Path


class RollbackStorage:
    """Store rollback requests, validations, and human decisions immutably."""

    def __init__(self, models_root: Path) -> None:
        self.root = models_root / "rollback_requests"

    def output_dir(self, request_id: str) -> Path:
        return self.root / request_id

    def publish_request(
        self, request: RollbackRequest, identity_hash: str
    ) -> tuple[RollbackBundle, bool]:
        existing = self.read(request.request_id)
        if existing is not None:
            if existing.manifest.identity_hash != identity_hash:
                raise DataValidationError("immutable rollback request identity differs")
            return existing, True
        output = self.output_dir(request.request_id)
        if output.exists():
            raise DataValidationError("incomplete rollback request cannot be overwritten")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=f".{request.request_id}."))
        try:
            request_path = staging / "request.json"
            atomic_write_json(request_path, request.model_dump(mode="json"))
            manifest = RollbackRequestManifest(
                request_id=request.request_id,
                identity_hash=identity_hash,
                request_sha256=file_sha256(request_path),
                created_at=request.created_at,
            )
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, output)
            bundle = self.read(request.request_id)
            if bundle is None:
                raise DataValidationError("published rollback request is incomplete")
            return bundle, False
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def read(self, request_id: str) -> RollbackBundle | None:
        output = self.output_dir(request_id)
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = RollbackRequestManifest.model_validate(_load_json(manifest_path))
            request = RollbackRequest.model_validate(_load_json(output / "request.json"))
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback request schema: {error}") from error
        if request.request_id != request_id or manifest.request_id != request_id:
            raise DataValidationError("rollback request identity is inconsistent")
        if file_sha256(output / "request.json") != manifest.request_sha256:
            raise DataValidationError("rollback request hash differs from manifest")
        identity = canonical_payload_hash(
            request.model_dump(mode="json", exclude={"request_id", "created_at"})
        )
        if manifest.identity_hash != identity:
            raise DataValidationError("rollback request logical identity differs")
        if request.request_id != f"rollback_{identity[:24]}":
            raise DataValidationError("rollback request_id is not derived from identity")
        return RollbackBundle(request, manifest, output)

    def publish_validation(
        self, result: RollbackValidationResult
    ) -> tuple[RollbackValidationResult, bool]:
        output = self.output_dir(result.request_id) / "validation"
        existing = self.read_validation(result.request_id)
        identity = canonical_payload_hash(result.model_dump(mode="json", exclude={"validated_at"}))
        if existing is not None:
            current, manifest = existing
            if manifest.identity_hash != identity:
                raise DataValidationError("immutable rollback validation identity differs")
            return current, True
        if output.exists():
            raise DataValidationError("incomplete rollback validation cannot be overwritten")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=".validation."))
        try:
            result_path = staging / "validation_result.json"
            atomic_write_json(result_path, result.model_dump(mode="json"))
            manifest = RollbackValidationManifest(
                request_id=result.request_id,
                identity_hash=identity,
                result_sha256=file_sha256(result_path),
                created_at=result.validated_at,
            )
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, output)
            stored = self.read_validation(result.request_id)
            if stored is None:
                raise DataValidationError("published rollback validation is incomplete")
            return stored[0], False
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def read_validation(
        self, request_id: str
    ) -> tuple[RollbackValidationResult, RollbackValidationManifest] | None:
        output = self.output_dir(request_id) / "validation"
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = RollbackValidationManifest.model_validate(_load_json(manifest_path))
            result = RollbackValidationResult.model_validate(
                _load_json(output / "validation_result.json")
            )
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback validation schema: {error}") from error
        if result.request_id != request_id or manifest.request_id != request_id:
            raise DataValidationError("rollback validation identity is inconsistent")
        if file_sha256(output / "validation_result.json") != manifest.result_sha256:
            raise DataValidationError("rollback validation result hash differs")
        identity = canonical_payload_hash(result.model_dump(mode="json", exclude={"validated_at"}))
        if identity != manifest.identity_hash:
            raise DataValidationError("rollback validation logical identity differs")
        return result, manifest

    def publish_approval(
        self,
        event: RollbackApprovalEvent,
        *,
        identity_hash: str,
        policy_hash: str,
    ) -> tuple[StoredRollbackApproval, bool]:
        existing = self.list_approvals(event.request_id)
        if existing:
            if existing[0].manifest.event_identity_hash != identity_hash:
                raise DataValidationError("rollback request has a different review decision")
            return existing[0], True
        root = self.output_dir(event.request_id) / "approval_events"
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise DataValidationError("incomplete rollback approval blocks publication")
        event_path = root / f"{event.event_id}.json"
        manifest_path = root / f"{event.event_id}.manifest.json"
        event_temp = _temp_path(root, f".{event.event_id}.event.")
        manifest_temp = _temp_path(root, f".{event.event_id}.manifest.")
        try:
            atomic_write_json(event_temp, event.model_dump(mode="json"))
            manifest = RollbackApprovalManifest(
                event_id=event.event_id,
                request_id=event.request_id,
                event_identity_hash=identity_hash,
                event_file_sha256=file_sha256(event_temp),
                policy_hash=policy_hash,
                created_at=event.created_at,
            )
            atomic_write_json(manifest_temp, manifest.model_dump(mode="json"))
            os.replace(event_temp, event_path)
            os.replace(manifest_temp, manifest_path)
            stored = self.read_approval(event.request_id, event.event_id)
            if stored is None:
                raise DataValidationError("published rollback approval is incomplete")
            return stored, False
        finally:
            event_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)

    def read_approval(self, request_id: str, event_id: str) -> StoredRollbackApproval | None:
        root = self.output_dir(request_id) / "approval_events"
        event_path = root / f"{event_id}.json"
        manifest_path = root / f"{event_id}.manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            event = RollbackApprovalEvent.model_validate(_load_json(event_path))
            manifest = RollbackApprovalManifest.model_validate(_load_json(manifest_path))
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback approval schema: {error}") from error
        if event.request_id != request_id or manifest.request_id != request_id:
            raise DataValidationError("rollback approval request identity differs")
        if event.event_id != event_id or manifest.event_id != event_id:
            raise DataValidationError("rollback approval event identity differs")
        if file_sha256(event_path) != manifest.event_file_sha256:
            raise DataValidationError("rollback approval event hash differs")
        if rollback_approval_identity(event, manifest.policy_hash) != manifest.event_identity_hash:
            raise DataValidationError("rollback approval logical identity differs")
        if event.event_id != f"review_{manifest.event_identity_hash[:24]}":
            raise DataValidationError("rollback approval event_id is not derived from identity")
        return StoredRollbackApproval(event, manifest, event_path, manifest_path)

    def list_approvals(self, request_id: str) -> tuple[StoredRollbackApproval, ...]:
        root = self.output_dir(request_id) / "approval_events"
        if not root.exists():
            return ()
        result: list[StoredRollbackApproval] = []
        for path in sorted(root.glob("*.manifest.json")):
            event_id = path.name.removesuffix(".manifest.json")
            stored = self.read_approval(request_id, event_id)
            if stored is not None:
                result.append(stored)
        return tuple(result)


def rollback_approval_identity(event: RollbackApprovalEvent, policy_hash: str) -> str:
    return canonical_payload_hash(
        {
            **event.model_dump(mode="json", exclude={"event_id", "created_at", "expires_at"}),
            "policy_hash": policy_hash,
        }
    )


def _temp_path(root: Path, prefix: str) -> Path:
    handle, value = tempfile.mkstemp(dir=root, prefix=prefix)
    os.close(handle)
    path = Path(value)
    path.unlink()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"rollback artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid rollback JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"rollback artifact must contain an object: {path}")
    return payload
