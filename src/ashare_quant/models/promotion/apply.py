"""Controlled atomic application of an approved promotion request."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.apply_validation import validate_apply_preconditions
from ashare_quant.models.promotion.champion_history import (
    build_champion_assignment,
    publish_champion_assignment,
)
from ashare_quant.models.promotion.registry_versions import (
    build_promoted_registry,
    load_registry_records,
    publish_registry_versions,
    restore_registry_atomically,
    switch_registry_atomically,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import DEFAULT_PRODUCTION_LOCK_PATH, production_lock
from ashare_quant.utils.manifest import atomic_write_json


class ApplyTransition(BaseModel):
    """Immutable state transition separate from the original request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    transition_id: str
    apply_id: str
    request_id: str
    state: Literal["APPLY_PENDING", "PROMOTED"]
    registry_version_id: str | None
    created_at: str


class ApplyManifest(BaseModel):
    """Commit marker proving every state-changing artifact was published."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_apply"] = "promotion_apply"
    apply_id: str
    request_id: str
    status: Literal["PROMOTED"] = "PROMOTED"
    candidate_model_id: str
    previous_champion_model_id: str
    approval_event_id: str
    approval_event_hash: str = Field(min_length=64, max_length=64)
    registry_version_id: str
    registry_file_hash: str = Field(min_length=64, max_length=64)
    champion_assignment_id: str
    champion_history_hash: str = Field(min_length=64, max_length=64)
    deployment_contract_hash: str = Field(min_length=64, max_length=64)
    activated_at: str
    manifest_written_last: Literal[True] = True


@dataclass(frozen=True, slots=True)
class PromotionApplyResult:
    """Operator-facing apply status."""

    request_id: str
    status: str
    apply_id: str | None
    model_id: str | None
    previous_champion_model_id: str | None
    registry_version_id: str | None
    champion_assignment_id: str | None
    manifest_path: Path | None
    idempotent: bool = False


class PromotionApplyService:
    """Apply an approved request under production then registry lock ordering."""

    def __init__(
        self,
        *,
        models_root: Path,
        reports_root: Path,
        production_lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.models_root = models_root
        self.reports_root = reports_root
        self.production_lock_path = production_lock_path
        self.registry_lock_path = models_root / ".registry.lock"
        self.clock = clock or (lambda: datetime.now(UTC))

    def apply(self, request_id: str) -> PromotionApplyResult:
        """Validate and commit one registry transition, never automatically."""

        existing = self._committed(request_id)
        if existing is not None:
            return _result(*existing, idempotent=True)
        command = f"ashare-quant models promotion apply --request-id {request_id}"
        with production_lock(self.production_lock_path, command=command):
            with production_lock(self.registry_lock_path, command=command):
                existing = self._committed(request_id)
                if existing is not None:
                    return _result(*existing, idempotent=True)
                self._recover_interrupted_switch(request_id)
                context = validate_apply_preconditions(
                    request_id=request_id,
                    models_root=self.models_root,
                    reports_root=self.reports_root,
                    now=self.clock(),
                )
                apply_identity = canonical_payload_hash(
                    {
                        "request_id": request_id,
                        "request_manifest_hash": file_sha256(
                            context.bundle.output_dir / "manifest.json"
                        ),
                        "approval_event_id": context.approval_event.event_id,
                        "approval_event_hash": context.approval_event_hash,
                        "registry_hash": context.registry_hash,
                        "deployment_contract_hash": (
                            context.bundle.contract.deployment_contract_hash
                        ),
                    }
                )
                apply_id = f"apply_{apply_identity[:24]}"
                apply_dir = context.bundle.output_dir / "apply" / apply_id
                activated_at = (
                    self._pending_activated_at(apply_dir)
                    or self.clock().astimezone(UTC).isoformat()
                )
                registry_version_id, registry_payload, _ = build_promoted_registry(
                    records=tuple(load_registry_records(self.models_root / "registry.json")),
                    candidate_model_id=context.candidate.model_id,
                    current_champion_model_id=context.champion.model_id,
                    parent_registry_hash=context.registry_hash,
                    request_id=request_id,
                    approval_event_id=context.approval_event.event_id,
                    activated_at=activated_at,
                )
                pending = ApplyTransition(
                    transition_id=f"{apply_id}_apply_pending",
                    apply_id=apply_id,
                    request_id=request_id,
                    state="APPLY_PENDING",
                    registry_version_id=registry_version_id,
                    created_at=activated_at,
                )
                _publish_payload(apply_dir / "apply_pending.json", pending.model_dump(mode="json"))
                old_version, new_version = publish_registry_versions(
                    models_root=self.models_root,
                    old_registry_path=self.models_root / "registry.json",
                    registry_version_id=registry_version_id,
                    new_payload=registry_payload,
                )
                assignment = build_champion_assignment(
                    deployment_slot=context.bundle.request.current_champion_assignment.deployment_slot,
                    model_id=context.candidate.model_id,
                    previous_champion_model_id=context.champion.model_id,
                    promotion_request_id=request_id,
                    approval_event_id=context.approval_event.event_id,
                    registry_version_id=registry_version_id,
                    activated_at=activated_at,
                )
                registry_switched = False
                commit_written = False
                history_path = (
                    self.models_root
                    / "champion_history"
                    / f"{assignment.champion_assignment_id}.json"
                )
                history_preexisting = history_path.exists()
                promoted_path = apply_dir / "promoted.json"
                promoted_preexisting = promoted_path.exists()
                try:
                    switch_registry_atomically(self.models_root / "registry.json", new_version)
                    registry_switched = True
                    history_path = publish_champion_assignment(self.models_root, assignment)
                    promoted = ApplyTransition(
                        transition_id=f"{apply_id}_promoted",
                        apply_id=apply_id,
                        request_id=request_id,
                        state="PROMOTED",
                        registry_version_id=registry_version_id,
                        created_at=activated_at,
                    )
                    _publish_payload(promoted_path, promoted.model_dump(mode="json"))
                    manifest = ApplyManifest(
                        apply_id=apply_id,
                        request_id=request_id,
                        candidate_model_id=context.candidate.model_id,
                        previous_champion_model_id=context.champion.model_id,
                        approval_event_id=context.approval_event.event_id,
                        approval_event_hash=context.approval_event_hash,
                        registry_version_id=registry_version_id,
                        registry_file_hash=file_sha256(new_version),
                        champion_assignment_id=assignment.champion_assignment_id,
                        champion_history_hash=file_sha256(history_path),
                        deployment_contract_hash=(context.bundle.contract.deployment_contract_hash),
                        activated_at=activated_at,
                    )
                    manifest_path = apply_dir / "manifest.json"
                    atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
                    commit_written = True
                except Exception:
                    if registry_switched and not commit_written:
                        restore_registry_atomically(self.models_root / "registry.json", old_version)
                    if not commit_written and not history_preexisting:
                        history_path.unlink(missing_ok=True)
                    if not commit_written and not promoted_preexisting:
                        promoted_path.unlink(missing_ok=True)
                    raise
                return _result(manifest, manifest_path)

    def status(self, request_id: str) -> PromotionApplyResult:
        """Return committed, pending, or missing apply state without mutation."""

        committed = self._committed(request_id)
        if committed is not None:
            manifest, manifest_path = committed
            result = _result(manifest, manifest_path, idempotent=True)
            if file_sha256(self.models_root / "registry.json") != manifest.registry_file_hash:
                return dataclass_replace(result, status="INVALID")
            records = load_registry_records(self.models_root / "registry.json")
            champion = next((item for item in records if item.status == "champion"), None)
            if champion is None or champion.model_id != manifest.candidate_model_id:
                return dataclass_replace(result, status="INVALID")
            return result
        apply_root = self.models_root / "promotion_requests" / request_id / "apply"
        pending = sorted(apply_root.glob("*/apply_pending.json")) if apply_root.exists() else []
        return PromotionApplyResult(
            request_id=request_id,
            status="APPLY_PENDING" if pending else "NOT_APPLIED",
            apply_id=pending[0].parent.name if pending else None,
            model_id=None,
            previous_champion_model_id=None,
            registry_version_id=None,
            champion_assignment_id=None,
            manifest_path=None,
        )

    def _committed(self, request_id: str) -> tuple[ApplyManifest, Path] | None:
        root = self.models_root / "promotion_requests" / request_id / "apply"
        manifests = sorted(root.glob("*/manifest.json")) if root.exists() else []
        if not manifests:
            return None
        if len(manifests) != 1:
            raise DataValidationError("promotion request has multiple apply commit markers")
        try:
            manifest = ApplyManifest.model_validate(_load_json(manifests[0]))
        except ValidationError as error:
            raise DataValidationError(f"invalid promotion apply manifest: {error}") from error
        if manifest.request_id != request_id:
            raise DataValidationError("promotion apply manifest request identity differs")
        history = self.models_root / "champion_history" / f"{manifest.champion_assignment_id}.json"
        version = self.models_root / "registry_versions" / f"{manifest.registry_version_id}.json"
        if file_sha256(history) != manifest.champion_history_hash:
            raise DataValidationError("Champion history hash differs from apply manifest")
        if file_sha256(version) != manifest.registry_file_hash:
            raise DataValidationError("registry version hash differs from apply manifest")
        return manifest, manifests[0]

    def _recover_interrupted_switch(self, request_id: str) -> None:
        """Restore the approved parent registry after a process-level interruption."""

        apply_root = self.models_root / "promotion_requests" / request_id / "apply"
        pending_paths = (
            sorted(apply_root.glob("*/apply_pending.json")) if apply_root.exists() else []
        )
        if not pending_paths:
            return
        if len(pending_paths) != 1:
            raise DataValidationError("promotion request has conflicting APPLY_PENDING states")
        try:
            pending = ApplyTransition.model_validate(_load_json(pending_paths[0]))
        except ValidationError as error:
            raise DataValidationError(f"invalid APPLY_PENDING transition: {error}") from error
        if pending.registry_version_id is None:
            raise DataValidationError("APPLY_PENDING transition lacks registry version identity")
        version_path = (
            self.models_root / "registry_versions" / f"{pending.registry_version_id}.json"
        )
        version = _load_json(version_path)
        parent_hash = version.get("parent_registry_hash")
        if not isinstance(parent_hash, str) or len(parent_hash) != 64:
            raise DataValidationError("pending registry version lacks parent registry hash")
        registry_path = self.models_root / "registry.json"
        current_hash = file_sha256(registry_path)
        if current_hash == parent_hash:
            return
        if current_hash != file_sha256(version_path):
            raise DataValidationError(
                "registry differs from both pending and approved parent versions"
            )
        parent_path = (
            self.models_root / "registry_versions" / f"registry_source_{parent_hash[:24]}.json"
        )
        if file_sha256(parent_path) != parent_hash:
            raise DataValidationError("pending apply parent registry backup is invalid")
        restore_registry_atomically(registry_path, parent_path)

    @staticmethod
    def _pending_activated_at(apply_dir: Path) -> str | None:
        path = apply_dir / "apply_pending.json"
        if not path.is_file():
            return None
        try:
            pending = ApplyTransition.model_validate(_load_json(path))
        except ValidationError as error:
            raise DataValidationError(f"invalid APPLY_PENDING transition: {error}") from error
        return pending.created_at


def _publish_payload(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != payload:
            raise DataValidationError(f"immutable apply artifact differs: {path}")
        return
    atomic_write_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"promotion apply artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid promotion apply JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"promotion apply artifact must contain an object: {path}")
    return payload


def _result(
    manifest: ApplyManifest,
    manifest_path: Path,
    idempotent: bool = False,
) -> PromotionApplyResult:
    return PromotionApplyResult(
        request_id=manifest.request_id,
        status=manifest.status,
        apply_id=manifest.apply_id,
        model_id=manifest.candidate_model_id,
        previous_champion_model_id=manifest.previous_champion_model_id,
        registry_version_id=manifest.registry_version_id,
        champion_assignment_id=manifest.champion_assignment_id,
        manifest_path=manifest_path,
        idempotent=idempotent,
    )


def dataclass_replace(value: PromotionApplyResult, *, status: str) -> PromotionApplyResult:
    """Replace only status without importing mutable lifecycle helpers."""

    return PromotionApplyResult(
        request_id=value.request_id,
        status=status,
        apply_id=value.apply_id,
        model_id=value.model_id,
        previous_champion_model_id=value.previous_champion_model_id,
        registry_version_id=value.registry_version_id,
        champion_assignment_id=value.champion_assignment_id,
        manifest_path=value.manifest_path,
        idempotent=value.idempotent,
    )
