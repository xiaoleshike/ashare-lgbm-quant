"""Governed creation, human review, and application of Champion rollback."""

from __future__ import annotations

import getpass
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from ashare_quant.config.settings import PromotionReviewSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.registry_versions import (
    build_rollback_registry,
    load_registry_records,
    publish_registry_versions,
    restore_registry_atomically,
    switch_registry_atomically,
)
from ashare_quant.models.promotion.review_policy import ReviewPolicy, parse_timestamp
from ashare_quant.models.promotion.rollback_schema import (
    RollbackApplyManifest,
    RollbackApplyPending,
    RollbackApprovalEvent,
    RollbackChampionAssignment,
    RollbackReason,
    RollbackRequest,
    RollbackValidationResult,
)
from ashare_quant.models.promotion.rollback_storage import (
    RollbackBundle,
    RollbackStorage,
    rollback_approval_identity,
)
from ashare_quant.models.promotion.rollback_validation import (
    build_rollback_state,
    current_artifact_set_hash,
    validate_rollback_approval,
    validate_rollback_request,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import DEFAULT_PRODUCTION_LOCK_PATH, production_lock
from ashare_quant.utils.manifest import atomic_write_json

type Clock = Callable[[], datetime]
type ReviewerProvider = Callable[[], str]


@dataclass(frozen=True, slots=True)
class RollbackWorkflowResult:
    request_id: str
    status: str
    target_model_id: str
    current_champion_model_id: str
    output_dir: Path
    event_id: str | None = None
    registry_version_id: str | None = None
    champion_assignment_id: str | None = None
    idempotent: bool = False


class RollbackService:
    """Apply no state change without an immutable request and human approval."""

    def __init__(
        self,
        *,
        models_root: Path,
        settings: PromotionReviewSettings,
        production_lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
        requester_provider: ReviewerProvider | None = None,
        reviewer_provider: ReviewerProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.models_root = models_root
        self.storage = RollbackStorage(models_root)
        self.policy = ReviewPolicy(settings)
        self.production_lock_path = production_lock_path
        self.registry_lock_path = models_root / ".registry.lock"
        self.requester_provider = requester_provider or getpass.getuser
        self.reviewer_provider = reviewer_provider or getpass.getuser
        self.clock = clock or (lambda: datetime.now(UTC))

    def create(
        self,
        *,
        model_id: str,
        reason_type: str,
        reason_description: str,
        deployment_slot: str = "daily_stock_ranker",
    ) -> RollbackWorkflowResult:
        """Freeze one rollback target and current Registry without changing either."""

        reason = RollbackReason(type=reason_type.strip(), description=reason_description.strip())
        state = build_rollback_state(
            models_root=self.models_root,
            target_model_id=model_id,
            deployment_slot=deployment_slot,
        )
        created_at = self.clock().astimezone(UTC).isoformat()
        provisional = RollbackRequest(
            request_id="pending",
            target_model_id=state.target.model_id,
            current_champion_model_id=state.current_champion.model_id,
            deployment_slot=deployment_slot,
            reason=reason,
            target_contract=state.target_contract,
            registry_hash=state.registry_hash,
            requester=self.requester_provider(),
            created_at=created_at,
        )
        identity = canonical_payload_hash(_request_identity(provisional))
        request = provisional.model_copy(update={"request_id": f"rollback_{identity[:24]}"})
        bundle, idempotent = self.storage.publish_request(request, identity)
        validate_rollback_request(bundle.request, self.models_root)
        return _result(bundle.request, bundle.output_dir, idempotent=idempotent)

    def validate(self, request_id: str) -> RollbackWorkflowResult:
        """Publish immutable validation after rechecking history and artifacts."""

        bundle = self._required_bundle(request_id)
        state = validate_rollback_request(bundle.request, self.models_root)
        result = RollbackValidationResult(
            request_id=request_id,
            request_hash=file_sha256(bundle.output_dir / "request.json"),
            registry_hash=state.registry_hash,
            target_artifact_hash=state.target_contract.artifact_set_hash,
            historical_assignment_id=state.target_contract.historical_assignment_id,
            checks=(
                "historical_champion",
                "same_deployment_slot",
                "artifact_hashes",
                "feature_hash",
                "execution_contract",
                "registry_precondition",
            ),
            validated_at=self.clock().astimezone(UTC).isoformat(),
        )
        _, idempotent = self.storage.publish_validation(result)
        return _result(
            bundle.request,
            bundle.output_dir,
            status="VALIDATED",
            idempotent=idempotent,
        )

    def review(self, request_id: str) -> RollbackWorkflowResult:
        """Check reviewer authorization without writing a decision."""

        bundle = self._required_bundle(request_id)
        if self.storage.read_validation(request_id) is None:
            raise DataValidationError("rollback request must be validated before review")
        reviewer = self.reviewer_provider()
        self.policy.validate_reviewer(requester=bundle.request.requester, reviewer=reviewer)
        self.policy.validate_request_not_expired(bundle.request.created_at, self.clock())
        existing = self.storage.list_approvals(request_id)
        if existing:
            return self.status(request_id)
        return _result(bundle.request, bundle.output_dir, status="READY_FOR_REVIEW")

    def approve(self, request_id: str, comments: str) -> RollbackWorkflowResult:
        return self._decide(request_id, "APPROVED", comments)

    def reject(self, request_id: str, comments: str) -> RollbackWorkflowResult:
        return self._decide(request_id, "REJECTED", comments)

    def status(self, request_id: str) -> RollbackWorkflowResult:
        """Report derived immutable state without changing Registry or artifacts."""

        bundle = self.storage.read(request_id)
        if bundle is None:
            return RollbackWorkflowResult(
                request_id,
                "MISSING",
                "",
                "",
                self.storage.output_dir(request_id),
            )
        committed = self._read_apply_manifest(bundle.output_dir)
        if committed is not None:
            return _result(
                bundle.request,
                bundle.output_dir,
                status="APPLIED",
                event_id=committed.approval_event_id,
                registry_version_id=committed.registry_version_id,
                champion_assignment_id=committed.champion_assignment_id,
                idempotent=True,
            )
        approvals = self.storage.list_approvals(request_id)
        if approvals:
            if len(approvals) != 1:
                raise DataValidationError("rollback request has conflicting decisions")
            stored = approvals[0]
            event = stored.event
            review_status: str = event.event_type
            validation_path = bundle.output_dir / "validation" / "validation_result.json"
            try:
                invalid = (
                    stored.manifest.policy_hash != self.policy.policy_hash
                    or event.request_hash != file_sha256(bundle.output_dir / "request.json")
                    or event.validation_result_hash != file_sha256(validation_path)
                    or event.registry_hash_at_review
                    != file_sha256(self.models_root / "registry.json")
                    or event.target_artifact_hash_at_review
                    != current_artifact_set_hash(bundle.request, self.models_root)
                )
            except (DataValidationError, OSError):
                invalid = True
            if invalid:
                review_status = "INVALID"
            elif event.event_type == "APPROVED" and self.clock().astimezone(UTC) > parse_timestamp(
                event.expires_at
            ):
                review_status = "APPROVAL_EXPIRED"
            return _result(
                bundle.request,
                bundle.output_dir,
                status=review_status,
                event_id=event.event_id,
                idempotent=True,
            )
        current_status = (
            "VALIDATED"
            if self.storage.read_validation(request_id) is not None
            else "REQUEST_CREATED"
        )
        return _result(bundle.request, bundle.output_dir, status=current_status)

    def apply(self, request_id: str) -> RollbackWorkflowResult:
        """Restore an approved historical Champion under fixed lock ordering."""

        bundle = self._required_bundle(request_id)
        committed = self._read_apply_manifest(bundle.output_dir)
        if committed is not None:
            return self.status(request_id)
        command = f"ashare-quant models promotion rollback-apply --request-id {request_id}"
        with production_lock(self.production_lock_path, command=command):
            with production_lock(self.registry_lock_path, command=command):
                committed = self._read_apply_manifest(bundle.output_dir)
                if committed is not None:
                    return self.status(request_id)
                self._recover_interrupted(bundle.output_dir)
                context = validate_rollback_approval(
                    request_id=request_id,
                    models_root=self.models_root,
                    policy=self.policy,
                    now=self.clock(),
                )
                activated_at = (
                    self._pending_time(bundle.output_dir)
                    or self.clock().astimezone(UTC).isoformat()
                )
                version_id, payload, _ = build_rollback_registry(
                    records=load_registry_records(self.models_root / "registry.json"),
                    target_model_id=context.state.target.model_id,
                    current_champion_model_id=context.state.current_champion.model_id,
                    parent_registry_hash=context.state.registry_hash,
                    request_id=request_id,
                    approval_event_id=context.approval_event.event_id,
                    activated_at=activated_at,
                )
                pending = RollbackApplyPending(
                    request_id=request_id,
                    registry_version_id=version_id,
                    created_at=activated_at,
                )
                _publish_immutable(
                    bundle.output_dir / "rollback_apply_pending.json",
                    pending.model_dump(mode="json"),
                )
                old_version, new_version = publish_registry_versions(
                    models_root=self.models_root,
                    old_registry_path=self.models_root / "registry.json",
                    registry_version_id=version_id,
                    new_payload=payload,
                )
                assignment = _rollback_assignment(
                    request_id=request_id,
                    deployment_slot=bundle.request.deployment_slot,
                    model_id=context.state.target.model_id,
                    previous_model_id=context.state.current_champion.model_id,
                    approval_event_id=context.approval_event.event_id,
                    registry_version_id=version_id,
                    activated_at=activated_at,
                )
                history_path = (
                    self.models_root
                    / "champion_history"
                    / f"{assignment.champion_assignment_id}.json"
                )
                history_existed = history_path.exists()
                switched = False
                committed_write = False
                try:
                    switch_registry_atomically(self.models_root / "registry.json", new_version)
                    switched = True
                    _publish_immutable(history_path, assignment.model_dump(mode="json"))
                    manifest = RollbackApplyManifest(
                        request_id=request_id,
                        target_model_id=context.state.target.model_id,
                        previous_champion_model_id=context.state.current_champion.model_id,
                        approval_event_id=context.approval_event.event_id,
                        approval_event_hash=context.approval_event_hash,
                        registry_version_id=version_id,
                        registry_file_hash=file_sha256(new_version),
                        champion_assignment_id=assignment.champion_assignment_id,
                        champion_history_hash=file_sha256(history_path),
                        activated_at=activated_at,
                    )
                    atomic_write_json(
                        bundle.output_dir / "rollback_apply_manifest.json",
                        manifest.model_dump(mode="json"),
                    )
                    committed_write = True
                except Exception:
                    if switched and not committed_write:
                        restore_registry_atomically(self.models_root / "registry.json", old_version)
                    if not committed_write and not history_existed:
                        history_path.unlink(missing_ok=True)
                    raise
                return _result(
                    bundle.request,
                    bundle.output_dir,
                    status="APPLIED",
                    event_id=context.approval_event.event_id,
                    registry_version_id=version_id,
                    champion_assignment_id=assignment.champion_assignment_id,
                )

    def _decide(
        self,
        request_id: str,
        event_type: Literal["APPROVED", "REJECTED"],
        comments: str,
    ) -> RollbackWorkflowResult:
        bundle = self._required_bundle(request_id)
        validation = self.storage.read_validation(request_id)
        if validation is None:
            raise DataValidationError("rollback request must be validated before decision")
        clean_comments = comments.strip()
        if not clean_comments:
            raise DataValidationError("rollback review comments must not be empty")
        reviewer = self.reviewer_provider()
        self.policy.validate_reviewer(requester=bundle.request.requester, reviewer=reviewer)
        self.policy.validate_request_not_expired(bundle.request.created_at, self.clock())
        validate_rollback_request(bundle.request, self.models_root)
        now = self.clock().astimezone(UTC)
        provisional = RollbackApprovalEvent(
            event_id="pending",
            event_type=event_type,
            request_id=request_id,
            request_hash=file_sha256(bundle.output_dir / "request.json"),
            validation_result_hash=file_sha256(
                bundle.output_dir / "validation" / "validation_result.json"
            ),
            registry_hash_at_review=file_sha256(self.models_root / "registry.json"),
            target_artifact_hash_at_review=current_artifact_set_hash(
                bundle.request, self.models_root
            ),
            reviewer=reviewer,
            requester=bundle.request.requester,
            decision="approved" if event_type == "APPROVED" else "rejected",
            comments=clean_comments,
            created_at=now.isoformat(),
            expires_at=self.policy.expires_at(now).isoformat(),
        )
        identity = rollback_approval_identity(provisional, self.policy.policy_hash)
        event = provisional.model_copy(update={"event_id": f"review_{identity[:24]}"})
        stored, idempotent = self.storage.publish_approval(
            event, identity_hash=identity, policy_hash=self.policy.policy_hash
        )
        return _result(
            bundle.request,
            bundle.output_dir,
            status=stored.event.event_type,
            event_id=stored.event.event_id,
            idempotent=idempotent,
        )

    def _required_bundle(self, request_id: str) -> RollbackBundle:
        bundle = self.storage.read(request_id)
        if bundle is None:
            raise DataValidationError(f"rollback request does not exist: {request_id}")
        return bundle

    def _read_apply_manifest(self, output_dir: Path) -> RollbackApplyManifest | None:
        path = output_dir / "rollback_apply_manifest.json"
        if not path.is_file():
            return None
        try:
            manifest = RollbackApplyManifest.model_validate(_load_json(path))
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback apply manifest: {error}") from error
        history = self.models_root / "champion_history" / f"{manifest.champion_assignment_id}.json"
        version = self.models_root / "registry_versions" / f"{manifest.registry_version_id}.json"
        if file_sha256(history) != manifest.champion_history_hash:
            raise DataValidationError("rollback Champion history hash differs")
        if file_sha256(version) != manifest.registry_file_hash:
            raise DataValidationError("rollback Registry version hash differs")
        return manifest

    def _recover_interrupted(self, output_dir: Path) -> None:
        path = output_dir / "rollback_apply_pending.json"
        if not path.is_file():
            return
        try:
            pending = RollbackApplyPending.model_validate(_load_json(path))
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback pending journal: {error}") from error
        version_path = (
            self.models_root / "registry_versions" / f"{pending.registry_version_id}.json"
        )
        version = _load_json(version_path)
        parent_hash = version.get("parent_registry_hash")
        if not isinstance(parent_hash, str) or len(parent_hash) != 64:
            raise DataValidationError("rollback pending Registry parent hash is invalid")
        registry_path = self.models_root / "registry.json"
        current_hash = file_sha256(registry_path)
        if current_hash == parent_hash:
            return
        if current_hash != file_sha256(version_path):
            raise DataValidationError("Registry differs from rollback and parent versions")
        parent = self.models_root / "registry_versions" / f"registry_source_{parent_hash[:24]}.json"
        if file_sha256(parent) != parent_hash:
            raise DataValidationError("rollback parent Registry backup is invalid")
        restore_registry_atomically(registry_path, parent)

    @staticmethod
    def _pending_time(output_dir: Path) -> str | None:
        path = output_dir / "rollback_apply_pending.json"
        if not path.is_file():
            return None
        try:
            return RollbackApplyPending.model_validate(_load_json(path)).created_at
        except ValidationError as error:
            raise DataValidationError(f"invalid rollback pending journal: {error}") from error


def _request_identity(request: RollbackRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude={"request_id", "created_at"})


def _rollback_assignment(
    *,
    request_id: str,
    deployment_slot: str,
    model_id: str,
    previous_model_id: str,
    approval_event_id: str,
    registry_version_id: str,
    activated_at: str,
) -> RollbackChampionAssignment:
    core = {
        "deployment_slot": deployment_slot,
        "model_id": model_id,
        "previous_champion_model_id": previous_model_id,
        "rollback_request_id": request_id,
        "approval_event_id": approval_event_id,
        "registry_version_id": registry_version_id,
    }
    return RollbackChampionAssignment(
        champion_assignment_id=f"champion_{canonical_payload_hash(core)[:24]}",
        deployment_slot=deployment_slot,
        model_id=model_id,
        previous_champion_model_id=previous_model_id,
        rollback_request_id=request_id,
        approval_event_id=approval_event_id,
        registry_version_id=registry_version_id,
        activated_at=activated_at,
    )


def _publish_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != payload:
            raise DataValidationError(f"immutable rollback artifact differs: {path}")
        return
    atomic_write_json(path, payload)


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


def _result(
    request: RollbackRequest,
    output_dir: Path,
    *,
    status: str | None = None,
    event_id: str | None = None,
    registry_version_id: str | None = None,
    champion_assignment_id: str | None = None,
    idempotent: bool = False,
) -> RollbackWorkflowResult:
    return RollbackWorkflowResult(
        request_id=request.request_id,
        status=status or request.status,
        target_model_id=request.target_model_id,
        current_champion_model_id=request.current_champion_model_id,
        output_dir=output_dir,
        event_id=event_id,
        registry_version_id=registry_version_id,
        champion_assignment_id=champion_assignment_id,
        idempotent=idempotent,
    )
