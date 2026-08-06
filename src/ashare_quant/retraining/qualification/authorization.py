"""Stage-specific authorization validation and lifecycle operations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.configuration import RetrainingQualificationPolicy
from ashare_quant.retraining.qualification.authorization_schemas import (
    AuthorizationConsumptionClaim,
    AuthorizationConsumptionReceipt,
    AuthorizationResult,
    AuthorizationRevocation,
    AuthorizationStage,
    AuthorizationState,
    QualificationAuthorization,
    QualificationAuthorizationStatus,
    RevocationResult,
)
from ashare_quant.retraining.qualification.authorization_storage import (
    QualificationAuthorizationStorage,
)
from ashare_quant.retraining.qualification.schemas import QualificationSnapshot


class QualificationAuthorizationBlockedError(DataValidationError):
    """A valid command that cannot authorize the qualification's current state."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


class QualificationAuthorizationConflictError(QualificationAuthorizationBlockedError):
    """An active authorization must be revoked before replacement."""


class QualificationAuthorizationService:
    """Issue and evaluate immutable single-attempt stage authorizations."""

    def __init__(
        self,
        *,
        qualification_root: Path,
        policy: RetrainingQualificationPolicy,
        now: Callable[[], datetime],
    ) -> None:
        self.storage = QualificationAuthorizationStorage(qualification_root)
        self.policy = policy
        self.now = now

    def create(
        self,
        snapshot: QualificationSnapshot,
        *,
        stage: AuthorizationStage,
        approved_by: str,
        reason: str,
        expires_at: str | None,
    ) -> AuthorizationResult:
        approved_by = approved_by.strip()
        reason = reason.strip()
        if not approved_by or not reason:
            raise DataValidationError("authorization approver and reason must be non-empty")
        required_state = _required_state(stage)
        if snapshot.summary.current_state != required_state:
            raise QualificationAuthorizationBlockedError(
                "AUTHORIZATION_STATE_NOT_PENDING",
                f"{stage} authorization requires state {required_state}",
            )
        if snapshot.manifest is None or snapshot.summary.static_qualification_policy_hash is None:
            raise QualificationAuthorizationBlockedError(
                "LEGACY_AUTHORIZATION_MIGRATION_REQUIRED",
                "start a new qualification before issuing authorization",
            )
        if stage == "shadow":
            self._require_shadow_lineage(snapshot)
        issued = self._now()
        explicit_expiry = self._expiry(issued, expires_at) if expires_at is not None else None
        records = self._evaluated_records(snapshot, stage)
        invalid = [item for item in records if item[3] in {"INVALID", "LEGACY_UNSUPPORTED"}]
        if invalid:
            identities = ",".join(item[0].authorization_id for item in invalid)
            raise DataValidationError(
                f"AUTHORIZATION_STORAGE_INVALID: recovery required for {identities}"
            )
        active = [item for item in records if item[3] == "ACTIVE"]
        if len(active) > 1:
            raise DataValidationError(
                "AUTHORIZATION_STORAGE_INVALID: multiple active authorizations conflict"
            )
        if active:
            existing, output, _, _, _ = active[0]
            expiry_matches = (
                explicit_expiry is None or existing.expires_at == explicit_expiry.isoformat()
            )
            if existing.approved_by == approved_by and existing.reason == reason and expiry_matches:
                return AuthorizationResult(existing.authorization_id, stage, "ACTIVE", output, True)
            raise QualificationAuthorizationConflictError(
                "ACTIVE_AUTHORIZATION_CONFLICT",
                f"stage={stage} existing_authorization_id={existing.authorization_id} "
                f"approved_by={existing.approved_by} expires_at={existing.expires_at}. "
                f"Revoke authorization {existing.authorization_id} before issuing a replacement.",
            )
        expiry = explicit_expiry or self._expiry(issued, None)
        authorization = self._authorization(
            snapshot,
            stage=stage,
            approved_by=approved_by,
            reason=reason,
            issued=issued,
            expiry=expiry,
        )
        output, idempotent = self.storage.publish_authorization(authorization)
        return AuthorizationResult(
            authorization.authorization_id, stage, "ACTIVE", output, idempotent
        )

    def revoke(
        self,
        snapshot: QualificationSnapshot,
        *,
        authorization_id: str,
        revoked_by: str,
        reason: str,
    ) -> RevocationResult:
        revoked_by = revoked_by.strip()
        reason = reason.strip()
        if not revoked_by or not reason:
            raise DataValidationError("revocation operator and reason must be non-empty")
        authorization, _, authorization_hash = self.storage.authorization(
            snapshot.summary.qualification_run_id, authorization_id
        )
        if authorization.qualification_run_id != snapshot.summary.qualification_run_id:
            raise DataValidationError("authorization belongs to another qualification")
        if self.storage.claims(snapshot.summary.qualification_run_id, authorization_id):
            return RevocationResult("", authorization_id, False, Path(), False)
        existing = self.storage.revocations(snapshot.summary.qualification_run_id, authorization_id)
        if existing:
            old, output, _ = existing[0]
            if old.revoked_by == revoked_by and old.reason == reason:
                return RevocationResult(old.revocation_id, authorization_id, True, output, True)
            raise DataValidationError("authorization already has a different revocation")
        identity = canonical_payload_hash(
            {
                "authorization_id": authorization_id,
                "authorization_sha256": authorization_hash,
                "qualification_run_id": snapshot.summary.qualification_run_id,
                "stage": authorization.stage,
                "revoked_by": revoked_by,
                "reason": reason,
            }
        )
        revocation = AuthorizationRevocation(
            revocation_id=f"revocation_{identity[:24]}",
            authorization_id=authorization_id,
            authorization_sha256=authorization_hash,
            qualification_run_id=snapshot.summary.qualification_run_id,
            stage=authorization.stage,
            revoked_by=revoked_by,
            reason=reason,
            revoked_at=self._now().isoformat(),
        )
        output, idempotent = self.storage.publish_revocation(revocation)
        return RevocationResult(
            revocation.revocation_id, authorization_id, True, output, idempotent
        )

    def evaluate(
        self, snapshot: QualificationSnapshot, *, stage: AuthorizationStage
    ) -> QualificationAuthorizationStatus:
        if snapshot.summary.static_qualification_policy_hash is None:
            return QualificationAuthorizationStatus(
                stage=stage,
                status="LEGACY_UNSUPPORTED",
                message="LEGACY_AUTHORIZATION_MIGRATION_REQUIRED",
            )
        candidates = self._evaluated_records(snapshot, stage)
        identities = {
            state: tuple(
                authorization.authorization_id
                for authorization, _, _, current, _ in candidates
                if current == state
            )
            for state in (
                "CONSUMED",
                "REVOKED",
                "EXPIRED",
                "STALE",
                "INVALID",
                "LEGACY_UNSUPPORTED",
            )
        }
        active = [item for item in candidates if item[3] == "ACTIVE"]
        if len(active) > 1:
            return QualificationAuthorizationStatus(
                stage=stage,
                status="INVALID",
                consumed_authorization_ids=identities["CONSUMED"],
                revoked_authorization_ids=identities["REVOKED"],
                expired_authorization_ids=identities["EXPIRED"],
                stale_authorization_ids=identities["STALE"],
                invalid_authorization_ids=identities["INVALID"],
                legacy_unsupported_authorization_ids=identities["LEGACY_UNSUPPORTED"],
                message="multiple active authorizations conflict",
            )
        if active:
            authorization, _, digest, _, message = active[0]
            return QualificationAuthorizationStatus(
                stage=stage,
                status="ACTIVE",
                authorization_id=authorization.authorization_id,
                expires_at=authorization.expires_at,
                authorization_sha256=digest,
                consumed_authorization_ids=identities["CONSUMED"],
                revoked_authorization_ids=identities["REVOKED"],
                expired_authorization_ids=identities["EXPIRED"],
                stale_authorization_ids=identities["STALE"],
                invalid_authorization_ids=identities["INVALID"],
                legacy_unsupported_authorization_ids=identities["LEGACY_UNSUPPORTED"],
                message=message,
            )
        if candidates:
            priority: tuple[AuthorizationState, ...] = (
                "INVALID",
                "LEGACY_UNSUPPORTED",
                "STALE",
                "REVOKED",
                "CONSUMED",
                "EXPIRED",
            )
            selected = next(
                item for state in priority for item in reversed(candidates) if item[3] == state
            )
            authorization, _, digest, status, message = selected
            return QualificationAuthorizationStatus(
                stage=stage,
                status=status,
                authorization_id=authorization.authorization_id,
                expires_at=authorization.expires_at,
                authorization_sha256=digest,
                consumed_authorization_ids=identities["CONSUMED"],
                revoked_authorization_ids=identities["REVOKED"],
                expired_authorization_ids=identities["EXPIRED"],
                stale_authorization_ids=identities["STALE"],
                invalid_authorization_ids=identities["INVALID"],
                legacy_unsupported_authorization_ids=identities["LEGACY_UNSUPPORTED"],
                message=message,
            )
        return QualificationAuthorizationStatus(
            stage=stage,
            status="REQUIRED",
            message=f"{stage} authorization is required",
        )

    def _evaluated_records(
        self,
        snapshot: QualificationSnapshot,
        stage: AuthorizationStage,
    ) -> tuple[tuple[QualificationAuthorization, Path, str, AuthorizationState, str], ...]:
        run_id = snapshot.summary.qualification_run_id
        evaluated: list[tuple[QualificationAuthorization, Path, str, AuthorizationState, str]] = []
        for authorization, output, digest in self.storage.authorizations(run_id, stage):
            invalid = self._lineage_error(snapshot, authorization, digest)
            if invalid is not None:
                evaluated.append((authorization, output, digest, "INVALID", invalid))
                continue
            state, message = self._authorization_state(snapshot, authorization, digest)
            evaluated.append((authorization, output, digest, state, message))
        return tuple(evaluated)

    def _lineage_error(
        self,
        snapshot: QualificationSnapshot,
        authorization: QualificationAuthorization,
        digest: str,
    ) -> str | None:
        run_id = snapshot.summary.qualification_run_id
        if authorization.qualification_run_id != run_id:
            return "authorization qualification lineage mismatch"
        if authorization.static_qualification_policy_hash != (
            snapshot.summary.static_qualification_policy_hash
        ):
            return "authorization static policy lineage mismatch"
        authorization_events = [
            event
            for event in snapshot.events
            if event.details.get("authorization_id") == authorization.authorization_id
            and event.details.get("authorization_sha256") == digest
            and event.details.get("reviewed_manifest_sha256")
            == authorization.qualification_snapshot_manifest_sha256
        ]
        if len(authorization_events) != 1:
            return "authorization audit event lineage mismatch"
        revocations = self.storage.revocations(run_id, authorization.authorization_id)
        if len(revocations) > 1:
            return "authorization has duplicate revocations"
        for revocation, _, _ in revocations:
            if (
                revocation.authorization_id != authorization.authorization_id
                or revocation.authorization_sha256 != digest
                or revocation.qualification_run_id != run_id
                or revocation.stage != authorization.stage
            ):
                return "authorization revocation lineage mismatch"
            matching_events = [
                event
                for event in snapshot.events
                if event.details.get("authorization_id") == authorization.authorization_id
                and event.details.get("revocation_id") == revocation.revocation_id
                and event.details.get("stage") == authorization.stage
            ]
            if len(matching_events) != 1:
                return "authorization revocation audit event lineage mismatch"
        claims = self.storage.claims(run_id, authorization.authorization_id)
        if len(claims) > 1:
            return "authorization has duplicate consumption claims"
        for claim, _, _ in claims:
            if (
                claim.authorization_id != authorization.authorization_id
                or claim.authorization_sha256 != digest
                or claim.qualification_run_id != run_id
                or claim.stage != authorization.stage
                or claim.static_policy_hash != authorization.static_qualification_policy_hash
            ):
                return "authorization consumption lineage mismatch"
            receipts = self.storage.receipts(
                run_id, authorization.authorization_id, claim.consumption_id
            )
            if not receipts:
                return "authorization consumption claim lacks terminal receipt"
            if len(receipts) > 1:
                return "authorization consumption has duplicate receipts"
            for receipt, _, _ in receipts:
                if (
                    receipt.authorization_id != authorization.authorization_id
                    or receipt.qualification_run_id != run_id
                    or receipt.stage != authorization.stage
                    or receipt.consumption_id != claim.consumption_id
                ):
                    return "authorization receipt lineage mismatch"
            matching_events = [
                event
                for event in snapshot.events
                if event.sequence == claim.stage_event_sequence
                and event.state
                == ("TRAINING" if authorization.stage == "training" else "SHADOW_ENROLLING")
            ]
            if len(matching_events) != 1:
                return "authorization consumption audit event lineage mismatch"
        if revocations and claims:
            return "authorization is both revoked and consumed"
        return None

    def claim(
        self,
        snapshot: QualificationSnapshot,
        *,
        stage: AuthorizationStage,
        attempt_identity: str,
        stage_event_sequence: int,
    ) -> tuple[AuthorizationConsumptionClaim, Path]:
        status = self.evaluate(snapshot, stage=stage)
        if status.status != "ACTIVE" or status.authorization_id is None:
            raise DataValidationError(f"active {stage} authorization is required: {status.message}")
        authorization, _, digest = self.storage.authorization(
            snapshot.summary.qualification_run_id, status.authorization_id
        )
        identity = canonical_payload_hash(
            {
                "authorization_id": authorization.authorization_id,
                "qualification_run_id": snapshot.summary.qualification_run_id,
                "stage": stage,
                "attempt_identity": attempt_identity,
                "stage_event_sequence": stage_event_sequence,
            }
        )
        claim = AuthorizationConsumptionClaim(
            consumption_id=f"consumption_{identity[:24]}",
            authorization_id=authorization.authorization_id,
            authorization_sha256=digest,
            qualification_run_id=snapshot.summary.qualification_run_id,
            stage=stage,
            qualification_snapshot_manifest_sha256=file_sha256(self._snapshot_manifest(snapshot)),
            stage_event_sequence=stage_event_sequence,
            consumed_at=self._now().isoformat(),
            runtime_capability_enabled=True,
            static_policy_hash=self.policy.static_policy_hash,
            attempt_identity=attempt_identity,
        )
        output, idempotent = self.storage.publish_claim(claim)
        if idempotent:
            raise DataValidationError("authorization has already been claimed")
        return claim, output

    def receipt(
        self,
        claim: AuthorizationConsumptionClaim,
        *,
        status: Literal["COMPLETED", "FAILED"],
        result_manifest: Path | None = None,
        error: str | None = None,
    ) -> Path:
        identity = canonical_payload_hash(
            {
                "consumption_id": claim.consumption_id,
                "status": status,
                "result_manifest_sha256": (
                    file_sha256(result_manifest) if result_manifest is not None else None
                ),
                "error": error,
            }
        )
        receipt = AuthorizationConsumptionReceipt(
            receipt_id=f"receipt_{identity[:24]}",
            consumption_id=claim.consumption_id,
            authorization_id=claim.authorization_id,
            qualification_run_id=claim.qualification_run_id,
            stage=claim.stage,
            completed_at=self._now().isoformat(),
            status=status,
            result_manifest_path=str(result_manifest) if result_manifest is not None else None,
            result_manifest_sha256=(
                file_sha256(result_manifest) if result_manifest is not None else None
            ),
            error=error,
        )
        output, _ = self.storage.publish_receipt(receipt)
        return output

    def _authorization_state(
        self,
        snapshot: QualificationSnapshot,
        authorization: QualificationAuthorization,
        digest: str,
    ) -> tuple[AuthorizationState, str]:
        run_id = snapshot.summary.qualification_run_id
        if authorization.qualification_run_id != run_id or authorization.stage not in {
            "training",
            "shadow",
        }:
            return "INVALID", "authorization identity mismatch"
        if self.storage.claims(run_id, authorization.authorization_id):
            return "CONSUMED", "authorization has been consumed"
        if self.storage.revocations(run_id, authorization.authorization_id):
            return "REVOKED", "authorization has been revoked"
        if self._now() >= _aware(authorization.expires_at):
            return "EXPIRED", "authorization has expired"
        if not self._binding_matches(snapshot, authorization, digest):
            return "STALE", "authorization snapshot binding is stale"
        return "ACTIVE", "authorization is active"

    def _binding_matches(
        self,
        snapshot: QualificationSnapshot,
        authorization: QualificationAuthorization,
        digest: str,
    ) -> bool:
        if snapshot.manifest is None:
            return False
        values_match = (
            authorization.qualification_identity_hash
            == snapshot.manifest.qualification_identity_hash
            and authorization.request_id == snapshot.summary.request_id
            and authorization.as_of == snapshot.summary.as_of
            and authorization.parent_model_id == snapshot.summary.parent_model_id
            and authorization.horizon == snapshot.summary.horizon
            and authorization.training_request_hash == snapshot.manifest.training_request_hash
            and authorization.static_qualification_policy_hash
            == snapshot.summary.static_qualification_policy_hash
            and authorization.checkpoint_results_sha256 == snapshot.manifest.checkpoints_sha256
            and authorization.source_inventory_sha256 == snapshot.manifest.inventory_sha256
            and authorization.invariant_results_sha256 == snapshot.manifest.invariants_sha256
            and authorization.frozen_retraining_policy_hash
            == snapshot.summary.frozen_retraining_policy_hash
            and authorization.frozen_lifecycle_policy_hash
            == snapshot.summary.frozen_lifecycle_policy_hash
            and authorization.frozen_promotion_policy_hash
            == snapshot.summary.frozen_promotion_policy_hash
            and authorization.frozen_config_hash == snapshot.summary.frozen_config_hash
            and authorization.model_id == snapshot.summary.model_id
            and authorization.training_run_id == snapshot.summary.training_run_id
            and authorization.validation_run_id == snapshot.summary.validation_run_id
            and authorization.qualification_snapshot_state == snapshot.summary.current_state
            and self._stage_artifacts_match(snapshot, authorization)
        )
        matching_events = [
            event
            for event in snapshot.events
            if event.details.get("authorization_id") == authorization.authorization_id
            and event.details.get("authorization_sha256") == digest
            and event.details.get("reviewed_manifest_sha256")
            == authorization.qualification_snapshot_manifest_sha256
        ]
        return (
            values_match and len(matching_events) == 1 and matching_events[0] == snapshot.events[-1]
        )

    def _stage_artifacts_match(
        self,
        snapshot: QualificationSnapshot,
        authorization: QualificationAuthorization,
    ) -> bool:
        if authorization.stage == "training":
            return True
        training = snapshot.checkpoints.get("training")
        validation = snapshot.checkpoints.get("validation")
        if training is None or validation is None:
            return False
        expected = (
            training.artifact_hashes.get("model"),
            training.artifact_hashes.get("registration"),
            validation.artifact_hashes.get("validation"),
        )
        bound = (
            authorization.model_artifact_manifest_sha256,
            authorization.candidate_registration_sha256,
            authorization.validation_manifest_sha256,
        )
        if expected != bound:
            return False
        for checkpoint in (training, validation):
            for path in checkpoint.artifact_paths:
                if not Path(path).is_file():
                    return False
            for path, digest in zip(
                checkpoint.artifact_paths,
                checkpoint.artifact_hashes.values(),
                strict=True,
            ):
                if file_sha256(Path(path)) != digest:
                    return False
        return True

    def _authorization(
        self,
        snapshot: QualificationSnapshot,
        *,
        stage: AuthorizationStage,
        approved_by: str,
        reason: str,
        issued: datetime,
        expiry: datetime,
    ) -> QualificationAuthorization:
        assert snapshot.manifest is not None
        capability_name: Literal["allow_real_training", "allow_real_shadow"] = (
            "allow_real_training" if stage == "training" else "allow_real_shadow"
        )
        logical = {
            "qualification_run_id": snapshot.summary.qualification_run_id,
            "qualification_identity_hash": snapshot.manifest.qualification_identity_hash,
            "stage": stage,
            "snapshot_manifest_sha256": file_sha256(self._snapshot_manifest(snapshot)),
            "approved_by": approved_by,
            "reason": reason,
            "expires_at": expiry.isoformat(),
            "static_policy_hash": self.policy.static_policy_hash,
            "phase": self.policy.phase,
            "model_id": snapshot.summary.model_id,
            "training_run_id": snapshot.summary.training_run_id,
            "validation_run_id": snapshot.summary.validation_run_id,
        }
        model_hashes = self._shadow_hashes(snapshot) if stage == "shadow" else (None, None, None)
        logical.update(
            {
                "model_artifact_manifest_sha256": model_hashes[0],
                "candidate_registration_sha256": model_hashes[1],
                "validation_manifest_sha256": model_hashes[2],
            }
        )
        identity = canonical_payload_hash(logical)
        return QualificationAuthorization(
            authorization_id=f"authorization_{identity[:24]}",
            qualification_run_id=snapshot.summary.qualification_run_id,
            qualification_identity_hash=snapshot.manifest.qualification_identity_hash,
            stage=stage,
            request_id=snapshot.summary.request_id,
            as_of=snapshot.summary.as_of,
            parent_model_id=snapshot.summary.parent_model_id,
            model_id=snapshot.summary.model_id,
            horizon=snapshot.summary.horizon,
            training_request_hash=snapshot.manifest.training_request_hash,
            qualification_snapshot_state=_required_state(stage),
            qualification_snapshot_manifest_path=str(self._snapshot_manifest(snapshot)),
            qualification_snapshot_manifest_sha256=file_sha256(self._snapshot_manifest(snapshot)),
            qualification_summary_sha256=snapshot.manifest.summary_sha256,
            qualification_events_sha256=snapshot.manifest.events_sha256,
            checkpoint_results_sha256=snapshot.manifest.checkpoints_sha256,
            source_inventory_sha256=snapshot.manifest.inventory_sha256,
            invariant_results_sha256=snapshot.manifest.invariants_sha256,
            static_qualification_policy_hash=self.policy.static_policy_hash,
            runtime_capability_name=capability_name,
            runtime_capability_enabled_at_authorization=bool(getattr(self.policy, capability_name)),
            runtime_capability_hash_at_authorization=self.policy.runtime_capability_hash,
            frozen_retraining_policy_hash=snapshot.summary.frozen_retraining_policy_hash,
            frozen_lifecycle_policy_hash=snapshot.summary.frozen_lifecycle_policy_hash,
            frozen_promotion_policy_hash=snapshot.summary.frozen_promotion_policy_hash,
            frozen_config_hash=snapshot.summary.frozen_config_hash,
            training_run_id=snapshot.summary.training_run_id,
            validation_run_id=snapshot.summary.validation_run_id,
            model_artifact_manifest_sha256=model_hashes[0],
            candidate_registration_sha256=model_hashes[1],
            validation_manifest_sha256=model_hashes[2],
            approved_by=approved_by,
            reason=reason,
            issued_at=issued.isoformat(),
            expires_at=expiry.isoformat(),
        )

    def _expiry(self, issued: datetime, expires_at: str | None) -> datetime:
        expiry = (
            issued + timedelta(minutes=self.policy.authorization.default_validity_minutes)
            if expires_at is None
            else _aware(expires_at)
        )
        if expiry <= issued:
            raise DataValidationError(
                "AUTHORIZATION_EXPIRY_INVALID: authorization expiration must be in the future"
            )
        maximum = issued + timedelta(minutes=self.policy.authorization.maximum_validity_minutes)
        if expiry > maximum:
            raise DataValidationError(
                "AUTHORIZATION_EXPIRY_INVALID: authorization expiration exceeds policy maximum"
            )
        return expiry

    def _shadow_hashes(self, snapshot: QualificationSnapshot) -> tuple[str, str, str]:
        assert snapshot.summary.model_id is not None
        training = snapshot.checkpoints.get("training")
        validation = snapshot.checkpoints.get("validation")
        if training is None or validation is None:
            raise DataValidationError("Shadow authorization lacks training or validation evidence")
        return (
            training.artifact_hashes["model"],
            training.artifact_hashes["registration"],
            validation.artifact_hashes["validation"],
        )

    def _require_shadow_lineage(self, snapshot: QualificationSnapshot) -> None:
        if not all(
            (
                snapshot.summary.model_id,
                snapshot.summary.training_run_id,
                snapshot.summary.validation_run_id,
            )
        ):
            raise DataValidationError("Shadow authorization requires completed validation lineage")
        self._shadow_hashes(snapshot)

    def _snapshot_manifest(self, snapshot: QualificationSnapshot) -> Path:
        return (
            self.storage.qualification_root
            / snapshot.summary.qualification_run_id
            / "manifest.json"
        )

    def _now(self) -> datetime:
        value = self.now()
        if value.tzinfo is None:
            raise DataValidationError("authorization clock must be timezone-aware")
        return value.astimezone(UTC)


def _required_state(
    stage: AuthorizationStage,
) -> Literal["TRAINING_PENDING_APPROVAL", "SHADOW_PENDING_APPROVAL"]:
    return "TRAINING_PENDING_APPROVAL" if stage == "training" else "SHADOW_PENDING_APPROVAL"


def _aware(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise DataValidationError(
            f"AUTHORIZATION_EXPIRY_INVALID: invalid authorization timestamp: {value}"
        ) from error
    if parsed.tzinfo is None:
        raise DataValidationError(
            "AUTHORIZATION_EXPIRY_INVALID: authorization timestamp must include a timezone"
        )
    return parsed.astimezone(UTC)
