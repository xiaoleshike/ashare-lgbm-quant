"""Explicit, operator-advanced operational qualification orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.compute import resolve_training_backend
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.performance_observation.storage import (
    read_observation_artifact,
)
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.retraining.execution.schemas import (
    CandidateRegistration,
    ExecutionResult,
    QualificationExecutionContext,
)
from ashare_quant.retraining.orchestration.dry_run import (
    LifecycleDryRunResult,
    LifecycleDryRunService,
)
from ashare_quant.retraining.orchestration.schemas import LifecycleInput
from ashare_quant.retraining.orchestration.service import RetrainingLifecycleOrchestrator
from ashare_quant.retraining.qualification.authorization import (
    QualificationAuthorizationBlockedError,
    QualificationAuthorizationService,
)
from ashare_quant.retraining.qualification.authorization_schemas import (
    AuthorizationResult,
    AuthorizationStage,
    QualificationAuthorizationStatus,
    RevocationResult,
)
from ashare_quant.retraining.qualification.invariants import (
    compare_protected_state,
    protected_state_inventory,
)
from ashare_quant.retraining.qualification.lifecycle import require_qualification_transition
from ashare_quant.retraining.qualification.preflight import run_preflight
from ashare_quant.retraining.qualification.recovery import inspect_qualification_recovery
from ashare_quant.retraining.qualification.schemas import (
    QualificationCheck,
    QualificationCheckpoint,
    QualificationEvent,
    QualificationManifest,
    QualificationRecovery,
    QualificationResult,
    QualificationSnapshot,
    QualificationState,
    QualificationSummary,
)
from ashare_quant.retraining.qualification.storage import QualificationStorage
from ashare_quant.retraining.readiness.schemas import ReadinessResult
from ashare_quant.retraining.shadow.schemas import RetrainedShadowResult
from ashare_quant.retraining.validation.schemas import RetrainingValidationResult
from ashare_quant.utils.manifest import config_hash, current_git_info


class DryRunRunner(Protocol):
    def run(self, request_id: str, *, as_of: str | None = None) -> LifecycleDryRunResult: ...


class ReadinessRunner(Protocol):
    def validate(self, as_of: str, *, request_id: str | None = None) -> ReadinessResult: ...


class ExecutionRunner(Protocol):
    def execute(
        self, request_id: str, *, qualification: QualificationExecutionContext | None = None
    ) -> ExecutionResult: ...


class ValidationRunner(Protocol):
    def validate(
        self, model_id: str, *, qualification: QualificationExecutionContext | None = None
    ) -> RetrainingValidationResult: ...


class ShadowRunner(Protocol):
    def predict(
        self,
        model_id: str,
        *,
        as_of: str | None = None,
        qualification: QualificationExecutionContext | None = None,
    ) -> RetrainedShadowResult: ...


class OperationalQualificationService:
    """Coordinate real checkpoints while requiring explicit operator advancement."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        retraining_policy_path: Path,
        promotion_policy_path: Path,
        lifecycle: RetrainingLifecycleOrchestrator | None = None,
        dry_run: DryRunRunner | None = None,
        readiness: ReadinessRunner | None = None,
        execution: ExecutionRunner | None = None,
        validation: ValidationRunner | None = None,
        shadow: ShadowRunner | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.retraining_policy_path = retraining_policy_path
        self.promotion_policy_path = promotion_policy_path
        self.lifecycle = lifecycle or RetrainingLifecycleOrchestrator(
            settings=settings,
            config_path=config_path,
            retraining_policy_path=retraining_policy_path,
            promotion_policy_path=promotion_policy_path,
        )
        self.policy = self.lifecycle.retraining_policy.qualification
        self.dry_run = dry_run or LifecycleDryRunService(self.lifecycle)
        self.readiness = readiness or self.lifecycle.readiness
        self.execution: ExecutionRunner = execution or cast(
            ExecutionRunner, self.lifecycle.execution
        )
        self.validation: ValidationRunner = validation or cast(
            ValidationRunner, self.lifecycle.validation
        )
        self.shadow: ShadowRunner = shadow or cast(ShadowRunner, self.lifecycle.shadow)
        self.now = now or (lambda: datetime.now(UTC))
        self.storage = QualificationStorage(settings.paths.reports)
        self.authorization = QualificationAuthorizationService(
            qualification_root=self.storage.root,
            policy=self.policy,
            now=self.now,
        )
        self.project_root = self.lifecycle.project_root
        self.lock_path = self.project_root / "runs" / ".retraining-qualification.lock"

    def start(self, request_id: str, *, as_of: str) -> QualificationResult:
        """Run preflight, controlled dry-run, and readiness, then stop before training."""

        with production_lock(self.lock_path, command=f"qualification-start {request_id}"):
            lifecycle_run_id, frozen = self.lifecycle.proposed_identity(request_id)
            self._require_request(frozen, as_of)
            identity_hash = self._identity(frozen, as_of)
            run_id = f"qualification_{identity_hash[:24]}"
            existing = self.storage.read(run_id)
            if existing is not None:
                return self._result(existing, idempotent=True)
            baseline = self._protected(as_of)
            snapshot = self._initialize(run_id, lifecycle_run_id, frozen, as_of, baseline)
            snapshot = self._publish(snapshot, frozen, identity_hash)
            snapshot = self._preflight(snapshot, frozen, identity_hash)
            if snapshot.summary.current_state != "PREFLIGHT_READY":
                return self._result(snapshot)
            snapshot = self._controlled_dry_run(snapshot, frozen, identity_hash)
            if snapshot.summary.current_state != "DRY_RUN_READY":
                return self._result(snapshot)
            snapshot = self._readiness(snapshot, frozen, identity_hash)
            return self._result(snapshot)

    def advance(self, run_id: str, *, target: str) -> QualificationResult:
        if target not in {"training", "validation", "shadow", "observation"}:
            raise DataValidationError(f"unsupported qualification checkpoint: {target}")
        if target not in self.policy.allowed_stop_points:
            raise DataValidationError(f"qualification checkpoint is disabled by policy: {target}")
        with production_lock(self.lock_path, command=f"qualification-advance {run_id} {target}"):
            snapshot = self._required(run_id)
            frozen = self.lifecycle.frozen_input_for_request(
                snapshot.summary.request_id,
                retraining_policy_hash=snapshot.summary.frozen_retraining_policy_hash,
                lifecycle_policy_hash=snapshot.summary.frozen_lifecycle_policy_hash,
                promotion_policy_hash=snapshot.summary.frozen_promotion_policy_hash,
                frozen_config_hash=snapshot.summary.frozen_config_hash,
            )
            identity_hash = self._manifest_identity(snapshot)
            after_training = snapshot.summary.training_run_id is not None
            self._validate_source_inventory(snapshot, allow_policy_drift=after_training)
            self._verify_invariants(snapshot, final=False)
            if target == "training":
                snapshot = self._training(snapshot, frozen, identity_hash)
            elif target == "validation":
                snapshot = self._validation(snapshot, frozen, identity_hash)
            elif target == "shadow":
                snapshot = self._shadow(snapshot, frozen, identity_hash)
            else:
                snapshot = self._observation(snapshot, frozen, identity_hash)
            return self._result(snapshot)

    def status(self, run_id: str) -> dict[str, Any]:
        snapshot = self.storage.read(run_id)
        if snapshot is None:
            return {"qualification_run_id": run_id, "status": "MISSING"}
        training = self.authorization.evaluate(snapshot, stage="training")
        shadow = self.authorization.evaluate(snapshot, stage="shadow")
        training_checkpoint = snapshot.checkpoints.get("training")
        training_compute = (
            training_checkpoint.metrics.get("training_compute")
            if training_checkpoint is not None
            else None
        )
        return {
            **snapshot.summary.model_dump(mode="json"),
            "status": "VALID",
            "output": str(self.storage.output_dir(run_id)),
            "current_static_qualification_policy_hash": self.policy.static_policy_hash,
            "static_policy_drift": (
                snapshot.summary.static_qualification_policy_hash != self.policy.static_policy_hash
            ),
            "runtime_training_enabled": self.policy.allow_real_training,
            "runtime_shadow_enabled": self.policy.allow_real_shadow,
            "runtime_capability_hash": self.policy.runtime_capability_hash,
            "requested_training_backend": self.settings.ranker.training_backend.device_type,
            "effective_training_backend": (
                training_compute.get("effective_device_type")
                if isinstance(training_compute, dict)
                else None
            ),
            "training_backend_probe_status": (
                training_compute.get("probe_status") if isinstance(training_compute, dict) else None
            ),
            "training_authorization_status": training.status,
            "training_authorization_id": training.authorization_id,
            "training_authorization_expires_at": training.expires_at,
            "shadow_authorization_status": shadow.status,
            "shadow_authorization_id": shadow.authorization_id,
            "shadow_authorization_expires_at": shadow.expires_at,
            "consumed_authorization_ids": sorted(
                set(training.consumed_authorization_ids) | set(shadow.consumed_authorization_ids)
            ),
            "revoked_authorization_ids": sorted(
                set(training.revoked_authorization_ids) | set(shadow.revoked_authorization_ids)
            ),
            "expired_authorization_ids": sorted(
                set(training.expired_authorization_ids) | set(shadow.expired_authorization_ids)
            ),
            "stale_authorization_ids": sorted(
                set(training.stale_authorization_ids) | set(shadow.stale_authorization_ids)
            ),
            "invalid_authorization_ids": sorted(
                set(training.invalid_authorization_ids) | set(shadow.invalid_authorization_ids)
            ),
            "legacy_authorization_compatible": (
                snapshot.summary.static_qualification_policy_hash is not None
            ),
        }

    def recovery(self, run_id: str) -> QualificationRecovery:
        result = inspect_qualification_recovery(
            self.storage,
            run_id,
            authorization_storage=self.authorization.storage,
            current_static_policy_hash=self.policy.static_policy_hash,
            now=self.now,
        )
        try:
            snapshot = self.storage.read(run_id)
        except (DataValidationError, OSError, ValueError):
            return result
        if snapshot is None:
            return result
        issues = list(result.issues)
        for stage in ("training", "shadow"):
            try:
                status = self.authorization.evaluate(snapshot, stage=stage)
            except (DataValidationError, OSError, ValueError) as error:
                issues.append(f"{stage} authorization storage invalid: {error}")
                continue
            if status.status in {"INVALID", "STALE", "LEGACY_UNSUPPORTED"}:
                issues.append(f"{stage} authorization {status.status}: {status.message}")
        if not issues:
            return result
        return QualificationRecovery(
            run_id,
            "ACTION_REQUIRED",
            tuple(dict.fromkeys(issues)),
            tuple(
                dict.fromkeys(
                    (*result.operator_actions, "inspect immutable authorization artifacts")
                )
            ),
        )

    def authorize(
        self,
        run_id: str,
        *,
        stage: AuthorizationStage,
        approved_by: str,
        reason: str,
        expires_at: str | None = None,
    ) -> AuthorizationResult:
        with production_lock(self.lock_path, command=f"qualification-authorize {run_id} {stage}"):
            snapshot = self._required(run_id)
            if snapshot.summary.static_qualification_policy_hash is None:
                raise QualificationAuthorizationBlockedError(
                    "LEGACY_AUTHORIZATION_MIGRATION_REQUIRED",
                    "start a new qualification before issuing authorization",
                )
            self._require_static_policy(snapshot)
            self._validate_source_inventory(snapshot, allow_policy_drift=False)
            self._verify_invariants(snapshot, final=False)
            result = self.authorization.create(
                snapshot,
                stage=stage,
                approved_by=approved_by,
                reason=reason,
                expires_at=expires_at,
            )
            if result.idempotent:
                return result
            authorization, _, digest = self.authorization.storage.authorization(
                run_id, result.authorization_id
            )
            frozen = self._frozen_from_summary(snapshot)
            identity_hash = self._manifest_identity(snapshot)
            snapshot = self._transition(
                snapshot,
                snapshot.summary.current_state,
                f"{stage} authorization recorded",
                {
                    "authorization_id": result.authorization_id,
                    "stage": stage,
                    "authorization_sha256": digest,
                    "approved_by": authorization.approved_by,
                    "expires_at": authorization.expires_at,
                    "reviewed_manifest_sha256": (
                        authorization.qualification_snapshot_manifest_sha256
                    ),
                },
            )
            self._publish(snapshot, frozen, identity_hash)
            return result

    def revoke_authorization(
        self,
        run_id: str,
        *,
        authorization_id: str,
        revoked_by: str,
        reason: str,
    ) -> RevocationResult:
        with production_lock(
            self.lock_path, command=f"qualification-revoke-authorization {run_id}"
        ):
            snapshot = self._required(run_id)
            result = self.authorization.revoke(
                snapshot,
                authorization_id=authorization_id,
                revoked_by=revoked_by,
                reason=reason,
            )
            if not result.effective or result.idempotent:
                return result
            authorization, _, _ = self.authorization.storage.authorization(run_id, authorization_id)
            frozen = self._frozen_from_summary(snapshot)
            snapshot = self._transition(
                snapshot,
                snapshot.summary.current_state,
                f"{authorization.stage} authorization revoked",
                {
                    "authorization_id": authorization_id,
                    "revocation_id": result.revocation_id,
                    "stage": authorization.stage,
                    "revoked_by": revoked_by.strip(),
                },
            )
            self._publish(snapshot, frozen, self._manifest_identity(snapshot))
            return result

    def authorization_status(
        self, run_id: str, *, stage: AuthorizationStage | None = None
    ) -> tuple[QualificationAuthorizationStatus, ...]:
        snapshot = self._required_for_inspection(run_id)
        stages: tuple[AuthorizationStage, ...] = (stage,) if stage else ("training", "shadow")
        return tuple(self.authorization.evaluate(snapshot, stage=item) for item in stages)

    def cancel(self, run_id: str, *, reason: str) -> QualificationResult:
        if not reason.strip():
            raise DataValidationError("qualification cancellation requires a reason")
        with production_lock(self.lock_path, command=f"qualification-cancel {run_id}"):
            snapshot = self._required(run_id)
            frozen = self._frozen_from_summary(snapshot)
            identity_hash = self._manifest_identity(snapshot)
            snapshot = self._transition(snapshot, "CANCELLED", reason)
            return self._result(self._publish(snapshot, frozen, identity_hash))

    def _preflight(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        snapshot = self._transition(snapshot, "PREFLIGHT_CHECKING", "preflight started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        git = current_git_info()
        checks, inventory = run_preflight(
            frozen=frozen,
            as_of=snapshot.summary.as_of,
            project_root=self.project_root,
            reports_root=self.settings.paths.reports,
            models_root=self.settings.paths.models,
            processed_root=self.settings.paths.processed_data,
            config_path=self.config_path,
            retraining_policy_path=self.retraining_policy_path,
            promotion_policy_path=self.promotion_policy_path,
            qualification_enabled=self.policy.enabled,
            require_clean_worktree=self.policy.require_clean_worktree,
            git_dirty=bool(git["dirty"]),
            minimum_free_disk_bytes=self.policy.minimum_free_disk_bytes,
            minimum_available_memory_bytes=self.policy.minimum_available_memory_bytes,
            production_lock_path=self.project_root / "runs" / ".production.lock",
            lifecycle_lock_path=self.lifecycle.lifecycle_lock,
        )
        checks = (*checks, self._qualification_daily_limit(snapshot))
        stale_paths: list[str] = []
        for root in (self.storage.staging_root, self.lifecycle.storage.staging_root):
            if root.is_dir():
                stale_paths.extend(str(path) for path in root.iterdir())
        checks = (
            *checks,
            QualificationCheck(
                name="stale_transaction_paths",
                status="FAIL" if stale_paths else "PASS",
                message=(
                    f"stale staging or backup paths require recovery: {stale_paths}"
                    if stale_paths
                    else "no stale transaction paths"
                ),
            ),
        )
        failed = any(item.status == "FAIL" for item in checks)
        checkpoint = QualificationCheckpoint(
            name="preflight",
            status="blocked" if failed else "success",
            checks=checks,
            artifact_paths=tuple(str(value["path"]) for value in inventory.values()),
            artifact_hashes={name: str(value["sha256"]) for name, value in inventory.items()},
            warnings=tuple(item.message for item in checks if item.status == "WARN"),
        )
        snapshot = replace(
            snapshot,
            checkpoints={**snapshot.checkpoints, "preflight": checkpoint},
            source_inventory=inventory,
        )
        snapshot = self._transition(
            snapshot,
            "PREFLIGHT_BLOCKED" if failed else "PREFLIGHT_READY",
            "preflight blocked" if failed else "preflight passed",
        )
        return self._publish(snapshot, frozen, identity_hash)

    def _controlled_dry_run(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        snapshot = self._transition(snapshot, "DRY_RUN_CHECKING", "controlled dry-run started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        result = self.dry_run.run(snapshot.summary.request_id, as_of=snapshot.summary.as_of)
        report_path = result.output_dir / "dry_run.json"
        manifest_path = result.output_dir / "manifest.json"
        report = _json(report_path)
        manifest = _json(manifest_path)
        valid = (
            result.status == "READY_TO_EXECUTE"
            and report.get("request_id") == snapshot.summary.request_id
            and report.get("as_of") == snapshot.summary.as_of
            and report.get("proposed_lifecycle_run_id")
            == snapshot.summary.proposed_lifecycle_run_id
            and report.get("no_mutation_confirmed") is True
            and report.get("cooldown_status") == "PASS"
            and report.get("budget_status") == "PASS"
            and report.get("lock_status") == "AVAILABLE"
            and report.get("source_hashes", {}).get("training_request")
            == frozen.training_request_hash
            and manifest.get("dry_run_sha256") == file_sha256(report_path)
        )
        checkpoint = QualificationCheckpoint(
            name="dry_run",
            status="success" if valid else "blocked",
            artifact_paths=(str(report_path), str(manifest_path)),
            artifact_hashes={
                "dry_run": file_sha256(report_path),
                "manifest": file_sha256(manifest_path),
            },
            metrics={"dry_run_id": result.dry_run_id, "status": result.status},
            error=None if valid else "controlled lifecycle dry-run is not executable",
        )
        snapshot = replace(
            snapshot,
            summary=snapshot.summary.model_copy(update={"dry_run_id": result.dry_run_id}),
            checkpoints={**snapshot.checkpoints, "dry_run": checkpoint},
        )
        snapshot = self._transition(
            snapshot,
            "DRY_RUN_READY" if valid else "DRY_RUN_BLOCKED",
            "controlled dry-run passed" if valid else "controlled dry-run blocked",
        )
        return self._publish(snapshot, frozen, identity_hash)

    def _readiness(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        snapshot = self._transition(snapshot, "READINESS_CHECKING", "readiness checkpoint started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        result = self.readiness.validate(
            snapshot.summary.as_of, request_id=snapshot.summary.request_id
        )
        report = result.report
        valid = (
            report.status == "READY"
            and report.request_id == snapshot.summary.request_id
            and report.as_of == snapshot.summary.as_of
            and report.request_hash == frozen.training_request_hash
            and report.feature_hash == snapshot.source_inventory["feature_manifest"]["sha256"]
            and report.universe_hash == snapshot.source_inventory["universe_manifest"]["sha256"]
            and report.label_hash == snapshot.source_inventory["label_manifest"]["sha256"]
            and all(report.checks.get(name) == "PASS" for name in report.checks)
        )
        manifest_path = result.output_dir / "manifest.json"
        checkpoint = QualificationCheckpoint(
            name="readiness",
            status="success" if valid else "failed",
            artifact_paths=(str(manifest_path),),
            artifact_hashes={"readiness": file_sha256(manifest_path)},
            metrics={
                "run_id": report.run_id,
                "production_run_id": report.production_run_id,
                "feature_hash": report.feature_hash,
                "universe_hash": report.universe_hash,
                "label_hash": report.label_hash,
            },
            error=None if valid else "retraining execution readiness is not READY",
        )
        snapshot = replace(
            snapshot,
            summary=snapshot.summary.model_copy(update={"readiness_run_id": report.run_id}),
            checkpoints={**snapshot.checkpoints, "readiness": checkpoint},
        )
        if not valid:
            snapshot = self._transition(snapshot, "READINESS_FAILED", "readiness failed")
        else:
            snapshot = self._transition(snapshot, "READINESS_READY", "readiness passed")
            snapshot = self._transition(
                snapshot, "TRAINING_PENDING_APPROVAL", "explicit training approval required"
            )
        return self._publish(snapshot, frozen, identity_hash)

    def _training(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        if snapshot.summary.current_state == "VALIDATION_PENDING_APPROVAL":
            return snapshot
        if snapshot.summary.current_state == "TRAINING_FAILED":
            snapshot = self._transition(
                snapshot,
                "TRAINING_PENDING_APPROVAL",
                "failed training attempt requires a new authorization",
            )
            return self._publish(snapshot, frozen, identity_hash)
        if snapshot.summary.current_state != "TRAINING_PENDING_APPROVAL":
            raise DataValidationError("qualification training cannot skip prior checkpoints")
        if not self.policy.allow_real_training:
            return snapshot
        self._require_static_policy(snapshot)
        resolve_training_backend(self.settings.ranker.training_backend)
        authorization_status = self.authorization.evaluate(snapshot, stage="training")
        if authorization_status.status != "ACTIVE":
            return snapshot
        controls = self.lifecycle.operational_controls()
        budget = controls.budget()
        cooldown = controls.cooldown(
            lifecycle_run_id=snapshot.summary.proposed_lifecycle_run_id,
            parent_model_id=snapshot.summary.parent_model_id,
            horizon=snapshot.summary.horizon,
        )
        if not budget.allowed or not cooldown.allowed:
            return snapshot
        attempt_identity = canonical_payload_hash(
            {
                "qualification_run_id": snapshot.summary.qualification_run_id,
                "stage": "training",
                "authorization_id": authorization_status.authorization_id,
                "next_event_sequence": len(snapshot.events) + 1,
            }
        )
        claim, _ = self.authorization.claim(
            snapshot,
            stage="training",
            attempt_identity=attempt_identity,
            stage_event_sequence=len(snapshot.events) + 1,
        )
        snapshot = self._transition(
            snapshot,
            "TRAINING",
            "qualification-only training started",
            {
                "budget": asdict(budget),
                "cooldown": asdict(cooldown),
                "authorization_id": claim.authorization_id,
                "consumption_id": claim.consumption_id,
                "runtime_capability_enabled": True,
            },
        )
        snapshot = self._publish(snapshot, frozen, identity_hash)
        context = self._qualification_context(snapshot)
        try:
            with production_lock(
                self.lifecycle.lifecycle_lock,
                command=f"qualification training {snapshot.summary.qualification_run_id}",
            ):
                result = self.execution.execute(
                    snapshot.summary.request_id,
                    qualification=context,
                )
            manifest_path = cast(Path, result.artifact_dir) / "manifest.json"
            registration_path = (
                self.settings.paths.models
                / "candidate_registrations"
                / result.model_id
                / "registration.json"
            )
            artifact = _json(manifest_path)
            registration = CandidateRegistration.model_validate(_json(registration_path))
            if (
                artifact.get("qualification_only") is not True
                or artifact.get("qualification_run_id") != snapshot.summary.qualification_run_id
                or registration.qualification_run_id != snapshot.summary.qualification_run_id
                or not registration.qualification_only
            ):
                raise DataValidationError("training output lacks qualification-only lineage")
            training_compute = artifact.get("training_compute")
            if not isinstance(training_compute, dict):
                raise DataValidationError("training output lacks compute backend provenance")
            checkpoint = QualificationCheckpoint(
                name="training",
                status="success",
                artifact_paths=(
                    str(result.output_dir / "manifest.json"),
                    str(manifest_path),
                    str(registration_path),
                ),
                artifact_hashes={
                    "execution": file_sha256(result.output_dir / "manifest.json"),
                    "model": file_sha256(manifest_path),
                    "registration": file_sha256(registration_path),
                },
                metrics={
                    "training_run_id": result.training_run_id,
                    "model_id": result.model_id,
                    "training_compute": training_compute,
                },
            )
            self.authorization.receipt(
                claim,
                status="COMPLETED",
                result_manifest=result.output_dir / "manifest.json",
            )
            snapshot = replace(
                snapshot,
                summary=snapshot.summary.model_copy(
                    update={"training_run_id": result.training_run_id, "model_id": result.model_id}
                ),
                checkpoints={**snapshot.checkpoints, "training": checkpoint},
            )
            snapshot = self._transition(snapshot, "TRAINING_COMPLETED", "training completed")
            snapshot = self._transition(
                snapshot,
                "VALIDATION_PENDING_APPROVAL",
                "explicit validation approval required",
            )
        except Exception as error:
            self.authorization.receipt(claim, status="FAILED", error=str(error))
            snapshot = replace(
                snapshot,
                checkpoints={
                    **snapshot.checkpoints,
                    "training": QualificationCheckpoint(
                        name="training",
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                    ),
                },
            )
            snapshot = self._transition(snapshot, "TRAINING_FAILED", str(error))
        return self._publish(snapshot, frozen, identity_hash)

    def _validation(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        if snapshot.summary.current_state == "SHADOW_PENDING_APPROVAL":
            return snapshot
        if snapshot.summary.current_state != "VALIDATION_PENDING_APPROVAL":
            raise DataValidationError("qualification validation cannot skip training")
        assert snapshot.summary.model_id is not None
        model_id = snapshot.summary.model_id
        snapshot = self._transition(snapshot, "VALIDATING", "qualification validation started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        try:
            with production_lock(
                self.lifecycle.lifecycle_lock,
                command=f"qualification validation {snapshot.summary.qualification_run_id}",
            ):
                result = self.validation.validate(
                    model_id,
                    qualification=self._qualification_context(snapshot),
                )
            manifest_path = result.output_dir / "manifest.json"
            manifest = _json(manifest_path)
            if (
                manifest.get("qualification_only") is not True
                or manifest.get("qualification_run_id") != snapshot.summary.qualification_run_id
                or manifest.get("training_run_id") != snapshot.summary.training_run_id
            ):
                raise DataValidationError("validation output lacks exact qualification lineage")
            checkpoint = QualificationCheckpoint(
                name="validation",
                status="success",
                artifact_paths=(str(manifest_path),),
                artifact_hashes={"validation": file_sha256(manifest_path)},
                metrics={"validation_run_id": result.run_id},
            )
            snapshot = replace(
                snapshot,
                summary=snapshot.summary.model_copy(update={"validation_run_id": result.run_id}),
                checkpoints={**snapshot.checkpoints, "validation": checkpoint},
            )
            snapshot = self._transition(snapshot, "VALIDATION_COMPLETED", "validation completed")
            snapshot = self._transition(
                snapshot, "SHADOW_PENDING_APPROVAL", "explicit Shadow approval required"
            )
        except Exception as error:
            snapshot = replace(
                snapshot,
                checkpoints={
                    **snapshot.checkpoints,
                    "validation": QualificationCheckpoint(
                        name="validation", status="failed", error=f"{type(error).__name__}: {error}"
                    ),
                },
            )
            snapshot = self._transition(snapshot, "VALIDATION_FAILED", str(error))
        return self._publish(snapshot, frozen, identity_hash)

    def _shadow(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        if snapshot.summary.current_state == "SHADOW_ENROLLED":
            return snapshot
        if snapshot.summary.current_state == "SHADOW_FAILED":
            snapshot = self._transition(
                snapshot,
                "SHADOW_PENDING_APPROVAL",
                "failed Shadow attempt requires a new authorization",
            )
            return self._publish(snapshot, frozen, identity_hash)
        if snapshot.summary.current_state != "SHADOW_PENDING_APPROVAL":
            raise DataValidationError("qualification Shadow cannot skip validation")
        if not self.policy.allow_real_shadow:
            return snapshot
        self._require_static_policy(snapshot)
        authorization_status = self.authorization.evaluate(snapshot, stage="shadow")
        if authorization_status.status != "ACTIVE":
            return snapshot
        assert snapshot.summary.model_id is not None
        model_id = snapshot.summary.model_id
        attempt_identity = canonical_payload_hash(
            {
                "qualification_run_id": snapshot.summary.qualification_run_id,
                "stage": "shadow",
                "authorization_id": authorization_status.authorization_id,
                "next_event_sequence": len(snapshot.events) + 1,
            }
        )
        claim, _ = self.authorization.claim(
            snapshot,
            stage="shadow",
            attempt_identity=attempt_identity,
            stage_event_sequence=len(snapshot.events) + 1,
        )
        snapshot = self._transition(snapshot, "SHADOW_ENROLLING", "Shadow enrollment started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        try:
            with production_lock(
                self.lifecycle.lifecycle_lock,
                command=f"qualification shadow {snapshot.summary.qualification_run_id}",
            ):
                result = self.shadow.predict(
                    model_id,
                    as_of=snapshot.summary.as_of,
                    qualification=self._qualification_context(snapshot),
                )
            manifest_path = result.output_dir / "manifest.json"
            manifest = _json(manifest_path)
            if (
                manifest.get("qualification_only") is not True
                or manifest.get("qualification_run_id") != snapshot.summary.qualification_run_id
                or manifest.get("promotion_forbidden") is not True
                or manifest.get("trading_forbidden") is not True
            ):
                raise DataValidationError("Shadow output lacks qualification isolation")
            checkpoint = QualificationCheckpoint(
                name="shadow",
                status="success",
                artifact_paths=(str(manifest_path),),
                artifact_hashes={"shadow": file_sha256(manifest_path)},
                metrics={
                    "shadow_run_id": result.shadow_run_id,
                    "production_run_id": manifest.get("production_run_id"),
                },
            )
            self.authorization.receipt(
                claim,
                status="COMPLETED",
                result_manifest=result.output_dir / "manifest.json",
            )
            snapshot = replace(
                snapshot,
                summary=snapshot.summary.model_copy(
                    update={
                        "shadow_run_id": result.shadow_run_id,
                        "production_run_id": str(manifest.get("production_run_id", "")),
                    }
                ),
                checkpoints={**snapshot.checkpoints, "shadow": checkpoint},
            )
            snapshot = self._transition(snapshot, "SHADOW_ENROLLED", "Shadow enrolled")
        except Exception as error:
            self.authorization.receipt(claim, status="FAILED", error=str(error))
            snapshot = replace(
                snapshot,
                checkpoints={
                    **snapshot.checkpoints,
                    "shadow": QualificationCheckpoint(
                        name="shadow", status="failed", error=f"{type(error).__name__}: {error}"
                    ),
                },
            )
            snapshot = self._transition(snapshot, "SHADOW_FAILED", str(error))
        return self._publish(snapshot, frozen, identity_hash)

    def _observation(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        if snapshot.summary.current_state == "QUALIFIED":
            return snapshot
        if snapshot.summary.current_state != "SHADOW_ENROLLED":
            raise DataValidationError("qualification observation cannot skip Shadow")
        snapshot = self._transition(
            snapshot, "OBSERVATION_CHECKING", "observation integration checked"
        )
        snapshot = self._publish(snapshot, frozen, identity_hash)
        sessions, paths, hashes = self._matching_observations(snapshot)
        state: QualificationState = (
            "OBSERVATION_ACCUMULATING" if sessions else "OBSERVATION_PENDING"
        )
        checkpoint = QualificationCheckpoint(
            name="observation",
            status="success",
            artifact_paths=tuple(paths),
            artifact_hashes=hashes,
            metrics={"mature_sessions": sessions, "historical_backfill": False},
        )
        snapshot = replace(
            snapshot,
            summary=snapshot.summary.model_copy(update={"observation_status": state}),
            checkpoints={**snapshot.checkpoints, "observation": checkpoint},
        )
        snapshot = self._transition(snapshot, state, f"observation status={state}")
        current = self._protected(snapshot.summary.as_of)
        baseline = cast(dict[str, dict[str, Any]], snapshot.invariant_results["baseline"])
        changed = compare_protected_state(baseline, current)
        snapshot = replace(
            snapshot,
            invariant_results={"baseline": baseline, "current": current, "changed": list(changed)},
        )
        if changed:
            snapshot = self._transition(
                snapshot,
                "FAILED",
                "protected safety invariants changed",
                {"changed": list(changed)},
            )
        else:
            snapshot = self._transition(
                snapshot, "QUALIFIED", "operational qualification completed"
            )
        return self._publish(snapshot, frozen, identity_hash)

    def _matching_observations(
        self, snapshot: QualificationSnapshot
    ) -> tuple[int, list[str], dict[str, str]]:
        dates: set[str] = set()
        paths: list[str] = []
        hashes: dict[str, str] = {}
        root = self.settings.paths.reports / "performance_observation"
        if not root.is_dir():
            return 0, paths, hashes
        for path in sorted(root.glob("*/observation.parquet")):
            artifact = read_observation_artifact(path.parent)
            if artifact is None:
                raise DataValidationError(
                    f"incomplete performance observation artifact: {path.parent}"
                )
            frame, _ = artifact
            required = {
                "qualification_run_id",
                "qualification_only",
                "training_run_id",
                "validation_run_id",
                "shadow_run_id",
                "label_status",
                "signal_date",
            }
            if not required.issubset(frame.columns):
                continue
            selected = frame.loc[
                frame["qualification_only"].fillna(False).astype(bool)
                & frame["qualification_run_id"]
                .astype(str)
                .eq(snapshot.summary.qualification_run_id)
                & frame["model_id"].astype(str).eq(str(snapshot.summary.model_id))
                & frame["horizon"].astype(int).eq(snapshot.summary.horizon)
                & frame["training_run_id"].astype(str).eq(str(snapshot.summary.training_run_id))
                & frame["validation_run_id"].astype(str).eq(str(snapshot.summary.validation_run_id))
                & frame["shadow_run_id"].astype(str).eq(str(snapshot.summary.shadow_run_id))
                & frame["label_status"].astype(str).eq("available")
                & pd.to_numeric(frame["future_excess_ret"], errors="coerce").notna()
            ]
            if selected.empty:
                continue
            dates.update(selected["signal_date"].astype(str))
            paths.append(str(path))
            manifest_path = path.parent / "manifest.json"
            paths.append(str(manifest_path))
            hashes[f"{path.parent.name}:parquet"] = file_sha256(path)
            hashes[f"{path.parent.name}:manifest"] = file_sha256(manifest_path)
        return len(dates), paths, hashes

    def _initialize(
        self,
        run_id: str,
        lifecycle_run_id: str,
        frozen: LifecycleInput,
        as_of: str,
        baseline: dict[str, dict[str, Any]],
    ) -> QualificationSnapshot:
        created = self._timestamp()
        target = frozen.request.target_models[0]
        summary = QualificationSummary(
            qualification_run_id=run_id,
            request_id=frozen.request.request_id,
            as_of=as_of,
            parent_model_id=target.model_id,
            horizon=target.horizon,
            current_state="CREATED",
            proposed_lifecycle_run_id=lifecycle_run_id,
            frozen_retraining_policy_hash=frozen.retraining_policy_hash,
            frozen_lifecycle_policy_hash=frozen.lifecycle_policy_hash,
            frozen_promotion_policy_hash=frozen.promotion_policy_hash,
            frozen_config_hash=frozen.frozen_config_hash,
            qualification_policy_hash=self.policy.static_policy_hash,
            static_qualification_policy_hash=self.policy.static_policy_hash,
            created_at=created,
            updated_at=created,
        )
        event = QualificationEvent(
            sequence=1,
            state="CREATED",
            created_at=created,
            message="qualification identity created",
        )
        return QualificationSnapshot(summary, (event,), {}, {}, {"baseline": baseline})

    def _identity(self, frozen: LifecycleInput, as_of: str) -> str:
        target = frozen.request.target_models[0]
        return canonical_payload_hash(
            {
                "training_request_hash": frozen.training_request_hash,
                "request_id": frozen.request.request_id,
                "as_of": as_of,
                "parent_model_id": target.model_id,
                "horizon": target.horizon,
                "retraining_policy_hash": frozen.retraining_policy_hash,
                "lifecycle_policy_hash": frozen.lifecycle_policy_hash,
                "promotion_policy_hash": frozen.promotion_policy_hash,
                "config_hash": frozen.frozen_config_hash,
                "qualification_policy_hash": self.policy.static_policy_hash,
                "phase": "2.8.2G",
            }
        )

    def _publish(
        self, snapshot: QualificationSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> QualificationSnapshot:
        git = current_git_info()
        manifest = QualificationManifest(
            qualification_identity_hash=identity_hash,
            qualification_run_id=snapshot.summary.qualification_run_id,
            request_id=snapshot.summary.request_id,
            as_of=snapshot.summary.as_of,
            current_state=snapshot.summary.current_state,
            training_request_hash=frozen.training_request_hash,
            qualification_policy_hash=snapshot.summary.qualification_policy_hash,
            static_qualification_policy_hash=(snapshot.summary.static_qualification_policy_hash),
            legacy_qualification_policy_hash=(snapshot.summary.legacy_qualification_policy_hash),
            retraining_policy_hash=snapshot.summary.frozen_retraining_policy_hash,
            lifecycle_policy_hash=snapshot.summary.frozen_lifecycle_policy_hash,
            promotion_policy_hash=snapshot.summary.frozen_promotion_policy_hash,
            config_hash=snapshot.summary.frozen_config_hash,
            source_hashes={
                name: str(value["sha256"])
                for name, value in snapshot.source_inventory.items()
                if isinstance(value.get("sha256"), str)
            },
            protected_invariant_hashes={
                name: cast(str | None, value.get("sha256"))
                for name, value in cast(
                    dict[str, dict[str, Any]], snapshot.invariant_results.get("baseline", {})
                ).items()
            },
            summary_sha256="0" * 64,
            events_sha256="0" * 64,
            checkpoints_sha256="0" * 64,
            inventory_sha256="0" * 64,
            invariants_sha256="0" * 64,
            report_sha256="0" * 64,
            git_commit=git["commit"],
            git_dirty=bool(git["dirty"]),
        )
        return self.storage.publish(snapshot, manifest)

    def _transition(
        self,
        snapshot: QualificationSnapshot,
        target: QualificationState,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> QualificationSnapshot:
        require_qualification_transition(snapshot.summary.current_state, target)
        created = self._timestamp()
        event = QualificationEvent(
            sequence=len(snapshot.events) + 1,
            state=target,
            created_at=created,
            message=message,
            details=details or {},
        )
        return replace(
            snapshot,
            summary=snapshot.summary.model_copy(
                update={"current_state": target, "updated_at": created}
            ),
            events=(*snapshot.events, event),
        )

    def _require_request(self, frozen: LifecycleInput, as_of: str) -> None:
        if frozen.request.as_of != as_of:
            raise DataValidationError("qualification as_of must match Training Request")
        if len(frozen.request.target_models) != 1:
            raise DataValidationError("qualification requires exactly one target model")

    def _require_static_policy(self, snapshot: QualificationSnapshot) -> None:
        if (
            self.lifecycle.retraining_policy.policy_hash
            != snapshot.summary.frozen_retraining_policy_hash
            or self.lifecycle.retraining_policy.lifecycle_policy_hash
            != snapshot.summary.frozen_lifecycle_policy_hash
            or self.lifecycle.promotion_policy.policy_hash
            != snapshot.summary.frozen_promotion_policy_hash
            or config_hash(self.config_path) != snapshot.summary.frozen_config_hash
            or self.policy.static_policy_hash != snapshot.summary.static_qualification_policy_hash
        ):
            raise DataValidationError("blocking policy or configuration drift before training")

    def _qualification_context(
        self, snapshot: QualificationSnapshot
    ) -> QualificationExecutionContext:
        return QualificationExecutionContext(
            qualification_run_id=snapshot.summary.qualification_run_id
        )

    def _qualification_daily_limit(self, snapshot: QualificationSnapshot) -> QualificationCheck:
        zone = ZoneInfo(self.settings.production.timezone)
        operational_date = self.now().astimezone(zone).date()
        counted: list[str] = []
        if self.storage.root.is_dir():
            for directory in sorted(path for path in self.storage.root.iterdir() if path.is_dir()):
                if (
                    directory.name == ".tmp"
                    or directory.name == snapshot.summary.qualification_run_id
                ):
                    continue
                previous = self.storage.read(directory.name)
                if previous is None:
                    raise DataValidationError(
                        f"incomplete qualification history blocks daily limit: {directory}"
                    )
                created = datetime.fromisoformat(previous.summary.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                if created.astimezone(zone).date() == operational_date:
                    counted.append(previous.summary.qualification_run_id)
        allowed = len(counted) < self.policy.maximum_qualification_runs_per_day
        return QualificationCheck(
            name="qualification_daily_limit",
            status="PASS" if allowed else "FAIL",
            message="available" if allowed else "daily qualification limit exhausted",
            details={
                "operational_date": operational_date.isoformat(),
                "configured_limit": self.policy.maximum_qualification_runs_per_day,
                "counted_run_ids": counted,
            },
        )

    def _protected(self, as_of: str) -> dict[str, dict[str, Any]]:
        return protected_state_inventory(
            project_root=self.project_root,
            models_root=self.settings.paths.models,
            reports_root=self.settings.paths.reports,
            paper_root=self.settings.paths.paper_trading,
            as_of=as_of,
        )

    def _verify_invariants(self, snapshot: QualificationSnapshot, *, final: bool) -> None:
        baseline = cast(dict[str, dict[str, Any]], snapshot.invariant_results["baseline"])
        changed = compare_protected_state(baseline, self._protected(snapshot.summary.as_of))
        if changed:
            raise DataValidationError(
                f"protected qualification invariants changed: {list(changed)}"
            )

    def _validate_source_inventory(
        self, snapshot: QualificationSnapshot, *, allow_policy_drift: bool
    ) -> None:
        mutable_after_training = {"config", "promotion_policy"}
        for name, source in snapshot.source_inventory.items():
            if name == "retraining_policy":
                continue
            if allow_policy_drift and name in mutable_after_training:
                continue
            path = Path(str(source.get("path", "")))
            digest = source.get("sha256")
            if not isinstance(digest, str) or file_sha256(path) != digest:
                raise DataValidationError(f"qualification source changed: {name}")
        if (
            self.lifecycle.retraining_policy.policy_hash
            != snapshot.summary.frozen_retraining_policy_hash
            or self.lifecycle.retraining_policy.lifecycle_policy_hash
            != snapshot.summary.frozen_lifecycle_policy_hash
            or self.policy.static_policy_hash != snapshot.summary.static_qualification_policy_hash
        ):
            raise DataValidationError("qualification static policy or lifecycle policy changed")

    def _frozen_from_summary(self, snapshot: QualificationSnapshot) -> LifecycleInput:
        return self.lifecycle.frozen_input_for_request(
            snapshot.summary.request_id,
            retraining_policy_hash=snapshot.summary.frozen_retraining_policy_hash,
            lifecycle_policy_hash=snapshot.summary.frozen_lifecycle_policy_hash,
            promotion_policy_hash=snapshot.summary.frozen_promotion_policy_hash,
            frozen_config_hash=snapshot.summary.frozen_config_hash,
        )

    def _required(self, run_id: str) -> QualificationSnapshot:
        snapshot = self.storage.read(run_id)
        if snapshot is None:
            raise DataValidationError(f"qualification does not exist: {run_id}")
        if snapshot.summary.current_state in {"CANCELLED", "QUALIFIED", "FAILED"}:
            if snapshot.summary.current_state == "QUALIFIED":
                return snapshot
            raise DataValidationError(
                f"qualification is terminal: {snapshot.summary.current_state}"
            )
        return snapshot

    def _required_for_inspection(self, run_id: str) -> QualificationSnapshot:
        snapshot = self.storage.read(run_id)
        if snapshot is None:
            raise DataValidationError(f"qualification does not exist: {run_id}")
        return snapshot

    def _manifest_identity(self, snapshot: QualificationSnapshot) -> str:
        if snapshot.manifest is None:
            raise DataValidationError("qualification manifest is missing")
        return snapshot.manifest.qualification_identity_hash

    def _result(
        self, snapshot: QualificationSnapshot, *, idempotent: bool = False
    ) -> QualificationResult:
        return QualificationResult(
            snapshot.summary.qualification_run_id,
            snapshot.summary.current_state,
            self.storage.output_dir(snapshot.summary.qualification_run_id),
            idempotent,
        )

    def _timestamp(self) -> str:
        return self.now().astimezone(UTC).isoformat()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required qualification artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid qualification JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"qualification JSON must contain an object: {path}")
    return payload
