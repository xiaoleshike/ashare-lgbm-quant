"""Governed end-to-end orchestration of retrained Challenger stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import (
    PromotionGatePolicy,
    load_promotion_gate_policy,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.retraining.configuration import RetrainingPolicy, load_retraining_policy
from ashare_quant.retraining.execution import GovernedRetrainingExecutionService
from ashare_quant.retraining.execution.schemas import CandidateRegistration, ExecutionResult
from ashare_quant.retraining.orchestration.controls import LifecycleOperationalControls
from ashare_quant.retraining.orchestration.lifecycle import require_transition
from ashare_quant.retraining.orchestration.recovery import inspect_lifecycle_recovery
from ashare_quant.retraining.orchestration.schemas import (
    LifecycleEvent,
    LifecycleInput,
    LifecycleManifest,
    LifecycleRunResult,
    LifecycleSnapshot,
    LifecycleState,
    LifecycleSummary,
    ObservationProgress,
    RecoveryInspection,
    StageResult,
)
from ashare_quant.retraining.orchestration.stages import (
    latest_successful_shadow_path,
    resolve_promotion_evidence_references,
    track_prospective_observations,
)
from ashare_quant.retraining.orchestration.storage import LifecycleStorage
from ashare_quant.retraining.orchestration.validation import validate_lifecycle_input
from ashare_quant.retraining.readiness import RetrainingExecutionReadinessValidator
from ashare_quant.retraining.readiness.schemas import ReadinessResult
from ashare_quant.retraining.shadow import RetrainedChallengerShadowService
from ashare_quant.retraining.shadow.schemas import RetrainedShadowResult
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.retraining.validation import RetrainingValidationService
from ashare_quant.retraining.validation.schemas import RetrainingValidationResult
from ashare_quant.utils.manifest import config_hash, current_git_info

StopPoint = str | None


class ReadinessRunner(Protocol):
    def validate(self, as_of: str, *, request_id: str | None = None) -> ReadinessResult: ...


class ExecutionRunner(Protocol):
    def execute(self, request_id: str) -> ExecutionResult: ...


class ValidationRunner(Protocol):
    def validate(self, model_id: str) -> RetrainingValidationResult: ...


class ShadowRunner(Protocol):
    def predict(self, model_id: str, *, as_of: str | None = None) -> RetrainedShadowResult: ...


class RetrainingLifecycleOrchestrator:
    """Coordinate immutable services without promotion or Champion mutation."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        retraining_policy_path: Path,
        promotion_policy_path: Path,
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
        self.retraining_policy: RetrainingPolicy = load_retraining_policy(retraining_policy_path)
        self.promotion_policy: PromotionGatePolicy = load_promotion_gate_policy(
            promotion_policy_path
        )
        resolved = config_path.resolve()
        self.project_root = (
            resolved.parent.parent if resolved.parent.name == "config" else Path.cwd().resolve()
        )
        self.lifecycle_lock = self.project_root / "runs" / ".retraining-lifecycle.lock"
        self.now = now or (lambda: datetime.now(UTC))
        self.request_storage = RetrainingRequestStorage(
            reports_root=settings.paths.reports,
            config_path=config_path,
            policy=self.retraining_policy,
            promotion_policy=self.promotion_policy,
        )
        self.storage = LifecycleStorage(settings.paths.reports)
        self.readiness = readiness or RetrainingExecutionReadinessValidator(
            settings=settings,
            config_path=config_path,
            project_root=self.project_root,
            retraining_policy_path=retraining_policy_path,
            promotion_policy_path=promotion_policy_path,
        )
        self.execution = execution or GovernedRetrainingExecutionService(
            settings=settings,
            config_path=config_path,
            retraining_policy_path=retraining_policy_path,
            promotion_policy_path=promotion_policy_path,
        )
        self.validation = validation or RetrainingValidationService(
            settings=settings,
            config_path=config_path,
        )
        self.shadow = shadow or RetrainedChallengerShadowService(
            settings=settings,
            config_path=config_path,
        )

    def run(self, request_id: str, *, stop_after: StopPoint = None) -> LifecycleRunResult:
        """Create one frozen identity or continue the existing request lifecycle."""

        _validate_stop(stop_after)
        with production_lock(self.lifecycle_lock, command=f"retraining lifecycle-run {request_id}"):
            existing = self.storage.find_by_request(request_id)
            if existing is not None:
                frozen = self._frozen_input(existing)
                if stop_after == "readiness":
                    return self._result(existing, idempotent=True)
                return self._continue(existing, frozen, stop_after=stop_after, recheck=True)
            frozen = self._input(request_id)
            identity_hash = self._identity_hash(frozen)
            run_id = f"retraining_lifecycle_{identity_hash[:24]}"
            readiness = self.readiness.validate(
                frozen.request.as_of, request_id=frozen.request.request_id
            )
            snapshot = self._initialize(frozen, readiness, run_id)
            snapshot = self._publish(snapshot, frozen, identity_hash)
            if readiness.report.status != "READY" or stop_after == "readiness":
                return self._result(snapshot)
            return self._continue(snapshot, frozen, stop_after=stop_after, recheck=False)

    def resume(self, lifecycle_run_id: str) -> LifecycleRunResult:
        """Resume the exact frozen identity without recursively calling ``run``."""

        with production_lock(
            self.lifecycle_lock, command=f"retraining lifecycle-resume {lifecycle_run_id}"
        ):
            snapshot = self._required_snapshot(lifecycle_run_id)
            frozen = self._frozen_input(snapshot)
            return self._continue(snapshot, frozen, stop_after=None, recheck=True)

    def revalidate_evidence(self, lifecycle_run_id: str) -> LifecycleRunResult:
        """Re-evaluate exact evidence under current policy without retraining."""

        with production_lock(
            self.lifecycle_lock,
            command=f"retraining lifecycle-revalidate-evidence {lifecycle_run_id}",
        ):
            snapshot = self._required_snapshot(lifecycle_run_id)
            frozen = self._frozen_input(snapshot)
            identity_hash = self._manifest_identity(snapshot)
            if snapshot.summary.mature_sessions < snapshot.summary.required_sessions:
                snapshot = self._policy_event(
                    snapshot,
                    ready=False,
                    warnings=("prospective observations remain insufficient",),
                )
            else:
                snapshot = self._evidence_readiness(snapshot, explicit_revalidation=True)
            snapshot = self._publish(snapshot, frozen, identity_hash)
            return self._result(snapshot)

    def status(self, lifecycle_run_id: str) -> dict[str, Any]:
        snapshot = self.storage.read(lifecycle_run_id)
        if snapshot is None:
            return {"lifecycle_run_id": lifecycle_run_id, "status": "MISSING"}
        current_policy_hash = self.promotion_policy.policy_hash
        evaluated = snapshot.summary.evaluated_promotion_policy_hash
        manifest = snapshot.manifest
        frozen_promotion = manifest.promotion_policy_hash if manifest else None
        frozen_retraining = manifest.retraining_policy_hash if manifest else None
        frozen_lifecycle = manifest.lifecycle_policy_hash if manifest else None
        promotion_drift = current_policy_hash != frozen_promotion
        return {
            **snapshot.summary.model_dump(mode="json"),
            "status": "COMPLETE_SNAPSHOT",
            "frozen_promotion_policy_hash": (
                snapshot.summary.frozen_promotion_policy_hash or frozen_promotion
            ),
            "frozen_retraining_policy_hash": frozen_retraining,
            "current_retraining_policy_hash": self.retraining_policy.policy_hash,
            "frozen_lifecycle_policy_hash": frozen_lifecycle,
            "current_lifecycle_policy_hash": self.retraining_policy.lifecycle_policy_hash,
            "frozen_config_hash": manifest.config_hash if manifest else None,
            "current_config_hash": config_hash(self.config_path),
            "current_promotion_policy_hash": current_policy_hash,
            "evaluated_promotion_policy_hash": evaluated,
            "policy_drift": promotion_drift,
            "evidence_stale": bool(evaluated and evaluated != current_policy_hash),
            "latest_successful_shadow_date": snapshot.summary.shadow_as_of,
            "latest_observation_cutoff": snapshot.summary.observation_cutoff,
            "output": str(self.storage.output_dir(lifecycle_run_id)),
        }

    def recovery(self, lifecycle_run_id: str) -> RecoveryInspection:
        return inspect_lifecycle_recovery(self.storage, lifecycle_run_id)

    def operational_controls(self) -> LifecycleOperationalControls:
        """Return shared fail-closed controls used by execution and dry-run."""

        return LifecycleOperationalControls(
            storage=self.storage,
            timezone=self.settings.production.timezone,
            max_daily_training_runs=self.retraining_policy.lifecycle.max_daily_training_runs,
            cooldown_days=self.retraining_policy.lifecycle.cooldown_days,
            now=self.now(),
        )

    def proposed_identity(self, request_id: str) -> tuple[str, LifecycleInput]:
        """Calculate creation identity without running readiness or any model service."""

        existing = self.storage.find_by_request(request_id)
        frozen = self._frozen_input(existing) if existing else self._input(request_id)
        identity = (
            self._manifest_identity(existing)
            if existing is not None
            else self._identity_hash(frozen)
        )
        return f"retraining_lifecycle_{identity[:24]}", frozen

    def _continue(
        self,
        snapshot: LifecycleSnapshot,
        frozen: LifecycleInput,
        *,
        stop_after: StopPoint,
        recheck: bool,
    ) -> LifecycleRunResult:
        identity_hash = self._manifest_identity(snapshot)
        self._require_identity(snapshot, frozen, identity_hash)
        self._validate_referenced_sources(snapshot)
        snapshot = self._upgrade_legacy_shadow(snapshot, frozen, identity_hash)
        if recheck and not any(event.state == "TRAINING" for event in snapshot.events):
            snapshot = self._recheck_readiness(snapshot, frozen, identity_hash)
        if stop_after == "readiness" or snapshot.summary.current_state == "READINESS_FAILED":
            return self._result(snapshot, idempotent=not recheck)
        snapshot = self._run_training(snapshot, frozen, identity_hash)
        if snapshot.summary.current_state not in _TRAINING_COMPLETE_STATES:
            return self._result(snapshot)
        if stop_after == "training":
            return self._result(snapshot)
        snapshot = self._run_validation(snapshot, frozen, identity_hash)
        if snapshot.summary.current_state not in _VALIDATION_COMPLETE_STATES:
            return self._result(snapshot)
        if stop_after == "validation":
            return self._result(snapshot)
        snapshot = self._run_shadow(snapshot, frozen, identity_hash)
        if snapshot.summary.current_state not in _SHADOW_OBSERVABLE_STATES:
            return self._result(snapshot)
        if stop_after == "shadow":
            return self._result(snapshot)
        snapshot = self._track_observation(snapshot, frozen, identity_hash)
        return self._result(snapshot)

    def _input(self, request_id: str) -> LifecycleInput:
        return validate_lifecycle_input(
            request_id=request_id,
            reports_root=self.settings.paths.reports,
            storage=self.request_storage,
            retraining_policy=self.retraining_policy,
            promotion_policy=self.promotion_policy,
            require_current_policy=True,
        )

    def _frozen_input(self, snapshot: LifecycleSnapshot) -> LifecycleInput:
        manifest = snapshot.manifest
        if manifest is None:
            raise DataValidationError("lifecycle manifest is required for frozen resume")
        try:
            current = validate_lifecycle_input(
                request_id=snapshot.summary.request_id,
                reports_root=self.settings.paths.reports,
                storage=self.request_storage,
                retraining_policy=self.retraining_policy,
                promotion_policy=self.promotion_policy,
                require_current_policy=False,
            )
        except DataValidationError:
            # Test fixtures may replace ``_input`` without publishing a request bundle.
            current = self._input(snapshot.summary.request_id)
        return replace(
            current,
            retraining_policy_hash=manifest.retraining_policy_hash,
            lifecycle_policy_hash=manifest.lifecycle_policy_hash,
            promotion_policy_hash=manifest.promotion_policy_hash,
            frozen_config_hash=manifest.config_hash,
        )

    def _identity_hash(self, frozen: LifecycleInput) -> str:
        return canonical_payload_hash(
            {
                "training_request_hash": frozen.training_request_hash,
                "retraining_policy_hash": frozen.retraining_policy_hash,
                "promotion_policy_hash": frozen.promotion_policy_hash,
                "lifecycle_policy_hash": frozen.lifecycle_policy_hash,
                "config_hash": frozen.frozen_config_hash,
            }
        )

    def _initialize(
        self, frozen: LifecycleInput, readiness: ReadinessResult, run_id: str
    ) -> LifecycleSnapshot:
        target = frozen.request.target_models[0]
        created = self._timestamp()
        events = (
            LifecycleEvent(
                sequence=1,
                state="REQUEST_ACCEPTED",
                created_at=created,
                message="immutable training request accepted",
            ),
            LifecycleEvent(
                sequence=2,
                state="READINESS_CHECKING",
                created_at=created,
                message="existing readiness service invoked",
            ),
            LifecycleEvent(
                sequence=3,
                state=(
                    "READINESS_READY" if readiness.report.status == "READY" else "READINESS_FAILED"
                ),
                created_at=created,
                message=f"readiness status={readiness.report.status}",
                details={"readiness_run_id": readiness.report.run_id},
            ),
        )
        summary = LifecycleSummary(
            lifecycle_run_id=run_id,
            request_id=frozen.request.request_id,
            model_id=None,
            parent_model_id=target.model_id,
            horizon=target.horizon,
            trigger_reasons=tuple(frozen.request.trigger_reason),
            current_state=events[-1].state,
            readiness_run_id=readiness.report.run_id,
            required_sessions=self.retraining_policy.lifecycle.required_sessions(target.horizon),
            frozen_retraining_policy_hash=frozen.retraining_policy_hash,
            frozen_lifecycle_policy_hash=frozen.lifecycle_policy_hash,
            frozen_promotion_policy_hash=frozen.promotion_policy_hash,
            current_promotion_policy_hash=self.promotion_policy.policy_hash,
            policy_drift=self.promotion_policy.policy_hash != frozen.promotion_policy_hash,
            operational_date=self.operational_controls().operational_date.isoformat(),
            created_at=created,
            updated_at=created,
        )
        readiness_path = readiness.output_dir / "manifest.json"
        stage = StageResult(
            stage="readiness",
            status="success" if readiness.report.status == "READY" else "failed",
            artifact_paths=(str(readiness_path),),
            artifact_hashes={"readiness": file_sha256(readiness_path)},
            metrics={"run_id": readiness.report.run_id},
            error=(
                None
                if readiness.report.status == "READY"
                else "; ".join(
                    item.message for item in readiness.report.check_details if item.status != "PASS"
                )
            ),
        )
        return LifecycleSnapshot(summary, events, {"readiness": stage})

    def _recheck_readiness(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        readiness = self.readiness.validate(
            frozen.request.as_of, request_id=frozen.request.request_id
        )
        stage = StageResult(
            stage="readiness_recheck",
            status="success" if readiness.report.status == "READY" else "failed",
            artifact_paths=(str(readiness.output_dir / "manifest.json"),),
            artifact_hashes={
                f"readiness:{readiness.report.run_id}": file_sha256(
                    readiness.output_dir / "manifest.json"
                )
            },
            metrics={"run_id": readiness.report.run_id},
        )
        snapshot = self._with_stage(snapshot, stage, merge_hashes=True)
        if snapshot.summary.current_state in {
            "READINESS_FAILED",
            "READINESS_READY",
            "TRAINING_COOLDOWN_BLOCKED",
            "TRAINING_BUDGET_BLOCKED",
        }:
            snapshot = self._transition(snapshot, "READINESS_CHECKING", "readiness recheck")
        if readiness.report.status != "READY":
            snapshot = self._transition(snapshot, "READINESS_FAILED", "readiness recheck failed")
        else:
            snapshot = self._transition(snapshot, "READINESS_READY", "readiness recheck passed")
        return self._publish(snapshot, frozen, identity_hash)

    def _run_training(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.current_state in _TRAINING_COMPLETE_STATES:
            return snapshot
        if snapshot.summary.current_state not in {
            "READINESS_READY",
            "TRAINING_FAILED",
            "TRAINING_COOLDOWN_BLOCKED",
            "TRAINING_BUDGET_BLOCKED",
        }:
            return snapshot
        controls = self.operational_controls()
        if not any(event.state == "TRAINING" for event in snapshot.events):
            cooldown = controls.cooldown(
                lifecycle_run_id=snapshot.summary.lifecycle_run_id,
                parent_model_id=snapshot.summary.parent_model_id,
                horizon=snapshot.summary.horizon,
            )
            if not cooldown.allowed:
                snapshot = self._transition(
                    snapshot,
                    "TRAINING_COOLDOWN_BLOCKED",
                    "lifecycle training cooldown is active",
                    details=_dataclass_dict(cooldown),
                    operational_date=cooldown.operational_date,
                )
                return self._publish(snapshot, frozen, identity_hash)
        budget = controls.budget()
        if not budget.allowed:
            snapshot = self._transition(
                snapshot,
                "TRAINING_BUDGET_BLOCKED",
                "daily training-attempt budget is exhausted",
                details=_dataclass_dict(budget),
                operational_date=budget.operational_date,
            )
            return self._publish(snapshot, frozen, identity_hash)
        snapshot = self._transition(
            snapshot,
            "TRAINING",
            "training execution started",
            details={"budget_decision": _dataclass_dict(budget)},
            operational_date=budget.operational_date,
        )
        snapshot = self._publish(snapshot, frozen, identity_hash)
        try:
            result = self.execution.execute(frozen.request.request_id)
            registration_path = (
                self.settings.paths.models
                / "candidate_registrations"
                / result.model_id
                / "registration.json"
            )
            registration = CandidateRegistration.model_validate(_json(registration_path))
            artifact_manifest = (
                result.artifact_dir / "manifest.json" if result.artifact_dir else None
            )
            if artifact_manifest is None or not artifact_manifest.is_file():
                raise DataValidationError("training result lacks immutable model artifact")
            stage = StageResult(
                stage="training",
                status="success",
                artifact_paths=(
                    str(result.output_dir / "manifest.json"),
                    str(artifact_manifest),
                    str(registration_path),
                ),
                artifact_hashes={
                    "execution": file_sha256(result.output_dir / "manifest.json"),
                    "model_artifact": file_sha256(artifact_manifest),
                    "candidate_registration": file_sha256(registration_path),
                },
                metrics={
                    "training_run_id": result.training_run_id,
                    "model_id": result.model_id,
                    "candidate_registration_id": registration.candidate_registration_id,
                },
            )
            snapshot = self._with_stage(snapshot, stage)
            snapshot = self._transition(
                snapshot,
                "TRAINING_COMPLETED",
                "candidate artifact and registration completed",
                {"training_run_id": result.training_run_id, "model_id": result.model_id},
                training_run_id=result.training_run_id,
                model_id=result.model_id,
            )
        except Exception as error:
            snapshot = self._with_stage(
                snapshot,
                StageResult(
                    stage=f"training_attempt:{len(snapshot.events)}",
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            snapshot = self._transition(
                snapshot, "TRAINING_FAILED", f"training failed: {type(error).__name__}: {error}"
            )
        return self._publish(snapshot, frozen, identity_hash)

    def _run_validation(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.current_state in _VALIDATION_COMPLETE_STATES:
            return snapshot
        if snapshot.summary.current_state not in {"TRAINING_COMPLETED", "VALIDATION_FAILED"}:
            return snapshot
        model_id = snapshot.summary.model_id
        assert model_id is not None
        snapshot = self._transition(snapshot, "VALIDATING", "candidate validation started")
        snapshot = self._publish(snapshot, frozen, identity_hash)
        try:
            result = self.validation.validate(model_id)
            manifest_path = result.output_dir / "manifest.json"
            manifest = _json(manifest_path)
            eligibility_path = result.output_dir / "shadow" / "eligibility.json"
            if (
                result.status != "COMPLETED"
                or manifest.get("training_run_id") != snapshot.summary.training_run_id
                or manifest.get("model_id") != model_id
                or _json(eligibility_path).get("shadow_eligible") is not True
            ):
                raise DataValidationError("validation result is not shadow eligible")
            stage = StageResult(
                stage="validation",
                status="success",
                artifact_paths=(str(manifest_path), str(eligibility_path)),
                artifact_hashes={
                    "validation": file_sha256(manifest_path),
                    "shadow_eligibility": file_sha256(eligibility_path),
                    "offline_validation": str(manifest["offline_validation_hash"]),
                    "executable_validation": str(manifest["executable_validation_hash"]),
                },
                metrics={"validation_run_id": result.run_id},
            )
            snapshot = self._with_stage(snapshot, stage)
            snapshot = self._transition(
                snapshot,
                "VALIDATION_COMPLETED",
                "offline, executable, and shadow eligibility validation completed",
                {"validation_run_id": result.run_id},
                validation_run_id=result.run_id,
            )
        except Exception as error:
            snapshot = self._with_stage(
                snapshot,
                StageResult(
                    stage=f"validation_attempt:{len(snapshot.events)}",
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            snapshot = self._transition(
                snapshot,
                "VALIDATION_FAILED",
                f"validation failed: {type(error).__name__}: {error}",
            )
        return self._publish(snapshot, frozen, identity_hash)

    def _run_shadow(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        assert snapshot.summary.model_id is not None
        model_id = snapshot.summary.model_id
        initial = not _has_successful_shadow(snapshot)
        if initial and snapshot.summary.current_state not in {
            "VALIDATION_COMPLETED",
            "SHADOW_FAILED",
        }:
            return snapshot
        if not initial and snapshot.summary.current_state not in _SHADOW_OBSERVABLE_STATES:
            return snapshot
        if initial:
            snapshot = self._transition(snapshot, "SHADOW_ENROLLING", "shadow enrollment started")
            snapshot = self._publish(snapshot, frozen, identity_hash)
        try:
            result = self.shadow.predict(
                model_id,
                as_of=frozen.request.as_of if initial else None,
            )
            manifest_path = result.output_dir / "manifest.json"
            manifest = _json(manifest_path)
            self._validate_shadow_manifest(snapshot, frozen, result, manifest)
            if not initial and result.shadow_run_id in snapshot.summary.successful_shadow_run_ids:
                return snapshot
            stage_name = "shadow_enrollment" if initial else "shadow_refresh"
            stage = StageResult(
                stage=stage_name,
                status="success",
                artifact_paths=(str(manifest_path),),
                artifact_hashes={f"shadow:{result.shadow_run_id}": file_sha256(manifest_path)},
                metrics={
                    "shadow_run_ids": [result.shadow_run_id],
                    "production_run_id": manifest.get("production_run_id"),
                    "as_of": result.as_of,
                },
            )
            snapshot = self._with_stage(snapshot, stage, merge_hashes=True, merge_metrics=True)
            shadow_ids = tuple(
                dict.fromkeys((*snapshot.summary.successful_shadow_run_ids, result.shadow_run_id))
            )
            if initial:
                snapshot = self._transition(
                    snapshot,
                    "SHADOW_ENROLLED",
                    "retrained Challenger enrolled in prospective shadow",
                    {"shadow_run_id": result.shadow_run_id, "as_of": result.as_of},
                    shadow_run_id=result.shadow_run_id,
                    production_run_id=str(manifest.get("production_run_id", "")),
                    shadow_as_of=result.as_of,
                    successful_shadow_run_ids=shadow_ids,
                )
            else:
                snapshot = self._transition(
                    snapshot,
                    snapshot.summary.current_state,
                    "daily retrained Shadow refresh succeeded",
                    {"shadow_run_id": result.shadow_run_id, "as_of": result.as_of},
                    shadow_run_id=result.shadow_run_id,
                    production_run_id=str(manifest.get("production_run_id", "")),
                    shadow_as_of=result.as_of,
                    successful_shadow_run_ids=shadow_ids,
                )
        except Exception as error:
            stage_name = "shadow_enrollment_attempt" if initial else "shadow_refresh_attempt"
            snapshot = self._with_stage(
                snapshot,
                StageResult(
                    stage=f"{stage_name}:{len(snapshot.events)}",
                    status="failed",
                    warnings=(
                        ()
                        if initial
                        else ("daily Shadow refresh failed; successful enrollment is preserved",)
                    ),
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            if initial:
                snapshot = self._transition(
                    snapshot,
                    "SHADOW_FAILED",
                    f"shadow enrollment failed: {type(error).__name__}: {error}",
                )
            else:
                snapshot = self._transition(
                    snapshot,
                    snapshot.summary.current_state,
                    "daily Shadow refresh failed; enrollment remains valid",
                    {"error": f"{type(error).__name__}: {error}"},
                )
        return self._publish(snapshot, frozen, identity_hash)

    def _track_observation(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.current_state == "SHADOW_ENROLLED":
            snapshot = self._transition(
                snapshot, "OBSERVATION_PENDING", "waiting for mature prospective observations"
            )
            snapshot = self._publish(snapshot, frozen, identity_hash)
        assert snapshot.summary.model_id is not None
        assert snapshot.summary.training_run_id is not None
        assert snapshot.summary.validation_run_id is not None
        progress = track_prospective_observations(
            reports_root=self.settings.paths.reports,
            model_id=snapshot.summary.model_id,
            horizon=snapshot.summary.horizon,
            training_run_id=snapshot.summary.training_run_id,
            validation_run_id=snapshot.summary.validation_run_id,
            required_sessions=snapshot.summary.required_sessions,
            accepted_shadow_run_ids=snapshot.summary.successful_shadow_run_ids,
        )
        previous = snapshot.stage_results.get("observation")
        if previous is not None:
            removed = sorted(set(previous.artifact_hashes) - set(progress.source_hashes))
            changed = sorted(
                key
                for key, digest in previous.artifact_hashes.items()
                if progress.source_hashes.get(key) not in {None, digest}
            )
            if removed or changed:
                raise DataValidationError(
                    "observation evidence changed or disappeared; "
                    f"removed={removed} changed={changed}"
                )
        stage = StageResult(
            stage="observation",
            status="success" if progress.mature_sessions else "pending",
            artifact_paths=tuple(progress.source_artifacts.values()),
            artifact_hashes=progress.source_hashes,
            metrics={
                "mature_sessions": progress.mature_sessions,
                "required_sessions": progress.required_sessions,
                "shadow_run_ids": list(progress.shadow_run_ids),
                "first_signal_date": progress.first_signal_date,
                "latest_signal_date": progress.latest_signal_date,
                "observation_cutoff": progress.observation_cutoff,
                "aggregate_hash": progress.aggregate_hash,
            },
        )
        changed_progress = previous != stage or any(
            (
                snapshot.summary.mature_sessions != progress.mature_sessions,
                snapshot.summary.first_observation_date != progress.first_signal_date,
                snapshot.summary.latest_observation_date != progress.latest_signal_date,
                snapshot.summary.observation_cutoff != progress.observation_cutoff,
                snapshot.summary.observation_evidence_hash != progress.aggregate_hash,
            )
        )
        if not changed_progress:
            return self._apply_policy_drift_if_needed(snapshot, frozen, identity_hash)
        added = sorted(
            set(progress.source_hashes) - set(previous.artifact_hashes if previous else {})
        )
        details = {
            "previous_mature_sessions": snapshot.summary.mature_sessions,
            "current_mature_sessions": progress.mature_sessions,
            "required_sessions": progress.required_sessions,
            "first_signal_date": progress.first_signal_date,
            "latest_signal_date": progress.latest_signal_date,
            "observation_cutoff": progress.observation_cutoff,
            "added_source_artifacts": added,
            "removed_source_artifacts": [],
            "previous_aggregate_hash": snapshot.summary.observation_evidence_hash,
            "current_aggregate_hash": progress.aggregate_hash,
            "accepted_shadow_run_ids": list(progress.shadow_run_ids),
        }
        snapshot = self._with_stage(snapshot, stage)
        target: LifecycleState = cast(LifecycleState, progress.status)
        if snapshot.summary.current_state in {"EVIDENCE_READY", "POLICY_REVIEW_REQUIRED"}:
            target = snapshot.summary.current_state
        snapshot = self._transition(
            snapshot,
            target,
            f"prospective observation progress={progress.mature_sessions}",
            details,
            observation_status=progress.status,
            mature_sessions=progress.mature_sessions,
            first_observation_date=progress.first_signal_date,
            latest_observation_date=progress.latest_signal_date,
            observation_cutoff=progress.observation_cutoff,
            observation_evidence_hash=progress.aggregate_hash,
        )
        if progress.status == "OBSERVATION_SUFFICIENT":
            snapshot = self._evidence_readiness(snapshot, explicit_revalidation=False)
        return self._publish(snapshot, frozen, identity_hash)

    def _apply_policy_drift_if_needed(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        drift = self.promotion_policy.policy_hash != frozen.promotion_policy_hash
        stale = bool(
            snapshot.summary.evaluated_promotion_policy_hash
            and snapshot.summary.evaluated_promotion_policy_hash
            != self.promotion_policy.policy_hash
        )
        if not drift and not stale:
            return snapshot
        if snapshot.summary.current_state in {"OBSERVATION_SUFFICIENT", "EVIDENCE_READY"}:
            snapshot = self._transition(
                snapshot,
                "POLICY_REVIEW_REQUIRED",
                "current Promotion Policy differs from lifecycle evidence policy",
                {
                    "frozen_policy_hash": frozen.promotion_policy_hash,
                    "current_policy_hash": self.promotion_policy.policy_hash,
                },
                promotion_evidence_status="POLICY_REVIEW_REQUIRED",
                current_promotion_policy_hash=self.promotion_policy.policy_hash,
                policy_drift=drift,
                evidence_stale=True,
            )
            return self._publish(snapshot, frozen, identity_hash)
        return snapshot

    def _evidence_readiness(
        self, snapshot: LifecycleSnapshot, *, explicit_revalidation: bool
    ) -> LifecycleSnapshot:
        frozen_hash = snapshot.summary.frozen_promotion_policy_hash or (
            snapshot.manifest.promotion_policy_hash if snapshot.manifest else ""
        )
        current_hash = self.promotion_policy.policy_hash
        if current_hash != frozen_hash and not explicit_revalidation:
            return self._policy_event(snapshot, ready=False, warnings=("Promotion Policy drift",))
        training = snapshot.stage_results["training"]
        validation = snapshot.stage_results["validation"]
        observation = snapshot.stage_results["observation"]
        shadow_path = latest_successful_shadow_path(snapshot.stage_results)
        progress = ObservationProgress(
            status="OBSERVATION_SUFFICIENT",
            mature_sessions=snapshot.summary.mature_sessions,
            required_sessions=snapshot.summary.required_sessions,
            source_artifacts={
                key: path
                for key, path in zip(
                    sorted(observation.artifact_hashes), observation.artifact_paths, strict=True
                )
            },
            source_hashes=observation.artifact_hashes,
            shadow_run_ids=tuple(
                str(value) for value in observation.metrics.get("shadow_run_ids", [])
            ),
            first_signal_date=snapshot.summary.first_observation_date,
            latest_signal_date=snapshot.summary.latest_observation_date,
            observation_cutoff=snapshot.summary.observation_cutoff,
            aggregate_hash=snapshot.summary.observation_evidence_hash,
        )
        ready, paths, hashes, warnings, references = resolve_promotion_evidence_references(
            reports_root=self.settings.paths.reports,
            lifecycle_run_id=snapshot.summary.lifecycle_run_id,
            request_id=snapshot.summary.request_id,
            model_id=str(snapshot.summary.model_id),
            parent_model_id=snapshot.summary.parent_model_id,
            horizon=snapshot.summary.horizon,
            training_run_id=str(snapshot.summary.training_run_id),
            validation_run_id=str(snapshot.summary.validation_run_id),
            execution_path=Path(training.artifact_paths[0]),
            validation_path=Path(validation.artifact_paths[0]),
            shadow_path=shadow_path,
            observation=progress,
            policy=self.promotion_policy,
        )
        evaluation_stage = StageResult(
            stage=f"promotion_evidence_evaluation:{current_hash[:16]}",
            status="success" if ready else "pending",
            artifact_paths=tuple(paths.values()),
            artifact_hashes=hashes,
            metrics={
                "status": "READY_FOR_PREPARATION" if ready else "NOT_READY",
                "evaluated_promotion_policy_hash": current_hash,
                "references": [item.model_dump(mode="json") for item in references],
                "observation_cutoff": snapshot.summary.observation_cutoff,
            },
            warnings=warnings,
        )
        snapshot = self._with_stage(snapshot, evaluation_stage)
        if ready:
            snapshot = self._with_stage(
                snapshot,
                evaluation_stage.model_copy(update={"stage": "promotion_evidence"}),
                merge_hashes=True,
                merge_metrics=True,
            )
        return self._policy_event(snapshot, ready=ready, warnings=warnings)

    def _policy_event(
        self, snapshot: LifecycleSnapshot, *, ready: bool, warnings: tuple[str, ...]
    ) -> LifecycleSnapshot:
        current_hash = self.promotion_policy.policy_hash
        frozen_hash = snapshot.summary.frozen_promotion_policy_hash or (
            snapshot.manifest.promotion_policy_hash if snapshot.manifest else current_hash
        )
        if ready:
            return self._transition(
                snapshot,
                "EVIDENCE_READY",
                "exact immutable evidence is ready for separate preparation",
                {"evaluated_policy_hash": current_hash},
                promotion_evidence_status="READY_FOR_PREPARATION",
                evaluated_promotion_policy_hash=current_hash,
                current_promotion_policy_hash=current_hash,
                policy_drift=current_hash != frozen_hash,
                evidence_stale=False,
            )
        target: LifecycleState = (
            "POLICY_REVIEW_REQUIRED"
            if current_hash != frozen_hash
            else snapshot.summary.current_state
        )
        return self._transition(
            snapshot,
            target,
            "promotion evidence is not ready under evaluated policy",
            {"warnings": list(warnings), "evaluated_policy_hash": current_hash},
            promotion_evidence_status=(
                "POLICY_REVIEW_REQUIRED" if current_hash != frozen_hash else "NOT_READY"
            ),
            evaluated_promotion_policy_hash=current_hash,
            current_promotion_policy_hash=current_hash,
            policy_drift=current_hash != frozen_hash,
            evidence_stale=bool(current_hash != frozen_hash),
        )

    def _transition(
        self,
        snapshot: LifecycleSnapshot,
        state: LifecycleState,
        message: str,
        details: dict[str, Any] | None = None,
        **summary_updates: object,
    ) -> LifecycleSnapshot:
        require_transition(snapshot.summary.current_state, state)
        timestamp = self._timestamp()
        event = LifecycleEvent(
            sequence=len(snapshot.events) + 1,
            state=state,
            created_at=timestamp,
            message=message,
            details=json.loads(json.dumps(details or {}, sort_keys=True)),
        )
        summary = snapshot.summary.model_copy(
            update={"current_state": state, "updated_at": timestamp, **summary_updates}
        )
        return LifecycleSnapshot(summary, (*snapshot.events, event), snapshot.stage_results)

    def _with_stage(
        self,
        snapshot: LifecycleSnapshot,
        stage: StageResult,
        *,
        merge_hashes: bool = False,
        merge_metrics: bool = False,
    ) -> LifecycleSnapshot:
        stages = dict(snapshot.stage_results)
        previous = stages.get(stage.stage)
        if previous is not None and (merge_hashes or merge_metrics):
            metrics = dict(previous.metrics)
            metrics.update(stage.metrics)
            if merge_metrics and "shadow_run_ids" in previous.metrics:
                metrics["shadow_run_ids"] = list(
                    dict.fromkeys(
                        [
                            *cast(list[str], previous.metrics["shadow_run_ids"]),
                            *cast(list[str], stage.metrics.get("shadow_run_ids", [])),
                        ]
                    )
                )
            stage = stage.model_copy(
                update={
                    "artifact_paths": tuple(
                        dict.fromkeys((*previous.artifact_paths, *stage.artifact_paths))
                    ),
                    "artifact_hashes": {**previous.artifact_hashes, **stage.artifact_hashes},
                    "metrics": metrics,
                }
            )
        stages[stage.stage] = stage
        return LifecycleSnapshot(snapshot.summary, snapshot.events, stages)

    def _publish(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        source_paths: dict[str, str] = {
            "training_request": str(
                self.request_storage.requests_root
                / frozen.request.request_id
                / "training_request.json"
            )
        }
        source_hashes = {"training_request": frozen.training_request_hash}
        for stage_name, stage in snapshot.stage_results.items():
            for index, path in enumerate(stage.artifact_paths):
                source_paths[f"{stage_name}:{index}"] = path
            source_hashes.update(
                {f"{stage_name}:{name}": digest for name, digest in stage.artifact_hashes.items()}
            )
        git = current_git_info()
        summary = snapshot.summary
        manifest = LifecycleManifest(
            lifecycle_identity_hash=identity_hash,
            lifecycle_run_id=summary.lifecycle_run_id,
            request_id=summary.request_id,
            model_id=summary.model_id,
            parent_model_id=summary.parent_model_id,
            horizon=summary.horizon,
            current_state=summary.current_state,
            readiness_run_id=summary.readiness_run_id,
            training_run_id=summary.training_run_id,
            validation_run_id=summary.validation_run_id,
            shadow_run_id=summary.shadow_run_id,
            production_run_id=summary.production_run_id,
            shadow_as_of=summary.shadow_as_of,
            successful_shadow_run_ids=summary.successful_shadow_run_ids,
            first_observation_date=summary.first_observation_date,
            latest_observation_date=summary.latest_observation_date,
            observation_cutoff=summary.observation_cutoff,
            observation_evidence_hash=summary.observation_evidence_hash,
            observation_status=summary.observation_status,
            mature_sessions=summary.mature_sessions,
            required_sessions=summary.required_sessions,
            promotion_evidence_status=summary.promotion_evidence_status,
            retraining_policy_hash=frozen.retraining_policy_hash,
            lifecycle_policy_hash=frozen.lifecycle_policy_hash,
            promotion_policy_hash=frozen.promotion_policy_hash,
            evaluated_promotion_policy_hash=summary.evaluated_promotion_policy_hash,
            current_promotion_policy_hash=self.promotion_policy.policy_hash,
            policy_drift=self.promotion_policy.policy_hash != frozen.promotion_policy_hash,
            evidence_stale=summary.evidence_stale,
            operational_date=summary.operational_date,
            evidence_hash=frozen.request.evidence_hash,
            training_request_hash=frozen.training_request_hash,
            source_artifacts=source_paths,
            source_hashes=source_hashes,
            summary_sha256="0" * 64,
            events_sha256="0" * 64,
            stage_results_sha256="0" * 64,
            report_sha256="0" * 64,
            git_commit=git["commit"],
            git_dirty=bool(git["dirty"]),
            config_hash=frozen.frozen_config_hash,
        )
        return self.storage.publish(snapshot, manifest)

    def _require_identity(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> None:
        manifest = snapshot.manifest
        if manifest is None or (
            manifest.lifecycle_identity_hash != identity_hash
            or manifest.request_id != frozen.request.request_id
            or manifest.training_request_hash != frozen.training_request_hash
            or manifest.retraining_policy_hash != frozen.retraining_policy_hash
            or manifest.lifecycle_policy_hash != frozen.lifecycle_policy_hash
            or manifest.promotion_policy_hash != frozen.promotion_policy_hash
        ):
            raise DataValidationError("existing lifecycle has conflicting immutable lineage")

    def _validate_referenced_sources(self, snapshot: LifecycleSnapshot) -> None:
        for name, stage in snapshot.stage_results.items():
            if stage.status != "success":
                continue
            expected = set(stage.artifact_hashes.values())
            for raw in stage.artifact_paths:
                path = Path(raw)
                if not path.is_file():
                    raise DataValidationError(f"lifecycle source disappeared: {name}: {path}")
                if file_sha256(path) not in expected:
                    raise DataValidationError(f"lifecycle source hash changed: {name}: {path}")

    def _validate_shadow_manifest(
        self,
        snapshot: LifecycleSnapshot,
        frozen: LifecycleInput,
        result: RetrainedShadowResult,
        manifest: dict[str, Any],
    ) -> None:
        expected = {
            "model_id": snapshot.summary.model_id,
            "model_origin": "retrained_challenger",
            "training_request_id": frozen.request.request_id,
            "training_run_id": snapshot.summary.training_run_id,
            "validation_run_id": snapshot.summary.validation_run_id,
            "access_policy": "prospective_production",
            "shadow_run_id": result.shadow_run_id,
        }
        mismatch = [key for key, value in expected.items() if manifest.get(key) != value]
        models = manifest.get("models")
        model = (
            next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("model_id") == snapshot.summary.model_id
                ),
                None,
            )
            if isinstance(models, list)
            else None
        )
        if model is not None and model.get("native_horizon") != snapshot.summary.horizon:
            mismatch.append("horizon")
        if mismatch:
            raise DataValidationError(f"retrained Shadow lineage is conflicting: {mismatch}")

    def _upgrade_legacy_shadow(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.successful_shadow_run_ids or not _has_successful_shadow(snapshot):
            return snapshot
        legacy = snapshot.stage_results.get("shadow")
        if legacy is None or legacy.status != "success" or not legacy.artifact_paths:
            raise DataValidationError("legacy lifecycle lacks verifiable successful Shadow lineage")
        shadow_ids: list[str] = []
        latest_as_of: str | None = None
        latest_production: str | None = None
        for raw in legacy.artifact_paths:
            payload = _json(Path(raw))
            expected = {
                "model_id": snapshot.summary.model_id,
                "model_origin": "retrained_challenger",
                "training_request_id": frozen.request.request_id,
                "training_run_id": snapshot.summary.training_run_id,
                "validation_run_id": snapshot.summary.validation_run_id,
                "access_policy": "prospective_production",
            }
            if any(payload.get(key) != value for key, value in expected.items()):
                raise DataValidationError("ambiguous legacy Shadow lineage requires recovery")
            shadow_id = payload.get("shadow_run_id")
            if not isinstance(shadow_id, str) or not shadow_id:
                raise DataValidationError("legacy Shadow manifest lacks shadow_run_id")
            shadow_ids.append(shadow_id)
            latest_as_of = str(payload.get("as_of") or latest_as_of or "") or None
            latest_production = (
                str(payload.get("production_run_id") or latest_production or "") or None
            )
        snapshot = self._transition(
            snapshot,
            snapshot.summary.current_state,
            "verified legacy successful Shadow enrollment",
            {"shadow_run_ids": sorted(set(shadow_ids))},
            successful_shadow_run_ids=tuple(sorted(set(shadow_ids))),
            shadow_run_id=shadow_ids[-1],
            shadow_as_of=latest_as_of or snapshot.summary.shadow_as_of,
            production_run_id=latest_production or snapshot.summary.production_run_id,
        )
        return self._publish(snapshot, frozen, identity_hash)

    def _required_snapshot(self, lifecycle_run_id: str) -> LifecycleSnapshot:
        snapshot = self.storage.read(lifecycle_run_id)
        if snapshot is None:
            raise DataValidationError(f"lifecycle run does not exist: {lifecycle_run_id}")
        return snapshot

    @staticmethod
    def _manifest_identity(snapshot: LifecycleSnapshot) -> str:
        if snapshot.manifest is None:
            raise DataValidationError("lifecycle manifest identity is missing")
        return snapshot.manifest.lifecycle_identity_hash

    def _result(
        self, snapshot: LifecycleSnapshot, *, idempotent: bool = False
    ) -> LifecycleRunResult:
        return LifecycleRunResult(
            snapshot.summary.lifecycle_run_id,
            snapshot.summary.request_id,
            snapshot.summary.current_state,
            snapshot.summary.model_id,
            self.storage.output_dir(snapshot.summary.lifecycle_run_id),
            idempotent,
        )

    def _timestamp(self) -> str:
        return self.now().astimezone(UTC).isoformat()


def _has_successful_shadow(snapshot: LifecycleSnapshot) -> bool:
    return any(
        stage.status == "success"
        for name, stage in snapshot.stage_results.items()
        if name in {"shadow", "shadow_enrollment", "shadow_refresh"}
    )


def _dataclass_dict(value: object) -> dict[str, Any]:
    slots = getattr(type(value), "__slots__", ())
    return {str(name): getattr(value, name) for name in slots}


def _validate_stop(value: StopPoint) -> None:
    if value not in {None, "readiness", "training", "validation", "shadow"}:
        raise DataValidationError(f"unsupported lifecycle stop point: {value}")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid lifecycle source JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"lifecycle source must contain an object: {path}")
    return value


_TRAINING_COMPLETE_STATES: frozenset[str] = frozenset(
    {
        "TRAINING_COMPLETED",
        "VALIDATING",
        "VALIDATION_FAILED",
        "VALIDATION_COMPLETED",
        "SHADOW_ENROLLING",
        "SHADOW_FAILED",
        "SHADOW_ENROLLED",
        "OBSERVATION_PENDING",
        "OBSERVATION_ACCUMULATING",
        "OBSERVATION_SUFFICIENT",
        "POLICY_REVIEW_REQUIRED",
        "EVIDENCE_READY",
    }
)
_VALIDATION_COMPLETE_STATES: frozenset[str] = frozenset(
    {
        "VALIDATION_COMPLETED",
        "SHADOW_ENROLLING",
        "SHADOW_FAILED",
        "SHADOW_ENROLLED",
        "OBSERVATION_PENDING",
        "OBSERVATION_ACCUMULATING",
        "OBSERVATION_SUFFICIENT",
        "POLICY_REVIEW_REQUIRED",
        "EVIDENCE_READY",
    }
)
_SHADOW_OBSERVABLE_STATES: frozenset[str] = frozenset(
    {
        "SHADOW_ENROLLED",
        "OBSERVATION_PENDING",
        "OBSERVATION_ACCUMULATING",
        "OBSERVATION_SUFFICIENT",
        "POLICY_REVIEW_REQUIRED",
        "EVIDENCE_READY",
    }
)
