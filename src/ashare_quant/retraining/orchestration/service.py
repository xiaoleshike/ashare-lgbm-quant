"""Governed end-to-end orchestration of retrained Challenger stages."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

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
    RecoveryInspection,
    StageResult,
)
from ashare_quant.retraining.orchestration.stages import (
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
        """Run or resume one deterministic lifecycle under its non-nested lock."""

        _validate_stop(stop_after)
        with production_lock(self.lifecycle_lock, command=f"retraining lifecycle-run {request_id}"):
            frozen = self._input(request_id)
            readiness = self.readiness.validate(
                frozen.request.as_of, request_id=frozen.request.request_id
            )
            identity_hash = canonical_payload_hash(
                {
                    "training_request_hash": frozen.training_request_hash,
                    "retraining_policy_hash": frozen.retraining_policy_hash,
                    "promotion_policy_hash": frozen.promotion_policy_hash,
                    "lifecycle_policy_hash": frozen.lifecycle_policy_hash,
                    "readiness_identity": readiness.report.run_id,
                    "config_hash": config_hash(self.config_path),
                }
            )
            run_id = f"retraining_lifecycle_{identity_hash[:24]}"
            snapshot = self.storage.read(run_id)
            if snapshot is None:
                snapshot = self._initialize(frozen, readiness, run_id)
                snapshot = self._publish(snapshot, frozen, identity_hash)
            else:
                self._require_identity(snapshot, frozen, readiness, identity_hash)
            if readiness.report.status != "READY":
                return self._result(snapshot)
            if stop_after == "readiness" or snapshot.summary.current_state == "EVIDENCE_READY":
                return self._result(snapshot, idempotent=True)
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
            if snapshot.summary.current_state not in {
                "SHADOW_ENROLLED",
                "OBSERVATION_PENDING",
                "OBSERVATION_ACCUMULATING",
                "OBSERVATION_SUFFICIENT",
                "EVIDENCE_READY",
            }:
                return self._result(snapshot)
            if stop_after == "shadow":
                return self._result(snapshot)
            snapshot = self._track_observation(snapshot, frozen, identity_hash)
            return self._result(snapshot)

    def resume(self, lifecycle_run_id: str) -> LifecycleRunResult:
        snapshot = self.storage.read(lifecycle_run_id)
        if snapshot is None:
            raise DataValidationError(f"lifecycle run does not exist: {lifecycle_run_id}")
        result = self.run(snapshot.summary.request_id)
        if result.lifecycle_run_id != lifecycle_run_id:
            raise DataValidationError("lifecycle logical identity changed; resume is prohibited")
        return result

    def status(self, lifecycle_run_id: str) -> dict[str, Any]:
        snapshot = self.storage.read(lifecycle_run_id)
        if snapshot is None:
            return {"lifecycle_run_id": lifecycle_run_id, "status": "MISSING"}
        return {
            **snapshot.summary.model_dump(mode="json"),
            "status": "COMPLETE_SNAPSHOT",
            "output": str(self.storage.output_dir(lifecycle_run_id)),
        }

    def recovery(self, lifecycle_run_id: str) -> RecoveryInspection:
        return inspect_lifecycle_recovery(self.storage, lifecycle_run_id)

    def _input(self, request_id: str) -> LifecycleInput:
        return validate_lifecycle_input(
            request_id=request_id,
            reports_root=self.settings.paths.reports,
            storage=self.request_storage,
            retraining_policy=self.retraining_policy,
            promotion_policy=self.promotion_policy,
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
        state = events[-1].state
        summary = LifecycleSummary(
            lifecycle_run_id=run_id,
            request_id=frozen.request.request_id,
            model_id=None,
            parent_model_id=target.model_id,
            horizon=target.horizon,
            trigger_reasons=tuple(frozen.request.trigger_reason),
            current_state=state,
            readiness_run_id=readiness.report.run_id,
            required_sessions=self.retraining_policy.lifecycle.required_sessions(target.horizon),
            created_at=created,
            updated_at=created,
        )
        readiness_path = readiness.output_dir / "manifest.json"
        stages = {
            "readiness": StageResult(
                stage="readiness",
                status="success" if readiness.report.status == "READY" else "failed",
                artifact_paths=(str(readiness_path),),
                artifact_hashes={"readiness": file_sha256(readiness_path)},
                metrics={"run_id": readiness.report.run_id},
                error=(
                    None
                    if readiness.report.status == "READY"
                    else "; ".join(
                        item.message
                        for item in readiness.report.check_details
                        if item.status != "PASS"
                    )
                ),
            )
        }
        return LifecycleSnapshot(summary, events, stages)

    def _run_training(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.current_state in _TRAINING_COMPLETE_STATES:
            return snapshot
        if snapshot.summary.current_state not in {"READINESS_READY", "TRAINING_FAILED"}:
            return snapshot
        self._enforce_daily_training_limit(snapshot)
        snapshot = self._transition(snapshot, "TRAINING", "training execution started")
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
                {
                    "training_run_id": result.training_run_id,
                    "model_id": result.model_id,
                },
                training_run_id=result.training_run_id,
                model_id=result.model_id,
            )
        except Exception as error:
            snapshot = self._with_stage(
                snapshot,
                StageResult(
                    stage="training",
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
                    stage="validation",
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
        initial = snapshot.summary.current_state in {"VALIDATION_COMPLETED", "SHADOW_FAILED"}
        if initial:
            snapshot = self._transition(snapshot, "SHADOW_ENROLLING", "shadow enrollment started")
            snapshot = self._publish(snapshot, frozen, identity_hash)
        elif snapshot.summary.current_state not in {
            "SHADOW_ENROLLED",
            "OBSERVATION_PENDING",
            "OBSERVATION_ACCUMULATING",
            "OBSERVATION_SUFFICIENT",
            "EVIDENCE_READY",
        }:
            return snapshot
        try:
            result = self.shadow.predict(
                model_id,
                as_of=frozen.request.as_of if initial else None,
            )
            manifest_path = result.output_dir / "manifest.json"
            manifest = _json(manifest_path)
            if (
                manifest.get("model_origin") != "retrained_challenger"
                or manifest.get("training_run_id") != snapshot.summary.training_run_id
                or manifest.get("validation_run_id") != snapshot.summary.validation_run_id
                or manifest.get("access_policy") != "prospective_production"
            ):
                raise DataValidationError("retrained shadow lineage is conflicting")
            stage = StageResult(
                stage="shadow",
                status="success",
                artifact_paths=(str(manifest_path),),
                artifact_hashes={f"shadow:{result.as_of}": file_sha256(manifest_path)},
                metrics={
                    "shadow_run_id": result.shadow_run_id,
                    "production_run_id": manifest.get("production_run_id"),
                    "as_of": result.as_of,
                },
            )
            snapshot = self._with_stage(snapshot, stage, merge_hashes=True)
            if initial:
                snapshot = self._transition(
                    snapshot,
                    "SHADOW_ENROLLED",
                    "retrained Challenger enrolled in prospective shadow",
                    {"shadow_run_id": result.shadow_run_id, "as_of": result.as_of},
                    shadow_run_id=result.shadow_run_id,
                    production_run_id=str(manifest.get("production_run_id", "")),
                    shadow_as_of=result.as_of,
                )
                snapshot = self._publish(snapshot, frozen, identity_hash)
            elif result.shadow_run_id != snapshot.summary.shadow_run_id:
                snapshot = self._replace_summary(
                    snapshot,
                    shadow_run_id=result.shadow_run_id,
                    production_run_id=str(manifest.get("production_run_id", "")),
                    shadow_as_of=result.as_of,
                )
                snapshot = self._publish(snapshot, frozen, identity_hash)
        except Exception as error:
            if not initial:
                stage = StageResult(
                    stage="shadow",
                    status="failed",
                    warnings=("daily shadow refresh failed; prior enrollment remains valid",),
                    error=f"{type(error).__name__}: {error}",
                )
                return self._publish(self._with_stage(snapshot, stage), frozen, identity_hash)
            snapshot = self._with_stage(
                snapshot,
                StageResult(
                    stage="shadow",
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            snapshot = self._transition(
                snapshot, "SHADOW_FAILED", f"shadow failed: {type(error).__name__}: {error}"
            )
            snapshot = self._publish(snapshot, frozen, identity_hash)
        return snapshot

    def _track_observation(
        self, snapshot: LifecycleSnapshot, frozen: LifecycleInput, identity_hash: str
    ) -> LifecycleSnapshot:
        if snapshot.summary.current_state == "EVIDENCE_READY":
            return snapshot
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
            },
        )
        changed = (
            progress.status != snapshot.summary.observation_status
            or progress.mature_sessions != snapshot.summary.mature_sessions
            or stage != snapshot.stage_results.get("observation")
        )
        snapshot = self._with_stage(snapshot, stage)
        if progress.status != snapshot.summary.current_state:
            snapshot = self._transition(
                snapshot,
                cast(LifecycleState, progress.status),
                f"prospective mature sessions={progress.mature_sessions}",
                {"required_sessions": progress.required_sessions},
                observation_status=progress.status,
                mature_sessions=progress.mature_sessions,
            )
        else:
            snapshot = self._replace_summary(
                snapshot,
                observation_status=progress.status,
                mature_sessions=progress.mature_sessions,
            )
        if progress.status == "OBSERVATION_SUFFICIENT":
            snapshot = self._evidence_readiness(snapshot)
        if changed or snapshot.summary.current_state == "EVIDENCE_READY":
            snapshot = self._publish(snapshot, frozen, identity_hash)
        return snapshot

    def _evidence_readiness(self, snapshot: LifecycleSnapshot) -> LifecycleSnapshot:
        training = snapshot.stage_results["training"]
        validation = snapshot.stage_results["validation"]
        shadow = snapshot.stage_results["shadow"]
        observation = snapshot.stage_results["observation"]
        from ashare_quant.retraining.orchestration.schemas import ObservationProgress

        progress = ObservationProgress(
            status="OBSERVATION_SUFFICIENT",
            mature_sessions=snapshot.summary.mature_sessions,
            required_sessions=snapshot.summary.required_sessions,
            source_artifacts={
                f"observation:{index}": path
                for index, path in enumerate(observation.artifact_paths)
            },
            source_hashes=observation.artifact_hashes,
            shadow_run_ids=tuple(
                str(value) for value in observation.metrics.get("shadow_run_ids", [])
            ),
        )
        ready, paths, hashes, warnings = resolve_promotion_evidence_references(
            reports_root=self.settings.paths.reports,
            model_id=str(snapshot.summary.model_id),
            execution_path=Path(training.artifact_paths[0]),
            validation_path=Path(validation.artifact_paths[0]),
            shadow_path=Path(shadow.artifact_paths[-1]),
            observation=progress,
            policy=self.promotion_policy,
        )
        stage = StageResult(
            stage="promotion_evidence",
            status="success" if ready else "pending",
            artifact_paths=tuple(paths.values()),
            artifact_hashes=hashes,
            metrics={"status": "READY_FOR_PREPARATION" if ready else "NOT_READY"},
            warnings=warnings,
        )
        snapshot = self._with_stage(snapshot, stage)
        if ready:
            return self._transition(
                snapshot,
                "EVIDENCE_READY",
                "immutable evidence inputs are ready for separate preparation",
                promotion_evidence_status="READY_FOR_PREPARATION",
            )
        return self._replace_summary(snapshot, promotion_evidence_status="NOT_READY")

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
            details=details or {},
        )
        summary = snapshot.summary.model_copy(
            update={"current_state": state, "updated_at": timestamp, **summary_updates}
        )
        return LifecycleSnapshot(summary, (*snapshot.events, event), snapshot.stage_results)

    def _replace_summary(self, snapshot: LifecycleSnapshot, **updates: object) -> LifecycleSnapshot:
        return LifecycleSnapshot(
            snapshot.summary.model_copy(update={"updated_at": self._timestamp(), **updates}),
            snapshot.events,
            snapshot.stage_results,
        )

    def _with_stage(
        self, snapshot: LifecycleSnapshot, stage: StageResult, *, merge_hashes: bool = False
    ) -> LifecycleSnapshot:
        stages = dict(snapshot.stage_results)
        previous = stages.get(stage.stage)
        if merge_hashes and previous is not None:
            stage = stage.model_copy(
                update={
                    "artifact_paths": tuple(
                        dict.fromkeys((*previous.artifact_paths, *stage.artifact_paths))
                    ),
                    "artifact_hashes": {**previous.artifact_hashes, **stage.artifact_hashes},
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
        manifest = LifecycleManifest(
            lifecycle_identity_hash=identity_hash,
            lifecycle_run_id=snapshot.summary.lifecycle_run_id,
            request_id=snapshot.summary.request_id,
            model_id=snapshot.summary.model_id,
            parent_model_id=snapshot.summary.parent_model_id,
            horizon=snapshot.summary.horizon,
            current_state=snapshot.summary.current_state,
            readiness_run_id=snapshot.summary.readiness_run_id,
            training_run_id=snapshot.summary.training_run_id,
            validation_run_id=snapshot.summary.validation_run_id,
            shadow_run_id=snapshot.summary.shadow_run_id,
            production_run_id=snapshot.summary.production_run_id,
            shadow_as_of=snapshot.summary.shadow_as_of,
            observation_status=snapshot.summary.observation_status,
            mature_sessions=snapshot.summary.mature_sessions,
            required_sessions=snapshot.summary.required_sessions,
            promotion_evidence_status=snapshot.summary.promotion_evidence_status,
            retraining_policy_hash=frozen.retraining_policy_hash,
            lifecycle_policy_hash=frozen.lifecycle_policy_hash,
            promotion_policy_hash=frozen.promotion_policy_hash,
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
            config_hash=config_hash(self.config_path),
        )
        return self.storage.publish(snapshot, manifest)

    def _require_identity(
        self,
        snapshot: LifecycleSnapshot,
        frozen: LifecycleInput,
        readiness: ReadinessResult,
        identity_hash: str,
    ) -> None:
        manifest = snapshot.manifest
        if manifest is None or (
            manifest.lifecycle_identity_hash != identity_hash
            or manifest.request_id != frozen.request.request_id
            or manifest.readiness_run_id != readiness.report.run_id
            or manifest.training_request_hash != frozen.training_request_hash
        ):
            raise DataValidationError("existing lifecycle has conflicting immutable lineage")

    def _enforce_daily_training_limit(self, snapshot: LifecycleSnapshot) -> None:
        if any(event.state == "TRAINING" for event in snapshot.events):
            return
        today = self.now().astimezone(UTC).date().isoformat()
        count = 0
        for path in self.storage.root.glob("*/lifecycle_events.parquet"):
            try:
                frame = pd.read_parquet(path, columns=["state", "created_at"])
            except (OSError, ValueError):
                continue
            count += int(
                (
                    frame["state"].astype(str).eq("TRAINING")
                    & frame["created_at"].astype(str).str.startswith(today)
                ).sum()
            )
        if count >= self.retraining_policy.lifecycle.max_daily_training_runs:
            raise DataValidationError("maximum daily governed training runs reached")

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
        "EVIDENCE_READY",
    }
)
