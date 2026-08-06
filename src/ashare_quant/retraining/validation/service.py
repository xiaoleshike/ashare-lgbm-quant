"""Governed retrained-Challenger validation orchestration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.retraining.execution.schemas import QualificationExecutionContext
from ashare_quant.retraining.validation.artifact_validation import (
    validate_candidate_artifact,
)
from ashare_quant.retraining.validation.evidence import build_validation_evidence
from ashare_quant.retraining.validation.executable import RetrainingExecutableValidator
from ashare_quant.retraining.validation.offline import RetrainingOfflineValidator
from ashare_quant.retraining.validation.schemas import (
    CandidateValidationContext,
    ExecutableValidationEvidence,
    OfflineValidationRun,
    RetrainingValidationManifest,
    RetrainingValidationResult,
    ShadowEligibilityEvidence,
)
from ashare_quant.retraining.validation.shadow import (
    RetrainingShadowEligibilityValidator,
)
from ashare_quant.retraining.validation.storage import RetrainingValidationStorage
from ashare_quant.utils.manifest import config_hash, current_git_info


class OfflineValidator(Protocol):
    def evaluate(self, context: CandidateValidationContext) -> OfflineValidationRun: ...


class ExecutableValidator(Protocol):
    def evaluate(
        self, context: CandidateValidationContext, offline: OfflineValidationRun
    ) -> ExecutableValidationEvidence: ...


class ShadowValidator(Protocol):
    def evaluate(self, context: CandidateValidationContext) -> ShadowEligibilityEvidence: ...


class RetrainingValidationService:
    """Produce immutable validation evidence without model state transitions."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        context_loader: Callable[..., CandidateValidationContext] = validate_candidate_artifact,
        offline_validator: OfflineValidator | None = None,
        executable_validator: ExecutableValidator | None = None,
        shadow_validator: ShadowValidator | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.context_loader = context_loader
        self.offline = offline_validator or RetrainingOfflineValidator(
            processed_root=settings.paths.processed_data,
            settings=settings,
        )
        self.executable = executable_validator or RetrainingExecutableValidator(
            raw_root=settings.paths.raw_data,
            processed_root=settings.paths.processed_data,
            settings=settings,
        )
        self.shadow = shadow_validator or RetrainingShadowEligibilityValidator()
        self.storage = RetrainingValidationStorage(settings.paths.reports)
        project_root = (
            config_path.resolve().parent.parent
            if config_path.resolve().parent.name == "config"
            else Path.cwd().resolve()
        )
        self.lock_path = project_root / "runs" / ".production.lock"

    def validate(
        self,
        model_id: str,
        *,
        qualification: QualificationExecutionContext | None = None,
    ) -> RetrainingValidationResult:
        with production_lock(self.lock_path, command=f"retraining validate --model-id {model_id}"):
            context = self.context_loader(
                model_id=model_id,
                models_root=self.settings.paths.models,
                reports_root=self.settings.paths.reports,
                processed_root=self.settings.paths.processed_data,
                config_path=self.config_path,
                allow_frozen_config=qualification is not None,
            )
            current_config = config_hash(self.config_path)
            if current_config is None:
                raise DataValidationError("VALIDATION_FAILED: config hash is unavailable")
            git = current_git_info()
            if qualification is None and context.artifact.qualification_only:
                raise DataValidationError(
                    "VALIDATION_FAILED: qualification-only artifact requires qualification context"
                )
            if qualification is not None and (
                not context.artifact.qualification_only
                or context.artifact.qualification_run_id != qualification.qualification_run_id
                or context.registration.qualification_run_id != qualification.qualification_run_id
            ):
                raise DataValidationError(
                    "VALIDATION_FAILED: qualification lineage does not match candidate"
                )
            identity = canonical_payload_hash(
                {
                    "model_id": model_id,
                    "artifact_hash": context.artifact.artifact_hash,
                    "candidate_registration_hash": context.candidate_registration_hash,
                    "execution_manifest_hash": context.execution_manifest_hash,
                    "feature_hash": context.artifact.feature_hash,
                    "universe_hash": context.artifact.universe_hash,
                    "label_hash": context.artifact.label_hash,
                    "evaluation_start": context.evaluation_start,
                    "evaluation_end": context.evaluation_end,
                    "config_hash": current_config,
                    "git_commit": git["commit"],
                    "qualification": (
                        qualification.model_dump(mode="json") if qualification else None
                    ),
                }
            )
            run_id = f"retraining_validation_{identity[:24]}"
            existing = self.storage.existing(run_id, identity, model_id)
            if existing is not None:
                return existing
            registry_path = self.settings.paths.models / "registry.json"
            registry_before = file_sha256(registry_path)
            offline = self.offline.evaluate(context)
            executable = self.executable.evaluate(context, offline)
            shadow = self.shadow.evaluate(context)
            evidence = build_validation_evidence(
                run_id=run_id,
                context=context,
                offline=offline.evidence,
                executable=executable,
                shadow=shadow,
                minimum_sessions=(self.settings.models.challenger_evaluation.minimum_labelled_days),
            )
            if file_sha256(registry_path) != registry_before:
                raise DataValidationError("VALIDATION_FAILED: registry changed during validation")
            manifest = RetrainingValidationManifest(
                validation_identity=identity,
                run_id=run_id,
                model_id=model_id,
                candidate_registration_id=context.registration.candidate_registration_id,
                training_run_id=context.artifact.training_run_id,
                feature_hash=context.artifact.feature_hash,
                universe_hash=context.artifact.universe_hash,
                label_hash=context.artifact.label_hash,
                config_hash=current_config,
                artifact_hash=context.artifact.artifact_hash,
                offline_validation_hash="0" * 64,
                executable_validation_hash="0" * 64,
                shadow_eligibility_hash="0" * 64,
                evidence_hash="0" * 64,
                promotion_ready=evidence.promotion_ready,
                git_commit=git["commit"],
                git_dirty=bool(git["dirty"]),
                qualification_run_id=(
                    qualification.qualification_run_id if qualification else None
                ),
                qualification_only=qualification is not None,
                qualification_phase=(qualification.qualification_phase if qualification else None),
                promotion_forbidden=qualification is not None,
                trading_forbidden=qualification is not None,
            )
            return self.storage.publish(
                manifest=manifest,
                offline=offline.evidence,
                executable=executable,
                shadow=shadow,
                evidence=evidence,
            )

    def status(self, run_id: str) -> dict[str, object]:
        return self.storage.status(run_id)
