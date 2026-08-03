"""Governed request-to-Challenger retraining execution service."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import load_promotion_gate_policy
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.retraining.configuration import load_retraining_policy
from ashare_quant.retraining.execution.dataset import RetrainingDatasetPreparer
from ashare_quant.retraining.execution.lifecycle import LifecycleJournal
from ashare_quant.retraining.execution.recovery import RecoveryResult, recover_interrupted
from ashare_quant.retraining.execution.schemas import (
    ExecutionResult,
    PreparedTrainingData,
    TrainedRanker,
)
from ashare_quant.retraining.execution.storage import RetrainingExecutionStorage
from ashare_quant.retraining.execution.trainer import GovernedRankerTrainer
from ashare_quant.retraining.execution.validator import (
    validate_execution_inputs,
    validate_prepared_training_data,
)
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.utils.manifest import config_hash, current_git_info


class DatasetPreparer(Protocol):
    def prepare(self, **kwargs: object) -> PreparedTrainingData: ...


class RankerTrainer(Protocol):
    def train(self, prepared: PreparedTrainingData) -> TrainedRanker: ...


class GovernedRetrainingExecutionService:
    """Train only immutable Challenger refreshes after a READY authorization."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        retraining_policy_path: Path,
        promotion_policy_path: Path,
        dataset_preparer: DatasetPreparer | None = None,
        trainer: RankerTrainer | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.retraining_policy = load_retraining_policy(retraining_policy_path)
        self.promotion_policy = load_promotion_gate_policy(promotion_policy_path)
        self.request_storage = RetrainingRequestStorage(
            reports_root=settings.paths.reports,
            config_path=config_path,
            policy=self.retraining_policy,
            promotion_policy=self.promotion_policy,
        )
        self.storage = RetrainingExecutionStorage(
            reports_root=settings.paths.reports,
            models_root=settings.paths.models,
        )
        self.dataset_preparer = dataset_preparer or RetrainingDatasetPreparer(
            processed_root=settings.paths.processed_data,
            reports_root=settings.paths.reports,
            settings=settings,
            config_path=config_path,
        )
        self.trainer = trainer or GovernedRankerTrainer(settings)
        project_root = (
            config_path.resolve().parent.parent
            if config_path.resolve().parent.name == "config"
            else Path.cwd().resolve()
        )
        # Serialize the frozen data read with production artifact publication.
        self.lock_path = project_root / "runs" / ".production.lock"

    def execute(self, request_id: str) -> ExecutionResult:
        """Execute one deterministic challenger_refresh without touching registry.json."""

        with production_lock(self.lock_path, command=f"retraining execute {request_id}"):
            context = validate_execution_inputs(
                request_id=request_id,
                reports_root=self.settings.paths.reports,
                processed_root=self.settings.paths.processed_data,
                models_root=self.settings.paths.models,
                request_storage=self.request_storage,
                retraining_policy=self.retraining_policy,
                promotion_policy=self.promotion_policy,
                config_path=self.config_path,
            )
            target = context.request.target_models[0]
            git = current_git_info()
            current_config_hash = config_hash(self.config_path)
            if current_config_hash is None:
                raise DataValidationError("configuration hash is unavailable")
            identity = {
                "request_hash": context.request_hash,
                "model_id": target.model_id,
                "horizon": target.horizon,
                "feature_hash": context.source_model.feature_hash,
                "feature_manifest_hash": context.readiness.feature_hash,
                "universe_hash": context.readiness.universe_hash,
                "config_hash": current_config_hash,
                "git_commit": git["commit"],
            }
            identity_hash = canonical_payload_hash(identity)
            training_run_id = f"retraining_{identity_hash[:24]}"
            model_id = f"challenger_refresh_h{target.horizon}_{identity_hash[:16]}"
            existing = self.storage.existing(training_run_id, model_id)
            if existing is not None:
                return existing
            journal = LifecycleJournal(self.storage.journal_root, training_run_id)
            events = journal.events()
            if events and events[-1].status not in {"FAILED", "INTERRUPTED"}:
                raise DataValidationError(
                    "retraining execution is incomplete; run recovery before retry"
                )
            journal.append("CREATED", "challenger_refresh identity frozen")
            registry_before = context.registry_hash
            try:
                prepared = self.dataset_preparer.prepare(
                    source_model=context.source_model,
                    horizon=target.horizon,
                    readiness=context.readiness,
                )
                validate_prepared_training_data(prepared, context)
                journal.append("DATA_READY")
                journal.append("TRAINING")
                trained = self.trainer.train(prepared)
                journal.append("ARTIFACT_VALIDATING")
                if file_sha256(self.settings.paths.models / "registry.json") != registry_before:
                    raise DataValidationError("Registry changed during retraining execution")
                result = self.storage.publish(
                    training_run_id=training_run_id,
                    model_id=model_id,
                    request_id=request_id,
                    request_hash=context.request_hash,
                    prepared=prepared,
                    trained=trained,
                    config_hash_value=current_config_hash,
                    git_commit=git["commit"],
                    git_dirty=bool(git["dirty"]),
                )
                journal.append("COMPLETED")
                return result
            except Exception as error:
                try:
                    journal.append("FAILED", f"{type(error).__name__}: {error}")
                except DataValidationError:
                    pass
                raise

    def status(self, training_run_id: str) -> dict[str, object]:
        journal = LifecycleJournal(self.storage.journal_root, training_run_id)
        events = journal.events()
        execution = self.storage.execution_root / training_run_id
        return {
            "training_run_id": training_run_id,
            "status": events[-1].status if events else "MISSING",
            "events": [event.model_dump(mode="json") for event in events],
            "artifact_published": (execution / "manifest.json").is_file(),
        }

    def recovery(self, training_run_id: str) -> RecoveryResult:
        with production_lock(self.lock_path, command=f"retraining recovery {training_run_id}"):
            return recover_interrupted(
                reports_root=self.settings.paths.reports,
                training_run_id=training_run_id,
            )
