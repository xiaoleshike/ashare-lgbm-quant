"""Transactional publication for retrained Challenger artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.alerts.storage import replace_targets_atomically
from ashare_quant.retraining.execution.artifact import validate_artifact, write_staged_artifact
from ashare_quant.retraining.execution.schemas import (
    CandidateRegistration,
    ExecutionResult,
    PreparedTrainingData,
    TrainedRanker,
)
from ashare_quant.utils.manifest import atomic_write_json


class RetrainingExecutionStorage:
    """Publish model, candidate registration, and completion record as one transaction."""

    def __init__(self, *, reports_root: Path, models_root: Path) -> None:
        self.reports_root = reports_root
        self.models_root = models_root
        self.execution_root = reports_root / "retraining" / "executions"
        self.journal_root = reports_root / "retraining" / "execution_journals"
        self.candidate_root = models_root / "candidate_registrations"

    def existing(self, training_run_id: str, model_id: str) -> ExecutionResult | None:
        execution = self.execution_root / training_run_id
        artifact = self.models_root / "challengers" / model_id
        registration = self.candidate_root / model_id
        present = (execution.exists(), artifact.exists(), registration.exists())
        if not any(present):
            return None
        if not all(present):
            raise DataValidationError("incomplete retraining publication exists")
        manifest = validate_artifact(artifact)
        execution_manifest = _json(execution / "manifest.json")
        candidate = CandidateRegistration.model_validate(_json(registration / "registration.json"))
        if (
            manifest.training_run_id != training_run_id
            or manifest.model_id != model_id
            or execution_manifest.get("training_run_id") != training_run_id
            or candidate.training_run_id != training_run_id
            or candidate.artifact_hash != manifest.artifact_hash
        ):
            raise DataValidationError("immutable retraining publication identity differs")
        return ExecutionResult(training_run_id, model_id, "COMPLETED", execution, artifact, True)

    def publish(
        self,
        *,
        training_run_id: str,
        model_id: str,
        request_id: str,
        request_hash: str,
        prepared: PreparedTrainingData,
        trained: TrainedRanker,
        config_hash_value: str,
        git_commit: str | None,
        git_dirty: bool,
    ) -> ExecutionResult:
        common = self.reports_root / "retraining"
        staging_root = common / ".tmp"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=staging_root, prefix=f"execution_{training_run_id}_"))
        try:
            staged_artifact = staging / "artifact"
            artifact_manifest = write_staged_artifact(
                directory=staged_artifact,
                model_id=model_id,
                training_run_id=training_run_id,
                request_hash=request_hash,
                prepared=prepared,
                trained=trained,
                config_hash_value=config_hash_value,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
            staged_registration = staging / "registration"
            staged_registration.mkdir()
            registration = CandidateRegistration(
                model_id=model_id,
                training_run_id=training_run_id,
                artifact_path=str((self.models_root / "challengers" / model_id).resolve()),
                artifact_hash=artifact_manifest.artifact_hash,
                feature_hash=artifact_manifest.feature_hash,
                horizon=artifact_manifest.horizon,
            )
            atomic_write_json(
                staged_registration / "registration.json", registration.model_dump(mode="json")
            )
            atomic_write_json(
                staged_registration / "manifest.json",
                {
                    "schema_version": 1,
                    "artifact_name": "retraining_candidate_registration_manifest",
                    "model_id": model_id,
                    "training_run_id": training_run_id,
                    "registration_sha256": file_sha256(staged_registration / "registration.json"),
                    "manifest_written_last": True,
                },
            )
            staged_execution = staging / "execution"
            staged_execution.mkdir()
            atomic_write_json(
                staged_execution / "dataset_manifest.json",
                prepared.dataset_manifest.model_dump(mode="json"),
            )
            atomic_write_json(
                staged_execution / "execution.json",
                {
                    "schema_version": 1,
                    "artifact_name": "governed_retraining_execution",
                    "training_run_id": training_run_id,
                    "request_id": request_id,
                    "model_id": model_id,
                    "training_type": "challenger_refresh",
                    "status": "COMPLETED",
                    "artifact_hash": artifact_manifest.artifact_hash,
                },
            )
            execution_hash = canonical_payload_hash(
                {
                    "dataset_manifest": file_sha256(staged_execution / "dataset_manifest.json"),
                    "execution": file_sha256(staged_execution / "execution.json"),
                    "artifact_hash": artifact_manifest.artifact_hash,
                }
            )
            atomic_write_json(
                staged_execution / "manifest.json",
                {
                    "schema_version": 1,
                    "artifact_name": "governed_retraining_execution_manifest",
                    "training_run_id": training_run_id,
                    "request_id": request_id,
                    "model_id": model_id,
                    "training_type": "challenger_refresh",
                    "status": "COMPLETED",
                    "execution_hash": execution_hash,
                    "artifact_hash": artifact_manifest.artifact_hash,
                    "manifest_written_last": True,
                },
            )
            targets = (
                (staged_artifact, self.models_root / "challengers" / model_id),
                (staged_registration, self.candidate_root / model_id),
                (staged_execution, self.execution_root / training_run_id),
            )
            if any(target.exists() for _, target in targets):
                raise DataValidationError("retraining target exists with a different identity")
            replace_targets_atomically(targets, backup_root=staging / "backups")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return ExecutionResult(
            training_run_id,
            model_id,
            "COMPLETED",
            self.execution_root / training_run_id,
            self.models_root / "challengers" / model_id,
        )


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required retraining publication is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"retraining publication must contain an object: {path}")
    return payload
