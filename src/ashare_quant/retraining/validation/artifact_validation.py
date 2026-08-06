"""Read-only lineage validation for governed retrained Challengers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger import (
    _load_and_validate_folds,
    _validate_experiment,
    _validate_fold_for_experiment,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import RegisteredModel
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.execution.artifact import validate_artifact
from ashare_quant.retraining.execution.schemas import CandidateRegistration, DatasetManifest
from ashare_quant.retraining.validation.schemas import CandidateValidationContext
from ashare_quant.utils.manifest import config_hash


def validate_candidate_artifact(
    *,
    model_id: str,
    models_root: Path,
    reports_root: Path,
    processed_root: Path,
    config_path: Path,
    require_current_processed_hashes: bool = True,
    allow_frozen_config: bool = False,
) -> CandidateValidationContext:
    """Validate all immutable training and candidate-registration identities."""

    artifact_dir = models_root / "challengers" / model_id
    artifact = validate_artifact(artifact_dir)
    if artifact.model_id != model_id or artifact.training_status != "completed":
        raise DataValidationError("VALIDATION_FAILED: candidate model identity is invalid")
    registration_dir = models_root / "candidate_registrations" / model_id
    registration_path = registration_dir / "registration.json"
    registration_manifest = _json(registration_dir / "manifest.json")
    try:
        registration = CandidateRegistration.model_validate(_json(registration_path))
    except ValueError as error:
        raise DataValidationError(
            f"VALIDATION_FAILED: invalid candidate registration: {error}"
        ) from error
    if (
        registration.model_id != model_id
        or registration.training_run_id != artifact.training_run_id
        or registration.artifact_hash != artifact.artifact_hash
        or registration.feature_hash != artifact.feature_hash
        or registration.horizon != artifact.horizon
        or registration.status != "candidate"
        or registration_manifest.get("registration_sha256") != file_sha256(registration_path)
    ):
        raise DataValidationError("VALIDATION_FAILED: candidate registration identity mismatch")
    execution_dir = reports_root / "retraining" / "executions" / artifact.training_run_id
    execution_manifest_path = execution_dir / "manifest.json"
    execution = _json(execution_manifest_path)
    if (
        execution.get("status") != "COMPLETED"
        or execution.get("model_id") != model_id
        or execution.get("artifact_hash") != artifact.artifact_hash
    ):
        raise DataValidationError("VALIDATION_FAILED: retraining execution identity mismatch")
    dataset_path = artifact_dir / "dataset_manifest.json"
    try:
        dataset = DatasetManifest.model_validate(_json(dataset_path))
    except ValueError as error:
        raise DataValidationError(
            f"VALIDATION_FAILED: invalid dataset manifest: {error}"
        ) from error
    current_config = config_hash(config_path)
    if current_config is None or (
        artifact.config_hash != current_config and not allow_frozen_config
    ):
        raise DataValidationError("VALIDATION_FAILED: candidate config hash mismatch")
    current_hashes = {
        "feature_manifest_hash": file_sha256(processed_root / "features_daily" / "_manifest.json"),
        "universe_hash": file_sha256(processed_root / "universe_daily" / "_manifest.json"),
        "label_hash": file_sha256(processed_root / "labels_forward" / "_manifest.json"),
    }
    for field, digest in current_hashes.items():
        if getattr(artifact, field) != getattr(dataset, field):
            raise DataValidationError(f"VALIDATION_FAILED: candidate {field} mismatch")
        if require_current_processed_hashes and getattr(artifact, field) != digest:
            raise DataValidationError(f"VALIDATION_FAILED: candidate {field} mismatch")
    feature_payload = _json(artifact_dir / "feature_list.json")
    raw_features = feature_payload.get("features")
    if not isinstance(raw_features, list) or not raw_features:
        raise DataValidationError("VALIDATION_FAILED: candidate feature list is empty")
    features = tuple(str(value) for value in raw_features)
    if feature_list_hash(features) != artifact.feature_hash:
        raise DataValidationError("VALIDATION_FAILED: candidate feature hash mismatch")
    if artifact.feature_list_hash != artifact.feature_hash:
        raise DataValidationError("VALIDATION_FAILED: inference feature identity mismatch")
    evaluation_start, evaluation_end, maximum_mature = _selection_evaluation(
        reports_root=reports_root,
        dataset=dataset,
        config_hash_value=artifact.config_hash if allow_frozen_config else current_config,
    )
    if artifact.validation_dates.get("end", "") >= evaluation_start:
        raise DataValidationError("VALIDATION_FAILED: evaluation is not after model validation")
    model = RegisteredModel(
        model_id=model_id,
        experiment_id=model_id,
        model_type="lightgbm_ranker",
        feature_hash=artifact.feature_hash,
        feature_count=len(features),
        training_date_range={
            "start": artifact.train_dates["start"],
            "end": artifact.validation_dates["end"],
        },
        validation_metrics={},
        test_metrics={},
        git_commit=artifact.git_commit,
        config_hash=artifact.config_hash,
        creation_time="governed_retraining",
        artifact_path=str(artifact_dir.resolve()),
        status="candidate",
    )
    return CandidateValidationContext(
        model=model,
        artifact_dir=artifact_dir,
        artifact=artifact,
        dataset=dataset,
        registration=registration,
        candidate_registration_hash=file_sha256(registration_path),
        execution_manifest_hash=file_sha256(execution_manifest_path),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        maximum_mature_evaluation_date=maximum_mature,
        fold_id=dataset.fold_id,
    )


def _selection_evaluation(
    *, reports_root: Path, dataset: DatasetManifest, config_hash_value: str
) -> tuple[str, str, str]:
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in sorted((reports_root / "horizon_experiments").glob("*/experiment_manifest.json")):
        plan = _json(path)
        if (
            plan.get("config_hash") != config_hash_value
            or plan.get("folds_manifest_hash") != dataset.fold_manifest_hash
        ):
            continue
        experiments = plan.get("experiments")
        if not isinstance(experiments, list):
            continue
        for raw in experiments:
            if not isinstance(raw, dict) or raw.get("horizon") != dataset.horizon:
                continue
            _validate_experiment(
                raw,
                feature_hash=dataset.feature_hash,
                universe_hash=dataset.universe_hash,
                config_hash_value=config_hash_value,
            )
            selection_ids = _fold_ids(raw.get("selection_period"))
            final_test_ids = _fold_ids(raw.get("final_test_period"))
            if dataset.fold_id in selection_ids and dataset.fold_id not in final_test_ids:
                candidates.append((plan, raw))
    if len(candidates) != 1:
        raise DataValidationError(
            "VALIDATION_FAILED: candidate fold is not uniquely proven as selection-only"
        )
    plan, experiment = candidates[0]
    folds_manifest = Path(str(plan["folds_manifest"]))
    _, folds = _load_and_validate_folds(
        folds_manifest,
        expected_manifest_hash=dataset.fold_manifest_hash,
        expected_folds_hash=str(plan["folds_hash"]),
        expected_feature_hash=dataset.feature_hash,
    )
    matches = [fold for fold in folds if str(fold.get("fold_id")) == dataset.fold_id]
    if len(matches) != 1:
        raise DataValidationError("VALIDATION_FAILED: candidate selection fold is missing")
    fold = matches[0]
    _validate_fold_for_experiment(experiment, fold)
    start, end = str(fold["evaluation_start"]), str(fold["evaluation_end"])
    maximum_mature = str(experiment.get("maximum_mature_evaluation_date", ""))
    if not maximum_mature or end > maximum_mature:
        raise DataValidationError("VALIDATION_FAILED: evaluation labels are not mature")
    return start, end, maximum_mature


def _fold_ids(period: object) -> set[str]:
    if not isinstance(period, dict) or not isinstance(period.get("folds"), list):
        return set()
    return {
        str(item.get("fold_id"))
        for item in period["folds"]
        if isinstance(item, dict) and item.get("fold_id")
    }


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"VALIDATION_FAILED: required artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"VALIDATION_FAILED: invalid JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"VALIDATION_FAILED: artifact must be an object: {path}")
    return payload
