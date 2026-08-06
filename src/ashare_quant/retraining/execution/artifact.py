"""Staged Challenger artifact writing and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.execution.schemas import (
    ChallengerArtifactManifest,
    DatasetManifest,
    PreparedTrainingData,
    QualificationExecutionContext,
    TrainedRanker,
)
from ashare_quant.utils.manifest import atomic_write_json


def write_staged_artifact(
    *,
    directory: Path,
    model_id: str,
    training_run_id: str,
    request_hash: str,
    prepared: PreparedTrainingData,
    trained: TrainedRanker,
    config_hash_value: str,
    git_commit: str | None,
    git_dirty: bool,
    qualification: QualificationExecutionContext | None = None,
) -> ChallengerArtifactManifest:
    """Write model bytes and manifest last inside an unpublished directory."""

    directory.mkdir(parents=True, exist_ok=False)
    trained.model.booster_.save_model(str(directory / "model.txt"))
    atomic_write_json(
        directory / "feature_list.json",
        {
            "features": list(prepared.features),
            "feature_count": len(prepared.features),
            "feature_hash": feature_list_hash(prepared.features),
        },
    )
    atomic_write_json(
        directory / "metrics.json",
        {
            "metric_scope": "selection-fold validation only; final test not loaded",
            "validation": trained.metrics,
            "test": {},
            "feature_importance": trained.importance,
        },
    )
    atomic_write_json(
        directory / "dataset_manifest.json",
        prepared.dataset_manifest.model_dump(mode="json"),
    )
    component_hashes = {
        name: file_sha256(directory / name)
        for name in ("model.txt", "feature_list.json", "metrics.json", "dataset_manifest.json")
    }
    artifact_hash = canonical_payload_hash(component_hashes)
    dataset: DatasetManifest = prepared.dataset_manifest
    manifest = ChallengerArtifactManifest(
        model_id=model_id,
        horizon=dataset.horizon,
        holding_period=prepared.holding_period,
        execution_rule=prepared.execution_rule,
        training_run_id=training_run_id,
        training_request_hash=request_hash,
        feature_hash=dataset.feature_hash,
        feature_list_hash=dataset.feature_hash,
        feature_manifest_hash=dataset.feature_manifest_hash,
        universe_hash=dataset.universe_hash,
        label_hash=dataset.label_hash,
        config_hash=config_hash_value,
        artifact_hash=artifact_hash,
        train_rows=len(prepared.train.frame),
        validation_rows=len(prepared.validation.frame),
        train_dates=dataset.train_dates,
        validation_dates=dataset.validation_dates,
        fold_manifest=dataset.fold_manifest,
        git_commit=git_commit,
        git_dirty=git_dirty,
        qualification_run_id=(qualification.qualification_run_id if qualification else None),
        qualification_only=qualification is not None,
        qualification_phase=(qualification.qualification_phase if qualification else None),
        qualification_source=(qualification.qualification_source if qualification else None),
        promotion_forbidden=qualification is not None,
        trading_forbidden=qualification is not None,
    )
    atomic_write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
    validate_artifact(directory, manifest)
    return manifest


def validate_artifact(
    directory: Path, expected: ChallengerArtifactManifest | None = None
) -> ChallengerArtifactManifest:
    required = (
        "model.txt",
        "feature_list.json",
        "metrics.json",
        "dataset_manifest.json",
        "manifest.json",
    )
    if any(not (directory / name).is_file() for name in required):
        raise DataValidationError("challenger artifact is incomplete")
    try:
        manifest = ChallengerArtifactManifest.model_validate(_json(directory / "manifest.json"))
        dataset = DatasetManifest.model_validate(_json(directory / "dataset_manifest.json"))
    except ValueError as error:
        raise DataValidationError(f"challenger artifact schema is invalid: {error}") from error
    feature_payload = _json(directory / "feature_list.json")
    features = feature_payload.get("features")
    if (
        not isinstance(features, list)
        or feature_list_hash(tuple(map(str, features))) != manifest.feature_hash
    ):
        raise DataValidationError("challenger artifact feature hash mismatch")
    hashes = {
        name: file_sha256(directory / name)
        for name in ("model.txt", "feature_list.json", "metrics.json", "dataset_manifest.json")
    }
    if canonical_payload_hash(hashes) != manifest.artifact_hash:
        raise DataValidationError("challenger artifact hash mismatch")
    if (
        dataset.horizon != manifest.horizon
        or dataset.feature_hash != manifest.feature_hash
        or manifest.feature_list_hash != manifest.feature_hash
        or manifest.holding_period != manifest.horizon
        or manifest.execution_rule != "next_open"
        or (expected is not None and manifest != expected)
    ):
        raise DataValidationError("challenger artifact execution identity mismatch")
    return manifest


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload
