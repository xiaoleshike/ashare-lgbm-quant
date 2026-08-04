"""Contracts for governed retraining execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
from pydantic import BaseModel, ConfigDict, Field

LifecycleStatus = Literal[
    "CREATED",
    "DATA_READY",
    "TRAINING",
    "ARTIFACT_VALIDATING",
    "COMPLETED",
    "FAILED",
    "INTERRUPTED",
]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_dataset_manifest"] = "retraining_dataset_manifest"
    feature_hash: str
    feature_manifest_hash: str
    universe_hash: str
    label_hash: str
    horizon: Literal[5, 10, 20, 60]
    label_name: str
    train_dates: dict[str, str]
    validation_dates: dict[str, str]
    fold_manifest: str
    fold_manifest_hash: str
    fold_id: str
    final_test_loaded: Literal[False] = False
    evaluation_labels_loaded: Literal[False] = False
    production_observation_loaded: Literal[False] = False


class ChallengerArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["governed_retraining_challenger"] = "governed_retraining_challenger"
    model_id: str
    model_role: Literal["challenger"] = "challenger"
    training_type: Literal["challenger_refresh"] = "challenger_refresh"
    horizon: Literal[5, 10, 20, 60]
    holding_period: int
    execution_rule: str
    training_run_id: str
    training_request_hash: str
    feature_hash: str
    feature_list_hash: str
    feature_manifest_hash: str
    universe_hash: str
    label_hash: str
    config_hash: str
    artifact_hash: str = Field(min_length=64, max_length=64)
    train_rows: int
    validation_rows: int
    training_status: Literal["completed"] = "completed"
    train_dates: dict[str, str]
    validation_dates: dict[str, str]
    fold_manifest: str
    git_commit: str | None
    git_dirty: bool
    manifest_written_last: Literal[True] = True


class CandidateRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_candidate_registration"] = (
        "retraining_candidate_registration"
    )
    model_id: str
    candidate_registration_id: str
    status: Literal["candidate"] = "candidate"
    training_type: Literal["challenger_refresh"] = "challenger_refresh"
    training_run_id: str
    artifact_path: str
    artifact_hash: str
    feature_hash: str
    horizon: Literal[5, 10, 20, 60]
    registry_json_modified: Literal[False] = False


class LifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    training_run_id: str
    sequence: int
    status: LifecycleStatus
    created_at: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    dataset_manifest: DatasetManifest
    features: tuple[str, ...]
    train: Any
    validation: Any
    holding_period: int
    execution_rule: str


@dataclass(frozen=True, slots=True)
class TrainedRanker:
    model: lgb.LGBMRanker
    metrics: dict[str, object]
    importance: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    training_run_id: str
    model_id: str
    status: str
    output_dir: Path
    artifact_dir: Path | None = None
    idempotent: bool = False
