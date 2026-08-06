"""Typed contracts for retrained Challenger lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.retraining.schemas import TrainingRequest

LifecycleState = Literal[
    "REQUEST_ACCEPTED",
    "READINESS_CHECKING",
    "READINESS_FAILED",
    "READINESS_READY",
    "TRAINING",
    "TRAINING_FAILED",
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
    "FAILED",
    "CANCELLED",
]
ObservationStatus = Literal[
    "NOT_STARTED",
    "OBSERVATION_PENDING",
    "OBSERVATION_ACCUMULATING",
    "OBSERVATION_SUFFICIENT",
]
PromotionEvidenceStatus = Literal["NOT_READY", "READY_FOR_PREPARATION"]


class LifecycleEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=1)
    state: LifecycleState
    created_at: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    status: Literal["success", "failed", "pending", "stopped"]
    artifact_paths: tuple[str, ...] = ()
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None


class LifecycleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retrained_challenger_lifecycle"] = "retrained_challenger_lifecycle"
    lifecycle_run_id: str
    request_id: str
    model_id: str | None
    model_origin: Literal["retrained_challenger"] = "retrained_challenger"
    parent_model_id: str
    horizon: Literal[5, 10, 20, 60]
    trigger_reasons: tuple[str, ...]
    current_state: LifecycleState
    readiness_run_id: str
    training_run_id: str | None = None
    validation_run_id: str | None = None
    shadow_run_id: str | None = None
    production_run_id: str | None = None
    shadow_as_of: str | None = None
    observation_status: ObservationStatus = "NOT_STARTED"
    mature_sessions: int = 0
    required_sessions: int
    promotion_evidence_status: PromotionEvidenceStatus = "NOT_READY"
    created_at: str
    updated_at: str


class LifecycleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retrained_challenger_lifecycle_manifest"] = (
        "retrained_challenger_lifecycle_manifest"
    )
    lifecycle_identity_hash: str = Field(min_length=64, max_length=64)
    lifecycle_run_id: str
    request_id: str
    model_id: str | None
    model_origin: Literal["retrained_challenger"] = "retrained_challenger"
    parent_model_id: str
    horizon: Literal[5, 10, 20, 60]
    current_state: LifecycleState
    readiness_run_id: str
    training_run_id: str | None
    validation_run_id: str | None
    shadow_run_id: str | None
    production_run_id: str | None
    shadow_as_of: str | None
    observation_status: ObservationStatus
    mature_sessions: int
    required_sessions: int
    promotion_evidence_status: PromotionEvidenceStatus
    retraining_policy_hash: str
    lifecycle_policy_hash: str
    promotion_policy_hash: str
    evidence_hash: str
    training_request_hash: str
    source_artifacts: dict[str, str]
    source_hashes: dict[str, str]
    summary_sha256: str = Field(min_length=64, max_length=64)
    events_sha256: str = Field(min_length=64, max_length=64)
    stage_results_sha256: str = Field(min_length=64, max_length=64)
    report_sha256: str = Field(min_length=64, max_length=64)
    git_commit: str | None
    git_dirty: bool
    config_hash: str | None
    manifest_written_last: Literal[True] = True


@dataclass(frozen=True, slots=True)
class LifecycleSnapshot:
    summary: LifecycleSummary
    events: tuple[LifecycleEvent, ...]
    stage_results: dict[str, StageResult]
    manifest: LifecycleManifest | None = None


@dataclass(frozen=True, slots=True)
class LifecycleRunResult:
    lifecycle_run_id: str
    request_id: str
    current_state: str
    model_id: str | None
    output_dir: Path
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class ObservationProgress:
    status: ObservationStatus
    mature_sessions: int
    required_sessions: int
    source_artifacts: dict[str, str]
    source_hashes: dict[str, str]
    shadow_run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryInspection:
    lifecycle_run_id: str
    status: str
    current_state: str | None
    complete: bool
    staging_paths: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class LifecycleInput:
    request: TrainingRequest
    training_request_hash: str
    retraining_policy_hash: str
    lifecycle_policy_hash: str
    promotion_policy_hash: str
