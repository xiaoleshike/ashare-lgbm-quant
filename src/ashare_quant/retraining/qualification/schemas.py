"""Immutable contracts for controlled operational qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

QualificationState = Literal[
    "CREATED",
    "PREFLIGHT_CHECKING",
    "PREFLIGHT_BLOCKED",
    "PREFLIGHT_READY",
    "DRY_RUN_CHECKING",
    "DRY_RUN_BLOCKED",
    "DRY_RUN_READY",
    "READINESS_CHECKING",
    "READINESS_FAILED",
    "READINESS_READY",
    "TRAINING_PENDING_APPROVAL",
    "TRAINING",
    "TRAINING_FAILED",
    "TRAINING_COMPLETED",
    "VALIDATION_PENDING_APPROVAL",
    "VALIDATING",
    "VALIDATION_FAILED",
    "VALIDATION_COMPLETED",
    "SHADOW_PENDING_APPROVAL",
    "SHADOW_ENROLLING",
    "SHADOW_FAILED",
    "SHADOW_ENROLLED",
    "OBSERVATION_CHECKING",
    "OBSERVATION_PENDING",
    "OBSERVATION_ACCUMULATING",
    "QUALIFIED",
    "FAILED",
    "CANCELLED",
]


class QualificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Literal["PASS", "FAIL", "WARN", "NOT_RUN"]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualificationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=1)
    state: QualificationState
    created_at: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class QualificationCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: Literal["success", "blocked", "failed", "pending", "cancelled"]
    checks: tuple[QualificationCheck, ...] = ()
    artifact_paths: tuple[str, ...] = ()
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None


class QualificationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["controlled_operational_qualification"] = (
        "controlled_operational_qualification"
    )
    qualification_run_id: str
    qualification_only: Literal[True] = True
    qualification_phase: Literal["2.8.2G"] = "2.8.2G"
    qualification_source: Literal["controlled_operational_qualification"] = (
        "controlled_operational_qualification"
    )
    request_id: str
    as_of: str
    parent_model_id: str
    horizon: Literal[5, 10, 20, 60]
    current_state: QualificationState
    proposed_lifecycle_run_id: str
    dry_run_id: str | None = None
    readiness_run_id: str | None = None
    training_run_id: str | None = None
    model_id: str | None = None
    validation_run_id: str | None = None
    shadow_run_id: str | None = None
    production_run_id: str | None = None
    observation_status: str = "NOT_STARTED"
    frozen_retraining_policy_hash: str
    frozen_lifecycle_policy_hash: str
    frozen_promotion_policy_hash: str
    frozen_config_hash: str | None
    qualification_policy_hash: str
    created_at: str
    updated_at: str


class QualificationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["controlled_operational_qualification_manifest"] = (
        "controlled_operational_qualification_manifest"
    )
    qualification_identity_hash: str = Field(min_length=64, max_length=64)
    qualification_run_id: str
    request_id: str
    as_of: str
    current_state: QualificationState
    training_request_hash: str
    qualification_policy_hash: str
    retraining_policy_hash: str
    lifecycle_policy_hash: str
    promotion_policy_hash: str
    config_hash: str | None
    source_hashes: dict[str, str]
    protected_invariant_hashes: dict[str, str | None]
    summary_sha256: str = Field(min_length=64, max_length=64)
    events_sha256: str = Field(min_length=64, max_length=64)
    checkpoints_sha256: str = Field(min_length=64, max_length=64)
    inventory_sha256: str = Field(min_length=64, max_length=64)
    invariants_sha256: str = Field(min_length=64, max_length=64)
    report_sha256: str = Field(min_length=64, max_length=64)
    git_commit: str | None
    git_dirty: bool
    manifest_written_last: Literal[True] = True


@dataclass(frozen=True, slots=True)
class QualificationSnapshot:
    summary: QualificationSummary
    events: tuple[QualificationEvent, ...]
    checkpoints: dict[str, QualificationCheckpoint]
    source_inventory: dict[str, dict[str, Any]]
    invariant_results: dict[str, Any]
    manifest: QualificationManifest | None = None


@dataclass(frozen=True, slots=True)
class QualificationResult:
    qualification_run_id: str
    state: str
    output_dir: Path
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class QualificationRecovery:
    qualification_run_id: str
    status: str
    issues: tuple[str, ...]
    operator_actions: tuple[str, ...]
