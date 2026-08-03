"""Typed contracts for governed retraining trigger requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ModelRole = Literal[
    "champion",
    "challenger_h5",
    "challenger_h10",
    "challenger_h20",
    "challenger_h60",
]
TriggerReason = Literal[
    "alpha_decay",
    "ic_decline",
    "feature_drift",
    "critical_alert",
    "manual_request",
]
EvaluationStatus = Literal[
    "TRIGGERED",
    "NO_ACTION_REQUIRED",
    "INSUFFICIENT_OBSERVATIONS",
    "DISABLED",
]


class EvidenceReference(BaseModel):
    """One immutable monitoring source bound into a training request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(min_length=64, max_length=64)
    artifact_name: str
    as_of: str
    identity_hash: str | None = None


class RetrainingEvidence(BaseModel):
    """Fixed allowlist of evidence accepted by the trigger engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    monitor_snapshot: EvidenceReference
    performance_observation: EvidenceReference
    alerts: EvidenceReference


class TrainingTarget(BaseModel):
    """One independently evaluated trainable model horizon."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    model_role: ModelRole
    horizon: Literal[5, 10, 20, 60]


class TrainingRequest(BaseModel):
    """Immutable hand-off contract for the future training orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["governed_training_request"] = "governed_training_request"
    request_id: str
    status: Literal["CREATED"] = "CREATED"
    created_at: str
    as_of: str
    target_models: tuple[TrainingTarget, ...] = Field(min_length=1, max_length=1)
    trigger_reason: tuple[TriggerReason, ...] = Field(min_length=1)
    evidence: RetrainingEvidence
    evidence_hash: str = Field(min_length=64, max_length=64)
    policy_hash: str = Field(min_length=64, max_length=64)
    policy_version: str
    generation_mode: Literal["automatic", "manual"]
    training_allowed: Literal[True] = True
    promotion_allowed: Literal[False] = False


class TrainingRequestManifest(BaseModel):
    """Commit marker written after the immutable request payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["governed_training_request_manifest"] = (
        "governed_training_request_manifest"
    )
    request_id: str
    model_id: str
    model_role: ModelRole
    horizon: Literal[5, 10, 20, 60]
    trigger_reasons: tuple[TriggerReason, ...]
    evidence_hashes: dict[str, str]
    evidence_hash: str = Field(min_length=64, max_length=64)
    policy_hash: str = Field(min_length=64, max_length=64)
    policy_version: str
    git_commit: str | None
    git_dirty: bool
    config_hash: str | None
    generated_at: str
    request_file_sha256: str = Field(min_length=64, max_length=64)
    manifest_written_last: Literal[True] = True


@dataclass(frozen=True, slots=True)
class RetrainingSources:
    """Validated monitoring inputs for one as-of date."""

    as_of: str
    monitor_manifest: dict[str, Any]
    health: dict[str, Any]
    performance_manifest: dict[str, Any]
    performance_metrics: pd.DataFrame
    alerts_manifest: dict[str, Any]
    alerts: dict[str, Any]
    evidence: RetrainingEvidence
    evidence_hash: str


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    """One horizon-isolated trigger evaluation."""

    model_id: str
    model_role: str
    horizon: int
    status: EvaluationStatus
    reasons: tuple[str, ...]
    observation_sessions: int
    required_sessions: int
    request_id: str | None = None
    output_dir: Path | None = None
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class RetrainingEvaluationResult:
    """Result of one automatic or manual trigger invocation."""

    as_of: str
    decisions: tuple[RetrainingDecision, ...]
    request_paths: tuple[Path, ...]
    warnings: tuple[str, ...] = ()

    @property
    def triggered_count(self) -> int:
        return sum(decision.status == "TRIGGERED" for decision in self.decisions)


@dataclass(frozen=True, slots=True)
class RetrainingValidationResult:
    """Read-only request validation result."""

    request_id: str
    valid: bool
    status: str
    error: str | None = None
