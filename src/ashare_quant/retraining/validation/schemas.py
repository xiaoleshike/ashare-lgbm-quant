"""Contracts for immutable retrained-Challenger validation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.models.registry import RegisteredModel
from ashare_quant.retraining.execution.schemas import (
    CandidateRegistration,
    ChallengerArtifactManifest,
    DatasetManifest,
)


class OfflineValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_offline_validation"] = "retraining_offline_validation"
    model_id: str
    horizon: int
    evaluation_start: str
    evaluation_end: str
    prediction_rows: int
    labelled_rows: int
    evaluation_sessions: int
    overall_metrics: dict[str, Any]
    stability_metrics: tuple[dict[str, Any], ...]
    final_test_loaded: Literal[False] = False
    production_observation_loaded: Literal[False] = False


class ExecutableValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_executable_validation"] = "retraining_executable_validation"
    model_id: str
    horizon: int
    holding_period: int
    execution_rule: Literal["next_open"] = "next_open"
    signal_dates: int
    minimum_signal_date: str
    maximum_signal_date: str
    top_n: tuple[int, ...]
    execution_config: dict[str, Any]
    metrics: dict[str, dict[str, float | int | None]]
    unresolved_holdings: Literal[False] = False
    labels_loaded: Literal[False] = False
    trading_state_modified: Literal[False] = False


class ShadowEligibilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_shadow_eligibility"] = "retraining_shadow_eligibility"
    model_id: str
    shadow_eligible: bool
    feature_hash_compatible: bool
    universe_compatible: bool
    deployment_contract_compatible: bool
    inference_adapter_available: bool
    production_prediction_generated: Literal[False] = False
    reasons: tuple[str, ...] = ()


class ValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retrained_challenger_validation_evidence"] = (
        "retrained_challenger_validation_evidence"
    )
    run_id: str
    model_id: str
    candidate_registration_id: str
    training_run_id: str
    offline_status: Literal["PASS"] = "PASS"
    executable_status: Literal["PASS"] = "PASS"
    shadow_eligible: bool
    promotion_ready: bool
    promotion_request_created: Literal[False] = False
    registry_modified: Literal[False] = False


class RetrainingValidationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retrained_challenger_validation_manifest"] = (
        "retrained_challenger_validation_manifest"
    )
    validation_identity: str = Field(min_length=64, max_length=64)
    run_id: str
    model_id: str
    candidate_registration_id: str
    training_run_id: str
    feature_hash: str
    universe_hash: str
    label_hash: str
    config_hash: str
    artifact_hash: str
    offline_validation_hash: str
    executable_validation_hash: str
    shadow_eligibility_hash: str
    evidence_hash: str
    promotion_ready: bool
    git_commit: str | None
    git_dirty: bool
    registry_modified: Literal[False] = False
    champion_modified: Literal[False] = False
    promotion_executed: Literal[False] = False
    trading_executed: Literal[False] = False
    manifest_written_last: Literal[True] = True
    qualification_run_id: str | None = None
    qualification_only: bool = False
    qualification_phase: str | None = None
    promotion_forbidden: bool = False
    trading_forbidden: bool = False


@dataclass(frozen=True, slots=True)
class CandidateValidationContext:
    model: RegisteredModel
    artifact_dir: Path
    artifact: ChallengerArtifactManifest
    dataset: DatasetManifest
    registration: CandidateRegistration
    candidate_registration_hash: str
    execution_manifest_hash: str
    evaluation_start: str
    evaluation_end: str
    maximum_mature_evaluation_date: str
    fold_id: str


@dataclass(frozen=True, slots=True)
class OfflineValidationRun:
    evidence: OfflineValidationEvidence
    predictions: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RetrainingValidationResult:
    run_id: str
    model_id: str
    status: str
    promotion_ready: bool
    output_dir: Path
    idempotent: bool = False
