"""Versioned immutable contracts for qualification stage authorization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AuthorizationStage = Literal["training", "shadow"]
AuthorizationState = Literal[
    "REQUIRED",
    "ACTIVE",
    "EXPIRED",
    "REVOKED",
    "CONSUMED",
    "STALE",
    "INVALID",
    "LEGACY_UNSUPPORTED",
]


class QualificationAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["qualification_stage_authorization"] = (
        "qualification_stage_authorization"
    )
    authorization_id: str
    qualification_run_id: str
    qualification_identity_hash: str
    qualification_phase: Literal["2.8.2G"] = "2.8.2G"
    stage: AuthorizationStage
    decision: Literal["APPROVED"] = "APPROVED"
    request_id: str
    as_of: str
    parent_model_id: str
    model_id: str | None
    horizon: Literal[5, 10, 20, 60]
    training_request_hash: str
    qualification_snapshot_state: Literal["TRAINING_PENDING_APPROVAL", "SHADOW_PENDING_APPROVAL"]
    qualification_snapshot_manifest_path: str
    qualification_snapshot_manifest_sha256: str
    qualification_summary_sha256: str
    qualification_events_sha256: str
    checkpoint_results_sha256: str
    source_inventory_sha256: str
    invariant_results_sha256: str
    static_qualification_policy_hash: str
    runtime_capability_name: Literal["allow_real_training", "allow_real_shadow"]
    runtime_capability_enabled_at_authorization: bool
    runtime_capability_hash_at_authorization: str
    frozen_retraining_policy_hash: str
    frozen_lifecycle_policy_hash: str
    frozen_promotion_policy_hash: str
    frozen_config_hash: str | None
    training_run_id: str | None
    validation_run_id: str | None
    model_artifact_manifest_sha256: str | None = None
    candidate_registration_sha256: str | None = None
    validation_manifest_sha256: str | None = None
    approved_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    issued_at: str
    expires_at: str
    single_use: Literal[True] = True
    promotion_forbidden: Literal[True] = True
    trading_forbidden: Literal[True] = True


class AuthorizationRevocation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["qualification_authorization_revocation"] = (
        "qualification_authorization_revocation"
    )
    revocation_id: str
    authorization_id: str
    authorization_sha256: str
    qualification_run_id: str
    stage: AuthorizationStage
    revoked_by: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    revoked_at: str


class AuthorizationConsumptionClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["qualification_authorization_consumption_claim"] = (
        "qualification_authorization_consumption_claim"
    )
    consumption_id: str
    authorization_id: str
    authorization_sha256: str
    qualification_run_id: str
    stage: AuthorizationStage
    qualification_snapshot_manifest_sha256: str
    stage_event_sequence: int = Field(ge=1)
    consumed_at: str
    runtime_capability_enabled: Literal[True] = True
    static_policy_hash: str
    attempt_identity: str
    status: Literal["CLAIMED"] = "CLAIMED"


class AuthorizationConsumptionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["qualification_authorization_consumption_receipt"] = (
        "qualification_authorization_consumption_receipt"
    )
    receipt_id: str
    consumption_id: str
    authorization_id: str
    qualification_run_id: str
    stage: AuthorizationStage
    completed_at: str
    status: Literal["COMPLETED", "FAILED"]
    result_manifest_path: str | None = None
    result_manifest_sha256: str | None = None
    error: str | None = None


class AuthorizationArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal[
        "qualification_authorization_manifest",
        "qualification_revocation_manifest",
        "qualification_consumption_claim_manifest",
        "qualification_consumption_receipt_manifest",
    ]
    identity: str
    payload_file: str
    payload_sha256: str
    manifest_written_last: Literal[True] = True


class QualificationAuthorizationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: AuthorizationStage
    status: AuthorizationState
    authorization_id: str | None = None
    expires_at: str | None = None
    authorization_sha256: str | None = None
    consumed_authorization_ids: tuple[str, ...] = ()
    revoked_authorization_ids: tuple[str, ...] = ()
    expired_authorization_ids: tuple[str, ...] = ()
    stale_authorization_ids: tuple[str, ...] = ()
    invalid_authorization_ids: tuple[str, ...] = ()
    legacy_unsupported_authorization_ids: tuple[str, ...] = ()
    message: str


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    authorization_id: str
    stage: AuthorizationStage
    status: AuthorizationState
    output_dir: Path
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class RevocationResult:
    revocation_id: str
    authorization_id: str
    effective: bool
    output_dir: Path
    idempotent: bool = False
