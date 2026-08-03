"""Immutable schemas for governed historical-Champion rollback."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RollbackReason(BaseModel):
    """Operator-supplied reason frozen into a rollback request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(min_length=1)
    description: str = Field(min_length=1)


class RollbackTargetContract(BaseModel):
    """Frozen target identity and deployment compatibility contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_type: str
    feature_hash: str
    horizon: int = Field(gt=0)
    holding_period: int = Field(gt=0)
    execution_rule: str
    historical_assignment_id: str
    artifact_hashes: dict[str, str]
    artifact_set_hash: str = Field(min_length=64, max_length=64)


class RollbackRequest(BaseModel):
    """One immutable request to restore a historical Champion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["model_rollback_request"] = "model_rollback_request"
    request_id: str
    request_type: Literal["rollback"] = "rollback"
    status: Literal["REQUEST_CREATED"] = "REQUEST_CREATED"
    target_model_id: str
    current_champion_model_id: str
    deployment_slot: str
    reason: RollbackReason
    target_contract: RollbackTargetContract
    registry_hash: str = Field(min_length=64, max_length=64)
    requester: str
    created_at: str


class RollbackRequestManifest(BaseModel):
    """Completion marker written last for one rollback request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["model_rollback_request_bundle"] = "model_rollback_request_bundle"
    request_id: str
    identity_hash: str = Field(min_length=64, max_length=64)
    request_sha256: str = Field(min_length=64, max_length=64)
    created_at: str
    manifest_written_last: Literal[True] = True


class RollbackValidationResult(BaseModel):
    """Immutable successful evidence validation bound to current state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_validation_result"] = "rollback_validation_result"
    request_id: str
    status: Literal["VALIDATED"] = "VALIDATED"
    request_hash: str = Field(min_length=64, max_length=64)
    registry_hash: str = Field(min_length=64, max_length=64)
    target_artifact_hash: str = Field(min_length=64, max_length=64)
    historical_assignment_id: str
    checks: tuple[str, ...]
    validated_at: str


class RollbackValidationManifest(BaseModel):
    """Completion marker for rollback validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_validation_manifest"] = "rollback_validation_manifest"
    request_id: str
    identity_hash: str = Field(min_length=64, max_length=64)
    result_sha256: str = Field(min_length=64, max_length=64)
    created_at: str
    manifest_written_last: Literal[True] = True


class RollbackApprovalEvent(BaseModel):
    """One append-only human rollback decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str
    event_type: Literal["APPROVED", "REJECTED"]
    request_id: str
    request_hash: str = Field(min_length=64, max_length=64)
    validation_result_hash: str = Field(min_length=64, max_length=64)
    registry_hash_at_review: str = Field(min_length=64, max_length=64)
    target_artifact_hash_at_review: str = Field(min_length=64, max_length=64)
    reviewer: str
    requester: str
    decision: Literal["approved", "rejected"]
    comments: str
    created_at: str
    expires_at: str


class RollbackApprovalManifest(BaseModel):
    """Completion marker written after a rollback approval event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_approval_event"] = "rollback_approval_event"
    event_id: str
    request_id: str
    event_identity_hash: str = Field(min_length=64, max_length=64)
    event_file_sha256: str = Field(min_length=64, max_length=64)
    policy_hash: str = Field(min_length=64, max_length=64)
    created_at: str
    manifest_written_last: Literal[True] = True


class RollbackChampionAssignment(BaseModel):
    """Immutable Champion history record produced by a rollback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_champion_assignment"] = "rollback_champion_assignment"
    champion_assignment_id: str
    deployment_slot: str
    model_id: str
    previous_champion_model_id: str
    rollback_request_id: str
    approval_event_id: str
    registry_version_id: str
    activated_at: str


class RollbackApplyPending(BaseModel):
    """Recovery journal written before switching registry.json."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_apply_pending"] = "rollback_apply_pending"
    request_id: str
    registry_version_id: str
    created_at: str


class RollbackApplyManifest(BaseModel):
    """Final commit marker for a completed rollback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["rollback_apply"] = "rollback_apply"
    request_id: str
    status: Literal["APPLIED"] = "APPLIED"
    target_model_id: str
    previous_champion_model_id: str
    approval_event_id: str
    approval_event_hash: str = Field(min_length=64, max_length=64)
    registry_version_id: str
    registry_file_hash: str = Field(min_length=64, max_length=64)
    champion_assignment_id: str
    champion_history_hash: str = Field(min_length=64, max_length=64)
    activated_at: str
    manifest_written_last: Literal[True] = True
