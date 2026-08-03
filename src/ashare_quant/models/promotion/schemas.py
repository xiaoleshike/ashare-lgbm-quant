"""Strict immutable schemas for model-promotion governance artifacts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type EvidenceType = Literal[
    "challenger_evaluation",
    "executable_validation",
    "shadow_prediction",
    "performance_observation",
    "monitoring_summary",
    "alerts",
]


class ModelIdentity(BaseModel):
    """Frozen registry and artifact identity for one model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    feature_hash: str = Field(min_length=1)
    artifact_manifest_sha256: str = Field(min_length=64, max_length=64)
    status: Literal["candidate", "champion", "retired"]


class EvidenceReference(BaseModel):
    """One physically hashed evidence artifact bounded by a review cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_type: EvidenceType
    source_path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    manifest_identity: dict[str, Any]
    evidence_date: str = Field(pattern=r"^\d{8}$")
    cutoff_date: str = Field(pattern=r"^\d{8}$")


class EvidenceSnapshot(BaseModel):
    """Canonical collection of all evidence frozen for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_evidence_snapshot"] = "promotion_evidence_snapshot"
    candidate_model_id: str
    evidence_cutoff_date: str = Field(pattern=r"^\d{8}$")
    sources: tuple[EvidenceReference, ...] = Field(min_length=6, max_length=6)
    evidence_snapshot_hash: str = Field(min_length=64, max_length=64)


class InferenceCompatibility(BaseModel):
    """Static deployment compatibility without loading or scoring a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compatible: Literal[True] = True
    model_type: str
    required_artifacts: tuple[str, ...]
    feature_count: int = Field(gt=0)


class DeploymentContract(BaseModel):
    """Frozen contract required by a future, separately approved deployment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_deployment_contract"] = "promotion_deployment_contract"
    model_id: str
    feature_hash: str = Field(min_length=1)
    horizon: int = Field(gt=0)
    holding_period: int = Field(gt=0)
    execution_rule: str = Field(min_length=1)
    inference_compatibility: InferenceCompatibility
    deployment_contract_hash: str = Field(min_length=64, max_length=64)


class ChampionAssignment(BaseModel):
    """Descriptive current assignment; applying a new assignment is out of scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    champion_assignment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    deployment_slot: str = Field(min_length=1)
    activated_at: str = Field(min_length=1)
    previous_assignment_id: str | None = None


class PromotionRequest(BaseModel):
    """Immutable request linking candidate, incumbent, evidence, and contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["model_promotion_request"] = "model_promotion_request"
    request_id: str = Field(min_length=1)
    state: Literal["promotion_requested"] = "promotion_requested"
    candidate: ModelIdentity
    current_champion: ModelIdentity
    current_champion_assignment: ChampionAssignment
    evidence_cutoff_date: str = Field(pattern=r"^\d{8}$")
    evidence_snapshot_hash: str = Field(min_length=64, max_length=64)
    deployment_contract_hash: str = Field(min_length=64, max_length=64)
    registry_hash: str = Field(min_length=64, max_length=64)
    created_time: str = Field(min_length=1)


class PromotionBundleManifest(BaseModel):
    """Completion marker written after every other request artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["model_promotion_governance_bundle"] = (
        "model_promotion_governance_bundle"
    )
    request_id: str
    status: Literal["complete"] = "complete"
    identity_hash: str = Field(min_length=64, max_length=64)
    artifact_hashes: dict[str, str]
    created_time: str
