"""Strict schemas for read-only promotion gate evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type GateCheckStatus = Literal["PASS", "FAIL", "WARNING"]
type GateStatus = Literal["PASS", "FAIL", "REVIEW_REQUIRED"]


class GateCheck(BaseModel):
    """One deterministic promotion eligibility check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    status: GateCheckStatus
    message: str = Field(min_length=1)
    evidence_hash: str = Field(min_length=64, max_length=64)


class GateResult(BaseModel):
    """Read-only outcome eligible for subsequent human review only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_gate_result"] = "promotion_gate_result"
    request_id: str
    candidate_model_id: str
    status: GateStatus
    checks: tuple[GateCheck, ...]
    policy_version: str = "legacy_v1"
    policy_hash: str = Field(min_length=64, max_length=64)
    created_at: str


class GateManifest(BaseModel):
    """Completion marker written after gate result and Markdown report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_gate_manifest"] = "promotion_gate_manifest"
    request_id: str
    gate_identity: str = Field(min_length=64, max_length=64)
    status: GateStatus
    policy_version: str = "legacy_v1"
    policy_hash: str = Field(min_length=64, max_length=64)
    source_request_manifest_hash: str = Field(min_length=64, max_length=64)
    artifact_hashes: dict[str, str]
    registry_modified: Literal[False] = False
    champion_modified: Literal[False] = False
    created_at: str
