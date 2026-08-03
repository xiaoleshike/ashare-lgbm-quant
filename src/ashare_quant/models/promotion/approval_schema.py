"""Immutable schemas for human promotion review events."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type ApprovalEventType = Literal["APPROVED", "REJECTED"]


class ApprovalEvent(BaseModel):
    """One append-only terminal human decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1)
    event_type: ApprovalEventType
    request_id: str
    request_hash: str = Field(min_length=64, max_length=64)
    gate_result_hash: str = Field(min_length=64, max_length=64)
    registry_hash_at_review: str = Field(min_length=64, max_length=64)
    reviewer: str = Field(min_length=1)
    requester: str = Field(min_length=1)
    decision: Literal["approved", "rejected"]
    comments: str = Field(min_length=1)
    created_at: str
    expires_at: str


class ApprovalEventManifest(BaseModel):
    """Completion marker written after an approval event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["promotion_approval_event"] = "promotion_approval_event"
    event_id: str
    request_id: str
    event_identity_hash: str = Field(min_length=64, max_length=64)
    event_file_sha256: str = Field(min_length=64, max_length=64)
    policy_hash: str = Field(min_length=64, max_length=64)
    created_at: str
    manifest_written_last: Literal[True] = True
