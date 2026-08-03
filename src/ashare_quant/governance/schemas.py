"""Typed contracts for immutable governance reports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type CheckStatus = Literal["PASS", "WARNING", "FAIL"]


class GovernanceCheck(BaseModel):
    """One deterministic read-only validation result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: CheckStatus
    message: str
    source_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GovernanceReport(BaseModel):
    """Published status, production-validation, or recovery report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: str
    report_type: Literal["status", "validation", "recovery"]
    status: CheckStatus
    generated_at: str
    summary: dict[str, Any]
    checks: tuple[GovernanceCheck, ...] = ()
    source_hashes: dict[str, str] = Field(default_factory=dict)
    read_only: Literal[True] = True


class GovernanceManifest(BaseModel):
    """Completion marker written after one governance report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["governance_report_manifest"] = "governance_report_manifest"
    report_type: Literal["status", "validation", "recovery"]
    snapshot_id: str
    report_sha256: str
    source_hashes: dict[str, str]
    created_at: str
    read_only: Literal[True] = True


def overall_status(checks: list[GovernanceCheck]) -> CheckStatus:
    """Return FAIL over WARNING over PASS."""

    if any(item.status == "FAIL" for item in checks):
        return "FAIL"
    if any(item.status == "WARNING" for item in checks):
        return "WARNING"
    return "PASS"
