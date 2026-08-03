"""Immutable contracts for retraining execution readiness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CheckStatus = Literal["PASS", "FAIL", "FAILED_POLICY_DRIFT", "NOT_RUN"]


class ReadinessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: CheckStatus
    message: str


class RetrainingReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_execution_readiness"] = "retraining_execution_readiness"
    run_id: str
    as_of: str
    request_id: str | None
    status: Literal["READY", "FAILED"]
    checks: dict[str, CheckStatus]
    check_details: tuple[ReadinessCheck, ...]
    production_run_id: str | None
    governance_snapshot_hash: str | None
    promotion_policy_hash: str
    request_hash: str | None
    read_only: Literal[True] = True


class RetrainingReadinessManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_execution_readiness_manifest"] = (
        "retraining_execution_readiness_manifest"
    )
    run_id: str
    as_of: str
    request_id: str | None
    status: Literal["READY", "FAILED"]
    checks: dict[str, CheckStatus]
    source_artifacts: tuple[str, ...]
    source_hashes: dict[str, str]
    promotion_policy_hash: str
    request_hash: str | None
    git_commit: str | None
    git_dirty: bool
    config_hash: str | None
    report_sha256: str = Field(min_length=64, max_length=64)
    markdown_sha256: str = Field(min_length=64, max_length=64)
    generated_at: str
    manifest_written_last: Literal[True] = True


@dataclass(frozen=True, slots=True)
class SchedulerContext:
    invocation_id: str
    production_run_id: str
    completed_time: str


@dataclass(frozen=True, slots=True)
class ClosedLoopContext:
    production_run_id: str
    shadow_run_id: str
    monitor_run_id: str
    research_run_id: str
    governance_snapshot_id: str


@dataclass(frozen=True, slots=True)
class GovernanceContext:
    snapshot_hash: str
    promotion_policy_hash: str
    promotion_policy_version: str
    previous_promotion_policy_hash: str | None = None
    previous_promotion_policy_version: str | None = None


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    report: RetrainingReadinessReport
    output_dir: Path
    idempotent: bool = False
