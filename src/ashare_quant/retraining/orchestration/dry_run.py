"""Controlled no-training lifecycle rehearsal and immutable reporting."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import detect_production_lock_owner
from ashare_quant.retraining.orchestration.service import RetrainingLifecycleOrchestrator
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info


class LifecycleDryRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_name: Literal["retraining_lifecycle_dry_run"] = "retraining_lifecycle_dry_run"
    dry_run_id: str
    request_id: str
    parent_model_id: str
    horizon: int
    as_of: str
    status: Literal["READY_TO_EXECUTE", "BLOCKED", "ERROR"]
    readiness_status: str
    scheduler_status: str
    closed_loop_status: str
    governance_status: str
    policy_status: str
    policy_drift: bool
    cooldown_status: str
    budget_status: str
    lock_status: str
    proposed_lifecycle_run_id: str
    proposed_training_run_id: str
    planned_stages: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    source_artifacts: dict[str, str]
    source_hashes: dict[str, str]
    no_mutation_confirmed: Literal[True] = True
    generated_at: str
    git_commit: str | None
    git_dirty: bool
    config_hash: str | None


@dataclass(frozen=True, slots=True)
class LifecycleDryRunResult:
    dry_run_id: str
    status: str
    output_dir: Path
    idempotent: bool = False


class LifecycleDryRunService:
    """Inspect readiness and controls without calling any execution-stage service."""

    def __init__(self, orchestrator: RetrainingLifecycleOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.root = orchestrator.settings.paths.reports / "retraining" / "lifecycle_dry_runs"

    def run(self, request_id: str, *, as_of: str | None = None) -> LifecycleDryRunResult:
        proposed_run_id, frozen = self.orchestrator.proposed_identity(request_id)
        resolved_as_of = as_of or frozen.request.as_of
        readiness = self.orchestrator.readiness.validate(resolved_as_of, request_id=request_id)
        controls = self.orchestrator.operational_controls()
        budget = controls.budget()
        cooldown = controls.cooldown(
            lifecycle_run_id=proposed_run_id,
            parent_model_id=frozen.request.target_models[0].model_id,
            horizon=frozen.request.target_models[0].horizon,
        )
        owner = detect_production_lock_owner(self.orchestrator.lifecycle_lock)
        check_map = readiness.report.checks
        policy_drift = (
            frozen.promotion_policy_hash != self.orchestrator.promotion_policy.policy_hash
        )
        blocked: list[str] = []
        if readiness.report.status != "READY":
            blocked.append("retraining readiness is not READY")
        if not cooldown.allowed:
            blocked.append("lifecycle cooldown is active")
        if not budget.allowed:
            blocked.append("daily training-attempt budget is exhausted")
        if owner is not None:
            blocked.append(f"lifecycle lock is held: {owner.describe()}")
        if policy_drift:
            blocked.append("Promotion Policy differs from frozen training request")
        identity = canonical_payload_hash(
            {
                "request_hash": frozen.training_request_hash,
                "as_of": resolved_as_of,
                "readiness_identity": readiness.report.run_id,
                "retraining_policy_hash": self.orchestrator.retraining_policy.policy_hash,
                "lifecycle_policy_hash": self.orchestrator.retraining_policy.lifecycle_policy_hash,
                "promotion_policy_hash": self.orchestrator.promotion_policy.policy_hash,
                "config_hash": config_hash(self.orchestrator.config_path),
            }
        )
        dry_run_id = f"lifecycle_dry_run_{identity[:24]}"
        request_path = (
            self.orchestrator.request_storage.requests_root / request_id / "training_request.json"
        )
        readiness_path = readiness.output_dir / "manifest.json"
        git = current_git_info()
        proposed_training_hash = canonical_payload_hash(
            {"lifecycle": proposed_run_id, "request": frozen.training_request_hash}
        )
        report = LifecycleDryRunReport(
            dry_run_id=dry_run_id,
            request_id=request_id,
            parent_model_id=frozen.request.target_models[0].model_id,
            horizon=frozen.request.target_models[0].horizon,
            as_of=resolved_as_of,
            status="BLOCKED" if blocked else "READY_TO_EXECUTE",
            readiness_status=readiness.report.status,
            scheduler_status=str(check_map.get("scheduler", "UNKNOWN")),
            closed_loop_status=str(check_map.get("closed_loop_manifest", "UNKNOWN")),
            governance_status=str(check_map.get("governance_snapshot", "UNKNOWN")),
            policy_status=str(check_map.get("promotion_policy", "UNKNOWN")),
            policy_drift=policy_drift,
            cooldown_status="PASS" if cooldown.allowed else "BLOCKED",
            budget_status="PASS" if budget.allowed else "BLOCKED",
            lock_status="AVAILABLE" if owner is None else "HELD",
            proposed_lifecycle_run_id=proposed_run_id,
            proposed_training_run_id=f"training_{proposed_training_hash[:24]}",
            planned_stages=("readiness", "training", "validation", "shadow", "observation"),
            blocked_reasons=tuple(blocked),
            warnings=(),
            source_artifacts={
                "training_request": str(request_path),
                "readiness": str(readiness_path),
            },
            source_hashes={
                "training_request": file_sha256(request_path),
                "readiness": file_sha256(readiness_path),
            },
            generated_at=frozen.request.created_at,
            git_commit=git["commit"],
            git_dirty=bool(git["dirty"]),
            config_hash=config_hash(self.orchestrator.config_path),
        )
        return self._publish(report, identity, budget=asdict(budget), cooldown=asdict(cooldown))

    def _publish(
        self,
        report: LifecycleDryRunReport,
        identity_hash: str,
        *,
        budget: dict[str, Any],
        cooldown: dict[str, Any],
    ) -> LifecycleDryRunResult:
        output = self.root / report.dry_run_id
        existing = _json(output / "manifest.json") if (output / "manifest.json").is_file() else None
        if existing is not None:
            if existing.get("identity_hash") != identity_hash:
                raise DataValidationError("dry-run identity cannot overwrite existing output")
            return LifecycleDryRunResult(report.dry_run_id, report.status, output, True)
        if output.exists():
            raise DataValidationError(f"incomplete lifecycle dry-run output exists: {output}")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".dry-run-"))
        try:
            atomic_write_json(staging / "dry_run.json", report.model_dump(mode="json"))
            atomic_write_json(
                staging / "execution_plan.json",
                {
                    "planned_stages": list(report.planned_stages),
                    "budget_decision": budget,
                    "cooldown_decision": cooldown,
                    "no_training": True,
                    "no_production_mutation": True,
                },
            )
            (staging / "report.md").write_text(_render(report), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "artifact_name": "retraining_lifecycle_dry_run_manifest",
                "dry_run_id": report.dry_run_id,
                "identity_hash": identity_hash,
                "status": report.status,
                "dry_run_sha256": file_sha256(staging / "dry_run.json"),
                "execution_plan_sha256": file_sha256(staging / "execution_plan.json"),
                "report_sha256": file_sha256(staging / "report.md"),
                "manifest_written_last": True,
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if file_sha256(staging / "dry_run.json") != manifest["dry_run_sha256"]:
                raise DataValidationError("staged lifecycle dry-run failed hash validation")
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return LifecycleDryRunResult(report.dry_run_id, report.status, output)


def _render(report: LifecycleDryRunReport) -> str:
    lines = [
        "# Retraining Lifecycle Dry Run",
        "",
        f"- Status: {report.status}",
        f"- Request: {report.request_id}",
        f"- Proposed lifecycle: {report.proposed_lifecycle_run_id}",
        f"- As-of: {report.as_of}",
        "- Mutation: none",
        "",
        "## Blocking Conditions",
        "",
    ]
    lines.extend(f"- {value}" for value in report.blocked_reasons)
    if not report.blocked_reasons:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid lifecycle dry-run manifest: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError("lifecycle dry-run manifest must be an object")
    return value
