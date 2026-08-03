"""Production closed-loop lineage validation."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.readiness.schemas import ClosedLoopContext, SchedulerContext
from ashare_quant.retraining.readiness.validators import SourceTracker, require_string

_REQUIRED_STAGES = {
    "data_update",
    "data_validate",
    "raw_freshness_gate",
    "universe_build",
    "universe_validate",
    "universe_readiness_gate",
    "features_build",
    "features_validate",
    "features_readiness_gate",
    "model_predict",
    "strategy_candidates",
    "publish_production_summary",
    "shadow_prediction",
    "monitoring",
    "research_agent",
    "governance_snapshot",
}


def validate_closed_loop(
    *,
    reports_root: Path,
    runs_root: Path,
    as_of: str,
    scheduler: SchedulerContext,
    tracker: SourceTracker,
) -> ClosedLoopContext:
    """Validate the exact successful non-dry-run production and closed-loop manifests."""

    matches = list(runs_root.glob(f"????????/{scheduler.production_run_id}/manifest.json"))
    if len(matches) != 1:
        raise DataValidationError("production run manifest is missing or ambiguous")
    run = tracker.json(matches[0], "production run manifest")
    if (
        run.get("status") != "success"
        or run.get("pipeline_type") != "production_daily"
        or str(run.get("resolved_as_of")) != as_of
        or run.get("run_id") != scheduler.production_run_id
        or "--dry-run" in str(run.get("command") or "")
        or run.get("scheduler_invocation_id") != scheduler.invocation_id
    ):
        raise DataValidationError("production run is failed, dry-run, incomplete, or unlinked")
    stages = run.get("stages")
    if not isinstance(stages, list):
        raise DataValidationError("production run lacks stage records")
    stage_status = {
        str(item.get("name")): str(item.get("status")) for item in stages if isinstance(item, dict)
    }
    incomplete = sorted(name for name in _REQUIRED_STAGES if stage_status.get(name) != "success")
    if incomplete:
        raise DataValidationError(f"production closed-loop stages are incomplete: {incomplete}")
    latest = reports_root / as_of / "closed_loop_manifest.json"
    closed = tracker.json(latest, "closed-loop manifest")
    immutable = reports_root / as_of / "closed_loop" / scheduler.production_run_id / "manifest.json"
    immutable_payload = tracker.json(immutable, "immutable closed-loop manifest")
    if closed != immutable_payload:
        raise DataValidationError("closed-loop projection differs from immutable manifest")
    if (
        closed.get("artifact_name") != "production_closed_loop_manifest"
        or closed.get("production_run_id") != scheduler.production_run_id
        or str(closed.get("as_of")) != as_of
    ):
        raise DataValidationError("closed-loop manifest identity is invalid")
    closed_stages = closed.get("stages")
    if not isinstance(closed_stages, list):
        raise DataValidationError("closed-loop manifest lacks stages")
    closed_status = {
        str(item.get("name")): str(item.get("status"))
        for item in closed_stages
        if isinstance(item, dict)
    }
    incomplete = sorted(name for name in _REQUIRED_STAGES if closed_status.get(name) != "success")
    if incomplete:
        raise DataValidationError(f"closed-loop manifest contains incomplete stages: {incomplete}")
    context = ClosedLoopContext(
        production_run_id=scheduler.production_run_id,
        shadow_run_id=require_string(closed, "shadow_run_id", "closed-loop manifest"),
        monitor_run_id=require_string(closed, "monitor_run_id", "closed-loop manifest"),
        research_run_id=require_string(closed, "research_run_id", "closed-loop manifest"),
        governance_snapshot_id=require_string(
            closed, "governance_snapshot_id", "closed-loop manifest"
        ),
    )
    for name, expected in (
        ("shadow_run_id", context.shadow_run_id),
        ("monitor_run_id", context.monitor_run_id),
        ("research_run_id", context.research_run_id),
        ("governance_snapshot_id", context.governance_snapshot_id),
    ):
        if run.get(name) != expected:
            raise DataValidationError(f"production run and closed-loop {name} differ")
    return context
