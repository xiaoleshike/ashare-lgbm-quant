"""Scheduler health validation for retraining execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.readiness.policy import RetrainingReadinessPolicy
from ashare_quant.retraining.readiness.schemas import SchedulerContext
from ashare_quant.retraining.readiness.validators import SourceTracker, require_string


def validate_scheduler(
    *,
    settings: AppSettings,
    project_root: Path,
    runs_root: Path,
    as_of: str,
    policy: RetrainingReadinessPolicy,
    tracker: SourceTracker,
    now: datetime,
) -> SchedulerContext:
    """Require one recent successful scheduler invocation for the requested session."""

    if not settings.production.scheduler.enabled:
        raise DataValidationError("production scheduler is disabled")
    for name in ("ashare-quant-production.service", "ashare-quant-production.timer"):
        tracker.track(project_root / "deploy" / "systemd" / name)
    candidates: list[tuple[str, Path, dict[str, object]]] = []
    for path in sorted((runs_root / "scheduler").glob("????????/*.json")):
        payload = tracker.json(path, "scheduler invocation")
        if str(payload.get("resolved_as_of")) == as_of:
            candidates.append((str(payload.get("completed_time") or ""), path, payload))
    if not candidates:
        raise DataValidationError(f"scheduler invocation is missing for as_of={as_of}")
    _, _, invocation = max(candidates, key=lambda item: (item[0], str(item[1])))
    if invocation.get("status") != "success" or invocation.get("skipped") is True:
        raise DataValidationError("latest scheduler invocation did not complete successfully")
    completed = require_string(invocation, "completed_time", "scheduler invocation")
    try:
        completed_at = datetime.fromisoformat(completed).astimezone(UTC)
    except ValueError as error:
        raise DataValidationError("scheduler completion time is invalid") from error
    if now.astimezone(UTC) - completed_at > timedelta(hours=policy.maximum_scheduler_age_hours):
        raise DataValidationError("latest scheduler invocation exceeds maximum allowed interval")
    attempts = invocation.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise DataValidationError("scheduler invocation has no audited pipeline attempt")
    retry_count = len(attempts) - 1
    if retry_count >= settings.production.scheduler.max_pipeline_attempts:
        raise DataValidationError("scheduler retry count exceeds configured maximum")
    return SchedulerContext(
        invocation_id=require_string(invocation, "invocation_id", "scheduler invocation"),
        production_run_id=require_string(invocation, "pipeline_run_id", "scheduler invocation"),
        completed_time=completed,
    )
