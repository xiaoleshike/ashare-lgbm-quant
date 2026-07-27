"""Session-aware scheduler facade for production and full-data operations."""

from __future__ import annotations

import json
import os
import time as time_module
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import ALL_DATASETS, get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.ingestion import DataIngestionService
from ashare_quant.data.quality_logging import append_validation_results
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.data.validation import DataValidator
from ashare_quant.orchestration.lock import (
    DEFAULT_PRODUCTION_LOCK_PATH,
    ProductionLockError,
    production_lock,
)
from ashare_quant.orchestration.production import ProductionPipelineResult
from ashare_quant.orchestration.publication import validate_production_publication
from ashare_quant.orchestration.run_manifest import (
    DEFAULT_RUNS_ROOT,
    create_run,
    record_failure,
    record_stage_end,
    record_stage_start,
    update_run_status,
)
from ashare_quant.utils.manifest import atomic_write_json

type SchedulerStatus = Literal["success", "failed", "skipped"]


class ProductionPipelineService(Protocol):
    """Production pipeline API used by the scheduler facade."""

    def run(
        self,
        as_of: str,
        *,
        dry_run: bool = False,
        invocation_source: str = "manual_cli",
        scheduler_trigger_time: str | None = None,
        scheduler_invocation_id: str | None = None,
        service_execution_id: str | None = None,
        timezone: str | None = None,
    ) -> ProductionPipelineResult: ...


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """Result of resolving one automatic daily scheduler trigger."""

    status: Literal["run", "skipped"]
    resolved_as_of: str | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SchedulerResult:
    """Terminal scheduler invocation outcome."""

    status: SchedulerStatus
    exit_code: int
    invocation_id: str
    invocation_manifest: Path
    resolved_as_of: str | None
    pipeline_run_id: str | None = None
    skipped_reason: str | None = None
    error_message: str | None = None


class SchedulerInvocation:
    """Atomically maintained audit record around one CLI or systemd invocation."""

    def __init__(
        self,
        *,
        runs_root: Path,
        source: str,
        timezone: str,
        requested_as_of: str | None,
        trigger_time: datetime,
        service_execution_id: str | None,
        command: str,
    ) -> None:
        invocation_id = (
            f"{trigger_time.astimezone(UTC).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"
        )
        local_day = trigger_time.astimezone(ZoneInfo(timezone)).strftime("%Y%m%d")
        self.invocation_id = invocation_id
        self.path = runs_root / "scheduler" / local_day / f"{invocation_id}.json"
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_name": "scheduler_invocation",
            "invocation_id": invocation_id,
            "invocation_source": source,
            "command": command,
            "requested_as_of": requested_as_of,
            "resolved_as_of": None,
            "timezone": timezone,
            "scheduler_trigger_time": trigger_time.isoformat(),
            "service_execution_id": service_execution_id,
            "status": "running",
            "skipped": False,
            "skipped_reason": None,
            "pipeline_run_id": None,
            "attempts": [],
            "error_message": None,
            "completed_time": None,
        }
        atomic_write_json(self.path, self.payload)

    def resolve(self, as_of: str | None) -> None:
        """Record the session selected by trade_cal."""

        self.payload["resolved_as_of"] = as_of
        atomic_write_json(self.path, self.payload)

    def add_attempt(self, run_id: str | None, status: str, error_message: str | None) -> None:
        """Append one independently manifested pipeline attempt."""

        attempts = self.payload["attempts"]
        if not isinstance(attempts, list):
            raise ValueError("scheduler invocation attempts must be a list")
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "pipeline_run_id": run_id,
                "status": status,
                "error_message": error_message,
            }
        )
        self.payload["pipeline_run_id"] = run_id
        atomic_write_json(self.path, self.payload)

    def finish(
        self,
        status: SchedulerStatus,
        *,
        skipped_reason: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Publish the terminal scheduler invocation state."""

        self.payload["status"] = status
        self.payload["skipped"] = status == "skipped"
        self.payload["skipped_reason"] = skipped_reason
        self.payload["error_message"] = error_message
        self.payload["completed_time"] = datetime.now(UTC).isoformat()
        atomic_write_json(self.path, self.payload)


class ProductionScheduler:
    """Resolve automatic dates, avoid duplicate publication, and audit attempts."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        raw_store: ParquetDataStore,
        pipeline: ProductionPipelineService,
        reports_root: Path,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        now: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time_module.sleep,
    ) -> None:
        self.settings = settings
        self.raw_store = raw_store
        self.pipeline = pipeline
        self.reports_root = reports_root
        self.runs_root = runs_root
        self._now = now or (lambda: datetime.now(UTC))
        self._sleeper = sleeper

    def run(
        self,
        requested_as_of: str | None,
        *,
        dry_run: bool = False,
        invocation_source: str | None = None,
    ) -> SchedulerResult:
        """Run one explicit or automatic production invocation."""

        trigger_time = self._now()
        source = invocation_source or os.environ.get("ASHARE_QUANT_INVOCATION_SOURCE", "manual_cli")
        command = "ashare-quant pipeline production"
        if requested_as_of is not None:
            command += f" --as-of {requested_as_of}"
        if dry_run:
            command += " --dry-run"
        invocation = SchedulerInvocation(
            runs_root=self.runs_root,
            source=source,
            timezone=self.settings.production.timezone,
            requested_as_of=requested_as_of,
            trigger_time=trigger_time,
            service_execution_id=os.environ.get("INVOCATION_ID"),
            command=command,
        )

        if requested_as_of is None:
            try:
                decision = resolve_automatic_production_date(
                    self.raw_store, self.settings, now=trigger_time
                )
            except Exception as error:  # noqa: BLE001 - preserve scheduler audit on bad calendar.
                message = f"{type(error).__name__}: {error}"
                invocation.finish("failed", error_message=message)
                return SchedulerResult(
                    "failed",
                    2,
                    invocation.invocation_id,
                    invocation.path,
                    None,
                    error_message=message,
                )
            invocation.resolve(decision.resolved_as_of)
            if decision.status == "skipped":
                invocation.finish("skipped", skipped_reason=decision.reason)
                return SchedulerResult(
                    "skipped",
                    0,
                    invocation.invocation_id,
                    invocation.path,
                    decision.resolved_as_of,
                    skipped_reason=decision.reason,
                )
            resolved = decision.resolved_as_of
            if resolved is None:
                raise AssertionError("run decision must include resolved_as_of")
            if (
                self.settings.production.scheduler.skip_if_already_successful
                and _has_valid_existing_publication(self.reports_root, self.runs_root, resolved)
            ):
                invocation.finish("skipped", skipped_reason="already_successful")
                return SchedulerResult(
                    "skipped",
                    0,
                    invocation.invocation_id,
                    invocation.path,
                    resolved,
                    skipped_reason="already_successful",
                )
        else:
            resolved = requested_as_of
            invocation.resolve(resolved)

        attempts = self.settings.production.scheduler.max_pipeline_attempts
        for attempt in range(1, attempts + 1):
            try:
                result = self.pipeline.run(
                    resolved,
                    dry_run=dry_run,
                    invocation_source=source,
                    scheduler_trigger_time=trigger_time.isoformat(),
                    scheduler_invocation_id=invocation.invocation_id,
                    service_execution_id=os.environ.get("INVOCATION_ID"),
                    timezone=self.settings.production.timezone,
                )
            except Exception as error:  # noqa: BLE001 - preserve scheduler audit on lock/runtime error.
                message = f"{type(error).__name__}: {error}"
                invocation.add_attempt(None, "failed", message)
                invocation.finish("failed", error_message=message)
                return SchedulerResult(
                    "failed",
                    3 if isinstance(error, ProductionLockError) else 2,
                    invocation.invocation_id,
                    invocation.path,
                    resolved,
                    error_message=message,
                )
            invocation.add_attempt(result.run.run_id, result.status, result.error_message)
            if result.status == "success":
                try:
                    if not dry_run:
                        validate_production_publication(
                            reports_root=self.reports_root,
                            runs_root=self.runs_root,
                            as_of=resolved,
                            expected_run_id=result.run.run_id,
                            run_manifest_path=result.run.manifest_path,
                        )
                except DataValidationError as error:
                    invocation.finish("failed", error_message=str(error))
                    return SchedulerResult(
                        "failed",
                        2,
                        invocation.invocation_id,
                        invocation.path,
                        resolved,
                        result.run.run_id,
                        error_message=str(error),
                    )
                invocation.finish("success")
                return SchedulerResult(
                    "success",
                    0,
                    invocation.invocation_id,
                    invocation.path,
                    resolved,
                    result.run.run_id,
                )
            if attempt >= attempts or not _is_transient_error(result.error_message):
                invocation.finish("failed", error_message=result.error_message)
                return SchedulerResult(
                    "failed",
                    result.exit_code or 2,
                    invocation.invocation_id,
                    invocation.path,
                    resolved,
                    result.run.run_id,
                    error_message=result.error_message,
                )
            self._sleeper(
                self.settings.production.scheduler.retry_backoff_seconds * (2 ** (attempt - 1))
            )
        raise AssertionError("scheduler attempt loop must return")


class FullDataUpdateScheduler:
    """Locked weekly all-dataset update using the latest completed trade_cal date."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        raw_store: ParquetDataStore,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.raw_store = raw_store
        self.runs_root = runs_root
        self.lock_path = lock_path
        self._now = now or (lambda: datetime.now(UTC))

    def run(self, *, invocation_source: str | None = None) -> SchedulerResult:
        """Update every configured dataset without using the system date as a session."""

        trigger = self._now()
        source = invocation_source or os.environ.get("ASHARE_QUANT_INVOCATION_SOURCE", "manual_cli")
        invocation = SchedulerInvocation(
            runs_root=self.runs_root,
            source=source,
            timezone=self.settings.production.timezone,
            requested_as_of=None,
            trigger_time=trigger,
            service_execution_id=os.environ.get("INVOCATION_ID"),
            command="ashare-quant pipeline full-update",
        )
        run = None
        try:
            as_of = latest_completed_trading_date(
                self.raw_store,
                self.settings.production.timezone,
                now=trigger,
            )
            invocation.resolve(as_of)
            with production_lock(self.lock_path, command="ashare-quant pipeline full-update"):
                run = create_run(
                    "ashare-quant pipeline full-update",
                    config_path=self.config_path,
                    runs_root=self.runs_root,
                    stages=("data_update", "data_validate"),
                    pipeline_type="scheduled_full_data_update",
                    as_of=as_of,
                    invocation_source=source,
                    resolved_as_of=as_of,
                    timezone=self.settings.production.timezone,
                    scheduler_trigger_time=trigger.isoformat(),
                    scheduler_invocation_id=invocation.invocation_id,
                    service_execution_id=os.environ.get("INVOCATION_ID"),
                )
                record_stage_start(run, "data_update")
                downloads = DataIngestionService(self.settings, self.raw_store).update(
                    ALL_DATASETS,
                    as_of,
                    refresh_snapshots=True,
                    repair_gaps=True,
                )
                record_stage_end(
                    run,
                    "data_update",
                    result={
                        "datasets": [
                            {
                                "dataset": item.dataset,
                                "rows_written": item.rows_written,
                                "skipped": item.skipped,
                                "message": item.message,
                            }
                            for item in downloads
                        ]
                    },
                )
                record_stage_start(run, "data_validate")
                validation = DataValidator(self.raw_store).validate_all(ALL_DATASETS)
                append_validation_results(self.settings.paths.data_quality_logs, validation)
                errors = [
                    error for result in validation if not result.ok for error in result.errors
                ]
                if errors:
                    record_failure(run, errors[0], stage_name="data_validate")
                    invocation.add_attempt(run.run_id, "failed", errors[0])
                    invocation.finish("failed", error_message=errors[0])
                    return SchedulerResult(
                        "failed",
                        1,
                        invocation.invocation_id,
                        invocation.path,
                        as_of,
                        run.run_id,
                        error_message=errors[0],
                    )
                record_stage_end(
                    run,
                    "data_validate",
                    result={"statuses": {item.dataset: item.status for item in validation}},
                )
                update_run_status(run, "success")
            invocation.add_attempt(run.run_id, "success", None)
            invocation.finish("success")
            return SchedulerResult(
                "success",
                0,
                invocation.invocation_id,
                invocation.path,
                as_of,
                run.run_id,
            )
        except Exception as error:  # noqa: BLE001 - scheduler must preserve terminal audit.
            message = f"{type(error).__name__}: {error}"
            if run is not None:
                try:
                    record_failure(run, error)
                except ValueError:
                    pass
                invocation.add_attempt(run.run_id, "failed", message)
            else:
                invocation.add_attempt(None, "failed", message)
            invocation.finish("failed", error_message=message)
            return SchedulerResult(
                "failed",
                2,
                invocation.invocation_id,
                invocation.path,
                self._resolved(invocation),
                run.run_id if run is not None else None,
                error_message=message,
            )

    @staticmethod
    def _resolved(invocation: SchedulerInvocation) -> str | None:
        value = invocation.payload.get("resolved_as_of")
        return value if isinstance(value, str) else None


def resolve_automatic_production_date(
    store: ParquetDataStore,
    settings: AppSettings,
    *,
    now: datetime,
) -> ScheduleDecision:
    """Resolve only today's ready trading session; never fall back to an older date."""

    if not settings.production.scheduler.enabled:
        return ScheduleDecision("skipped", None, "scheduler_disabled")
    timezone = ZoneInfo(settings.production.timezone)
    local_now = now.astimezone(timezone)
    today = local_now.strftime("%Y%m%d")
    open_dates = _open_trade_dates(store)
    if today not in open_dates:
        return ScheduleDecision("skipped", None, "non_trading_day")
    ready_time = time.fromisoformat(settings.production.market_data_ready_time)
    if local_now.time().replace(tzinfo=None) < ready_time:
        return ScheduleDecision("skipped", today, "market_data_not_ready")
    return ScheduleDecision("run", today)


def latest_completed_trading_date(
    store: ParquetDataStore,
    timezone: str,
    *,
    now: datetime,
) -> str:
    """Return the latest trade_cal session completed by the current local time."""

    local_now = now.astimezone(ZoneInfo(timezone))
    cutoff_date = local_now.date()
    if local_now.time().replace(tzinfo=None) < time(15, 0):
        cutoff_date -= timedelta(days=1)
    cutoff = cutoff_date.strftime("%Y%m%d")
    dates = sorted(date for date in _open_trade_dates(store) if date <= cutoff)
    if not dates:
        raise DataValidationError("trade_cal contains no completed trading session")
    return dates[-1]


def _open_trade_dates(store: ParquetDataStore) -> set[str]:
    calendar = store.read_dataset(get_dataset_spec("trade_cal"))
    required = {"cal_date", "is_open"}
    if calendar.empty or not required.issubset(calendar.columns):
        raise DataValidationError("trade_cal with cal_date and is_open is required")
    return set(calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"].astype(str))


def _has_valid_existing_publication(
    reports_root: Path,
    runs_root: Path,
    as_of: str,
) -> bool:
    try:
        validate_production_publication(
            reports_root=reports_root,
            runs_root=runs_root,
            as_of=as_of,
        )
    except DataValidationError:
        return False
    return True


def _is_transient_error(message: str | None) -> bool:
    if not message:
        return False
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "timeout",
            "timed out",
            "connection error",
            "connecterror",
            "connection reset",
            "temporarily unavailable",
            "rate limit",
            "too many requests",
            "http 429",
            "service unavailable",
        )
    )


def read_scheduler_invocation(path: Path) -> dict[str, Any]:
    """Read one scheduler invocation for operator tooling and tests."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"scheduler invocation must be a JSON object: {path}")
    return payload
