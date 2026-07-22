"""Locked daily production orchestration without inference or trading stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.orchestration.lock import (
    DEFAULT_PRODUCTION_LOCK_PATH,
    production_lock,
)
from ashare_quant.orchestration.run_manifest import (
    DEFAULT_RUNS_ROOT,
    ProductionRun,
    create_run,
    record_failure,
    record_stage_end,
    record_stage_start,
    update_run_context,
    update_run_status,
    update_source_provenance,
)
from ashare_quant.utils.manifest import read_manifest

SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
MARKET_CLOSE_TIME = time(15, 0)

type StageExecutor = Callable[[tuple[str, ...]], int | StageResult]
type AsOfResolver = Callable[[str | None], str]
type ReadinessExecutor = Callable[[str, str], GateResult]


@dataclass(frozen=True, slots=True)
class DailyPipelineStage:
    """One ordered hard-gate stage and its existing CLI arguments."""

    name: str
    arguments: tuple[str, ...]
    artifact_name: str | None = None
    readiness_gate: str | None = None


@dataclass(frozen=True, slots=True)
class DailyPipelineResult:
    """Terminal outcome for one daily orchestration attempt."""

    run: ProductionRun
    status: str
    exit_code: int
    as_of: str | None
    failed_stage: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """Normalized result returned by every orchestration stage."""

    status: Literal["success", "failed"]
    artifact_paths: tuple[str, ...] = ()
    metrics: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable run-manifest representation."""

        payload = {
            "status": self.status,
            "artifact_paths": list(self.artifact_paths),
            "metrics": self.metrics or {},
            "warnings": list(self.warnings),
        }
        # Keep existing structured gate/build fields readable by older run tooling.
        for key, value in (self.metrics or {}).items():
            payload.setdefault(key, value)
        return payload


@dataclass(slots=True)
class DailyPipelineContext:
    """Execution dependencies for reusable daily stages."""

    run: ProductionRun
    as_of: str | None
    processed_root: Path
    executor: StageExecutor
    readiness_executor: ReadinessExecutor
    as_of_resolver: AsOfResolver


@dataclass(frozen=True, slots=True)
class DailyStagesResult:
    """Outcome of the lock-free, manifest-reusing daily stage sequence."""

    status: Literal["success", "failed"]
    as_of: str | None
    exit_code: int
    failed_stage: str | None = None
    error_message: str | None = None


class DailyPipelineStages:
    """Execute daily stages using a caller-owned lock and run manifest."""

    def execute(self, context: DailyPipelineContext) -> DailyStagesResult:
        """Run the daily hard stages without acquiring a lock or creating a run."""

        as_of = context.as_of
        stages = daily_pipeline_stages(as_of)
        for stage_index in range(len(stages)):
            if stage_index == 1 and as_of is None:
                try:
                    as_of = context.as_of_resolver(None)
                except Exception as error:  # noqa: BLE001 - stage boundary records all failures.
                    message = _exception_message(error)
                    record_failure(context.run, error)
                    return DailyStagesResult("failed", None, 2, error_message=message)
                stages = daily_pipeline_stages(as_of)
            stage = stages[stage_index]
            record_stage_start(context.run, stage.name)
            try:
                result, exit_code = _execute_daily_stage(stage, as_of, context)
            except Exception as error:  # noqa: BLE001 - stage boundary records all failures.
                message = _exception_message(error)
                record_failure(context.run, error, stage_name=stage.name)
                return DailyStagesResult(
                    "failed",
                    as_of,
                    2,
                    failed_stage=stage.name,
                    error_message=message,
                )
            result_payload = result.to_dict()
            if exit_code != 0 or result.status == "failed":
                message = result.error_message or (
                    f"stage {stage.name} returned exit code {exit_code}"
                )
                record_stage_end(
                    context.run,
                    stage.name,
                    status="failed",
                    error_message=message,
                    result=result_payload,
                )
                return DailyStagesResult(
                    "failed",
                    as_of,
                    exit_code or 1,
                    failed_stage=stage.name,
                    error_message=message,
                )
            try:
                if stage.artifact_name is not None:
                    artifact_manifest = (result.metrics or {}).get("artifact_manifest")
                    if isinstance(artifact_manifest, dict):
                        update_source_provenance(
                            context.run, stage.artifact_name, artifact_manifest
                        )
                update_run_context(
                    context.run,
                    artifact_paths=result.artifact_paths,
                    warnings=result.warnings,
                )
                record_stage_end(context.run, stage.name, result=result_payload)
            except Exception as error:  # noqa: BLE001 - provenance is a hard stage boundary.
                message = _exception_message(error)
                record_failure(context.run, error, stage_name=stage.name)
                return DailyStagesResult(
                    "failed",
                    as_of,
                    2,
                    failed_stage=stage.name,
                    error_message=message,
                )
        return DailyStagesResult("success", as_of, 0)


class DailyPipelineOrchestrator:
    """Run existing data, universe, and feature commands under one production lock."""

    def __init__(
        self,
        *,
        executor: StageExecutor,
        as_of_resolver: AsOfResolver,
        config_path: Path,
        processed_root: Path,
        readiness_executor: ReadinessExecutor,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    ) -> None:
        self._executor = executor
        self._as_of_resolver = as_of_resolver
        self._config_path = config_path
        self._processed_root = processed_root
        self._readiness_executor = readiness_executor
        self._runs_root = runs_root
        self._lock_path = lock_path
        self._stages = DailyPipelineStages()

    def run(self, requested_as_of: str | None = None) -> DailyPipelineResult:
        """Execute all hard stages in order and stop at the first failure."""

        command = "ashare-quant pipeline daily"
        if requested_as_of is not None:
            command = f"{command} --as-of {requested_as_of}"
        with production_lock(self._lock_path, command=command):
            run = create_run(
                command,
                config_path=self._config_path,
                runs_root=self._runs_root,
                stages=tuple(stage.name for stage in daily_pipeline_stages(requested_as_of)),
                upstream_manifests=load_upstream_manifests(self._processed_root),
            )
            as_of: str | None = None
            if requested_as_of is not None:
                try:
                    as_of = self._as_of_resolver(requested_as_of)
                except Exception as error:  # noqa: BLE001 - record resolver failure.
                    message = _exception_message(error)
                    record_failure(run, error)
                    return DailyPipelineResult(run, "failed", 2, None, error_message=message)

            stages_result = self._stages.execute(
                DailyPipelineContext(
                    run=run,
                    as_of=as_of,
                    processed_root=self._processed_root,
                    executor=self._executor,
                    readiness_executor=self._readiness_executor,
                    as_of_resolver=self._as_of_resolver,
                )
            )
            if stages_result.status == "failed":
                return DailyPipelineResult(
                    run,
                    "failed",
                    stages_result.exit_code,
                    stages_result.as_of,
                    failed_stage=stages_result.failed_stage,
                    error_message=stages_result.error_message,
                )

            update_run_status(run, "success")
            return DailyPipelineResult(run, "success", 0, stages_result.as_of)


def _execute_daily_stage(
    stage: DailyPipelineStage,
    as_of: str | None,
    context: DailyPipelineContext,
) -> tuple[StageResult, int]:
    if stage.readiness_gate is not None:
        gate = context.readiness_executor(stage.readiness_gate, as_of or "")
        metrics = gate.to_dict()
        error_message = gate.hard_failures[0] if gate.hard_failures else None
        result = StageResult(
            status="success" if gate.ready else "failed",
            metrics=metrics,
            warnings=gate.warnings,
            error_message=error_message,
        )
        return result, 0 if gate.ready else 1

    raw_result = context.executor(stage.arguments)
    if isinstance(raw_result, StageResult):
        return raw_result, 0 if raw_result.status == "success" else 1
    metrics = stage_result(stage, raw_result, as_of or "auto", context.processed_root)
    artifacts = (
        (str(context.processed_root / stage.artifact_name),)
        if stage.artifact_name is not None
        else ()
    )
    return (
        StageResult(
            status="success" if raw_result == 0 else "failed",
            artifact_paths=artifacts,
            metrics=metrics,
        ),
        raw_result,
    )


def daily_pipeline_stages(as_of: str | None) -> tuple[DailyPipelineStage, ...]:
    """Return the fixed stage order, substituting the resolved as-of date when known."""

    date = as_of or "<pending>"
    data_update_arguments: tuple[str, ...] = ("data", "update", "--repair-gaps")
    if as_of is not None:
        data_update_arguments += ("--end-date", as_of)
    return (
        DailyPipelineStage("data_update", data_update_arguments),
        DailyPipelineStage("data_validate", ("data", "validate")),
        DailyPipelineStage("raw_freshness_gate", (), readiness_gate="raw_freshness_gate"),
        DailyPipelineStage(
            "universe_build",
            ("universe", "build", "--start-date", date, "--end-date", date),
            artifact_name="universe_daily",
        ),
        DailyPipelineStage(
            "universe_validate",
            ("universe", "validate", "--start-date", date, "--end-date", date),
        ),
        DailyPipelineStage("universe_readiness_gate", (), readiness_gate="universe_readiness_gate"),
        DailyPipelineStage(
            "features_build",
            ("features", "build", "--start-date", date, "--end-date", date),
            artifact_name="features_daily",
        ),
        DailyPipelineStage(
            "features_validate",
            ("features", "validate", "--start-date", date, "--end-date", date),
        ),
        DailyPipelineStage("features_readiness_gate", (), readiness_gate="features_readiness_gate"),
    )


def resolve_completed_trading_date(
    store: ParquetDataStore,
    requested_as_of: str | None,
    *,
    now: datetime | None = None,
) -> str:
    """Resolve an explicit or latest completed open date from authoritative ``trade_cal``."""

    calendar = store.read_dataset(get_dataset_spec("trade_cal"))
    required = {"cal_date", "is_open"}
    if calendar.empty or not required.issubset(calendar.columns):
        raise DataValidationError("trade_cal with cal_date and is_open is required")
    local_now = (now or datetime.now(UTC)).astimezone(SHANGHAI_TIMEZONE)
    today = local_now.strftime("%Y%m%d")
    completed_cutoff = today if local_now.time() >= MARKET_CLOSE_TIME else _previous_date(today)
    open_dates: list[str] = sorted(
        str(value)
        for value in calendar.loc[calendar["is_open"].astype(int).eq(1), "cal_date"]
        .dropna()
        .unique()
    )
    completed_dates = [date for date in open_dates if date <= completed_cutoff]
    if not completed_dates:
        raise DataValidationError("trade_cal contains no completed open trading date")
    if requested_as_of is None:
        return completed_dates[-1]
    _validate_date(requested_as_of)
    if requested_as_of > completed_cutoff:
        raise DataValidationError(f"requested as-of date is not completed yet: {requested_as_of}")
    if requested_as_of not in set(open_dates):
        raise DataValidationError(
            f"requested as-of date is not an open trading day: {requested_as_of}"
        )
    return requested_as_of


def load_upstream_manifests(processed_root: Path) -> dict[str, dict[str, Any]]:
    """Load pre-run processed artifact manifests when they are available."""

    manifests: dict[str, dict[str, Any]] = {}
    for artifact_name in ("universe_daily", "features_daily", "labels_forward"):
        manifest = read_manifest(processed_root / artifact_name)
        if manifest is not None:
            manifests[artifact_name] = manifest
    return manifests


def stage_result(
    stage: DailyPipelineStage,
    exit_code: int,
    as_of: str,
    processed_root: Path,
) -> dict[str, Any]:
    """Return compact command and artifact evidence for one completed stage."""

    result: dict[str, Any] = {
        "command": "ashare-quant " + " ".join(stage.arguments),
        "exit_code": exit_code,
        "as_of": as_of,
    }
    if stage.artifact_name is not None:
        artifact_manifest = read_manifest(processed_root / stage.artifact_name)
        result["artifact_manifest"] = artifact_manifest
        if artifact_manifest is not None:
            result["incremental_build"] = artifact_manifest.get("build_scope")
            result["canonical_artifact"] = artifact_manifest.get("canonical_artifact")
    return result


def _validate_date(value: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise DataValidationError(f"invalid YYYYMMDD date: {value}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise DataValidationError(f"invalid YYYYMMDD date: {value}")


def _previous_date(value: str) -> str:
    parsed = datetime.strptime(value, "%Y%m%d")
    return (parsed - timedelta(days=1)).strftime("%Y%m%d")


def _exception_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
