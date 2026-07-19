"""Locked daily production orchestration without inference or trading stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
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
    update_run_status,
    update_source_provenance,
)
from ashare_quant.utils.manifest import read_manifest

SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")
MARKET_CLOSE_TIME = time(15, 0)

type StageExecutor = Callable[[tuple[str, ...]], int]
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

            for stage_index in range(len(daily_pipeline_stages(as_of))):
                if stage_index == 1 and as_of is None:
                    try:
                        as_of = self._as_of_resolver(None)
                    except Exception as error:  # noqa: BLE001 - record resolver failure.
                        message = _exception_message(error)
                        record_failure(run, error)
                        return DailyPipelineResult(run, "failed", 2, None, error_message=message)
                stage = daily_pipeline_stages(as_of)[stage_index]
                record_stage_start(run, stage.name)
                try:
                    if stage.readiness_gate is not None:
                        gate_result = self._readiness_executor(stage.readiness_gate, as_of or "")
                        exit_code = 0 if gate_result.ready else 1
                        result = gate_result.to_dict()
                    else:
                        exit_code = self._executor(stage.arguments)
                        result = stage_result(
                            stage,
                            exit_code,
                            as_of or "auto",
                            self._processed_root,
                        )
                except Exception as error:  # noqa: BLE001 - stage boundary records all failures.
                    message = _exception_message(error)
                    record_failure(run, error, stage_name=stage.name)
                    return DailyPipelineResult(
                        run,
                        "failed",
                        2,
                        as_of,
                        failed_stage=stage.name,
                        error_message=message,
                    )
                if exit_code != 0:
                    failures = result.get("hard_failures")
                    detail = failures[0] if isinstance(failures, list) and failures else None
                    message = detail or f"stage {stage.name} returned exit code {exit_code}"
                    record_stage_end(
                        run,
                        stage.name,
                        status="failed",
                        error_message=message,
                        result=result,
                    )
                    return DailyPipelineResult(
                        run,
                        "failed",
                        exit_code,
                        as_of,
                        failed_stage=stage.name,
                        error_message=message,
                    )
                if stage.artifact_name is not None:
                    try:
                        artifact_manifest = result.get("artifact_manifest")
                        if isinstance(artifact_manifest, dict):
                            update_source_provenance(run, stage.artifact_name, artifact_manifest)
                        record_stage_end(run, stage.name, result=result)
                    except Exception as error:  # noqa: BLE001 - provenance is a hard gate.
                        message = _exception_message(error)
                        record_failure(run, error, stage_name=stage.name)
                        return DailyPipelineResult(
                            run,
                            "failed",
                            2,
                            as_of,
                            failed_stage=stage.name,
                            error_message=message,
                        )
                else:
                    record_stage_end(run, stage.name, result=result)

            update_run_status(run, "success")
            return DailyPipelineResult(run, "success", 0, as_of)


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
