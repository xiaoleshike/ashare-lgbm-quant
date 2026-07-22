"""Single-lock daily production orchestration over existing service APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import DEFAULT_DATASETS
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.ingestion import DataIngestionService
from ashare_quant.data.quality_logging import append_validation_results
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.data.validation import DataValidator
from ashare_quant.features import FeatureBuilder, FeatureStore, FeatureValidator
from ashare_quant.models.inference import InferenceResult, ProductionInferenceEngine
from ashare_quant.models.production_observation import (
    ProductionObservationRecorder,
    ProductionObservationResult,
)
from ashare_quant.orchestration.daily import (
    AsOfResolver,
    DailyPipelineContext,
    DailyPipelineStages,
    ReadinessExecutor,
    StageResult,
    daily_pipeline_stages,
    load_upstream_manifests,
)
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
)
from ashare_quant.research.daily_report import DailyReportResult, DailyResearchReportGenerator
from ashare_quant.research.decision_support import (
    DecisionSupportResult,
    InvestmentDecisionSupport,
)
from ashare_quant.research.explainability.engine import ExplainabilityEngine
from ashare_quant.research.explainability.schemas import ExplainabilityResult
from ashare_quant.strategy.candidate_selector import CandidateSelectionResult, CandidateSelector
from ashare_quant.universe import UniverseBuilder, UniverseStore, UniverseValidator
from ashare_quant.utils.manifest import (
    atomic_write_json,
    parquet_artifact_statistics,
    processed_source_fingerprint,
    raw_source_fingerprints,
    utc_now_iso,
    write_build_manifest,
)

PRODUCTION_STAGE_NAMES = (
    *(stage.name for stage in daily_pipeline_stages("20000101")),
    "model_predict",
    "strategy_candidates",
    "research_report",
    "research_explain",
    "research_decision",
    "production_observation",
    "publish_production_summary",
)
DRY_RUN_STAGE_NAMES = ("dry_run_validation", "manifest_planning")


class ReadinessService(Protocol):
    """Read-only readiness API required by dry-run mode."""

    def check_all(self, as_of: str) -> tuple[GateResult, ...]: ...


@dataclass(frozen=True, slots=True)
class ProductionPipelineResult:
    """Terminal result for one production pipeline attempt."""

    run: ProductionRun
    status: str
    exit_code: int
    as_of: str
    failed_stage: str | None = None
    error_message: str | None = None
    summary_path: Path | None = None


class ProductionDailyStageExecutor:
    """Invoke existing daily services directly without entering the CLI."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        raw_store: ParquetDataStore,
        universe_store: UniverseStore,
        feature_store: FeatureStore,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.raw_store = raw_store
        self.universe_store = universe_store
        self.feature_store = feature_store

    def __call__(self, arguments: tuple[str, ...]) -> StageResult:
        """Dispatch one known daily stage to its existing Python service."""

        command = arguments[:2]
        if command == ("data", "update"):
            return self._data_update(arguments)
        if command == ("data", "validate"):
            return self._data_validate()
        if command == ("universe", "build"):
            return self._universe_build(_argument(arguments, "--start-date"))
        if command == ("universe", "validate"):
            return self._universe_validate(_argument(arguments, "--start-date"))
        if command == ("features", "build"):
            return self._features_build(_argument(arguments, "--start-date"))
        if command == ("features", "validate"):
            return self._features_validate(_argument(arguments, "--start-date"))
        raise ValueError(f"unsupported direct daily stage: {' '.join(arguments)}")

    def _data_update(self, arguments: tuple[str, ...]) -> StageResult:
        end_date = _optional_argument(arguments, "--end-date")
        downloads = DataIngestionService(self.settings, self.raw_store).update(
            DEFAULT_DATASETS,
            end_date,
            refresh_snapshots=False,
            repair_gaps="--repair-gaps" in arguments,
        )
        validation = DataValidator(self.raw_store).validate_all(DEFAULT_DATASETS)
        append_validation_results(self.settings.paths.data_quality_logs, validation)
        errors = [error for result in validation if not result.ok for error in result.errors]
        warnings = tuple(warning for result in validation for warning in result.warnings)
        return StageResult(
            status="failed" if errors else "success",
            metrics={
                "datasets": [
                    {
                        "dataset": item.dataset,
                        "rows_written": item.rows_written,
                        "skipped": item.skipped,
                        "message": item.message,
                    }
                    for item in downloads
                ],
                "validation_statuses": {item.dataset: item.status for item in validation},
            },
            warnings=warnings,
            error_message=errors[0] if errors else None,
        )

    def _data_validate(self) -> StageResult:
        validation = DataValidator(self.raw_store).validate_all(DEFAULT_DATASETS)
        errors = [error for result in validation if not result.ok for error in result.errors]
        return StageResult(
            status="failed" if errors else "success",
            metrics={"statuses": {item.dataset: item.status for item in validation}},
            warnings=tuple(warning for item in validation for warning in item.warnings),
            error_message=errors[0] if errors else None,
        )

    def _universe_build(self, as_of: str) -> StageResult:
        started = utc_now_iso()
        result = UniverseBuilder(self.raw_store, self.universe_store, self.settings).build(
            as_of, as_of
        )
        if not result.validation.ok:
            return StageResult(
                "failed",
                warnings=result.validation.warnings,
                error_message=result.validation.errors[0],
            )
        statistics = parquet_artifact_statistics(self.universe_store.dataset_dir)
        manifest = write_build_manifest(
            self.universe_store.dataset_dir,
            artifact_name="universe_daily",
            build_started_at=started,
            config_path=self.config_path,
            start_date=as_of,
            end_date=as_of,
            row_count=result.rows_written,
            canonical_statistics=statistics,
            partitions_changed=result.partitions_changed,
            source_fingerprints=raw_source_fingerprints(
                self.raw_store,
                (
                    "stock_basic",
                    "trade_cal",
                    "daily",
                    "daily_basic",
                    "suspend_d",
                    "stk_limit",
                    "namechange",
                ),
            ),
        )
        return StageResult(
            "success",
            artifact_paths=(str(self.universe_store.dataset_dir),),
            metrics={
                "rows_built": result.rows_built,
                "rows_written": result.rows_written,
                "partitions_changed": result.partitions_changed,
                "artifact_manifest": manifest,
            },
            warnings=result.validation.warnings,
        )

    def _universe_validate(self, as_of: str) -> StageResult:
        result = UniverseValidator(self.universe_store).validate(as_of, as_of)
        return StageResult(
            "success" if result.ok else "failed",
            metrics={"validated_date": as_of},
            warnings=result.warnings,
            error_message=result.errors[0] if result.errors else None,
        )

    def _features_build(self, as_of: str) -> StageResult:
        started = utc_now_iso()
        result = FeatureBuilder(
            self.raw_store,
            self.universe_store,
            self.feature_store,
            self.settings,
        ).build(as_of, as_of)
        universe_statistics = parquet_artifact_statistics(self.universe_store.dataset_dir)
        sources = raw_source_fingerprints(
            self.raw_store,
            (
                "trade_cal",
                "daily",
                "adj_factor",
                "daily_basic",
                "index_daily",
                "fina_indicator",
                "income",
                "balancesheet",
                "cashflow",
            ),
        )
        sources["universe_daily"] = processed_source_fingerprint(
            self.universe_store.dataset_dir,
            rows=universe_statistics.row_count,
            partitions=universe_statistics.partition_count,
            min_date=universe_statistics.min_date,
            max_date=universe_statistics.max_date,
        )
        statistics = parquet_artifact_statistics(self.feature_store.dataset_dir)
        feature_count = len(set(statistics.column_names) - {"trade_date", "ts_code"})
        manifest = write_build_manifest(
            self.feature_store.dataset_dir,
            artifact_name="features_daily",
            build_started_at=started,
            config_path=self.config_path,
            start_date=as_of,
            end_date=as_of,
            row_count=result.rows_written,
            canonical_statistics=statistics,
            partitions_changed=result.partitions_changed,
            source_fingerprints=sources,
            extra={"feature_count": feature_count},
        )
        return StageResult(
            "success",
            artifact_paths=(str(self.feature_store.dataset_dir),),
            metrics={
                "rows_built": result.rows_built,
                "rows_written": result.rows_written,
                "feature_count": result.feature_count,
                "elapsed_seconds": result.elapsed_seconds,
                "artifact_manifest": manifest,
            },
        )

    def _features_validate(self, as_of: str) -> StageResult:
        result = FeatureValidator(self.feature_store).validate(as_of, as_of)
        return StageResult(
            "success" if result.ok else "failed",
            metrics={"rows": result.rows, "validated_date": as_of},
            warnings=result.warnings,
            error_message=result.errors[0] if result.errors else None,
        )


class ProductionPipeline:
    """Run the complete read-only-after-build production research workflow."""

    def __init__(
        self,
        *,
        config_path: Path,
        processed_root: Path,
        reports_root: Path,
        daily_executor: ProductionDailyStageExecutor,
        daily_stages: DailyPipelineStages,
        readiness: ReadinessService,
        readiness_executor: ReadinessExecutor,
        as_of_resolver: AsOfResolver,
        inference: ProductionInferenceEngine,
        candidates: CandidateSelector,
        research_report: DailyResearchReportGenerator,
        explainability: ExplainabilityEngine,
        decision_support: InvestmentDecisionSupport,
        observation: ProductionObservationRecorder,
        runs_root: Path = DEFAULT_RUNS_ROOT,
        lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    ) -> None:
        self.config_path = config_path
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.daily_executor = daily_executor
        self.daily_stages = daily_stages
        self.readiness = readiness
        self.readiness_executor = readiness_executor
        self.as_of_resolver = as_of_resolver
        self.inference = inference
        self.candidates = candidates
        self.research_report = research_report
        self.explainability = explainability
        self.decision_support = decision_support
        self.observation = observation
        self.runs_root = runs_root
        self.lock_path = lock_path

    def run(self, as_of: str, *, dry_run: bool = False) -> ProductionPipelineResult:
        """Execute one production date under a single lock and manifest."""

        command = f"ashare-quant pipeline production --as-of {as_of}"
        if dry_run:
            command += " --dry-run"
        with production_lock(self.lock_path, command=command):
            run = create_run(
                command,
                config_path=self.config_path,
                runs_root=self.runs_root,
                stages=DRY_RUN_STAGE_NAMES if dry_run else PRODUCTION_STAGE_NAMES,
                upstream_manifests=load_upstream_manifests(self.processed_root),
                pipeline_type="production_daily",
                as_of=as_of,
            )
            try:
                resolved_as_of = self.as_of_resolver(as_of)
            except Exception as error:  # noqa: BLE001 - persist invalid-session failure.
                record_failure(run, error)
                return ProductionPipelineResult(
                    run, "failed", 2, as_of, error_message=_exception_message(error)
                )
            if dry_run:
                return self._run_dry(run, resolved_as_of)

            daily = self.daily_stages.execute(
                DailyPipelineContext(
                    run=run,
                    as_of=resolved_as_of,
                    processed_root=self.processed_root,
                    executor=self.daily_executor,
                    readiness_executor=self.readiness_executor,
                    as_of_resolver=self.as_of_resolver,
                )
            )
            if daily.status == "failed":
                return ProductionPipelineResult(
                    run,
                    "failed",
                    daily.exit_code,
                    resolved_as_of,
                    daily.failed_stage,
                    daily.error_message,
                )

            state: dict[str, Any] = {"artifacts": [], "warnings": []}
            stages = (
                ("model_predict", lambda: self._predict(resolved_as_of, state)),
                ("strategy_candidates", lambda: self._candidates(resolved_as_of, state)),
                ("research_report", lambda: self._report(resolved_as_of)),
                ("research_explain", lambda: self._explain(resolved_as_of)),
                ("research_decision", lambda: self._decision(resolved_as_of)),
                ("production_observation", lambda: self._observation(resolved_as_of, state)),
                (
                    "publish_production_summary",
                    lambda: self._publish_summary(run, resolved_as_of, state),
                ),
            )
            for stage_name, operation in stages:
                failure = self._execute_stage(run, stage_name, operation, state)
                if failure is not None:
                    return ProductionPipelineResult(
                        run,
                        "failed",
                        2,
                        resolved_as_of,
                        stage_name,
                        failure,
                    )
            update_run_status(run, "success")
            return ProductionPipelineResult(
                run,
                "success",
                0,
                resolved_as_of,
                summary_path=self.reports_root / resolved_as_of / "production_summary.json",
            )

    def _run_dry(self, run: ProductionRun, as_of: str) -> ProductionPipelineResult:
        def validate() -> StageResult:
            gates = self.readiness.check_all(as_of)
            failures = [failure for gate in gates for failure in gate.hard_failures]
            warnings = tuple(warning for gate in gates for warning in gate.warnings)
            return StageResult(
                "failed" if failures else "success",
                metrics={"gates": [gate.to_dict() for gate in gates]},
                warnings=warnings,
                error_message=failures[0] if failures else None,
            )

        expected = self._expected_artifacts(as_of)
        operations = (
            ("dry_run_validation", validate),
            (
                "manifest_planning",
                lambda: StageResult(
                    "success",
                    metrics={
                        "planned_stages": list(PRODUCTION_STAGE_NAMES),
                        "planned_artifact_paths": list(expected),
                    },
                ),
            ),
        )
        state: dict[str, Any] = {"artifacts": [], "warnings": []}
        for stage_name, operation in operations:
            failure = self._execute_stage(run, stage_name, operation, state, verify_artifacts=False)
            if failure is not None:
                return ProductionPipelineResult(run, "failed", 2, as_of, stage_name, failure)
        update_run_status(run, "success")
        return ProductionPipelineResult(run, "success", 0, as_of)

    def _execute_stage(
        self,
        run: ProductionRun,
        stage_name: str,
        operation: Callable[[], StageResult],
        state: dict[str, Any],
        *,
        verify_artifacts: bool = True,
    ) -> str | None:
        record_stage_start(run, stage_name)
        try:
            result: StageResult = operation()
            if result.status == "failed":
                raise DataValidationError(result.error_message or f"{stage_name} failed")
            if verify_artifacts:
                _require_artifacts(result.artifact_paths)
            state["artifacts"].extend(result.artifact_paths)
            state["warnings"].extend(result.warnings)
            update_run_context(
                run,
                model_id=state.get("model_id"),
                artifact_paths=result.artifact_paths,
                warnings=result.warnings,
            )
            record_stage_end(run, stage_name, result=result.to_dict())
            return None
        except Exception as error:  # noqa: BLE001 - hard production stage boundary.
            record_failure(run, error, stage_name=stage_name)
            return _exception_message(error)

    def _predict(self, as_of: str, state: dict[str, Any]) -> StageResult:
        result: InferenceResult = self.inference.predict(as_of)
        state["model_id"] = result.model_id
        paths = tuple(
            str(result.output_dir / name)
            for name in ("predictions.parquet", "ranking.csv", "summary.json", "manifest.json")
        )
        return StageResult(
            "success",
            paths,
            {"prediction_count": result.prediction_count, "model_id": result.model_id},
        )

    def _candidates(self, as_of: str, state: dict[str, Any]) -> StageResult:
        result: CandidateSelectionResult = self.candidates.select(as_of)
        state["candidate_result"] = result
        report_dir = result.output_path.parent
        paths = (str(result.output_path), str(report_dir / "candidates_manifest.json"))
        return StageResult(
            "success",
            paths,
            {
                "candidate_count": result.candidate_count,
                "filtered_counts": result.filtered_counts,
                "model_id": result.model_id,
            },
        )

    def _report(self, as_of: str) -> StageResult:
        result: DailyReportResult = self.research_report.generate(as_of)
        return StageResult(
            "success",
            (str(result.report_path), str(result.summary_path)),
            {"candidate_count": result.candidate_count, "model_id": result.model_id},
            result.warnings,
        )

    def _explain(self, as_of: str) -> StageResult:
        result: ExplainabilityResult = self.explainability.explain(as_of)
        return StageResult(
            "success",
            (result.json_path, result.markdown_path),
            {
                "candidate_count": result.candidate_count,
                "model_id": result.model_id,
                "method": result.method,
            },
        )

    def _decision(self, as_of: str) -> StageResult:
        result: DecisionSupportResult = self.decision_support.generate(as_of)
        return StageResult(
            "success",
            (str(result.json_path), str(result.markdown_path)),
            {"candidate_count": result.candidate_count, "model_id": result.model_id},
        )

    def _observation(self, as_of: str, state: dict[str, Any]) -> StageResult:
        result: ProductionObservationResult = self.observation.record(as_of)
        state["observation_path"] = str(result.output_path)
        return StageResult(
            "success",
            (str(result.output_path),),
            {"candidate_count": result.candidate_count, "model_id": result.model_id},
        )

    def _publish_summary(
        self, run: ProductionRun, as_of: str, state: dict[str, Any]
    ) -> StageResult:
        candidate_result = state.get("candidate_result")
        if not isinstance(candidate_result, CandidateSelectionResult):
            raise DataValidationError("candidate result is unavailable for production summary")
        candidates = candidate_result.candidates.sort_values(
            ["rank", "ts_code"], kind="mergesort"
        ).head(20)
        summary_path = self.reports_root / as_of / "production_summary.json"
        payload = {
            "schema_version": 1,
            "artifact_name": "production_daily_summary",
            "run_id": run.run_id,
            "as_of": as_of,
            "model_id": candidate_result.model_id,
            "candidate_count": candidate_result.candidate_count,
            "top_candidates": candidates.loc[:, ["rank", "ts_code", "prediction_score"]].to_dict(
                "records"
            ),
            "artifacts": list(dict.fromkeys(state["artifacts"])),
            "observation_log_path": state.get("observation_path"),
            "completed_time": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(summary_path, payload)
        return StageResult("success", (str(summary_path),), {"candidate_count": len(candidates)})

    def _expected_artifacts(self, as_of: str) -> tuple[str, ...]:
        report_dir = self.reports_root / as_of
        return tuple(
            str(report_dir / name)
            for name in (
                "predictions.parquet",
                "ranking.csv",
                "candidates.csv",
                "daily_report.md",
                "research_summary.json",
                "explanations.json",
                "explanations.md",
                "decision.json",
                "decision_report.md",
                "production_summary.json",
            )
        )


def _argument(arguments: tuple[str, ...], name: str) -> str:
    value = _optional_argument(arguments, name)
    if value is None:
        raise ValueError(f"daily stage requires {name}")
    return value


def _optional_argument(arguments: tuple[str, ...], name: str) -> str | None:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError):
        return None


def _require_artifacts(paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise DataValidationError(f"stage did not publish required artifacts: {missing}")


def _exception_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"
