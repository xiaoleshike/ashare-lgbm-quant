"""Command-line entry points for pipeline operations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ashare_quant.backtest import (
    BacktestDiagnosticEngine,
    BacktestRunner,
    HistoricalBacktestEngine,
)
from ashare_quant.config import load_settings
from ashare_quant.data.datasets import ALL_DATASETS, DEFAULT_DATASETS
from ashare_quant.data.exceptions import DataIngestionError, DataValidationError
from ashare_quant.data.ingestion import DataIngestionService, GapReport, build_store
from ashare_quant.data.quality_logging import append_quality_event, append_validation_results
from ashare_quant.data.validation import DataValidator, ValidationResult
from ashare_quant.diagnostics import FeatureDiagnosticPipeline
from ashare_quant.diagnostics.pipeline import ChronologicalSplit
from ashare_quant.features import (
    FEATURE_REGISTRY,
    FeatureBuilder,
    FeatureStore,
    FeatureValidationResult,
    FeatureValidator,
)
from ashare_quant.labels import LabelBuilder, LabelStore, LabelValidator
from ashare_quant.labels.validation import LabelValidationResult
from ashare_quant.models import (
    ModelDriftDiagnosticEngine,
    ModelRegistry,
    MultiHorizonExperimentPlanner,
    ProductionInferenceEngine,
    ProductionObservationRecorder,
    ProductionRankerTrainer,
    PurgedWalkForwardPlanner,
    RankerBaselineRunner,
)
from ashare_quant.orchestration import (
    DEFAULT_PRODUCTION_LOCK_PATH,
    DailyPipelineOrchestrator,
    FreshnessService,
    GateResult,
    ProductionLockError,
    resolve_completed_trading_date,
    run_with_production_lock,
)
from ashare_quant.research import (
    DailyResearchReportGenerator,
    ExplainabilityEngine,
    InvestmentDecisionSupport,
)
from ashare_quant.strategy import CandidateSelector
from ashare_quant.universe import UniverseBuilder, UniverseStore, UniverseValidator
from ashare_quant.universe.validation import UniverseValidationResult
from ashare_quant.utils import configure_logging
from ashare_quant.utils.manifest import (
    artifact_manifest_status,
    parquet_artifact_statistics,
    processed_source_fingerprint,
    raw_source_fingerprints,
    utc_now_iso,
    write_build_manifest,
)

LOGGER = logging.getLogger(__name__)


def run_production_cli_command(
    operation: Callable[[], int],
    *,
    lock_path: Path = DEFAULT_PRODUCTION_LOCK_PATH,
    command: str | None = None,
) -> int:
    """Run a future production pipeline CLI handler under the repository lock."""

    try:
        return run_with_production_lock(operation, lock_path=lock_path, command=command)
    except ProductionLockError as error:
        print(f"production run blocked: {error}", file=sys.stderr)
        return 3


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="ashare-quant")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate local configuration and runtime prerequisites.")
    subparsers.add_parser("config-check", help="Load and validate configuration.")

    add_data_parser(subparsers)
    add_universe_parser(subparsers)
    add_labels_parser(subparsers)
    add_features_parser(subparsers)
    add_diagnostics_parser(subparsers)
    add_models_parser(subparsers)
    add_strategy_parser(subparsers)
    add_research_parser(subparsers)
    add_backtest_parser(subparsers)
    add_pipeline_parser(subparsers)
    return parser


def add_data_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add raw data ingestion commands."""

    data_parser = subparsers.add_parser("data", help="Manage Tushare raw data ingestion.")
    data_parser.add_argument(
        "--storage-root",
        default=None,
        help="Override the configured canonical Parquet root.",
    )
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    init_parser = data_subparsers.add_parser("init", help="Run full historical data download.")
    add_dataset_args(init_parser)
    add_date_range_args(init_parser)

    update_parser = data_subparsers.add_parser("update", help="Run daily incremental update.")
    add_dataset_args(update_parser)
    update_parser.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")
    update_parser.add_argument(
        "--refresh-snapshots",
        action="store_true",
        help="Refresh existing snapshot datasets selected by --dataset/--all-datasets.",
    )
    update_parser.add_argument(
        "--repair-gaps",
        action="store_true",
        help="Repair missing trade_cal-derived trading dates before normal incremental update.",
    )

    gaps_parser = data_subparsers.add_parser(
        "gaps", help="Report missing trade_cal-derived trading dates without downloading."
    )
    add_dataset_args(gaps_parser)
    add_date_range_args(gaps_parser)

    validate_parser = data_subparsers.add_parser(
        "validate", help="Validate local Parquet datasets."
    )
    add_dataset_args(validate_parser)

    data_subparsers.add_parser("status", help="Show local dataset status.")


def add_universe_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add point-in-time universe commands."""

    universe_parser = subparsers.add_parser(
        "universe", help="Build and validate point-in-time A-share universes."
    )
    universe_parser.add_argument(
        "--storage-root",
        default=None,
        help="Override the configured canonical raw Parquet root.",
    )
    universe_parser.add_argument(
        "--output-root",
        default=None,
        help="Override the configured processed output root.",
    )
    universe_subparsers = universe_parser.add_subparsers(dest="universe_command", required=True)

    build_parser = universe_subparsers.add_parser(
        "build", help="Build daily universe rows for a date range."
    )
    build_parser.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD start date.")
    build_parser.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD end date.")

    validate_parser = universe_subparsers.add_parser(
        "validate", help="Validate stored daily universe rows."
    )
    validate_parser.add_argument(
        "--start-date", default=None, help="Inclusive YYYYMMDD start date."
    )
    validate_parser.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")

    status_parser = universe_subparsers.add_parser("status", help="Show daily universe status.")
    status_parser.add_argument("--date", default=None, help="Optional YYYYMMDD date.")


def add_labels_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add executable forward-return label commands."""

    labels_parser = subparsers.add_parser(
        "labels", help="Build and validate executable forward-return labels."
    )
    labels_parser.add_argument(
        "--storage-root",
        default=None,
        help="Override the configured canonical raw Parquet root.",
    )
    labels_parser.add_argument(
        "--processed-root",
        default=None,
        help="Override the configured processed data root.",
    )
    labels_subparsers = labels_parser.add_subparsers(dest="labels_command", required=True)

    build_parser = labels_subparsers.add_parser(
        "build", help="Build executable forward-return labels."
    )
    build_parser.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD start date.")
    build_parser.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD end date.")
    build_parser.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated trading-day horizons, for example 5,10,20,60.",
    )

    validate_parser = labels_subparsers.add_parser(
        "validate", help="Validate stored executable labels."
    )
    validate_parser.add_argument(
        "--start-date", default=None, help="Inclusive YYYYMMDD start date."
    )
    validate_parser.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")

    status_parser = labels_subparsers.add_parser("status", help="Show label status.")
    status_parser.add_argument("--date", default=None, help="Optional YYYYMMDD date.")
    status_parser.add_argument("--horizon", type=int, default=None, help="Optional horizon.")


def add_features_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add point-in-time feature engineering commands."""

    features_parser = subparsers.add_parser(
        "features", help="Build and inspect point-in-time feature matrices."
    )
    features_parser.add_argument(
        "--storage-root",
        default=None,
        help="Override the configured canonical raw Parquet root.",
    )
    features_parser.add_argument(
        "--processed-root",
        default=None,
        help="Override the configured processed data root.",
    )
    features_subparsers = features_parser.add_subparsers(dest="features_command", required=True)

    build_parser = features_subparsers.add_parser("build", help="Build daily feature rows.")
    build_parser.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD start date.")
    build_parser.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD end date.")

    status_parser = features_subparsers.add_parser("status", help="Show feature matrix status.")
    status_parser.add_argument("--date", default=None, help="Optional YYYYMMDD date.")

    validate_parser = features_subparsers.add_parser(
        "validate", help="Validate stored production feature rows."
    )
    validate_parser.add_argument(
        "--start-date", default=None, help="Inclusive YYYYMMDD start date."
    )
    validate_parser.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")

    features_subparsers.add_parser("registry", help="Show registered feature metadata summary.")


def add_diagnostics_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add leakage-controlled feature diagnostics commands."""

    parser = subparsers.add_parser(
        "diagnostics", help="Diagnose and select robust production features."
    )
    parser.add_argument(
        "--processed-root", default=None, help="Override the configured processed data root."
    )
    parser.add_argument(
        "--reports-root", default=None, help="Override the configured reports root."
    )
    commands = parser.add_subparsers(dest="diagnostics_command", required=True)
    run_parser = commands.add_parser(
        "run", help="Run train/validation diagnostics and one-time frozen test evaluation."
    )
    for period in ("train", "validation", "test"):
        run_parser.add_argument(f"--{period}-start", required=True, help="Inclusive YYYYMMDD date.")
        run_parser.add_argument(f"--{period}-end", required=True, help="Inclusive YYYYMMDD date.")
    run_parser.add_argument("--horizon", type=int, default=None, help="Configured label horizon.")
    commands.add_parser("status", help="Show the latest feature diagnostics report.")


def add_models_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add controlled model experiment and lifecycle commands."""

    parser = subparsers.add_parser("models", help="Run controlled baseline model experiments.")
    parser.add_argument(
        "--processed-root", default=None, help="Override the configured processed data root."
    )
    parser.add_argument(
        "--storage-root", default=None, help="Override the configured canonical raw Parquet root."
    )
    parser.add_argument(
        "--output-root", default=None, help="Override the configured model artifact root."
    )
    parser.add_argument(
        "--reports-root", default=None, help="Override the configured report output root."
    )
    commands = parser.add_subparsers(dest="models_command", required=True)
    ranker = commands.add_parser(
        "ranker-baseline", help="Run fixed top-50 and robust-subset lambdarank experiments."
    )
    ranker.add_argument(
        "--recommended-features",
        default=None,
        help="Override diagnostics latest.json or recommended_features.json path.",
    )
    ranker.add_argument(
        "--robust-features",
        default=None,
        help="Override the manually maintained robust feature-list JSON path.",
    )
    production = commands.add_parser(
        "train-production", help="Train the final production Ranker on the approved full period."
    )
    production.add_argument(
        "--feature-list",
        default=None,
        help="Override the configured frozen robust feature-list JSON path.",
    )
    commands.add_parser("list", help="List registered models, including retired models.")
    commands.add_parser("champion", help="Show the current production champion model.")
    promote = commands.add_parser("promote", help="Explicitly promote a validated candidate.")
    promote.add_argument("model_id", help="Registered model identifier.")
    retire = commands.add_parser("retire", help="Retire a registered model.")
    retire.add_argument("model_id", help="Registered model identifier.")
    predict = commands.add_parser("predict", help="Score one completed session with the champion.")
    predict.add_argument("--as-of", required=True, help="Completed session in YYYYMMDD.")
    diagnostics = commands.add_parser(
        "diagnostics", help="Run read-only diagnostics for registered model artifacts."
    )
    diagnostic_commands = diagnostics.add_subparsers(
        dest="models_diagnostics_command", required=True
    )
    drift = diagnostic_commands.add_parser(
        "drift", help="Diagnose feature, score, and feature-response drift."
    )
    drift.add_argument("--model-id", required=True, help="Registered model identifier.")
    drift.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD date.")
    drift.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD date.")
    walk_forward = commands.add_parser(
        "walk-forward-plan",
        help="Build a purged chronological fold plan without fitting models.",
    )
    walk_forward.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD date.")
    walk_forward.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD date.")
    walk_forward.add_argument(
        "--scheme", required=True, choices=("expanding", "rolling"), help="Training window scheme."
    )
    walk_forward.add_argument(
        "--model-id", default=None, help="Registered model identity; defaults to champion."
    )
    walk_forward.add_argument(
        "--purge-days", type=int, default=None, help="Override purged trading sessions."
    )
    walk_forward.add_argument(
        "--embargo-days", type=int, default=None, help="Override embargo trading sessions."
    )
    walk_forward.add_argument(
        "--rolling-years", type=int, default=None, help="Override rolling window years."
    )
    horizon_plan = commands.add_parser(
        "horizon-plan",
        help="Bind configured label horizons to an existing purged fold plan.",
    )
    horizon_plan.add_argument(
        "--folds-manifest",
        default=None,
        help="Existing walk-forward manifest.json; defaults to latest compatible plan.",
    )
    observation = commands.add_parser(
        "observation-log",
        help="Record existing prediction and candidate rankings without trading actions.",
    )
    observation.add_argument("--as-of", required=True, help="Prediction date in YYYYMMDD.")


def add_strategy_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add optional model-score candidate selection commands."""

    parser = subparsers.add_parser("strategy", help="Filter model scores into research candidates.")
    parser.add_argument(
        "--storage-root", default=None, help="Override the configured canonical raw Parquet root."
    )
    parser.add_argument(
        "--processed-root", default=None, help="Override the configured processed data root."
    )
    parser.add_argument(
        "--reports-root", default=None, help="Override the configured report input/output root."
    )
    commands = parser.add_subparsers(dest="strategy_command", required=True)
    candidates = commands.add_parser(
        "candidates", help="Apply configured signal-date filters to production predictions."
    )
    candidates.add_argument("--as-of", required=True, help="Prediction date in YYYYMMDD.")


def add_research_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add deterministic human-readable quantitative research reports."""

    parser = subparsers.add_parser("research", help="Generate descriptive research reports.")
    parser.add_argument(
        "--storage-root", default=None, help="Override the configured canonical raw Parquet root."
    )
    parser.add_argument(
        "--processed-root", default=None, help="Override the configured processed data root."
    )
    parser.add_argument(
        "--reports-root", default=None, help="Override the configured report input/output root."
    )
    parser.add_argument(
        "--models-root", default=None, help="Override the configured model registry root."
    )
    commands = parser.add_subparsers(dest="research_command", required=True)
    report = commands.add_parser("report", help="Generate one daily quantitative research report.")
    report.add_argument("--as-of", required=True, help="Candidate date in YYYYMMDD.")
    explain = commands.add_parser("explain", help="Explain unchanged champion-model scores.")
    explain.add_argument("--as-of", required=True, help="Candidate date in YYYYMMDD.")
    decision = commands.add_parser(
        "decision", help="Generate human-review investment decision support."
    )
    decision.add_argument("--as-of", required=True, help="Candidate date in YYYYMMDD.")


def add_backtest_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add executable backtest commands."""

    parser = subparsers.add_parser(
        "backtest", help="Run executable portfolio simulations from model scores."
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Override the configured canonical raw Parquet root.",
    )
    parser.add_argument(
        "--processed-root", default=None, help="Override the configured processed data root."
    )
    parser.add_argument(
        "--models-root", default=None, help="Override the configured model artifact root."
    )
    parser.add_argument(
        "--output-root", default=None, help="Override the configured backtest output root."
    )
    commands = parser.add_subparsers(dest="backtest_command", required=True)
    run_parser = commands.add_parser("run", help="Run Top-N executable backtests.")
    run_parser.add_argument(
        "--model-dir",
        default=None,
        help="Saved Ranker experiment directory. Defaults to latest experiment_b_robust_*.",
    )
    run_parser.add_argument("--start-date", required=True, help="Inclusive YYYYMMDD start date.")
    run_parser.add_argument("--end-date", required=True, help="Inclusive YYYYMMDD end date.")
    run_parser.add_argument(
        "--top-n",
        default=None,
        help="Comma-separated Top-N variants, default from config such as 10,20,50.",
    )
    historical = commands.add_parser(
        "historical", help="Run an OOS historical champion selection backtest."
    )
    historical.add_argument(
        "--period", default=None, help="Configured period such as 2020-2023 or 2023-2026."
    )
    historical.add_argument("--start-date", default=None, help="Inclusive YYYYMMDD start date.")
    historical.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")
    historical.add_argument(
        "--top-n", default=None, help="Comma-separated Top-N variants; defaults to 10,20,50."
    )
    diagnostics = commands.add_parser(
        "diagnostics", help="Diagnose alpha in one immutable historical backtest run."
    )
    diagnostics.add_argument("--run-id", required=True, help="Historical backtest run ID.")


def add_pipeline_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add locked production orchestration commands."""

    parser = subparsers.add_parser("pipeline", help="Run locked production workflows.")
    commands = parser.add_subparsers(dest="pipeline_command", required=True)
    daily = commands.add_parser("daily", help="Update and validate daily production artifacts.")
    daily.add_argument(
        "--as-of",
        default=None,
        help="Completed open trading date in YYYYMMDD; defaults to latest completed date.",
    )
    readiness = commands.add_parser(
        "readiness", help="Run read-only raw, universe, and feature readiness gates."
    )
    readiness.add_argument("--as-of", required=True, help="Completed session in YYYYMMDD.")


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add common dataset-selection arguments."""

    parser.add_argument(
        "--dataset",
        action="append",
        choices=ALL_DATASETS,
        help="Dataset to process. Repeat to select multiple datasets.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Process every configured dataset, including extended optional datasets.",
    )


def add_date_range_args(parser: argparse.ArgumentParser) -> None:
    """Add common date range arguments."""

    parser.add_argument("--start-date", default=None, help="Inclusive YYYYMMDD start date.")
    parser.add_argument("--end-date", default=None, help="Inclusive YYYYMMDD end date.")


def run_doctor(config_path: str | None) -> int:
    """Run lightweight environment checks without contacting external services."""

    settings = load_settings(config_path)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    LOGGER.info("configuration loaded", extra={"project": settings.project_name})
    if not settings.has_tushare_token:
        LOGGER.warning("TUSHARE_TOKEN is not set; data ingestion commands will fail later")
    return 0


def run_config_check(config_path: str | None) -> int:
    """Validate configuration and print a non-secret summary."""

    settings = load_settings(config_path)
    print(f"project={settings.project_name}")
    print(f"environment={settings.environment}")
    print(f"tushare_token_set={settings.has_tushare_token}")
    return 0


def selected_datasets(args: argparse.Namespace) -> tuple[str, ...]:
    """Return CLI-selected datasets with conservative defaults."""

    if args.all_datasets:
        return ALL_DATASETS
    return tuple(args.dataset) if args.dataset else DEFAULT_DATASETS


def run_data_command(args: argparse.Namespace) -> int:
    """Run one data subcommand."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    store = build_store(args.storage_root, settings)

    if args.data_command == "status":
        for status in store.all_statuses():
            print(
                f"{status.name}: exists={status.exists} rows={status.rows} "
                f"partitions={status.partitions} min_date={status.min_date} "
                f"max_date={status.max_date} "
                f"snapshot_updated_at={status.snapshot_updated_at} "
                f"snapshot_age_days={status.snapshot_age_days}"
            )
        return 0

    dataset_names = selected_datasets(args)
    if args.data_command == "gaps":
        reports = DataIngestionService(settings=settings, store=store).scan_gaps(
            dataset_names, args.start_date, args.end_date
        )
        print_gap_reports(reports)
        return 1 if any(report.has_gaps for report in reports) else 0

    if args.data_command == "validate":
        results = DataValidator(store).validate_all(dataset_names)
        print_validation_results(results)
        return 0 if all(result.ok for result in results) else 1

    service = DataIngestionService(settings=settings, store=store)
    try:
        if args.data_command == "init":
            download_results = service.init(dataset_names, args.start_date, args.end_date)
        elif args.data_command == "update":
            download_results = service.update(
                dataset_names, args.end_date, args.refresh_snapshots, args.repair_gaps
            )
        else:
            raise ValueError(f"Unsupported data command: {args.data_command}")
    except DataIngestionError as error:
        append_quality_event(
            settings.paths.data_quality_logs,
            {
                "event": "data_ingestion_failed",
                "severity": "error",
                "message": str(error),
                "datasets": list(dataset_names),
            },
        )
        print(f"data ingestion failed: {error}", file=sys.stderr)
        return 2

    validation_results = DataValidator(store).validate_all(dataset_names)
    append_validation_results(settings.paths.data_quality_logs, validation_results)
    launch_baostock_previous_day_check(args.config, settings.paths.data_quality_logs)

    for result in download_results:
        print(
            f"{result.dataset}: rows_written={result.rows_written} "
            f"skipped={result.skipped} message={result.message}"
        )
    return 0 if all(result.ok for result in validation_results) else 1


def run_universe_command(args: argparse.Namespace) -> int:
    """Run one universe subcommand."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    raw_store = build_store(args.storage_root, settings)
    output_root = (
        settings.paths.processed_data if args.output_root is None else Path(args.output_root)
    )
    universe_store = UniverseStore(output_root)

    if args.universe_command == "status":
        status = universe_store.status(args.date)
        print(
            f"universe_daily: exists={status.exists} rows={status.rows} "
            f"partitions={status.partitions} min_date={status.min_date} max_date={status.max_date} "
            f"in_base_universe={status.in_base_universe} "
            f"in_model_universe={status.in_model_universe} can_buy={status.can_buy} "
            f"can_sell={status.can_sell}"
        )
        print_manifest_status(
            "universe_daily",
            universe_store.dataset_dir,
            effective_config_path(args.config),
        )
        return 0

    if args.universe_command == "validate":
        validation_result = UniverseValidator(universe_store).validate(
            args.start_date, args.end_date
        )
        print_universe_validation_result(validation_result)
        return 0 if validation_result.ok else 1

    if args.universe_command == "build":
        builder = UniverseBuilder(raw_store, universe_store, settings)
        build_started_at = utc_now_iso()
        try:
            build_result = builder.build(args.start_date, args.end_date)
        except DataValidationError as error:
            print(f"universe build failed: {error}", file=sys.stderr)
            return 2
        print(
            f"universe_daily: rows_built={build_result.rows_built} "
            f"rows_written={build_result.rows_written} "
            f"start_date={build_result.start_date} end_date={build_result.end_date}"
        )
        print_universe_validation_result(build_result.validation)
        if build_result.validation.ok:
            canonical_statistics = parquet_artifact_statistics(universe_store.dataset_dir)
            write_build_manifest(
                universe_store.dataset_dir,
                artifact_name="universe_daily",
                build_started_at=build_started_at,
                config_path=effective_config_path(args.config),
                start_date=build_result.start_date,
                end_date=build_result.end_date,
                row_count=build_result.rows_written,
                canonical_statistics=canonical_statistics,
                partitions_changed=build_result.partitions_changed,
                source_fingerprints=raw_source_fingerprints(
                    raw_store,
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
        return 0 if build_result.validation.ok else 1

    raise ValueError(f"Unsupported universe command: {args.universe_command}")


def run_labels_command(args: argparse.Namespace) -> int:
    """Run one labels subcommand."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    raw_store = build_store(args.storage_root, settings)
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    universe_store = UniverseStore(processed_root)
    label_store = LabelStore(processed_root)

    if args.labels_command == "status":
        status = label_store.status(args.date, args.horizon)
        print(
            f"labels_forward: exists={status.exists} rows={status.rows} "
            f"partitions={status.partitions} min_date={status.min_date} max_date={status.max_date} "
            f"available={status.available} unavailable={status.unavailable}"
        )
        print_manifest_status(
            "labels_forward",
            label_store.dataset_dir,
            effective_config_path(args.config),
        )
        return 0

    if args.labels_command == "validate":
        validation_result = LabelValidator(
            label_store,
            settings.labels.quantile_buckets,
            universe_store,
            settings.labels.horizons,
        ).validate(args.start_date, args.end_date)
        print_label_validation_result(validation_result)
        return 0 if validation_result.ok else 1

    if args.labels_command == "build":
        horizons = parse_horizons(args.horizons) if args.horizons else settings.labels.horizons
        builder = LabelBuilder(raw_store, universe_store, label_store, settings)
        build_started_at = utc_now_iso()
        try:
            build_result = builder.build(args.start_date, args.end_date, horizons)
        except DataValidationError as error:
            print(f"labels build failed: {error}", file=sys.stderr)
            return 2
        print(
            f"labels_forward: rows_built={build_result.rows_built} "
            f"rows_written={build_result.rows_written} "
            f"start_date={build_result.start_date} end_date={build_result.end_date} "
            f"horizons={','.join(str(horizon) for horizon in build_result.horizons)}"
        )
        print_label_validation_result(build_result.validation)
        if build_result.validation.ok:
            universe_status = universe_store.status()
            source_fingerprints = raw_source_fingerprints(
                raw_store,
                ("trade_cal", "daily", "adj_factor", "stk_limit", "index_daily"),
            )
            source_fingerprints["universe_daily"] = processed_source_fingerprint(
                universe_store.dataset_dir,
                rows=universe_status.rows,
                partitions=universe_status.partitions,
                min_date=universe_status.min_date,
                max_date=universe_status.max_date,
            )
            write_build_manifest(
                label_store.dataset_dir,
                artifact_name="labels_forward",
                build_started_at=build_started_at,
                config_path=effective_config_path(args.config),
                start_date=build_result.start_date,
                end_date=build_result.end_date,
                row_count=build_result.rows_written,
                source_fingerprints=source_fingerprints,
                extra={"label_horizons": list(build_result.horizons)},
            )
        return 0 if build_result.validation.ok else 1

    raise ValueError(f"Unsupported labels command: {args.labels_command}")


def run_features_command(args: argparse.Namespace) -> int:
    """Run one features subcommand."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    raw_store = build_store(args.storage_root, settings)
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    feature_store = FeatureStore(processed_root)

    if args.features_command == "registry":
        family_counts: dict[str, int] = {}
        for spec in FEATURE_REGISTRY:
            family_counts[spec.family] = family_counts.get(spec.family, 0) + 1
        print(f"feature_count={len(FEATURE_REGISTRY)}")
        for family, count in sorted(family_counts.items()):
            print(f"{family}: {count}")
        return 0

    if args.features_command == "status":
        status = feature_store.status(args.date)
        print(
            f"features_daily: exists={status.exists} rows={status.rows} "
            f"partitions={status.partitions} min_date={status.min_date} max_date={status.max_date} "
            f"feature_count={status.feature_count}"
        )
        print_manifest_status(
            "features_daily",
            feature_store.dataset_dir,
            effective_config_path(args.config),
        )
        return 0

    if args.features_command == "validate":
        validation_result = FeatureValidator(feature_store).validate(args.start_date, args.end_date)
        print_feature_validation_result(validation_result)
        return 0 if validation_result.ok else 1

    if args.features_command == "build":
        universe_store = UniverseStore(processed_root)
        builder = FeatureBuilder(raw_store, universe_store, feature_store, settings)
        build_started_at = utc_now_iso()
        try:
            result = builder.build(args.start_date, args.end_date)
        except DataValidationError as error:
            print(f"features build failed: {error}", file=sys.stderr)
            return 2
        print(
            f"features_daily: rows_built={result.rows_built} rows_written={result.rows_written} "
            f"feature_count={result.feature_count} elapsed_seconds={result.elapsed_seconds:.3f}"
        )
        missing_preview = sorted(
            result.missing_value_stats.items(), key=lambda item: item[1], reverse=True
        )[:10]
        for name, ratio in missing_preview:
            print(f"missing_ratio {name}={ratio:.4f}")
        universe_statistics = parquet_artifact_statistics(universe_store.dataset_dir)
        source_fingerprints = raw_source_fingerprints(
            raw_store,
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
        source_fingerprints["universe_daily"] = processed_source_fingerprint(
            universe_store.dataset_dir,
            rows=universe_statistics.row_count,
            partitions=universe_statistics.partition_count,
            min_date=universe_statistics.min_date,
            max_date=universe_statistics.max_date,
        )
        canonical_statistics = parquet_artifact_statistics(feature_store.dataset_dir)
        canonical_feature_count = len(
            set(canonical_statistics.column_names) - {"trade_date", "ts_code"}
        )
        write_build_manifest(
            feature_store.dataset_dir,
            artifact_name="features_daily",
            build_started_at=build_started_at,
            config_path=effective_config_path(args.config),
            start_date=result.start_date,
            end_date=result.end_date,
            row_count=result.rows_written,
            canonical_statistics=canonical_statistics,
            partitions_changed=result.partitions_changed,
            source_fingerprints=source_fingerprints,
            extra={"feature_count": canonical_feature_count},
        )
        return 0

    raise ValueError(f"Unsupported features command: {args.features_command}")


def run_diagnostics_command(args: argparse.Namespace) -> int:
    """Run one feature diagnostics subcommand."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    reports_root = settings.paths.reports if args.reports_root is None else Path(args.reports_root)
    latest_path = reports_root / "feature_diagnostics" / "latest.json"
    if args.diagnostics_command == "status":
        if not latest_path.exists():
            print("feature_diagnostics: exists=False")
            return 0
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        report_dir = Path(str(latest["report_dir"]))
        manifest_path = report_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection = json.loads(
            (report_dir / "recommended_features.json").read_text(encoding="utf-8")
        )
        print(
            f"feature_diagnostics: exists=True run_id={latest['run_id']} "
            f"recommended_count={selection['recommended_feature_count']} "
            f"git_commit={manifest['git_commit']} config_hash={manifest['config_hash']}"
        )
        return 0
    if args.diagnostics_command == "run":
        split = ChronologicalSplit(
            train_start=args.train_start,
            train_end=args.train_end,
            validation_start=args.validation_start,
            validation_end=args.validation_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )
        pipeline = FeatureDiagnosticPipeline(
            processed_root,
            reports_root,
            settings,
            Path(effective_config_path(args.config)),
        )
        try:
            result = pipeline.run(split, args.horizon)
        except (DataValidationError, ValueError) as error:
            print(f"diagnostics run failed: {error}", file=sys.stderr)
            return 2
        print(
            f"feature_diagnostics: report_dir={result.report_dir} "
            f"recommended_count={result.recommended_count} train_rows={result.train_rows} "
            f"validation_rows={result.validation_rows} test_rows={result.test_rows}"
        )
        return 0
    raise ValueError(f"Unsupported diagnostics command: {args.diagnostics_command}")


def run_models_command(args: argparse.Namespace) -> int:
    """Run one model experiment command."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    output_root = settings.paths.models if args.output_root is None else Path(args.output_root)
    reports_root = settings.paths.reports if args.reports_root is None else Path(args.reports_root)
    if args.models_command == "observation-log":
        recorder = ProductionObservationRecorder(reports_root)
        try:
            observation = recorder.record(args.as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"production observation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"production_observation: date={observation.prediction_date} "
            f"model_id={observation.model_id} candidates={observation.candidate_count} "
            f"output={observation.output_path}"
        )
        return 0
    if args.models_command == "horizon-plan":
        horizon_planner = MultiHorizonExperimentPlanner(
            raw_root=(
                settings.paths.parquet_store
                if args.storage_root is None
                else Path(args.storage_root)
            ),
            models_root=output_root,
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            horizon_result = horizon_planner.build(
                folds_manifest=(None if args.folds_manifest is None else Path(args.folds_manifest))
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"horizon experiment planning failed: {error}", file=sys.stderr)
            return 2
        print(
            f"horizon_experiment_plan: run_id={horizon_result.run_id} "
            f"source_model_id={horizon_result.source_model_id} "
            f"experiments={horizon_result.experiment_count} "
            f"folds_manifest={horizon_result.folds_manifest} "
            f"output={horizon_result.output_dir}"
        )
        return 0
    if args.models_command == "walk-forward-plan":
        raw_root = (
            settings.paths.parquet_store if args.storage_root is None else Path(args.storage_root)
        )
        walk_forward_planner = PurgedWalkForwardPlanner(
            raw_root=raw_root,
            models_root=output_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            walk_forward_result = walk_forward_planner.build(
                start_date=args.start_date,
                end_date=args.end_date,
                scheme=args.scheme,
                model_id=args.model_id,
                purge_days=args.purge_days,
                embargo_days=args.embargo_days,
                rolling_years=args.rolling_years,
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"walk-forward planning failed: {error}", file=sys.stderr)
            return 2
        print(
            f"walk_forward_plan: run_id={walk_forward_result.run_id} "
            f"scheme={walk_forward_result.scheme} model_id={walk_forward_result.model_id} "
            f"folds={walk_forward_result.fold_count} output={walk_forward_result.output_dir}"
        )
        return 0
    if args.models_command == "diagnostics":
        if args.models_diagnostics_command != "drift":
            raise ValueError(
                f"Unsupported models diagnostics command: {args.models_diagnostics_command}"
            )
        drift_engine = ModelDriftDiagnosticEngine(
            processed_root=processed_root,
            models_root=output_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            drift_result = drift_engine.run(
                model_id=args.model_id,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"model drift diagnostics failed: {error}", file=sys.stderr)
            return 2
        print(
            f"model_drift_diagnostics: run_id={drift_result.run_id} "
            f"model_id={drift_result.model_id} features={drift_result.feature_count} "
            f"months={drift_result.months} output={drift_result.output_dir}"
        )
        return 0
    if args.models_command == "predict":
        config_path = Path(effective_config_path(args.config))
        raw_store = build_store(args.storage_root, settings)
        universe_store = UniverseStore(processed_root)
        feature_store = FeatureStore(processed_root)
        freshness = FreshnessService(
            settings,
            raw_store,
            universe_store,
            feature_store,
            config_path=config_path,
        )
        inference_engine = ProductionInferenceEngine(
            registry=ModelRegistry(output_root),
            processed_root=processed_root,
            reports_root=reports_root,
            config_path=config_path,
            freshness=freshness,
        )
        try:
            inference_result = inference_engine.predict(args.as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"production prediction failed: {error}", file=sys.stderr)
            return 2
        print(
            f"prediction_output: date={inference_result.as_of} "
            f"model_id={inference_result.model_id} "
            f"stocks={inference_result.prediction_count} "
            f"output={inference_result.output_dir / 'predictions.parquet'}"
        )
        return 0
    if args.models_command in {"list", "champion", "promote", "retire"}:
        registry = ModelRegistry(output_root)
        try:
            if args.models_command == "list":
                print("model_id\tstatus\tcreated_time\ttest_rank_ic\ttest_sharpe\tfeature_count")
                for model in registry.list_models():
                    print(
                        f"{model.model_id}\t{model.status}\t{model.creation_time}\t"
                        f"{_format_registry_metric(model.test_metrics, 'rank_ic')}\t"
                        f"{_format_registry_metric(model.test_metrics, 'sharpe')}\t"
                        f"{model.feature_count}"
                    )
                return 0
            if args.models_command == "champion":
                champion = registry.get_champion()
                if champion is None:
                    print("model_champion: none")
                else:
                    print(
                        f"model_champion: model_id={champion.model_id} "
                        f"model_type={champion.model_type} artifact_path={champion.artifact_path}"
                    )
                return 0
            if args.models_command == "promote":
                promoted = registry.promote_model(args.model_id)
                print(
                    f"model_promoted: model_id={promoted.model_id} "
                    f"model_type={promoted.model_type} status={promoted.status}"
                )
                return 0
            retired = registry.retire_model(args.model_id)
            print(f"model_retired: model_id={retired.model_id} status={retired.status}")
            return 0
        except (DataValidationError, OSError, ProductionLockError) as error:
            print(f"model registry operation failed: {error}", file=sys.stderr)
            return 2
    if args.models_command == "ranker-baseline":
        runner = RankerBaselineRunner(
            processed_root,
            output_root,
            settings,
            Path(effective_config_path(args.config)),
        )
        try:
            results = runner.run(
                None if args.recommended_features is None else Path(args.recommended_features),
                None if args.robust_features is None else Path(args.robust_features),
            )
        except (DataValidationError, ValueError) as error:
            print(f"ranker baseline failed: {error}", file=sys.stderr)
            return 2
        for experiment_result in results:
            print(
                f"{experiment_result.experiment_name}: "
                f"experiment_id={experiment_result.experiment_id} "
                f"features={experiment_result.feature_count} "
                f"validation_rank_ic={experiment_result.validation_rank_ic:.6f} "
                f"test_rank_ic={experiment_result.test_rank_ic:.6f} "
                f"output={experiment_result.output_dir}"
            )
        return 0
    if args.models_command == "train-production":
        trainer = ProductionRankerTrainer(
            processed_root,
            output_root,
            settings,
            Path(effective_config_path(args.config)),
        )
        try:
            production_result = trainer.train(
                None if args.feature_list is None else Path(args.feature_list)
            )
        except (DataValidationError, ValueError) as error:
            print(f"production training failed: {error}", file=sys.stderr)
            return 2
        print(
            f"production_ranker: output={production_result.output_dir} "
            f"features={production_result.feature_count} "
            f"train_rows={production_result.train_rows} "
            f"train_groups={production_result.train_groups} "
            f"train_start={production_result.train_start} train_end={production_result.train_end}"
        )
        return 0
    raise ValueError(f"Unsupported models command: {args.models_command}")


def _format_registry_metric(metrics: dict[str, object], name: str) -> str:
    value = metrics.get(name)
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.6f}"


def run_strategy_command(args: argparse.Namespace) -> int:
    """Run one optional post-inference candidate selection command."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    if args.strategy_command != "candidates":
        raise ValueError(f"Unsupported strategy command: {args.strategy_command}")
    raw_root = (
        settings.paths.parquet_store if args.storage_root is None else Path(args.storage_root)
    )
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    reports_root = settings.paths.reports if args.reports_root is None else Path(args.reports_root)
    selector = CandidateSelector(
        raw_root=raw_root,
        processed_root=processed_root,
        reports_root=reports_root,
        config_path=Path(effective_config_path(args.config)),
        settings=settings.strategy.candidate_selection,
    )
    try:
        result = selector.select(args.as_of)
    except (DataValidationError, OSError, ValueError) as error:
        print(f"strategy candidate selection failed: {error}", file=sys.stderr)
        return 2
    print(
        f"strategy_candidates: date={result.as_of} candidates={result.candidate_count} "
        f"output={result.output_path}"
    )
    return 0


def run_research_command(args: argparse.Namespace) -> int:
    """Generate one optional post-candidate research report."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    raw_root = (
        settings.paths.parquet_store if args.storage_root is None else Path(args.storage_root)
    )
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    reports_root = settings.paths.reports if args.reports_root is None else Path(args.reports_root)
    if args.research_command == "decision":
        support = InvestmentDecisionSupport(
            raw_root=raw_root,
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings.research.decision_support,
        )
        try:
            decision_result = support.generate(args.as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"investment decision support failed: {error}", file=sys.stderr)
            return 2
        print(
            f"decision_support: date={decision_result.as_of} "
            f"candidates={decision_result.candidate_count} model_id={decision_result.model_id} "
            f"output={decision_result.json_path}"
        )
        return 0
    if args.research_command == "explain":
        models_root = settings.paths.models if args.models_root is None else Path(args.models_root)
        engine = ExplainabilityEngine(
            registry=ModelRegistry(models_root),
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings.research.explainability,
        )
        try:
            explanation_result = engine.explain(args.as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"research explanation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"research_explanations: date={explanation_result.as_of} "
            f"candidates={explanation_result.candidate_count} "
            f"model_id={explanation_result.model_id} method={explanation_result.method} "
            f"output={explanation_result.json_path}"
        )
        return 0
    if args.research_command != "report":
        raise ValueError(f"Unsupported research command: {args.research_command}")
    generator = DailyResearchReportGenerator(
        raw_root=raw_root,
        processed_root=processed_root,
        reports_root=reports_root,
        settings=settings.research.daily_report,
    )
    try:
        report_result = generator.generate(args.as_of)
    except (DataValidationError, OSError, ValueError) as error:
        print(f"daily research report failed: {error}", file=sys.stderr)
        return 2
    print(
        f"daily_research_report: date={report_result.as_of} "
        f"candidates={report_result.candidate_count} "
        f"model_id={report_result.model_id} output={report_result.report_path}"
    )
    return 0


def run_backtest_command(args: argparse.Namespace) -> int:
    """Run one executable backtest command."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    raw_root = (
        settings.paths.parquet_store if args.storage_root is None else Path(args.storage_root)
    )
    processed_root = (
        settings.paths.processed_data if args.processed_root is None else Path(args.processed_root)
    )
    models_root = settings.paths.models if args.models_root is None else Path(args.models_root)
    output_root = settings.paths.backtests if args.output_root is None else Path(args.output_root)
    if args.backtest_command == "diagnostics":
        historical_root = settings.paths.reports / "backtest"
        diagnostic_root = (
            settings.paths.reports / "backtest_diagnostics"
            if args.output_root is None
            else Path(args.output_root)
        )
        diagnostic_engine = BacktestDiagnosticEngine(
            processed_root=processed_root,
            backtest_root=historical_root,
            output_root=diagnostic_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            diagnostic_result = diagnostic_engine.run(args.run_id)
        except (DataValidationError, ValueError) as error:
            print(f"backtest diagnostics failed: {error}", file=sys.stderr)
            return 2
        print(
            f"backtest_diagnostics: run_id={diagnostic_result.run_id} "
            f"predictions={diagnostic_result.prediction_rows} "
            f"labelled={diagnostic_result.labelled_rows} "
            f"ic_days={diagnostic_result.ic_days} output={diagnostic_result.output_dir}"
        )
        return 0
    if args.backtest_command == "historical":
        historical_output = (
            settings.paths.reports / "backtest"
            if args.output_root is None
            else Path(args.output_root)
        )
        historical_engine = HistoricalBacktestEngine(
            raw_root=raw_root,
            processed_root=processed_root,
            output_root=historical_output,
            models_root=models_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            historical_result = historical_engine.run(
                period=args.period,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=None if args.top_n is None else parse_top_n(args.top_n),
            )
        except (DataValidationError, ValueError) as error:
            print(f"historical backtest failed: {error}", file=sys.stderr)
            return 2
        print(
            f"historical_backtest: run_id={historical_result.run_id} "
            f"model_id={historical_result.model_id} period={historical_result.start_date}.."
            f"{historical_result.end_date} output={historical_result.output_dir}"
        )
        return 0
    if args.backtest_command == "run":
        runner = BacktestRunner(
            raw_root,
            processed_root,
            models_root,
            output_root,
            settings,
            Path(effective_config_path(args.config)),
        )
        try:
            result = runner.run(
                model_dir=None if args.model_dir is None else Path(args.model_dir),
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=None if args.top_n is None else parse_top_n(args.top_n),
            )
        except (DataValidationError, ValueError) as error:
            print(f"backtest run failed: {error}", file=sys.stderr)
            return 2
        print(
            f"backtest: experiment_id={result.experiment_id} output={result.output_dir} "
            f"top_n={','.join(str(value) for value in result.top_n)}"
        )
        for top_n, metrics in result.metrics.items():
            print(
                f"top{top_n}: annual_return={metrics.get('annual_return')} "
                f"sharpe={metrics.get('sharpe')} "
                f"max_drawdown={metrics.get('maximum_drawdown')}"
            )
        return 0
    raise ValueError(f"Unsupported backtest command: {args.backtest_command}")


def run_pipeline_command(args: argparse.Namespace) -> int:
    """Run one locked production orchestration command."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    config_path = Path(effective_config_path(args.config))
    raw_store = build_store(None, settings)
    universe_store = UniverseStore(settings.paths.processed_data)
    feature_store = FeatureStore(settings.paths.processed_data)
    freshness = FreshnessService(
        settings,
        raw_store,
        universe_store,
        feature_store,
        config_path=config_path,
    )

    if args.pipeline_command == "readiness":
        try:
            as_of = resolve_completed_trading_date(raw_store, args.as_of)
            results = freshness.check_all(as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"pipeline_readiness: NOT_READY error={error}", file=sys.stderr)
            return 1
        print_readiness_results(results)
        return 0 if all(result.ready for result in results) else 1

    if args.pipeline_command != "daily":
        raise ValueError(f"Unsupported pipeline command: {args.pipeline_command}")

    def execute_stage(arguments: tuple[str, ...]) -> int:
        return main(("--config", str(config_path), *arguments))

    orchestrator = DailyPipelineOrchestrator(
        executor=execute_stage,
        as_of_resolver=lambda requested: resolve_completed_trading_date(raw_store, requested),
        config_path=config_path,
        processed_root=settings.paths.processed_data,
        readiness_executor=lambda gate, as_of: execute_readiness_gate(freshness, gate, as_of),
    )
    try:
        result = orchestrator.run(args.as_of)
    except ProductionLockError as error:
        print(f"production run blocked: {error}", file=sys.stderr)
        return 3
    if result.status == "success":
        print(
            f"pipeline_daily: status=success run_id={result.run.run_id} "
            f"as_of={result.as_of} manifest={result.run.manifest_path}"
        )
        return 0
    print(
        f"pipeline_daily: status=failed run_id={result.run.run_id} "
        f"as_of={result.as_of} failed_stage={result.failed_stage} "
        f"message={result.error_message} manifest={result.run.manifest_path}",
        file=sys.stderr,
    )
    return result.exit_code or 2


def execute_readiness_gate(
    service: FreshnessService,
    gate_name: str,
    as_of: str,
) -> GateResult:
    """Dispatch one named pipeline gate to the shared readiness service."""

    methods = {
        "raw_freshness_gate": service.check_raw,
        "universe_readiness_gate": service.check_universe,
        "features_readiness_gate": service.check_features,
    }
    try:
        method = methods[gate_name]
    except KeyError as error:
        raise ValueError(f"unsupported readiness gate: {gate_name}") from error
    return method(as_of)


def print_readiness_results(results: Sequence[GateResult]) -> None:
    """Print compact human-readable readiness output while retaining structured APIs."""

    ready = all(result.ready for result in results)
    print(f"pipeline_readiness: {'READY' if ready else 'NOT_READY'}")
    for result in results:
        print(
            f"{result.gate}: ready={result.ready} expected_as_of={result.expected_as_of} "
            f"hard_failures={len(result.hard_failures)} warnings={len(result.warnings)}"
        )
        for failure in result.hard_failures:
            print(f"  failure: {failure}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        row_counts = result.details.get("row_counts")
        if row_counts is not None:
            print(f"  row_counts: {json.dumps(row_counts, sort_keys=True)}")
        actual_dates = result.details.get("actual_max_dates")
        if actual_dates is not None:
            print(f"  actual_max_dates: {json.dumps(actual_dates, sort_keys=True)}")
        artifact_manifest = result.details.get("artifact_manifest")
        if artifact_manifest is not None:
            print(f"  artifact_manifest: {json.dumps(artifact_manifest, sort_keys=True)}")
        missingness = result.details.get("missingness_summary")
        if missingness is not None:
            print(f"  missingness_summary: {json.dumps(missingness, sort_keys=True)}")


def parse_horizons(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integer horizons."""

    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"horizons must be positive integers: {value}")
    return horizons


def parse_top_n(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive Top-N values."""

    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"top-n values must be positive integers: {value}")
    if len(set(values)) != len(values):
        raise ValueError(f"top-n values must not contain duplicates: {value}")
    return values


def launch_baostock_previous_day_check(config_path: str | None, log_root: object) -> None:
    """Start the baostock previous-trading-day checker without blocking ingestion."""

    script = "scripts/data_checks/run_baostock_previous_day_check.py"
    command = [sys.executable, script, "--log-root", str(log_root)]
    if config_path is not None:
        command.extend(["--config", config_path])
    output_dir = log_root / "background"  # type: ignore[operator]
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "baostock_previous_day_check.out"
    stderr_path = output_dir / "baostock_previous_day_check.err"
    try:
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)  # noqa: S603
    except OSError as error:
        append_quality_event(
            log_root,  # type: ignore[arg-type]
            {
                "event": "baostock_previous_day_check_launch_failed",
                "severity": "error",
                "message": str(error),
                "command": command,
            },
        )


def print_validation_results(results: Sequence[ValidationResult]) -> None:
    """Print compact validation output."""

    for result in results:
        print(f"{result.dataset}: ok={result.ok} status={result.status}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        for error in result.errors:
            print(f"  error: {error}")


def print_gap_reports(reports: Sequence[GapReport]) -> None:
    """Print compact gap-scan output."""

    for report in reports:
        print(
            f"{report.dataset}: gaps={report.has_gaps} skipped={report.skipped} "
            f"expected_dates={report.expected_dates} "
            f"missing_dates={len(report.missing_dates)} "
            f"excluded_before_inception={report.excluded_before_inception} "
            f"start_date={report.start_date} end_date={report.end_date} "
            f"message={report.message}"
        )
        if report.missing_by_entity:
            for entity, dates in sorted(report.missing_by_entity.items()):
                preview = ",".join(dates[:10])
                print(f"  {entity}: missing={len(dates)} first={preview}")
        if report.excluded_before_inception_by_entity:
            for entity, dates in sorted(report.excluded_before_inception_by_entity.items()):
                preview = ",".join(dates[:10])
                print(f"  {entity}: excluded_before_inception={len(dates)} first={preview}")
        elif report.missing_dates:
            preview = ",".join(report.missing_dates[:20])
            print(f"  first={preview}")


def print_universe_validation_result(result: UniverseValidationResult) -> None:
    """Print compact universe validation output."""

    print(f"validation: ok={result.ok}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        print(f"  error: {error}")


def print_label_validation_result(result: LabelValidationResult) -> None:
    """Print compact label validation output."""

    print(f"validation: ok={result.ok}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        print(f"  error: {error}")


def print_feature_validation_result(result: FeatureValidationResult) -> None:
    """Print compact production feature validation output."""

    print(f"validation: ok={result.ok} rows={result.rows}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    for error in result.errors:
        print(f"  error: {error}")


def print_manifest_status(artifact_name: str, artifact_dir: Path, config_path: str | None) -> None:
    """Print compact processed artifact provenance status."""

    status = artifact_manifest_status(artifact_dir, config_path=config_path)
    print(
        f"{artifact_name}_manifest: exists={status.exists} stale={status.stale} "
        f"artifact_git={status.artifact_git_revision} current_git={status.current_git_revision} "
        f"config_hash_match={status.config_hash_match} reason={status.reason}"
    )


def effective_config_path(config_arg: str | None) -> str:
    """Return the config path used by load_settings for manifest hashing."""

    return config_arg or os.environ.get("ASHARE_QUANT_CONFIG", "config/default.yaml")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor(args.config)
    if args.command == "config-check":
        return run_config_check(args.config)
    if args.command == "data":
        return run_data_command(args)
    if args.command == "universe":
        return run_universe_command(args)
    if args.command == "labels":
        return run_labels_command(args)
    if args.command == "features":
        return run_features_command(args)
    if args.command == "diagnostics":
        return run_diagnostics_command(args)
    if args.command == "models":
        return run_models_command(args)
    if args.command == "strategy":
        return run_strategy_command(args)
    if args.command == "research":
        return run_research_command(args)
    if args.command == "backtest":
        return run_backtest_command(args)
    if args.command == "pipeline":
        return run_pipeline_command(args)
    raise ValueError(f"Unsupported command: {args.command}")
