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
from ashare_quant.backtest.executable_validation import ExecutableOOSValidationEngine
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
from ashare_quant.governance import DailyGovernanceSnapshotService, GovernanceService
from ashare_quant.labels import LabelBuilder, LabelStore, LabelValidator
from ashare_quant.labels.validation import LabelValidationResult
from ashare_quant.models import (
    ChallengerEvaluationEngine,
    ChallengerPredictionEngine,
    ChallengerTrainer,
    HumanReviewService,
    ModelDriftDiagnosticEngine,
    ModelRegistry,
    MultiHorizonEnsembleEngine,
    MultiHorizonExperimentPlanner,
    ProductionInferenceEngine,
    ProductionObservationRecorder,
    ProductionRankerTrainer,
    PromotionApplyService,
    PromotionEvidencePaths,
    PromotionEvidenceResolver,
    PromotionGateEngine,
    PromotionGatePolicy,
    PromotionGovernanceService,
    PurgedWalkForwardPlanner,
    RankerBaselineRunner,
    RollbackService,
    load_promotion_gate_policy,
)
from ashare_quant.models.shadow import ShadowPredictionService
from ashare_quant.monitoring import (
    AlertService,
    MonitoringService,
    PerformanceMonitoringService,
)
from ashare_quant.monitoring.performance_observation import (
    PerformanceObservationService,
)
from ashare_quant.orchestration import (
    DEFAULT_PRODUCTION_LOCK_PATH,
    DailyPipelineOrchestrator,
    DailyPipelineStages,
    FreshnessService,
    GateResult,
    ProductionLockError,
    resolve_completed_trading_date,
    run_with_production_lock,
)
from ashare_quant.orchestration.production import (
    ProductionDailyStageExecutor,
    ProductionPipeline,
)
from ashare_quant.orchestration.scheduler import (
    FullDataUpdateScheduler,
    ProductionScheduler,
)
from ashare_quant.paper_trading import PaperTradingService
from ashare_quant.research import (
    DailyResearchReportGenerator,
    ExplainabilityEngine,
    InvestmentDecisionSupport,
)
from ashare_quant.research.agent import ResearchAgentService
from ashare_quant.retraining import RetrainingTriggerService
from ashare_quant.retraining.execution import GovernedRetrainingExecutionService
from ashare_quant.retraining.orchestration import RetrainingLifecycleOrchestrator
from ashare_quant.retraining.orchestration.dry_run import LifecycleDryRunService
from ashare_quant.retraining.readiness import RetrainingExecutionReadinessValidator
from ashare_quant.retraining.shadow import RetrainedChallengerShadowService
from ashare_quant.retraining.validation import RetrainingValidationService
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
    add_research_agent_parser(subparsers)
    add_backtest_parser(subparsers)
    add_paper_trading_parser(subparsers)
    add_monitor_parser(subparsers)
    add_governance_parser(subparsers)
    add_retraining_parser(subparsers)
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
    promotion = commands.add_parser(
        "promotion", help="Create and inspect immutable promotion governance requests."
    )
    promotion_commands = promotion.add_subparsers(dest="models_promotion_command", required=True)
    promotion_create = promotion_commands.add_parser(
        "create", help="Freeze a candidate promotion request without applying it."
    )
    promotion_create.add_argument("--model-id", required=True)
    promotion_create.add_argument("--evidence-cutoff-date", required=True)
    promotion_create.add_argument("--deployment-slot", default="daily_stock_ranker")
    promotion_prepare = promotion_commands.add_parser(
        "prepare", help="Discover and freeze immutable candidate promotion evidence."
    )
    promotion_prepare.add_argument("--model-id", required=True)
    promotion_prepare.add_argument(
        "--lifecycle-run-id", help="Use exact retrained lifecycle evidence only."
    )
    for argument in (
        "challenger-evaluation",
        "executable-validation",
        "shadow-prediction",
        "performance-observation",
        "monitoring-summary",
        "alerts",
    ):
        promotion_create.add_argument(f"--{argument}", required=True)
    for command_name in ("validate", "status", "review", "review-status"):
        command = promotion_commands.add_parser(
            command_name, help=f"{command_name.title()} an immutable promotion request."
        )
        command.add_argument("--request-id", required=True)
    for command_name in ("apply", "apply-status"):
        command = promotion_commands.add_parser(
            command_name, help=f"{command_name.title()} an approved promotion request."
        )
        command.add_argument("--request-id", required=True)
        if command_name == "apply":
            command.add_argument("--dry-run", action="store_true")
    rollback_create = promotion_commands.add_parser(
        "rollback-create", help="Create an immutable historical-Champion rollback request."
    )
    rollback_create.add_argument("--model-id", required=True)
    rollback_create.add_argument("--reason-file", required=True)
    rollback_create.add_argument("--reason-type", default="operator_requested")
    rollback_create.add_argument("--deployment-slot", default="daily_stock_ranker")
    for command_name in ("rollback-validate", "rollback-apply"):
        command = promotion_commands.add_parser(
            command_name, help=f"{command_name.title()} a governed rollback request."
        )
        command.add_argument("--request-id", required=True)
    for command_name in ("approve", "reject"):
        command = promotion_commands.add_parser(
            command_name, help=f"Append an immutable {command_name} review event."
        )
        command.add_argument("--request-id", required=True)
        command.add_argument("--comments-file", required=True)
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
    challenger = commands.add_parser(
        "train-challenger",
        help="Train immutable candidate Rankers from a multi-horizon experiment plan.",
    )
    challenger_selection = challenger.add_mutually_exclusive_group(required=True)
    challenger_selection.add_argument(
        "--experiment-id",
        help="Horizon experiment ID, name, or alias such as experiment_c_h20.",
    )
    challenger_selection.add_argument(
        "--all-horizons",
        action="store_true",
        help="Train one independent challenger for every horizon in the plan.",
    )
    challenger.add_argument(
        "--experiment-manifest",
        default=None,
        help="Specific horizon experiment_manifest.json; defaults to the latest plan.",
    )
    challenger_prediction = commands.add_parser(
        "predict-challenger",
        help="Publish immutable mature final-test predictions for one challenger.",
    )
    challenger_prediction.add_argument("--model-id", required=True)
    challenger_evaluation = commands.add_parser(
        "evaluate-challenger",
        help="Compare one challenger with the current champion on identical rows.",
    )
    challenger_evaluation.add_argument("--model-id", required=True)
    ensemble_evaluation = commands.add_parser(
        "evaluate-ensemble",
        help="Evaluate a fixed equal-weight multi-horizon rank ensemble.",
    )
    ensemble_evaluation.add_argument(
        "--model-id",
        action="append",
        required=True,
        help="One challenger model ID; repeat for horizons 5, 10, 20, and 60.",
    )
    observation = commands.add_parser(
        "observation-log",
        help="Record existing prediction and candidate rankings without trading actions.",
    )
    observation.add_argument("--as-of", required=True, help="Prediction date in YYYYMMDD.")
    shadow_predict = commands.add_parser(
        "shadow-predict",
        help="Publish prospective Champion/challenger/ensemble shadow scores.",
    )
    shadow_predict.add_argument("--as-of", required=True, help="Production session in YYYYMMDD.")
    shadow_status = commands.add_parser(
        "shadow-status", help="Inspect one immutable shadow prediction bundle."
    )
    shadow_status.add_argument("--as-of", required=True, help="Production session in YYYYMMDD.")
    shadow_validate = commands.add_parser(
        "shadow-validate", help="Run read-only shadow readiness validation."
    )
    shadow_validate.add_argument("--as-of", required=True, help="Production session in YYYYMMDD.")


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


def add_research_agent_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the isolated immutable-artifact research agent."""

    parser = subparsers.add_parser(
        "research-agent",
        help="Summarize immutable research artifacts without model or trading access.",
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help="Override the configured reports root.",
    )
    commands = parser.add_subparsers(dest="research_agent_command", required=True)
    for name, help_text in (
        ("generate", "Generate or idempotently reuse one research-agent report."),
        ("validate", "Validate sources and one published research-agent report."),
        ("status", "Inspect one published research-agent report."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--as-of", required=True, help="Research date in YYYYMMDD.")


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
    executable = commands.add_parser(
        "executable-validation",
        help="Compare Champion and Challenger using executable OOS portfolio rules.",
    )
    executable.add_argument(
        "--model-id",
        action="append",
        required=True,
        help="Repeat twice: current champion (or 'champion') and one Challenger model ID.",
    )
    executable.add_argument(
        "--top-n",
        default="10,20,50",
        help="Must contain the fixed fair-comparison variants 10,20,50.",
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
    production = commands.add_parser(
        "production", help="Run the complete daily prediction and research workflow."
    )
    production.add_argument(
        "--as-of",
        default=None,
        help="Completed session in YYYYMMDD; scheduler mode resolves today's ready session.",
    )
    production.add_argument(
        "--dry-run",
        action="store_true",
        help="Acquire the lock and validate/plan without publishing predictions or reports.",
    )
    readiness = commands.add_parser(
        "readiness", help="Run read-only raw, universe, and feature readiness gates."
    )
    readiness.add_argument("--as-of", required=True, help="Completed session in YYYYMMDD.")
    commands.add_parser(
        "full-update",
        help="Run a locked all-dataset refresh and gap repair through the latest session.",
    )


def add_paper_trading_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add broker-free virtual-account commands."""

    parser = subparsers.add_parser(
        "paper-trading", help="Manage isolated append-only virtual portfolios."
    )
    parser.add_argument("--root", default=None, help="Override the paper-trading ledger root.")
    commands = parser.add_subparsers(dest="paper_trading_command", required=True)
    commands.add_parser("init", help="Create configured virtual accounts idempotently.")
    rebalance = commands.add_parser(
        "rebalance", help="Create T+1 target orders from one production summary."
    )
    rebalance.add_argument("--as-of", required=True, help="Signal date in YYYYMMDD.")
    rebalance.add_argument(
        "--production-summary",
        default=None,
        help="Override reports/YYYYMMDD/production_summary.json.",
    )
    execute = commands.add_parser("execute", help="Execute due virtual orders at next open.")
    execute.add_argument("--as-of", required=True, help="Execution date in YYYYMMDD.")
    report = commands.add_parser("report", help="Publish the daily virtual portfolio report.")
    report.add_argument("--as-of", required=True, help="Report date in YYYYMMDD.")


def add_monitor_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add read-only production monitoring commands."""

    parser = subparsers.add_parser(
        "monitor", help="Inspect immutable production and paper-trading artifacts."
    )
    commands = parser.add_subparsers(dest="monitor_command", required=True)
    run = commands.add_parser("run", help="Publish one read-only monitoring snapshot.")
    run.add_argument("--as-of", required=True, help="Production session in YYYYMMDD.")
    observe = commands.add_parser(
        "observe",
        help="Join mature prospective shadow predictions to realized labels.",
    )
    observe.add_argument("--as-of", required=True, help="Maturity cutoff in YYYYMMDD.")
    performance = commands.add_parser(
        "performance",
        help="Aggregate mature prospective observations into monitoring metrics.",
    )
    performance.add_argument("--as-of", required=True, help="Observation cutoff in YYYYMMDD.")
    performance_status = commands.add_parser(
        "performance-status",
        help="Inspect one published performance-monitor artifact.",
    )
    performance_status.add_argument(
        "--as-of", required=True, help="Observation cutoff in YYYYMMDD."
    )
    performance_validate = commands.add_parser(
        "performance-validate",
        help="Validate performance observations without publishing.",
    )
    performance_validate.add_argument(
        "--as-of", required=True, help="Observation cutoff in YYYYMMDD."
    )
    alerts = commands.add_parser(
        "alerts",
        help="Evaluate monitoring metrics and publish alert lifecycle events.",
    )
    alerts.add_argument("--as-of", required=True, help="Monitoring date in YYYYMMDD.")
    alerts_status = commands.add_parser(
        "alerts-status",
        help="Inspect one published alert artifact.",
    )
    alerts_status.add_argument("--as-of", required=True, help="Monitoring date in YYYYMMDD.")
    alerts_validate = commands.add_parser(
        "alerts-validate",
        help="Validate alert inputs without publishing.",
    )
    alerts_validate.add_argument("--as-of", required=True, help="Monitoring date in YYYYMMDD.")


def add_governance_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add read-only production governance commands."""

    parser = subparsers.add_parser(
        "governance", help="Inspect production governance and recovery readiness."
    )
    commands = parser.add_subparsers(dest="governance_command", required=True)
    commands.add_parser("status", help="Publish a read-only governance overview.")
    commands.add_parser(
        "validate-production", help="Validate current production and governance integrity."
    )
    commands.add_parser(
        "validate-recovery", help="Validate registry and interrupted-run recovery inputs."
    )
    recover = commands.add_parser(
        "recover-registry", help="Preview the latest recoverable immutable Registry version."
    )
    recover.add_argument("--dry-run", action="store_true", required=True)


def add_retraining_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add governed, non-training retraining trigger commands."""

    parser = subparsers.add_parser(
        "retraining", help="Evaluate immutable monitoring evidence for retraining needs."
    )
    commands = parser.add_subparsers(dest="retraining_command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate all monitored model horizons.")
    evaluate.add_argument("--as-of", required=True, help="Monitoring date in YYYYMMDD.")
    create = commands.add_parser(
        "create-request", help="Create one manually requested governed training request."
    )
    create.add_argument("--model-id", required=True)
    create.add_argument("--as-of", required=True, help="Monitoring date in YYYYMMDD.")
    commands.add_parser("status", help="List immutable training-request history.")
    validate = commands.add_parser(
        "validate", help="Validate one request or one retrained Challenger."
    )
    validation_identity = validate.add_mutually_exclusive_group(required=True)
    validation_identity.add_argument("--request-id")
    validation_identity.add_argument("--model-id")
    readiness = commands.add_parser(
        "readiness", help="Validate governance before any retraining execution."
    )
    readiness.add_argument("--as-of", required=True, help="Production date in YYYYMMDD.")
    readiness.add_argument(
        "--request-id",
        help="Explicit request identity; required when a date has multiple requests.",
    )
    execute = commands.add_parser(
        "execute", help="Execute one READY request as an immutable Challenger refresh."
    )
    execute.add_argument("--request-id", required=True)
    execution_status = commands.add_parser(
        "execution-status", help="Inspect one governed retraining execution."
    )
    execution_status.add_argument("--run-id", required=True)
    recovery = commands.add_parser(
        "recovery", help="Mark an interrupted execution and clean unpublished staging."
    )
    recovery.add_argument("--run-id", required=True)
    validation_status = commands.add_parser(
        "validation-status", help="Inspect immutable retrained-Challenger validation evidence."
    )
    validation_status.add_argument("--run-id", required=True)
    shadow = commands.add_parser(
        "shadow", help="Publish a validated retrained Challenger prospective shadow sidecar."
    )
    shadow.add_argument("--model-id", required=True)
    shadow.add_argument("--as-of", help="Production date; defaults to latest complete shadow date.")
    shadow_status = commands.add_parser(
        "shadow-status", help="Inspect a retrained Challenger shadow sidecar."
    )
    shadow_status.add_argument("--model-id", required=True)
    shadow_status.add_argument("--as-of", help="Production date; defaults to latest model sidecar.")
    lifecycle_run = commands.add_parser(
        "lifecycle-run", help="Run one governed retrained Challenger lifecycle."
    )
    lifecycle_run.add_argument("--request-id", required=True)
    lifecycle_run.add_argument(
        "--stop-after", choices=("readiness", "training", "validation", "shadow")
    )
    lifecycle_status = commands.add_parser(
        "lifecycle-status", help="Inspect one retrained Challenger lifecycle snapshot."
    )
    lifecycle_status.add_argument("--run-id", required=True)
    lifecycle_resume = commands.add_parser(
        "lifecycle-resume", help="Resume one unambiguous governed lifecycle."
    )
    lifecycle_resume.add_argument("--run-id", required=True)
    lifecycle_recovery = commands.add_parser(
        "lifecycle-recovery", help="Inspect lifecycle recovery state without repair."
    )
    lifecycle_recovery.add_argument("--run-id", required=True)
    lifecycle_revalidate = commands.add_parser(
        "lifecycle-revalidate-evidence",
        help="Revalidate exact lifecycle evidence under the current Promotion Policy.",
    )
    lifecycle_revalidate.add_argument("--run-id", required=True)
    lifecycle_dry_run = commands.add_parser(
        "lifecycle-dry-run", help="Inspect one lifecycle plan without training or mutation."
    )
    lifecycle_dry_run.add_argument("--request-id", required=True)
    lifecycle_dry_run.add_argument("--as-of", help="Optional readiness date in YYYYMMDD.")


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
    maybe_launch_baostock_previous_day_check(
        settings.data.run_baostock_post_ingestion_check,
        args.config,
        settings.paths.data_quality_logs,
    )

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
    if args.models_command == "promotion":
        service = PromotionGovernanceService(
            models_root=output_root,
            reports_root=reports_root,
        )
        try:
            if args.models_promotion_command == "prepare":
                prepared = PromotionEvidenceResolver(
                    models_root=output_root,
                    reports_root=reports_root,
                ).prepare(args.model_id, lifecycle_run_id=args.lifecycle_run_id)
                print(
                    f"promotion_prepared: request_id={prepared.request_id} "
                    f"candidate={prepared.candidate_model_id} "
                    f"cutoff={prepared.evidence_cutoff_date} "
                    f"idempotent={prepared.idempotent} "
                    f"manifest={prepared.evidence_manifest_path}"
                )
                return 0
            if args.models_promotion_command == "rollback-create":
                reason_path = Path(args.reason_file)
                if not reason_path.is_file():
                    raise DataValidationError(f"rollback reason file does not exist: {reason_path}")
                rollback_result = RollbackService(
                    models_root=output_root,
                    settings=settings.promotion,
                ).create(
                    model_id=args.model_id,
                    reason_type=args.reason_type,
                    reason_description=reason_path.read_text(encoding="utf-8"),
                    deployment_slot=args.deployment_slot,
                )
                print(
                    f"rollback_request: request_id={rollback_result.request_id} "
                    f"target={rollback_result.target_model_id} "
                    f"champion={rollback_result.current_champion_model_id} "
                    f"status={rollback_result.status} "
                    f"idempotent={rollback_result.idempotent} "
                    f"output={rollback_result.output_dir}"
                )
                return 0
            if args.models_promotion_command in {"rollback-validate", "rollback-apply"}:
                rollback_service = RollbackService(
                    models_root=output_root,
                    settings=settings.promotion,
                )
                rollback_result = (
                    rollback_service.validate(args.request_id)
                    if args.models_promotion_command == "rollback-validate"
                    else rollback_service.apply(args.request_id)
                )
                print(
                    f"rollback: request_id={rollback_result.request_id} "
                    f"target={rollback_result.target_model_id} "
                    f"status={rollback_result.status} "
                    f"registry_version={rollback_result.registry_version_id} "
                    f"assignment={rollback_result.champion_assignment_id} "
                    f"idempotent={rollback_result.idempotent}"
                )
                return 0
            if args.models_promotion_command == "create":
                promotion_result = service.create(
                    model_id=args.model_id,
                    evidence_cutoff_date=args.evidence_cutoff_date,
                    deployment_slot=args.deployment_slot,
                    evidence_paths=PromotionEvidencePaths(
                        challenger_evaluation=Path(args.challenger_evaluation),
                        executable_validation=Path(args.executable_validation),
                        shadow_prediction=Path(args.shadow_prediction),
                        performance_observation=Path(args.performance_observation),
                        monitoring_summary=Path(args.monitoring_summary),
                        alerts=Path(args.alerts),
                    ),
                )
                print(
                    f"promotion_request: request_id={promotion_result.request_id} "
                    f"candidate={promotion_result.candidate_model_id} "
                    f"champion={promotion_result.champion_model_id} "
                    f"status={promotion_result.status} "
                    f"idempotent={promotion_result.idempotent} "
                    f"output={promotion_result.output_dir}"
                )
                return 0
            if args.models_promotion_command == "validate":
                resolved_evidence = (
                    output_root / "promotion_requests" / args.request_id / "evidence_manifest.json"
                )
                gate_result = PromotionGateEngine(
                    models_root=output_root,
                    reports_root=reports_root,
                    policy=(
                        load_promotion_gate_policy(Path("config/promotion_policy.yaml"))
                        if resolved_evidence.is_file()
                        else PromotionGatePolicy()
                    ),
                ).evaluate(args.request_id)
                print(
                    f"promotion_gate: request_id={gate_result.request_id} "
                    f"candidate={gate_result.candidate_model_id} "
                    f"status={gate_result.status} checks={gate_result.checks} "
                    f"idempotent={gate_result.idempotent} output={gate_result.output_dir}"
                )
                return 1 if gate_result.status == "FAIL" else 0
            if args.models_promotion_command in {"apply", "apply-status"}:
                apply_service = PromotionApplyService(
                    models_root=output_root,
                    reports_root=reports_root,
                )
                if args.models_promotion_command == "apply" and args.dry_run:
                    preview = apply_service.dry_run(args.request_id)
                    print(
                        f"promotion_apply_dry_run: request_id={preview.request_id} "
                        f"current={preview.current_champion_model_id} "
                        f"target={preview.target_champion_model_id} "
                        f"registry_hash={preview.registry_hash}"
                    )
                    for change in preview.registry_changes:
                        print(
                            f"  model={change['model_id']} "
                            f"status={change['from_status']}->{change['to_status']}"
                        )
                    for path in preview.files_affected:
                        print(f"  would_write={path}")
                    return 0
                apply_result = (
                    apply_service.apply(args.request_id)
                    if args.models_promotion_command == "apply"
                    else apply_service.status(args.request_id)
                )
                print(
                    f"promotion_apply: request_id={apply_result.request_id} "
                    f"status={apply_result.status} model_id={apply_result.model_id} "
                    f"previous_champion={apply_result.previous_champion_model_id} "
                    f"registry_version={apply_result.registry_version_id} "
                    f"assignment={apply_result.champion_assignment_id} "
                    f"idempotent={apply_result.idempotent} "
                    f"manifest={apply_result.manifest_path}"
                )
                return 0 if apply_result.status == "PROMOTED" else 1
            if args.models_promotion_command in {
                "review",
                "approve",
                "reject",
                "review-status",
            }:
                is_rollback = (
                    output_root / "rollback_requests" / args.request_id / "manifest.json"
                ).is_file()
                if is_rollback:
                    rollback_review = RollbackService(
                        models_root=output_root,
                        settings=settings.promotion,
                    )
                    if args.models_promotion_command == "review":
                        rollback_result = rollback_review.review(args.request_id)
                    elif args.models_promotion_command == "review-status":
                        rollback_result = rollback_review.status(args.request_id)
                    else:
                        comments_path = Path(args.comments_file)
                        if not comments_path.is_file():
                            raise DataValidationError(
                                f"review comments file does not exist: {comments_path}"
                            )
                        comments = comments_path.read_text(encoding="utf-8")
                        rollback_result = (
                            rollback_review.approve(args.request_id, comments)
                            if args.models_promotion_command == "approve"
                            else rollback_review.reject(args.request_id, comments)
                        )
                    print(
                        f"rollback_review: request_id={rollback_result.request_id} "
                        f"status={rollback_result.status} "
                        f"event_id={rollback_result.event_id} "
                        f"idempotent={rollback_result.idempotent}"
                    )
                    return 1 if rollback_result.status in {"INVALID", "APPROVAL_EXPIRED"} else 0
                review_service = HumanReviewService(
                    models_root=output_root,
                    reports_root=reports_root,
                    settings=settings.promotion,
                )
                if args.models_promotion_command == "review":
                    review_result = review_service.review(args.request_id)
                elif args.models_promotion_command == "review-status":
                    review_result = review_service.status(args.request_id)
                else:
                    comments_path = Path(args.comments_file)
                    if not comments_path.is_file():
                        raise DataValidationError(
                            f"review comments file does not exist: {comments_path}"
                        )
                    comments = comments_path.read_text(encoding="utf-8")
                    review_result = (
                        review_service.approve(args.request_id, comments)
                        if args.models_promotion_command == "approve"
                        else review_service.reject(args.request_id, comments)
                    )
                print(
                    f"promotion_review: request_id={review_result.request_id} "
                    f"status={review_result.status} reviewer={review_result.reviewer} "
                    f"requester={review_result.requester} "
                    f"event_id={review_result.event_id} "
                    f"idempotent={review_result.idempotent}"
                )
                return 1 if review_result.status in {"INVALID", "APPROVAL_EXPIRED"} else 0
            status_result = service.status(args.request_id)
            print(
                f"promotion_status: request_id={status_result.request_id} "
                f"status={status_result.status} output={status_result.output_dir}"
            )
            return 0 if status_result.status == "complete" else 1
        except (DataValidationError, OSError, ProductionLockError, ValueError) as error:
            print(f"promotion governance failed: {error}", file=sys.stderr)
            return 2
    if args.models_command in {"shadow-predict", "shadow-status", "shadow-validate"}:
        shadow = ShadowPredictionService(
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
            registry=ModelRegistry(output_root),
            processed_root=processed_root,
            reports_root=reports_root,
        )
        try:
            if args.models_command == "shadow-predict":
                result = shadow.predict(args.as_of)
                print(
                    f"shadow_predictions: as_of={result.as_of} "
                    f"production_run_id={result.production_run_id} "
                    f"shadow_run_id={result.shadow_run_id} rows={result.prediction_rows} "
                    f"models={result.model_count} idempotent={result.idempotent} "
                    f"output={result.output_dir}"
                )
                return 0
            if args.models_command == "shadow-status":
                status = shadow.status(args.as_of)
                print(
                    f"shadow_status: as_of={status['as_of']} status={status['status']} "
                    f"shadow_run_id={status['shadow_run_id']} "
                    f"rows={status['prediction_rows']} output={status['output']}"
                )
                return 0 if status["status"] == "complete" else 1
            ready, failures, checks = shadow.validate(args.as_of)
            print(
                f"shadow_readiness: as_of={args.as_of} "
                f"status={'READY' if ready else 'NOT_READY'} checks={checks}"
            )
            for failure in failures:
                print(f"  failure: {failure}")
            return 0 if ready else 1
        except (DataValidationError, OSError, ValueError) as error:
            print(f"shadow prediction failed: {error}", file=sys.stderr)
            return 2
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
    if args.models_command == "train-challenger":
        challenger_trainer = ChallengerTrainer(
            processed_root=processed_root,
            models_root=output_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            challenger_results = challenger_trainer.train(
                experiment_id=args.experiment_id,
                all_horizons=args.all_horizons,
                experiment_manifest=(
                    None if args.experiment_manifest is None else Path(args.experiment_manifest)
                ),
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"challenger training failed: {error}", file=sys.stderr)
            return 2
        for challenger_result in challenger_results:
            print(
                f"challenger_trained: model_id={challenger_result.model_id} "
                f"horizon={challenger_result.horizon} "
                f"train_rows={challenger_result.training_rows} "
                f"validation_rows={challenger_result.validation_rows} "
                f"validation_rank_ic={challenger_result.validation_rank_ic:.6f} "
                f"status=candidate output={challenger_result.output_dir}"
            )
        return 0
    if args.models_command == "predict-challenger":
        prediction_engine = ChallengerPredictionEngine(
            registry=ModelRegistry(output_root),
            processed_root=processed_root,
            reports_root=reports_root,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            prediction_result = prediction_engine.predict(args.model_id)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"challenger prediction failed: {error}", file=sys.stderr)
            return 2
        print(
            f"challenger_predictions: model_id={prediction_result.model_id} "
            f"horizon={prediction_result.horizon} rows={prediction_result.prediction_rows} "
            f"dates={prediction_result.prediction_dates} "
            f"output={prediction_result.output_dir}"
        )
        return 0
    if args.models_command == "evaluate-challenger":
        evaluation_engine = ChallengerEvaluationEngine(
            registry=ModelRegistry(output_root),
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            evaluation_result = evaluation_engine.evaluate(args.model_id)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"challenger evaluation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"challenger_evaluation: run_id={evaluation_result.run_id} "
            f"champion={evaluation_result.champion_model_id} "
            f"challenger={evaluation_result.challenger_model_id} "
            f"horizon={evaluation_result.horizon} "
            f"manual_review={evaluation_result.eligible_for_manual_review} "
            f"output={evaluation_result.output_dir}"
        )
        return 0
    if args.models_command == "evaluate-ensemble":
        ensemble_engine = MultiHorizonEnsembleEngine(
            registry=ModelRegistry(output_root),
            processed_root=processed_root,
            reports_root=reports_root,
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            ensemble_result = ensemble_engine.evaluate(args.model_id)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"ensemble evaluation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"ensemble_evaluation: run_id={ensemble_result.run_id} "
            f"models={','.join(ensemble_result.model_ids)} "
            f"rows={ensemble_result.prediction_rows} "
            f"dates={ensemble_result.prediction_dates} "
            f"output={ensemble_result.output_dir}"
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


def run_research_agent_command(args: argparse.Namespace) -> int:
    """Run one isolated research-agent operation."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    reports_root = settings.paths.reports if args.reports_root is None else Path(args.reports_root)
    service = ResearchAgentService(
        settings=settings.research.agent,
        config_path=Path(effective_config_path(args.config)),
        reports_root=reports_root,
    )
    try:
        if args.research_agent_command == "generate":
            result = service.generate(args.as_of)
            print(
                f"research_agent: as_of={result.as_of} mode={result.generation_mode} "
                f"run_id={result.run_id} idempotent={result.idempotent} "
                f"output={result.output_dir}"
            )
            return 0
        validation = (
            service.validate(args.as_of)
            if args.research_agent_command == "validate"
            else service.status(args.as_of)
        )
    except (DataValidationError, OSError, ValueError) as error:
        print(f"research agent failed: {error}", file=sys.stderr)
        return 2
    stream = sys.stdout if validation.valid else sys.stderr
    print(
        f"research_agent_{args.research_agent_command}: as_of={validation.as_of} "
        f"valid={validation.valid} exists={validation.exists} "
        f"mode={validation.generation_mode} error={validation.error}",
        file=stream,
    )
    return 0 if validation.valid else 2


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
    if args.backtest_command == "executable-validation":
        validation_engine = ExecutableOOSValidationEngine(
            raw_root=raw_root,
            processed_root=processed_root,
            models_root=models_root,
            reports_root=(
                settings.paths.reports if args.output_root is None else Path(args.output_root)
            ),
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            validation_result = validation_engine.run(
                args.model_id,
                top_n=parse_top_n(args.top_n),
            )
        except (DataValidationError, ValueError) as error:
            print(f"executable validation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"executable_validation: run_id={validation_result.run_id} "
            f"champion={validation_result.champion_model_id} "
            f"challenger={validation_result.challenger_model_id} "
            f"horizon={validation_result.horizon} output={validation_result.output_dir}"
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

    if args.pipeline_command == "production":
        models_root = settings.paths.models
        reports_root = settings.paths.reports
        pipeline = ProductionPipeline(
            config_path=config_path,
            processed_root=settings.paths.processed_data,
            reports_root=reports_root,
            daily_executor=ProductionDailyStageExecutor(
                settings=settings,
                config_path=config_path,
                raw_store=raw_store,
                universe_store=universe_store,
                feature_store=feature_store,
            ),
            daily_stages=DailyPipelineStages(),
            readiness=freshness,
            readiness_executor=lambda gate, as_of: execute_readiness_gate(freshness, gate, as_of),
            as_of_resolver=lambda requested: resolve_completed_trading_date(raw_store, requested),
            inference=ProductionInferenceEngine(
                registry=ModelRegistry(models_root),
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
                config_path=config_path,
                freshness=freshness,
            ),
            candidates=CandidateSelector(
                raw_root=settings.paths.parquet_store,
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
                config_path=config_path,
                settings=settings.strategy.candidate_selection,
            ),
            research_report=DailyResearchReportGenerator(
                raw_root=settings.paths.parquet_store,
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
                settings=settings.research.daily_report,
            ),
            explainability=ExplainabilityEngine(
                registry=ModelRegistry(models_root),
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
                settings=settings.research.explainability,
            ),
            decision_support=InvestmentDecisionSupport(
                raw_root=settings.paths.parquet_store,
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
                settings=settings.research.decision_support,
            ),
            observation=ProductionObservationRecorder(reports_root),
            paper_trading=PaperTradingService(
                settings=settings,
                config_path=config_path,
                registry=ModelRegistry(models_root),
                raw_root=settings.paths.parquet_store,
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
            )
            if settings.paper_trading.enabled
            else None,
            shadow_prediction=ShadowPredictionService(
                settings=settings,
                config_path=config_path,
                registry=ModelRegistry(models_root),
                processed_root=settings.paths.processed_data,
                reports_root=reports_root,
            ),
            monitoring=MonitoringService(
                settings=settings,
                config_path=config_path,
                reports_root=reports_root,
                paper_root=settings.paths.paper_trading,
            ),
            retraining=RetrainingTriggerService(
                reports_root=reports_root,
                config_path=config_path,
                policy_path=_retraining_policy_path(config_path),
                promotion_policy_path=_promotion_policy_path(config_path),
            ),
            research_agent=ResearchAgentService(
                settings=settings.research.agent,
                config_path=config_path,
                reports_root=reports_root,
            ),
            governance_snapshot=DailyGovernanceSnapshotService(
                settings=settings,
                config_path=config_path,
                project_root=(
                    config_path.resolve().parent.parent
                    if config_path.resolve().parent.name == "config"
                    else Path.cwd().resolve()
                ),
                promotion_policy_path=_promotion_policy_path(config_path),
            ),
        )
        scheduler = ProductionScheduler(
            settings=settings,
            raw_store=raw_store,
            pipeline=pipeline,
            reports_root=reports_root,
        )
        try:
            scheduler_result = scheduler.run(args.as_of, dry_run=args.dry_run)
        except ProductionLockError as error:
            print(f"production run blocked: {error}", file=sys.stderr)
            return 3
        if scheduler_result.status == "skipped":
            print(
                "pipeline_production: status=skipped "
                f"as_of={scheduler_result.resolved_as_of} "
                f"reason={scheduler_result.skipped_reason} "
                f"invocation={scheduler_result.invocation_manifest}"
            )
            return 0
        if scheduler_result.status == "success":
            print(
                "pipeline_production: status=success "
                f"run_id={scheduler_result.pipeline_run_id} "
                f"as_of={scheduler_result.resolved_as_of} dry_run={args.dry_run} "
                f"invocation={scheduler_result.invocation_manifest}"
            )
            return 0
        print(
            f"pipeline_production: status=failed run_id={scheduler_result.pipeline_run_id} "
            f"as_of={scheduler_result.resolved_as_of} "
            f"message={scheduler_result.error_message} "
            f"invocation={scheduler_result.invocation_manifest}",
            file=sys.stderr,
        )
        return scheduler_result.exit_code or 2

    if args.pipeline_command == "full-update":
        updater = FullDataUpdateScheduler(
            settings=settings,
            config_path=config_path,
            raw_store=raw_store,
        )
        try:
            update_result = updater.run()
        except ProductionLockError as error:
            print(f"full update blocked: {error}", file=sys.stderr)
            return 3
        stream = sys.stdout if update_result.status == "success" else sys.stderr
        print(
            f"pipeline_full_update: status={update_result.status} "
            f"as_of={update_result.resolved_as_of} run_id={update_result.pipeline_run_id} "
            f"message={update_result.error_message} "
            f"invocation={update_result.invocation_manifest}",
            file=stream,
        )
        return update_result.exit_code

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
        daily_result = orchestrator.run(args.as_of)
    except ProductionLockError as error:
        print(f"production run blocked: {error}", file=sys.stderr)
        return 3
    if daily_result.status == "success":
        print(
            f"pipeline_daily: status=success run_id={daily_result.run.run_id} "
            f"as_of={daily_result.as_of} manifest={daily_result.run.manifest_path}"
        )
        return 0
    print(
        f"pipeline_daily: status=failed run_id={daily_result.run.run_id} "
        f"as_of={daily_result.as_of} failed_stage={daily_result.failed_stage} "
        f"message={daily_result.error_message} manifest={daily_result.run.manifest_path}",
        file=sys.stderr,
    )
    return daily_result.exit_code or 2


def run_paper_trading_command(args: argparse.Namespace) -> int:
    """Run one paper-trading operation under the repository production lock."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    service = PaperTradingService(
        settings=settings,
        config_path=Path(effective_config_path(args.config)),
        registry=ModelRegistry(settings.paths.models),
        raw_root=settings.paths.parquet_store,
        processed_root=settings.paths.processed_data,
        reports_root=settings.paths.reports,
        paper_root=None if args.root is None else Path(args.root),
    )

    def operation() -> int:
        try:
            if args.paper_trading_command == "init":
                init_result = service.init()
                print(
                    f"paper_trading_init: accounts={init_result.account_count} "
                    f"created={init_result.created_count} root={init_result.root}"
                )
                return 0
            if args.paper_trading_command == "rebalance":
                rebalance_result = service.rebalance(
                    args.as_of,
                    production_summary_path=(
                        None if args.production_summary is None else Path(args.production_summary)
                    ),
                )
                print(
                    f"paper_trading_rebalance: as_of={rebalance_result.as_of} "
                    f"execution_rule={rebalance_result.execution_rule} "
                    f"orders_written={rebalance_result.orders_written} "
                    f"root={rebalance_result.root}"
                )
                return 0
            if args.paper_trading_command == "execute":
                execution_result = service.execute(args.as_of)
                print(
                    f"paper_trading_execute: as_of={execution_result.as_of} "
                    f"trades_written={execution_result.trades_written} "
                    f"equity_rows_written={execution_result.equity_rows_written} "
                    f"root={execution_result.root}"
                )
                return 0
            if args.paper_trading_command == "report":
                report_result = service.report(args.as_of)
                print(
                    f"paper_trading_report: as_of={report_result.as_of} "
                    f"portfolios={report_result.portfolio_count} "
                    f"output={report_result.report_path}"
                )
                return 0
            raise ValueError(f"Unsupported paper-trading command: {args.paper_trading_command}")
        except (DataValidationError, OSError, ValueError) as error:
            print(f"paper trading failed: {error}", file=sys.stderr)
            return 2

    return run_production_cli_command(
        operation,
        command=f"ashare-quant paper-trading {args.paper_trading_command}",
    )


def run_monitor_command(args: argparse.Namespace) -> int:
    """Run read-only monitoring without acquiring or modifying trading state."""

    settings = load_settings(args.config)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    if args.monitor_command == "observe":
        observation_service = PerformanceObservationService(
            raw_root=settings.paths.parquet_store,
            processed_root=settings.paths.processed_data,
            reports_root=settings.paths.reports,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            observation_result = observation_service.run(args.as_of)
        except (DataValidationError, OSError, ValueError) as error:
            print(f"performance observation failed: {error}", file=sys.stderr)
            return 2
        print(
            f"performance_observation: as_of={observation_result.observation_as_of} "
            f"rows={observation_result.observation_rows} "
            f"available={observation_result.available_rows} "
            f"idempotent={observation_result.idempotent} "
            f"output={observation_result.output_dir}"
        )
        return 0
    if args.monitor_command in {
        "performance",
        "performance-status",
        "performance-validate",
    }:
        performance_service = PerformanceMonitoringService(
            reports_root=settings.paths.reports,
            config_path=Path(effective_config_path(args.config)),
        )
        try:
            if args.monitor_command == "performance":
                performance_result = performance_service.run(args.as_of)
                print(
                    f"performance_monitor: as_of={performance_result.as_of} "
                    f"models={performance_result.model_count} "
                    f"observations={performance_result.observation_rows} "
                    f"idempotent={performance_result.idempotent} "
                    f"output={performance_result.output_dir}"
                )
                return 0
            validation = (
                performance_service.status(args.as_of)
                if args.monitor_command == "performance-status"
                else performance_service.validate(args.as_of)
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"performance monitor failed: {error}", file=sys.stderr)
            return 2
        stream = sys.stdout if validation.valid else sys.stderr
        print(
            f"performance_monitor_{args.monitor_command.rsplit('-', 1)[-1]}: "
            f"as_of={validation.as_of} valid={validation.valid} "
            f"exists={validation.exists} models={validation.model_count} "
            f"observations={validation.observation_rows} error={validation.error}",
            file=stream,
        )
        return 0 if validation.valid else 2
    if args.monitor_command in {"alerts", "alerts-status", "alerts-validate"}:
        alert_service = AlertService(
            settings=settings,
            config_path=Path(effective_config_path(args.config)),
            reports_root=settings.paths.reports,
        )
        try:
            if args.monitor_command == "alerts":
                alert_result = alert_service.run(args.as_of)
                print(
                    f"monitor_alerts: as_of={alert_result.as_of} "
                    f"alerts={alert_result.alert_count} "
                    f"critical={alert_result.critical_count} "
                    f"idempotent={alert_result.idempotent} "
                    f"output={alert_result.output_dir}"
                )
                return 0
            alert_validation = (
                alert_service.status(args.as_of)
                if args.monitor_command == "alerts-status"
                else alert_service.validate(args.as_of)
            )
        except (DataValidationError, OSError, ValueError) as error:
            print(f"alert evaluation failed: {error}", file=sys.stderr)
            return 2
        stream = sys.stdout if alert_validation.valid else sys.stderr
        print(
            f"monitor_{args.monitor_command.replace('-', '_')}: "
            f"as_of={alert_validation.as_of} valid={alert_validation.valid} "
            f"exists={alert_validation.exists} alerts={alert_validation.alert_count} "
            f"error={alert_validation.error}",
            file=stream,
        )
        return 0 if alert_validation.valid else 2
    if args.monitor_command != "run":
        raise ValueError(f"Unsupported monitor command: {args.monitor_command}")
    monitoring_service = MonitoringService(
        settings=settings,
        config_path=Path(effective_config_path(args.config)),
    )
    try:
        result = monitoring_service.run(args.as_of)
    except (DataValidationError, OSError, ValueError) as error:
        print(f"monitoring failed: {error}", file=sys.stderr)
        return 2
    print(
        f"monitor_run: as_of={result.as_of} run_id={result.run_id} "
        f"predictions={result.prediction_count} portfolios={result.portfolio_count} "
        f"performance_models={result.performance_model_count} "
        f"alerts={result.alert_count} "
        f"output={result.output_dir}"
    )
    return 0


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


def maybe_launch_baostock_previous_day_check(
    enabled: bool,
    config_path: str | None,
    log_root: object,
) -> None:
    """Launch the optional cross-provider check only when explicitly enabled."""

    if not enabled:
        LOGGER.info("post-ingestion baostock validation disabled")
        return
    launch_baostock_previous_day_check(config_path, log_root)


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


def run_governance_command(args: argparse.Namespace) -> int:
    """Run read-only governance status or validation operations."""

    settings = load_settings(args.config)
    config_path = Path(effective_config_path(args.config))
    service = GovernanceService(settings=settings, config_path=config_path)
    try:
        if args.governance_command == "status":
            result = service.status()
            _print_governance_status(result.report.summary)
            print(f"report: {result.report_path}")
            return 0
        if args.governance_command == "recover-registry":
            preview = service.recover_registry_dry_run()
            print("Registry Recovery Dry Run")
            print(f"latest_valid_registry: {preview.latest_valid_registry}")
            print(f"registry_hash: {preview.registry_hash}")
            print(f"champion_model_id: {preview.champion_model_id}")
            print(f"champion_assignment_id: {preview.champion_assignment_id}")
            print(f"registry_version_id: {preview.registry_version_id}")
            print(f"transition_manifest: {preview.transition_manifest}")
            print(f"corrupted_versions: {list(preview.corrupted_versions)}")
            return 0
        if args.governance_command == "validate-production":
            result = service.validate_production()
        elif args.governance_command == "validate-recovery":
            result = service.validate_recovery()
        else:
            raise ValueError(f"unsupported governance command: {args.governance_command}")
    except (DataValidationError, OSError, ValueError) as error:
        print(f"governance {args.governance_command} failed: {error}", file=sys.stderr)
        return 2
    print(f"{result.report.status}: {result.report.artifact_name}")
    for check in result.report.checks:
        print(f"  {check.status} {check.name}: {check.message}")
    print(f"report: {result.report_path}")
    return 1 if result.report.status == "FAIL" else 0


def run_retraining_command(args: argparse.Namespace) -> int:
    """Evaluate or validate governed training requests without training models."""

    settings = load_settings(args.config)
    config_path = Path(effective_config_path(args.config))
    try:
        if args.retraining_command in {
            "lifecycle-run",
            "lifecycle-status",
            "lifecycle-resume",
            "lifecycle-recovery",
            "lifecycle-revalidate-evidence",
            "lifecycle-dry-run",
        }:
            lifecycle = RetrainingLifecycleOrchestrator(
                settings=settings,
                config_path=config_path,
                retraining_policy_path=_retraining_policy_path(config_path),
                promotion_policy_path=_promotion_policy_path(config_path),
            )
            if args.retraining_command == "lifecycle-status":
                status = lifecycle.status(args.run_id)
                print(json.dumps(status, sort_keys=True, default=str))
                return 0 if status.get("status") != "MISSING" else 1
            if args.retraining_command == "lifecycle-recovery":
                recovery = lifecycle.recovery(args.run_id)
                print(
                    json.dumps(
                        {
                            "lifecycle_run_id": recovery.lifecycle_run_id,
                            "status": recovery.status,
                            "current_state": recovery.current_state,
                            "complete": recovery.complete,
                            "staging_paths": recovery.staging_paths,
                            "message": recovery.message,
                        },
                        sort_keys=True,
                    )
                )
                return 0 if recovery.status == "CLEAN" else 1
            if args.retraining_command == "lifecycle-dry-run":
                dry_run = LifecycleDryRunService(lifecycle).run(args.request_id, as_of=args.as_of)
                print(
                    f"retraining_lifecycle_dry_run: dry_run_id={dry_run.dry_run_id} "
                    f"status={dry_run.status} output={dry_run.output_dir} "
                    f"idempotent={dry_run.idempotent}"
                )
                return 0 if dry_run.status == "READY_TO_EXECUTE" else 1
            if args.retraining_command == "lifecycle-revalidate-evidence":
                result = lifecycle.revalidate_evidence(args.run_id)
                print(
                    f"retraining_lifecycle_evidence: run_id={result.lifecycle_run_id} "
                    f"state={result.current_state} output={result.output_dir}"
                )
                return 0 if result.current_state == "EVIDENCE_READY" else 1
            result = (
                lifecycle.resume(args.run_id)
                if args.retraining_command == "lifecycle-resume"
                else lifecycle.run(args.request_id, stop_after=args.stop_after)
            )
            print(
                f"retraining_lifecycle: run_id={result.lifecycle_run_id} "
                f"request_id={result.request_id} state={result.current_state} "
                f"model_id={result.model_id} output={result.output_dir} "
                f"idempotent={result.idempotent}"
            )
            return (
                1
                if result.current_state.endswith("_FAILED")
                or result.current_state in {"FAILED", "CANCELLED"}
                else 0
            )
        if args.retraining_command in {"shadow", "shadow-status"}:
            shadow = RetrainedChallengerShadowService(
                settings=settings,
                config_path=config_path,
            )
            if args.retraining_command == "shadow-status":
                shadow_status = shadow.status(args.model_id, as_of=args.as_of)
                print(json.dumps(shadow_status, sort_keys=True))
                return 0 if shadow_status["status"] == "complete" else 1
            shadow_result = shadow.predict(args.model_id, as_of=args.as_of)
            print(
                f"retraining_shadow: model_id={shadow_result.model_id} "
                f"as_of={shadow_result.as_of} shadow_run_id={shadow_result.shadow_run_id} "
                f"rows={shadow_result.prediction_rows} output={shadow_result.output_dir} "
                f"idempotent={shadow_result.idempotent}"
            )
            return 0
        if args.retraining_command == "validation-status" or (
            args.retraining_command == "validate" and args.model_id is not None
        ):
            validation = RetrainingValidationService(
                settings=settings,
                config_path=config_path,
            )
            if args.retraining_command == "validation-status":
                print(json.dumps(validation.status(args.run_id), sort_keys=True))
                return 0
            validation_result = validation.validate(args.model_id)
            print(
                f"retraining_validation: status={validation_result.status} "
                f"run_id={validation_result.run_id} model_id={validation_result.model_id} "
                f"promotion_ready={validation_result.promotion_ready} "
                f"output={validation_result.output_dir} "
                f"idempotent={validation_result.idempotent}"
            )
            return 0
        if args.retraining_command in {"execute", "execution-status", "recovery"}:
            execution = GovernedRetrainingExecutionService(
                settings=settings,
                config_path=config_path,
                retraining_policy_path=_retraining_policy_path(config_path),
                promotion_policy_path=_promotion_policy_path(config_path),
            )
            if args.retraining_command == "execute":
                execution_result = execution.execute(args.request_id)
                print(
                    f"retraining_execution: status={execution_result.status} "
                    f"run_id={execution_result.training_run_id} "
                    f"model_id={execution_result.model_id} "
                    f"output={execution_result.output_dir} "
                    f"idempotent={execution_result.idempotent}"
                )
                return 0
            if args.retraining_command == "execution-status":
                print(json.dumps(execution.status(args.run_id), sort_keys=True))
                return 0
            recovery_result = execution.recovery(args.run_id)
            print(
                f"retraining_recovery: run_id={recovery_result.training_run_id} "
                f"status={recovery_result.status} retry_allowed={recovery_result.retry_allowed} "
                f"staging_paths={len(recovery_result.staging_paths)}"
            )
            return 0
        service = RetrainingTriggerService(
            reports_root=settings.paths.reports,
            config_path=config_path,
            policy_path=_retraining_policy_path(config_path),
            promotion_policy_path=_promotion_policy_path(config_path),
        )
        if args.retraining_command == "readiness":
            project_root = (
                config_path.resolve().parent.parent
                if config_path.resolve().parent.name == "config"
                else Path.cwd().resolve()
            )
            readiness_result = RetrainingExecutionReadinessValidator(
                settings=settings,
                config_path=config_path,
                project_root=project_root,
                retraining_policy_path=_retraining_policy_path(config_path),
                promotion_policy_path=_promotion_policy_path(config_path),
            ).validate(args.as_of, request_id=args.request_id)
            print(
                f"retraining_readiness: status={readiness_result.report.status} "
                f"as_of={readiness_result.report.as_of} "
                f"request_id={readiness_result.report.request_id} "
                f"output={readiness_result.output_dir}"
            )
            for check in readiness_result.report.check_details:
                print(f"  {check.status} {check.name}: {check.message}")
            return 0 if readiness_result.report.status == "READY" else 1
        if args.retraining_command == "evaluate":
            evaluation_result = service.evaluate(args.as_of)
            print(
                json.dumps(
                    {
                        "status": (
                            "TRIGGERED"
                            if evaluation_result.triggered_count
                            else "NO_ACTION_REQUIRED"
                        ),
                        "as_of": evaluation_result.as_of,
                        "triggered_count": evaluation_result.triggered_count,
                        "decisions": [
                            {
                                "model_id": item.model_id,
                                "model_role": item.model_role,
                                "horizon": item.horizon,
                                "status": item.status,
                                "reasons": list(item.reasons),
                                "request_id": item.request_id,
                            }
                            for item in evaluation_result.decisions
                        ],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.retraining_command == "create-request":
            manual_result = service.create_request(model_id=args.model_id, as_of=args.as_of)
            item = manual_result.decisions[0]
            print(
                f"retraining_request: status={item.status} model_id={item.model_id} "
                f"horizon={item.horizon} request_id={item.request_id}"
            )
            return 0
        if args.retraining_command == "status":
            rows = service.status()
            print(json.dumps({"requests": rows}, sort_keys=True, default=str))
            return 0
        if args.retraining_command == "validate":
            assert args.request_id is not None
            request_validation_result = service.validate(args.request_id)
            stream = sys.stdout if request_validation_result.valid else sys.stderr
            print(
                f"retraining_validation: request_id={request_validation_result.request_id} "
                f"valid={request_validation_result.valid} "
                f"status={request_validation_result.status} "
                f"error={request_validation_result.error}",
                file=stream,
            )
            return 0 if request_validation_result.valid else 2
    except (DataValidationError, OSError, ProductionLockError, ValueError) as error:
        print(f"retraining {args.retraining_command} failed: {error}", file=sys.stderr)
        return 2
    raise ValueError(f"unsupported retraining command: {args.retraining_command}")


def _retraining_policy_path(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return (
        resolved.parent / "retraining_policy.yaml"
        if resolved.parent.name == "config"
        else Path("config/retraining_policy.yaml")
    )


def _promotion_policy_path(config_path: Path) -> Path:
    resolved = config_path.resolve()
    candidate = resolved.parent / "promotion_policy.yaml"
    return candidate if candidate.is_file() else Path("config/promotion_policy.yaml")


def _print_governance_status(summary: dict[str, object]) -> None:
    """Render the stable operator-facing governance overview."""

    labels = (
        ("Production", "production"),
        ("Champion", "champion"),
        ("Monitoring", "monitoring"),
        ("Paper Trading", "paper_trading"),
        ("Promotion", "promotion"),
        ("Rollback", "rollback"),
        ("Research Agent", "research_agent"),
    )
    print("Governance Status")
    for title, key in labels:
        print(f"\n{title}:")
        payload = summary.get(key)
        if isinstance(payload, dict):
            for name, value in payload.items():
                print(f"{name}: {value}")


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
    if args.command == "research-agent":
        return run_research_agent_command(args)
    if args.command == "backtest":
        return run_backtest_command(args)
    if args.command == "paper-trading":
        return run_paper_trading_command(args)
    if args.command == "monitor":
        return run_monitor_command(args)
    if args.command == "governance":
        return run_governance_command(args)
    if args.command == "retraining":
        return run_retraining_command(args)
    if args.command == "pipeline":
        return run_pipeline_command(args)
    raise ValueError(f"Unsupported command: {args.command}")
