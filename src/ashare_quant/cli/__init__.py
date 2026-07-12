"""Command-line entry points for pipeline operations."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ashare_quant.config import load_settings
from ashare_quant.data.datasets import ALL_DATASETS, DEFAULT_DATASETS
from ashare_quant.data.exceptions import DataIngestionError, DataValidationError
from ashare_quant.data.ingestion import DataIngestionService, GapReport, build_store
from ashare_quant.data.quality_logging import append_quality_event, append_validation_results
from ashare_quant.data.validation import DataValidator, ValidationResult
from ashare_quant.features import FEATURE_REGISTRY, FeatureBuilder, FeatureStore
from ashare_quant.labels import LabelBuilder, LabelStore, LabelValidator
from ashare_quant.labels.validation import LabelValidationResult
from ashare_quant.universe import UniverseBuilder, UniverseStore, UniverseValidator
from ashare_quant.universe.validation import UniverseValidationResult
from ashare_quant.utils import configure_logging
from ashare_quant.utils.manifest import (
    artifact_manifest_status,
    processed_source_fingerprint,
    raw_source_fingerprints,
    utc_now_iso,
    write_build_manifest,
)

LOGGER = logging.getLogger(__name__)


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
        help="Comma-separated trading-day horizons, for example 3,5,10.",
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

    features_subparsers.add_parser("registry", help="Show registered feature metadata summary.")


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
            write_build_manifest(
                universe_store.dataset_dir,
                artifact_name="universe_daily",
                build_started_at=build_started_at,
                config_path=effective_config_path(args.config),
                start_date=build_result.start_date,
                end_date=build_result.end_date,
                row_count=build_result.rows_written,
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
        universe_status = universe_store.status()
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
            rows=universe_status.rows,
            partitions=universe_status.partitions,
            min_date=universe_status.min_date,
            max_date=universe_status.max_date,
        )
        write_build_manifest(
            feature_store.dataset_dir,
            artifact_name="features_daily",
            build_started_at=build_started_at,
            config_path=effective_config_path(args.config),
            start_date=result.start_date,
            end_date=result.end_date,
            row_count=result.rows_written,
            source_fingerprints=source_fingerprints,
            extra={"feature_count": result.feature_count},
        )
        return 0

    raise ValueError(f"Unsupported features command: {args.features_command}")


def parse_horizons(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integer horizons."""

    horizons = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f"horizons must be positive integers: {value}")
    return horizons


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
            f"start_date={report.start_date} end_date={report.end_date} "
            f"message={report.message}"
        )
        if report.missing_by_entity:
            for entity, dates in sorted(report.missing_by_entity.items()):
                preview = ",".join(dates[:10])
                print(f"  {entity}: missing={len(dates)} first={preview}")
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
    raise ValueError(f"Unsupported command: {args.command}")
