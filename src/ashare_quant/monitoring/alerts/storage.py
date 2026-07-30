"""Validated monitoring inputs and transactional alert publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.alerts.evaluator import deterministic_alert_id
from ashare_quant.monitoring.alerts.schemas import (
    AlertBuild,
    AlertSeverity,
    AlertState,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame

ALERT_HISTORY_COLUMNS: tuple[str, ...] = (
    "alert_id",
    "alert_type",
    "severity",
    "status",
    "first_seen",
    "last_seen",
    "model_id",
    "portfolio_id",
    "metric_name",
    "metric_value",
    "threshold",
    "source_artifact_hash",
    "created_at",
)


def load_monitor_metrics(
    reports_root: Path,
    as_of: str,
) -> tuple[dict[str, Any], DataFrame, DataFrame, dict[str, str]]:
    """Load and hash only health, performance, and portfolio monitoring outputs."""

    root = reports_root / "model_monitor" / as_of
    monitor_manifest = _load_json(root / "manifest.json", "monitor manifest")
    if (
        monitor_manifest.get("schema_version") != 1
        or monitor_manifest.get("artifact_name") != "production_monitor_manifest"
        or str(monitor_manifest.get("as_of")) != as_of
    ):
        raise DataValidationError("invalid production monitor manifest identity")
    health_path = root / "health.json"
    performance_path = root / "performance" / "performance_metrics.parquet"
    performance_manifest_path = root / "performance" / "manifest.json"
    portfolio_path = root / "portfolio_metrics.parquet"
    expected = monitor_manifest.get("monitor_metric_file_hashes")
    if not isinstance(expected, dict):
        raise DataValidationError("monitor manifest lacks metric file hashes")
    paths = {
        "health": health_path,
        "performance_metrics": performance_path,
        "performance_manifest": performance_manifest_path,
        "portfolio_metrics": portfolio_path,
    }
    for name, path in paths.items():
        if file_sha256(path) != expected.get(name):
            raise DataValidationError(f"monitor metric source hash mismatch: {name}")
    performance_manifest = _load_json(performance_manifest_path, "performance manifest")
    if (
        performance_manifest.get("schema_version") != 1
        or performance_manifest.get("artifact_name") != "performance_monitor"
        or performance_manifest.get("status") != "success"
        or str(performance_manifest.get("as_of")) != as_of
    ):
        raise DataValidationError("invalid performance monitor manifest identity")
    if file_sha256(performance_path) != performance_manifest.get("metrics_file_sha256"):
        raise DataValidationError("performance metrics differ from performance manifest")
    health = _load_json(health_path, "health metrics")
    performance = pd.read_parquet(performance_path)
    portfolios = pd.read_parquet(portfolio_path)
    return (
        health,
        performance,
        portfolios,
        metric_source_hashes(
            health,
            performance,
            portfolios,
        ),
    )


def metric_source_hashes(
    health: dict[str, Any],
    performance: DataFrame,
    portfolios: DataFrame,
) -> dict[str, str]:
    """Hash logical metric contents consistently for standalone and integrated runs."""

    performance_records = _records(
        performance.sort_values(["model_id", "horizon"], kind="mergesort")
        if not performance.empty
        else performance
    )
    portfolio_records = _records(
        portfolios.sort_values("portfolio_id", kind="mergesort")
        if not portfolios.empty
        else portfolios
    )
    return {
        "health.json": canonical_payload_hash(health),
        "performance/performance_metrics.parquet": canonical_payload_hash(performance_records),
        "portfolio_metrics.parquet": canonical_payload_hash(portfolio_records),
    }


def load_alert_history(history_path: Path, before: str) -> DataFrame:
    """Load prior append-only lifecycle events strictly before as-of."""

    if not history_path.is_file():
        return pd.DataFrame(columns=list(ALERT_HISTORY_COLUMNS))
    frame = pd.read_parquet(history_path)
    _validate_history(frame)
    dates = frame["last_seen"].astype(str)
    if (dates > before).any():
        raise DataValidationError("alert history contains future lifecycle events")
    return frame.loc[dates < before].copy()


def alert_history_hash(frame: DataFrame) -> str:
    """Hash ordered history rows for deterministic lifecycle identity."""

    ordered = (
        frame.loc[:, list(ALERT_HISTORY_COLUMNS)].sort_values(
            ["last_seen", "alert_id"], kind="mergesort"
        )
        if not frame.empty
        else frame.reindex(columns=list(ALERT_HISTORY_COLUMNS))
    )
    return canonical_payload_hash(_records(ordered))


def write_alert_build(output_dir: Path, built: AlertBuild) -> None:
    """Write one alert subtree with its manifest last."""

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_alert_payload(built.alerts_payload)
    atomic_write_json(output_dir / "alerts.json", built.alerts_payload)
    (output_dir / "alert_report.md").write_text(built.report, encoding="utf-8")
    completed = {
        **built.manifest,
        "alerts_file_sha256": file_sha256(output_dir / "alerts.json"),
        "report_file_sha256": file_sha256(output_dir / "alert_report.md"),
    }
    atomic_write_json(output_dir / "manifest.json", completed)


def publish_alert_build(
    *,
    output_dir: Path,
    history_path: Path,
    built: AlertBuild,
) -> None:
    """Transactionally publish alert output and complete append-only history."""

    common_root = output_dir.parents[1]
    common_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(dir=common_root, prefix=".alert-publish-"))
    try:
        staged_output = staging_root / "alerts"
        staged_history = staging_root / "alert_history.parquet"
        write_alert_build(staged_output, built)
        built.history.to_parquet(staged_history, index=False)
        _validate_history(pd.read_parquet(staged_history))
        replace_targets_atomically(
            ((staged_output, output_dir), (staged_history, history_path)),
            backup_root=staging_root / "backups",
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def replace_targets_atomically(
    replacements: tuple[tuple[Path, Path], ...],
    *,
    backup_root: Path,
) -> None:
    """Replace multiple files/directories with rollback on any move failure."""

    backup_root.mkdir(parents=True, exist_ok=False)
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, (_, target) in enumerate(replacements):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = backup_root / str(index)
                os.replace(target, backup)
                backups.append((backup, target))
        for staged, target in replacements:
            os.replace(staged, target)
            published.append(target)
    except BaseException:
        for target in reversed(published):
            _remove(target)
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    for backup, _ in backups:
        _remove(backup)


def read_alert_output(output_dir: Path) -> dict[str, Any] | None:
    """Read and verify one complete alert output."""

    alerts_path = output_dir / "alerts.json"
    report_path = output_dir / "alert_report.md"
    manifest_path = output_dir / "manifest.json"
    if not alerts_path.is_file() or not report_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path, "alert manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != "alert_engine"
        or manifest.get("status") != "success"
    ):
        raise DataValidationError("invalid alert manifest identity")
    if file_sha256(alerts_path) != manifest.get("alerts_file_sha256"):
        raise DataValidationError("alerts JSON hash mismatch")
    if file_sha256(report_path) != manifest.get("report_file_sha256"):
        raise DataValidationError("alert report hash mismatch")
    validate_alert_payload(_load_json(alerts_path, "alerts JSON"))
    return manifest


def validate_alert_payload(payload: dict[str, Any]) -> None:
    """Validate alert schema, enums, deterministic IDs, and daily uniqueness."""

    if payload.get("schema_version") != 1 or payload.get("artifact_name") != "monitoring_alerts":
        raise DataValidationError("invalid alerts JSON identity")
    alerts = payload.get("alerts")
    if not isinstance(alerts, list):
        raise DataValidationError("alerts JSON lacks alerts list")
    required = {
        "alert_id",
        "alert_type",
        "severity",
        "status",
        "first_seen",
        "last_seen",
        "model_id",
        "portfolio_id",
        "metric_name",
        "metric_value",
        "threshold",
        "source_artifact_hash",
        "created_at",
    }
    identifiers: list[str] = []
    for raw in alerts:
        if not isinstance(raw, dict):
            raise DataValidationError("alert record must be an object")
        missing = sorted(required - set(raw))
        if missing:
            raise DataValidationError(f"alert record lacks fields: {missing}")
        try:
            AlertSeverity(str(raw["severity"]))
            AlertState(str(raw["status"]))
        except ValueError as error:
            raise DataValidationError(f"alert enum is invalid: {error}") from error
        expected_id = deterministic_alert_id(
            str(raw["alert_type"]),
            _optional_string(raw["model_id"]),
            _optional_string(raw["portfolio_id"]),
            str(raw["metric_name"]),
        )
        if raw["alert_id"] != expected_id:
            raise DataValidationError("alert_id violates deterministic identity contract")
        if not isinstance(raw["metric_value"], int | float) or not isinstance(
            raw["threshold"], int | float
        ):
            raise DataValidationError("alert metric and threshold must be numeric")
        identifiers.append(str(raw["alert_id"]))
    if len(set(identifiers)) != len(identifiers):
        raise DataValidationError("alerts JSON contains duplicate alert identities")


def _validate_history(frame: DataFrame) -> None:
    missing = sorted(set(ALERT_HISTORY_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"alert history lacks columns: {missing}")
    if not set(frame["severity"].astype(str)).issubset({value.value for value in AlertSeverity}):
        raise DataValidationError("alert history contains invalid severity")
    if not set(frame["status"].astype(str)).issubset({value.value for value in AlertState}):
        raise DataValidationError("alert history contains invalid lifecycle state")
    duplicates = frame.duplicated(["alert_id", "last_seen"], keep=False)
    if duplicates.any():
        raise DataValidationError("alert history contains duplicate lifecycle identities")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"{description} must be an object")
    return value


def _records(frame: DataFrame) -> list[dict[str, Any]]:
    normalized = frame.astype(object).where(frame.notna(), None)
    return cast(list[dict[str, Any]], normalized.to_dict("records"))


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
