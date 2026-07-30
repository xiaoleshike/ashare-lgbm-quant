"""Read-only alert orchestration over monitoring outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.monitoring.alerts.evaluator import evaluate_alerts
from ashare_quant.monitoring.alerts.lifecycle import append_history, apply_lifecycle
from ashare_quant.monitoring.alerts.reporting import (
    build_alert_payload,
    render_alert_report,
)
from ashare_quant.monitoring.alerts.rules import configured_rules
from ashare_quant.monitoring.alerts.schemas import (
    AlertBuild,
    AlertMonitorResult,
    AlertSeverity,
    AlertValidationResult,
)
from ashare_quant.monitoring.alerts.storage import (
    alert_history_hash,
    load_alert_history,
    load_monitor_metrics,
    metric_source_hashes,
    publish_alert_build,
    read_alert_output,
)
from ashare_quant.utils.manifest import config_hash

type DataFrame = pd.DataFrame


class AlertService:
    """Evaluate alert rules without model, label, inference, or trading access."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        reports_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.reports_root = reports_root or settings.paths.reports

    def build(self, as_of: str) -> AlertBuild:
        """Load only published monitoring metrics and build lifecycle events."""

        health, performance, portfolios, source_hashes = load_monitor_metrics(
            self.reports_root,
            as_of,
        )
        return self.build_from_metrics(
            as_of=as_of,
            health=health,
            performance=performance,
            portfolios=portfolios,
            source_hashes=source_hashes,
        )

    def build_from_metrics(
        self,
        *,
        as_of: str,
        health: dict[str, Any],
        performance: DataFrame,
        portfolios: DataFrame,
        source_hashes: dict[str, str] | None = None,
    ) -> AlertBuild:
        """Build from Monitoring Core's in-memory metrics for atomic integration."""

        _validate_date(as_of)
        resolved_hashes = source_hashes or metric_source_hashes(
            health,
            performance,
            portfolios,
        )
        source_hash = canonical_payload_hash(resolved_hashes)
        history_path = self.history_path
        prior_history = load_alert_history(history_path, as_of)
        evaluation = evaluate_alerts(
            health=health,
            performance=performance,
            portfolios=portfolios,
            rules=configured_rules(self.settings.monitoring.alerts),
            source_artifact_hash=source_hash,
        )
        alerts = apply_lifecycle(evaluation, prior_history, as_of)
        history = append_history(prior_history, alerts)
        payload = build_alert_payload(as_of, alerts, evaluation.warnings)
        severity_counts = {
            severity.value: sum(alert.severity == severity for alert in alerts)
            for severity in AlertSeverity
        }
        source_metrics = [
            {"path": path, "hash": digest} for path, digest in sorted(resolved_hashes.items())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_name": "alert_engine",
            "as_of": as_of,
            "source_metrics": source_metrics,
            "source_artifact_hash": source_hash,
            "previous_history_hash": alert_history_hash(prior_history),
            "resulting_history_hash": alert_history_hash(history),
            "config_hash": config_hash(self.config_path),
            "alert_count": len(alerts),
            "severity_counts": severity_counts,
            "warnings": list(evaluation.warnings),
            "labels_read": False,
            "models_modified": False,
            "trading_modified": False,
            "status": "success",
        }
        manifest["identity_hash"] = canonical_payload_hash(
            {
                "as_of": as_of,
                "source_metrics": source_metrics,
                "previous_history_hash": manifest["previous_history_hash"],
                "config_hash": manifest["config_hash"],
            }
        )
        return AlertBuild(
            as_of=as_of,
            alerts=alerts,
            alerts_payload=payload,
            report=render_alert_report(payload),
            manifest=manifest,
            history=history,
        )

    def run(self, as_of: str) -> AlertMonitorResult:
        """Build and transactionally publish alerts plus history."""

        built = self.build(as_of)
        output_dir = self.output_dir(as_of)
        existing = read_alert_output(output_dir)
        if existing is not None:
            if existing.get("identity_hash") != built.manifest["identity_hash"]:
                raise DataValidationError(
                    "existing alerts have different immutable source identity"
                )
            return AlertMonitorResult(
                as_of,
                output_dir,
                len(built.alerts),
                sum(alert.severity == AlertSeverity.CRITICAL for alert in built.alerts),
                idempotent=True,
            )
        if output_dir.exists():
            raise DataValidationError(f"incomplete alert output exists: {output_dir}")
        publish_alert_build(
            output_dir=output_dir,
            history_path=self.history_path,
            built=built,
        )
        return AlertMonitorResult(
            as_of,
            output_dir,
            len(built.alerts),
            sum(alert.severity == AlertSeverity.CRITICAL for alert in built.alerts),
        )

    def validate(self, as_of: str) -> AlertValidationResult:
        """Validate monitoring inputs and rules without publishing."""

        try:
            built = self.build(as_of)
        except (DataValidationError, OSError, ValueError) as error:
            return AlertValidationResult(as_of, False, False, 0, error=str(error))
        return AlertValidationResult(
            as_of,
            True,
            self.output_dir(as_of).is_dir(),
            len(built.alerts),
            tuple(str(value) for value in built.manifest["warnings"]),
        )

    def status(self, as_of: str) -> AlertValidationResult:
        """Validate one published alert output."""

        output_dir = self.output_dir(as_of)
        try:
            manifest = read_alert_output(output_dir)
        except (DataValidationError, OSError, ValueError) as error:
            return AlertValidationResult(
                as_of,
                False,
                output_dir.exists(),
                0,
                error=str(error),
            )
        if manifest is None:
            return AlertValidationResult(
                as_of,
                False,
                False,
                0,
                error="alert output is missing",
            )
        return AlertValidationResult(
            as_of,
            True,
            True,
            int(manifest.get("alert_count", 0)),
            tuple(str(value) for value in manifest.get("warnings", [])),
        )

    def output_dir(self, as_of: str) -> Path:
        return self.reports_root / "model_monitor" / as_of / "alerts"

    @property
    def history_path(self) -> Path:
        return self.reports_root / "model_monitor" / "history" / "alert_history.parquet"


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"alert as_of must use YYYYMMDD: {value}")
