"""Atomic orchestration for read-only production monitoring."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.monitoring.alerts.schemas import AlertBuild
from ashare_quant.monitoring.alerts.service import AlertService
from ashare_quant.monitoring.alerts.storage import (
    metric_source_hashes,
    replace_targets_atomically,
    write_alert_build,
)
from ashare_quant.monitoring.health import build_health_metrics
from ashare_quant.monitoring.performance.schemas import PerformanceBuild
from ashare_quant.monitoring.performance.service import (
    PerformanceMonitoringService,
    write_performance_build,
)
from ashare_quant.monitoring.portfolio import build_portfolio_metrics
from ashare_quant.monitoring.reporting import build_monitor_summary, render_monitor_report
from ashare_quant.monitoring.schemas import MonitoringResult
from ashare_quant.monitoring.validation import validate_monitoring_sources
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info


class MonitoringService:
    """Read immutable production/paper artifacts and publish monitoring outputs."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        reports_root: Path | None = None,
        paper_root: Path | None = None,
        performance_service: PerformanceMonitoringService | None = None,
        alert_service: AlertService | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.reports_root = reports_root or settings.paths.reports
        self.paper_root = paper_root or settings.paths.paper_trading
        self.performance_service = performance_service or PerformanceMonitoringService(
            reports_root=self.reports_root,
            config_path=config_path,
        )
        self.alert_service = alert_service or AlertService(
            settings=settings,
            config_path=config_path,
            reports_root=self.reports_root,
        )

    def run(self, as_of: str) -> MonitoringResult:
        """Validate, calculate, and atomically publish one monitoring snapshot."""

        sources, predictions, _ = validate_monitoring_sources(
            as_of=as_of,
            reports_root=self.reports_root,
            config_path=self.config_path,
        )
        health = build_health_metrics(sources, predictions)
        performance = self.performance_service.build(as_of)
        portfolio_ids = tuple(
            portfolio.portfolio_id for portfolio in self.settings.paper_trading.portfolios
        )
        portfolios, paper_hashes = build_portfolio_metrics(
            as_of=as_of,
            paper_root=self.paper_root,
            portfolio_ids=portfolio_ids,
        )
        alerts = self.alert_service.build_from_metrics(
            as_of=as_of,
            health=health.to_dict(),
            performance=performance.metrics,
            portfolios=portfolios,
            source_hashes=metric_source_hashes(
                health.to_dict(),
                performance.metrics,
                portfolios,
            ),
        )
        summary = build_monitor_summary(
            health,
            performance.summary,
            portfolios,
            alerts.alerts_payload,
        )
        all_hashes = {
            **sources.source_hashes,
            **{
                f"observation:{k}": v
                for k, v in performance.manifest["source_observation_hashes"].items()
            },
            **{f"paper:{k}": v for k, v in paper_hashes.items()},
        }
        aggregate_hash = _payload_hash(all_hashes)
        champion_assignment = _champion_assignment(
            self.settings.paths.models,
            sources.model_id,
        )
        run_id = f"monitor_{as_of}_{aggregate_hash[:16]}"
        completed_at = str(
            sources.production_summary.get("completed_time")
            or sources.prediction_manifest.get("generation_time")
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_name": "production_monitor_manifest",
            "run_id": run_id,
            "as_of": as_of,
            "git_commit": current_git_info()["commit"],
            "config_hash": config_hash(self.config_path),
            "source_artifact_hash": aggregate_hash,
            "source_artifact_hashes": all_hashes,
            "model_id": sources.model_id,
            "champion_assignment": champion_assignment,
            "performance_identity_contract": "model_id_and_horizon_are_never_mixed",
            "feature_hash": sources.feature_hash,
            "monitored_portfolio_ids": list(portfolio_ids),
            "row_counts": {
                "predictions": len(predictions),
                "candidates": int(sources.production_summary.get("candidate_count", 0)),
                "portfolio_metrics": len(portfolios),
                "performance_metrics": len(performance.metrics),
                "alerts": len(alerts.alerts),
            },
            "drift_reference": sources.drift_reference,
            "completed_at": completed_at,
            "read_only_contract": summary["scope"],
        }
        output_dir = self.reports_root / "model_monitor" / as_of
        _publish(
            output_dir,
            health.to_dict(),
            performance,
            portfolios,
            alerts,
            summary,
            manifest,
        )
        return MonitoringResult(
            as_of=as_of,
            run_id=run_id,
            output_dir=output_dir,
            portfolio_count=len(portfolios),
            prediction_count=len(predictions),
            performance_model_count=len(performance.metrics),
            alert_count=len(alerts.alerts),
        )


def _publish(
    output_dir: Path,
    health: dict[str, Any],
    performance: PerformanceBuild,
    portfolios: pd.DataFrame,
    alerts: AlertBuild,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging_root = Path(temporary)
        staging = staging_root / "monitor"
        staging.mkdir()
        atomic_write_json(staging / "health.json", health)
        write_performance_build(staging / "performance", performance)
        portfolios.to_parquet(staging / "portfolio_metrics.parquet", index=False)
        write_alert_build(staging / "alerts", alerts)
        atomic_write_json(staging / "monitor_summary.json", summary)
        (staging / "monitor_report.md").write_text(
            render_monitor_report(summary),
            encoding="utf-8",
        )
        completed_manifest = {
            **manifest,
            "monitor_metric_file_hashes": {
                "health": file_sha256(staging / "health.json"),
                "performance_metrics": file_sha256(
                    staging / "performance" / "performance_metrics.parquet"
                ),
                "performance_manifest": file_sha256(staging / "performance" / "manifest.json"),
                "portfolio_metrics": file_sha256(staging / "portfolio_metrics.parquet"),
            },
        }
        atomic_write_json(staging / "manifest.json", completed_manifest)
        staged_history = staging_root / "alert_history.parquet"
        alerts.history.to_parquet(staged_history, index=False)
        replace_targets_atomically(
            (
                (staging, output_dir),
                (staged_history, output_dir.parent / "history" / "alert_history.parquet"),
            ),
            backup_root=staging_root / "backups",
        )


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _champion_assignment(models_root: Path, model_id: str) -> dict[str, Any] | None:
    """Return current immutable deployment transition metadata when available."""

    root = models_root / "champion_history"
    matches: list[dict[str, Any]] = []
    if not root.exists():
        return None
    for path in root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("model_id") == model_id:
            matches.append(payload)
    if not matches:
        return None
    current = max(matches, key=lambda item: str(item.get("activated_at") or ""))
    return {
        "assignment_id": current.get("champion_assignment_id"),
        "previous_champion_model_id": current.get("previous_champion_model_id"),
        "new_champion_model_id": current.get("model_id"),
        "effective_date": current.get("activated_at"),
        "deployment_slot": current.get("deployment_slot"),
    }
