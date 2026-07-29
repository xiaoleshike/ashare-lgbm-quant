"""Atomic orchestration for read-only production monitoring."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
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
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.reports_root = reports_root or settings.paths.reports
        self.paper_root = paper_root or settings.paths.paper_trading
        self.performance_service = performance_service or PerformanceMonitoringService(
            reports_root=self.reports_root,
            config_path=config_path,
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
        summary = build_monitor_summary(health, performance.summary, portfolios)
        all_hashes = {
            **sources.source_hashes,
            **{
                f"observation:{k}": v
                for k, v in performance.manifest["source_observation_hashes"].items()
            },
            **{f"paper:{k}": v for k, v in paper_hashes.items()},
        }
        aggregate_hash = _payload_hash(all_hashes)
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
            "feature_hash": sources.feature_hash,
            "monitored_portfolio_ids": list(portfolio_ids),
            "row_counts": {
                "predictions": len(predictions),
                "candidates": int(sources.production_summary.get("candidate_count", 0)),
                "portfolio_metrics": len(portfolios),
                "performance_metrics": len(performance.metrics),
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
        )


def _publish(
    output_dir: Path,
    health: dict[str, Any],
    performance: PerformanceBuild,
    portfolios: pd.DataFrame,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        atomic_write_json(staging / "health.json", health)
        write_performance_build(staging / "performance", performance)
        portfolios.to_parquet(staging / "portfolio_metrics.parquet", index=False)
        atomic_write_json(staging / "monitor_summary.json", summary)
        (staging / "monitor_report.md").write_text(
            render_monitor_report(summary),
            encoding="utf-8",
        )
        atomic_write_json(staging / "manifest.json", manifest)
        _replace_directory(staging, output_dir)


def _replace_directory(staging: Path, output_dir: Path) -> None:
    """Atomically switch a complete monitor tree and restore the prior tree on failure."""

    backup_parent = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".monitor-backup-"))
    backup = backup_parent / output_dir.name
    moved_previous = False
    try:
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_previous = True
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if moved_previous and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if backup_parent.exists():
            shutil.rmtree(backup_parent)


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
