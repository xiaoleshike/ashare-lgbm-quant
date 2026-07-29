"""Atomic orchestration for read-only production monitoring."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.monitoring.health import build_health_metrics
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
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.reports_root = reports_root or settings.paths.reports
        self.paper_root = paper_root or settings.paths.paper_trading

    def run(self, as_of: str) -> MonitoringResult:
        """Validate, calculate, and atomically publish one monitoring snapshot."""

        sources, predictions, _ = validate_monitoring_sources(
            as_of=as_of,
            reports_root=self.reports_root,
            config_path=self.config_path,
        )
        health = build_health_metrics(sources, predictions)
        portfolio_ids = tuple(
            portfolio.portfolio_id for portfolio in self.settings.paper_trading.portfolios
        )
        portfolios, paper_hashes = build_portfolio_metrics(
            as_of=as_of,
            paper_root=self.paper_root,
            portfolio_ids=portfolio_ids,
        )
        summary = build_monitor_summary(health, portfolios)
        all_hashes = {**sources.source_hashes, **{f"paper:{k}": v for k, v in paper_hashes.items()}}
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
            },
            "drift_reference": sources.drift_reference,
            "completed_at": completed_at,
            "read_only_contract": summary["scope"],
        }
        output_dir = self.reports_root / "model_monitor" / as_of
        _publish(output_dir, health.to_dict(), portfolios, summary, manifest)
        return MonitoringResult(
            as_of=as_of,
            run_id=run_id,
            output_dir=output_dir,
            portfolio_count=len(portfolios),
            prediction_count=len(predictions),
        )


def _publish(
    output_dir: Path,
    health: dict[str, Any],
    portfolios: pd.DataFrame,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        atomic_write_json(staging / "health.json", health)
        portfolios.to_parquet(staging / "portfolio_metrics.parquet", index=False)
        atomic_write_json(staging / "monitor_summary.json", summary)
        (staging / "monitor_report.md").write_text(
            render_monitor_report(summary),
            encoding="utf-8",
        )
        atomic_write_json(staging / "manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "health.json",
            "portfolio_metrics.parquet",
            "monitor_summary.json",
            "monitor_report.md",
        ):
            os.replace(staging / filename, output_dir / filename)
        os.replace(staging / "manifest.json", output_dir / "manifest.json")


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
