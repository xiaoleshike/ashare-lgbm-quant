"""Strict allowlist loader for retraining trigger evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.schemas import RetrainingSources
from ashare_quant.retraining.validators import (
    evidence_hash,
    validate_as_of,
    validate_monitoring_bundle,
)


def load_retraining_sources(reports_root: Path, as_of: str) -> RetrainingSources:
    """Load only Monitoring, Alert, and summarized observation artifacts."""

    validate_as_of(as_of)
    root = reports_root / "model_monitor" / as_of
    paths = {
        "monitor_manifest": root / "manifest.json",
        "health": root / "health.json",
        "performance_metrics": root / "performance" / "performance_metrics.parquet",
        "performance_manifest": root / "performance" / "manifest.json",
        "alerts": root / "alerts" / "alerts.json",
        "alerts_manifest": root / "alerts" / "manifest.json",
    }
    monitor_manifest = _json(paths["monitor_manifest"], "monitor manifest")
    health = _json(paths["health"], "health metrics")
    performance_manifest = _json(paths["performance_manifest"], "performance manifest")
    alerts_manifest = _json(paths["alerts_manifest"], "alerts manifest")
    alerts = _json(paths["alerts"], "alerts")
    try:
        performance = pd.read_parquet(paths["performance_metrics"])
    except (OSError, ValueError) as error:
        raise DataValidationError(f"cannot read performance metrics: {error}") from error
    evidence = validate_monitoring_bundle(
        reports_root=reports_root,
        as_of=as_of,
        monitor_manifest=monitor_manifest,
        health=health,
        performance_manifest=performance_manifest,
        performance_metrics=performance,
        alerts_manifest=alerts_manifest,
        alerts=alerts,
        paths=paths,
    )
    return RetrainingSources(
        as_of=as_of,
        monitor_manifest=monitor_manifest,
        health=health,
        performance_manifest=performance_manifest,
        performance_metrics=performance.sort_values(
            ["model_id", "horizon"], kind="mergesort"
        ).reset_index(drop=True),
        alerts_manifest=alerts_manifest,
        alerts=alerts,
        evidence=evidence,
        evidence_hash=evidence_hash(evidence),
    )


def _json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required retraining {description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid retraining {description}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"retraining {description} must contain an object")
    return payload
