"""Read-only health measurements for one production publication."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.schemas import HealthMetrics, MonitoringSources

type DataFrame = pd.DataFrame


def build_health_metrics(sources: MonitoringSources, predictions: DataFrame) -> HealthMetrics:
    """Measure production coverage and score shape without loading labels."""

    readiness = sources.prediction_manifest.get("readiness")
    if not isinstance(readiness, list):
        raise DataValidationError("prediction manifest lacks structured readiness results")
    universe_gate = _gate(readiness, "universe_readiness_gate")
    feature_gate = _gate(readiness, "features_readiness_gate")
    universe_counts = _mapping(universe_gate.get("row_counts"), "universe row counts")
    feature_counts = _mapping(feature_gate.get("row_counts"), "feature row counts")
    missingness = _float_mapping(feature_gate.get("missingness_summary"))

    scores = pd.to_numeric(predictions["prediction_score"], errors="coerce")
    finite = scores[np.isfinite(scores)]
    if finite.empty:
        raise DataValidationError("prediction scores contain no finite values")
    percentiles = {
        name: float(finite.quantile(quantile))
        for name, quantile in (
            ("p01", 0.01),
            ("p10", 0.10),
            ("p50", 0.50),
            ("p90", 0.90),
            ("p99", 0.99),
        )
    }
    duplicated = int(finite.duplicated(keep=False).sum())
    feature_rows = int(feature_counts.get("features", 0))
    eligible_rows = int(feature_counts.get("eligible_after_hard_features", 0))
    return HealthMetrics(
        as_of=sources.as_of,
        model_id=sources.model_id,
        universe_size=int(universe_counts.get("rows", 0)),
        model_universe_size=int(universe_counts.get("in_model_universe", 0)),
        prediction_count=len(predictions),
        candidate_count=int(sources.production_summary.get("candidate_count", 0)),
        feature_coverage=eligible_rows / feature_rows if feature_rows else 0.0,
        feature_missing_ratios=missingness,
        score_mean=float(finite.mean()),
        score_std=float(finite.std(ddof=0)),
        score_percentiles=percentiles,
        score_spread=percentiles["p90"] - percentiles["p10"],
        duplicate_score_ratio=duplicated / len(finite),
        unique_score_ratio=float(finite.nunique(dropna=True) / len(finite)),
        drift_reference=sources.drift_reference,
    )


def _gate(readiness: list[object], name: str) -> dict[str, Any]:
    for item in readiness:
        if isinstance(item, dict) and item.get("gate") == name:
            return item
    raise DataValidationError(f"prediction manifest lacks readiness gate: {name}")


def _mapping(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"prediction manifest lacks {description}")
    return value


def _float_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): float(ratio)
        for name, ratio in sorted(value.items())
        if isinstance(ratio, int | float)
    }
