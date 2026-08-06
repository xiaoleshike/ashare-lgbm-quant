"""Typed contracts for model performance monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

PERFORMANCE_METRIC_COLUMNS: tuple[str, ...] = (
    "model_id",
    "model_role",
    "model_origin",
    "horizon",
    "feature_hash",
    "universe_hash",
    "observation_rows",
    "available_rows",
    "sessions",
    "pearson_ic",
    "rank_ic",
    "icir",
    "positive_ic_ratio",
    "top10_average_excess_ret",
    "top10_hit_rate",
    "top20_average_excess_ret",
    "top20_hit_rate",
    "top50_average_excess_ret",
    "top50_hit_rate",
    "decile_monotonicity",
    "rolling_20_ic_mean",
    "rolling_20_ic_std",
    "rolling_20_icir",
    "rolling_20_positive_ic_ratio",
    "rolling_60_ic_mean",
    "rolling_60_ic_std",
    "rolling_60_icir",
    "rolling_60_positive_ic_ratio",
    "rolling_120_ic_mean",
    "rolling_120_ic_std",
    "rolling_120_icir",
    "rolling_120_positive_ic_ratio",
    "alpha_decay_ratio",
    "top10_decay_ratio",
)


@dataclass(frozen=True, slots=True)
class ObservationSources:
    """Validated append-only observation history and its lineage."""

    observations: Any
    source_manifests: tuple[dict[str, Any], ...]
    source_hashes: dict[str, str]
    model_lineage: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PerformanceBuild:
    """In-memory deterministic performance-monitor output."""

    as_of: str
    metrics: Any
    summary: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PerformanceMonitorResult:
    """Published performance-monitor artifact."""

    as_of: str
    output_dir: Path
    model_count: int
    observation_rows: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class PerformanceValidationResult:
    """Read-only validation/status result."""

    as_of: str
    valid: bool
    exists: bool
    model_count: int
    observation_rows: int
    warnings: tuple[str, ...] = ()
    error: str | None = None
