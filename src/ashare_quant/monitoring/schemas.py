"""Typed output contracts for the monitoring layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MonitoringSources:
    """Validated immutable inputs consumed by one monitoring run."""

    as_of: str
    model_id: str
    feature_hash: str
    production_summary: dict[str, Any]
    prediction_manifest: dict[str, Any]
    candidate_manifest: dict[str, Any]
    predictions_path: Path
    candidates_path: Path
    source_hashes: dict[str, str]
    drift_reference: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class HealthMetrics:
    """Same-session production health measurements."""

    as_of: str
    model_id: str
    universe_size: int
    model_universe_size: int
    prediction_count: int
    candidate_count: int
    feature_coverage: float
    feature_missing_ratios: dict[str, float]
    score_mean: float
    score_std: float
    score_percentiles: dict[str, float]
    score_spread: float
    duplicate_score_ratio: float
    drift_reference: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioMetrics:
    """One isolated paper portfolio measured through an as-of date."""

    as_of: str
    portfolio_id: str
    nav: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    turnover: float
    transaction_cost_ratio: float
    position_count: int
    max_position_weight: float
    top5_concentration: float
    cash_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return one deterministic tabular record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MonitoringResult:
    """Published monitoring artifact identity."""

    as_of: str
    run_id: str
    output_dir: Path
    portfolio_count: int
    prediction_count: int
    performance_model_count: int = 0
