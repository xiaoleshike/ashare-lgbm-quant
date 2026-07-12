"""Typed application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SecretStr, model_validator


class PathSettings(BaseModel):
    """Filesystem locations used by the research pipeline."""

    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    parquet_store: Path = Path("data/parquet")
    duckdb_path: Path = Path("data/ashare_quant.duckdb")
    reports: Path = Path("reports")
    data_quality_logs: Path = Path("logs/data_quality")


class LoggingSettings(BaseModel):
    """Structured logging options."""

    level: str = "INFO"
    json_logs: bool = True


class DataSettings(BaseModel):
    """Data provider and ingestion control settings."""

    provider: Literal["tushare"] = "tushare"
    retry_attempts: int = Field(default=3, ge=1)
    rate_limit_per_minute: int = Field(default=180, ge=1)
    endpoint_rate_limits_per_minute: dict[str, PositiveInt] = Field(default_factory=dict)
    request_interval_seconds: float = Field(default=0.0, ge=0)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_max_seconds: float = Field(default=60.0, ge=0)
    default_start_date: str = "20100101"
    calendar_exchange: str = "SSE"
    index_codes: tuple[str, ...] = ("000001.SH", "000300.SH", "399001.SZ", "399006.SZ")
    fund_markets: tuple[str, ...] = ("E",)
    hs_types: tuple[str, ...] = ("SH", "SZ")
    stock_list_statuses: tuple[str, ...] = ("L", "D", "P")
    date_range_chunk_years: int = Field(default=1, ge=1)
    tushare_page_size: int = Field(default=6000, ge=1, le=6000)
    finance_revision_lookback_days: int = Field(default=550, ge=1)
    snapshot_refresh_policy: Literal["manual", "always", "ttl_days"] = "manual"
    snapshot_refresh_ttl_days: int = Field(default=7, ge=1)


class BacktestSettings(BaseModel):
    """Default assumptions for later out-of-sample backtests."""

    execution: Literal["next_open", "next_vwap"] = "next_open"
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    commission_bps: float = Field(default=3.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)


class UniverseSettings(BaseModel):
    """Point-in-time universe and tradability construction rules."""

    min_list_trading_days: int = Field(default=180, ge=0)
    liquidity_window_days: int = Field(default=20, ge=1)
    min_avg_amount: float = Field(default=30_000.0, ge=0)
    require_full_liquidity_window: bool = True
    min_price: float | None = Field(default=None, gt=0)
    exclude_st: bool = True
    mark_limit_up_not_buyable: bool = True
    mark_limit_down_not_sellable: bool = True
    price_tolerance: float = Field(default=1e-6, ge=0)


class LabelSettings(BaseModel):
    """Executable forward-return label construction rules."""

    horizons: tuple[int, ...] = (3, 5, 10)
    benchmark_index_code: str = "000300.SH"
    quantile_buckets: int = Field(default=5, ge=2)
    skip_unbuyable_entry: bool = True
    allow_limit_up_entry: bool = False
    allow_limit_down_exit: bool = False
    delay_unsellable_exit: bool = False
    max_exit_delay_days: int = Field(default=5, ge=0)
    price_tolerance: float = Field(default=1e-6, ge=0)
    price_adjustment: Literal["open_times_adj_factor"] = "open_times_adj_factor"


class FeatureSettings(BaseModel):
    """Point-in-time feature construction rules."""

    return_windows: tuple[int, ...] = (1, 3, 5, 10, 20, 60, 120)
    short_windows: tuple[int, ...] = (1, 2, 3, 5)
    medium_windows: tuple[int, ...] = (5, 10, 20, 60)
    long_windows: tuple[int, ...] = (20, 60, 120)
    quantile_buckets: int = Field(default=5, ge=2)
    min_period_fraction: float = Field(default=0.6, ge=0.1, le=1.0)
    min_traded_observation_fraction: float = Field(default=0.6, ge=0.1, le=1.0)
    benchmark_index_code: str = "000300.SH"
    include_fundamentals: bool = True


class AppSettings(BaseModel):
    """Top-level validated settings.

    `tushare_token` is intentionally sourced only from the process environment so
    secrets are not embedded in YAML files or committed to version control.
    """

    model_config = ConfigDict(extra="forbid")

    environment: str = "development"
    project_name: str = "ashare-lgbm-quant"
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    labels: LabelSettings = Field(default_factory=LabelSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    tushare_token: SecretStr | None = None

    @property
    def has_tushare_token(self) -> bool:
        """Return whether a Tushare token is available without exposing it."""

        return self.tushare_token is not None

    @model_validator(mode="after")
    def validate_benchmark_index_configuration(self) -> AppSettings:
        """Ensure benchmark-dependent stages use an index that ingestion downloads."""

        label_benchmark = self.labels.benchmark_index_code
        feature_benchmark = self.features.benchmark_index_code
        if label_benchmark != feature_benchmark:
            raise ValueError(
                "labels.benchmark_index_code and features.benchmark_index_code must match"
            )
        if label_benchmark not in self.data.index_codes:
            raise ValueError(
                "configured benchmark_index_code must be included in data.index_codes "
                f"for index_daily ingestion: {label_benchmark}"
            )
        return self


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file as a mapping."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return loaded


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load validated settings from YAML plus environment-only secrets."""

    path = Path(config_path or os.environ.get("ASHARE_QUANT_CONFIG", "config/default.yaml"))
    values = load_yaml_config(path)
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        values["tushare_token"] = token
    return AppSettings.model_validate(values)
