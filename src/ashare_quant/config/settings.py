"""Typed application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class PathSettings(BaseModel):
    """Filesystem locations used by the research pipeline."""

    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    parquet_store: Path = Path("data/parquet")
    duckdb_path: Path = Path("data/ashare_quant.duckdb")
    reports: Path = Path("reports")


class LoggingSettings(BaseModel):
    """Structured logging options."""

    level: str = "INFO"
    json_logs: bool = True


class DataSettings(BaseModel):
    """Data provider and ingestion control settings."""

    provider: Literal["tushare"] = "tushare"
    retry_attempts: int = Field(default=3, ge=1)
    rate_limit_per_minute: int = Field(default=180, ge=1)


class BacktestSettings(BaseModel):
    """Default assumptions for later out-of-sample backtests."""

    execution: Literal["next_open", "next_vwap"] = "next_open"
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    commission_bps: float = Field(default=3.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)


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
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    tushare_token: SecretStr | None = None

    @property
    def has_tushare_token(self) -> bool:
        """Return whether a Tushare token is available without exposing it."""

        return self.tushare_token is not None


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
