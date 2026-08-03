"""Typed application configuration loaded from YAML and environment variables."""

from __future__ import annotations

import os
from datetime import datetime, time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SecretStr, model_validator


class PathSettings(BaseModel):
    """Filesystem locations used by the research pipeline."""

    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    parquet_store: Path = Path("data/parquet")
    duckdb_path: Path = Path("data/ashare_quant.duckdb")
    reports: Path = Path("reports")
    models: Path = Path("models")
    backtests: Path = Path("backtests")
    paper_trading: Path = Path("paper_trading")
    data_quality_logs: Path = Path("logs/data_quality")


class LoggingSettings(BaseModel):
    """Structured logging options."""

    level: str = "INFO"
    json_logs: bool = True


class DataSettings(BaseModel):
    """Data provider and ingestion control settings."""

    provider: Literal["tushare"] = "tushare"
    run_baostock_post_ingestion_check: bool = False
    retry_attempts: int = Field(default=3, ge=1)
    rate_limit_per_minute: int = Field(default=180, ge=1)
    endpoint_rate_limits_per_minute: dict[str, PositiveInt] = Field(default_factory=dict)
    request_interval_seconds: float = Field(default=0.0, ge=0)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_max_seconds: float = Field(default=60.0, ge=0)
    default_start_date: str = "20100101"
    calendar_exchange: str = "SSE"
    index_codes: tuple[str, ...] = ("000001.SH", "000300.SH", "399001.SZ", "399006.SZ")
    index_first_available_dates: dict[str, str] = Field(default_factory=dict)
    fund_markets: tuple[str, ...] = ("E",)
    hs_types: tuple[str, ...] = ("SH", "SZ")
    stock_list_statuses: tuple[str, ...] = ("L", "D", "P")
    date_range_chunk_years: int = Field(default=1, ge=1)
    tushare_page_size: int = Field(default=6000, ge=1, le=6000)
    finance_revision_lookback_days: int = Field(default=550, ge=1)
    snapshot_refresh_policy: Literal["manual", "always", "ttl_days"] = "manual"
    snapshot_refresh_ttl_days: int = Field(default=7, ge=1)

    @model_validator(mode="after")
    def validate_index_first_available_dates(self) -> DataSettings:
        """Validate optional per-index inception boundaries used by gap detection."""

        unknown_codes = sorted(set(self.index_first_available_dates) - set(self.index_codes))
        if unknown_codes:
            raise ValueError(
                "data.index_first_available_dates contains codes not present in "
                f"data.index_codes: {unknown_codes}"
            )
        for code, value in self.index_first_available_dates.items():
            try:
                parsed = datetime.strptime(value, "%Y%m%d")
            except ValueError as error:
                raise ValueError(
                    f"data.index_first_available_dates[{code}] must be YYYYMMDD: {value}"
                ) from error
            if parsed.strftime("%Y%m%d") != value:
                raise ValueError(
                    f"data.index_first_available_dates[{code}] must be YYYYMMDD: {value}"
                )
        return self


class HistoricalBacktestPeriodSettings(BaseModel):
    """Named chronological evaluation period for a frozen champion model."""

    start_date: str
    end_date: str


class HistoricalBacktestSettings(BaseModel):
    """Research reporting rules layered over the executable backtest engine."""

    top_n: tuple[PositiveInt, ...] = (10, 20, 50)
    holding_period_days: PositiveInt = 5
    require_out_of_sample: bool = True
    bull_annual_return_threshold: float = 0.10
    bear_annual_return_threshold: float = -0.10
    periods: dict[str, HistoricalBacktestPeriodSettings] = Field(
        default_factory=lambda: {
            "2015-2020": HistoricalBacktestPeriodSettings(
                start_date="20150101", end_date="20191231"
            ),
            "2020-2023": HistoricalBacktestPeriodSettings(
                start_date="20200101", end_date="20221231"
            ),
            "2023-2026": HistoricalBacktestPeriodSettings(
                start_date="20230101", end_date="20261231"
            ),
        }
    )

    @model_validator(mode="after")
    def validate_historical_settings(self) -> HistoricalBacktestSettings:
        """Require unique Top-N values and ordered regime/period boundaries."""

        if len(set(self.top_n)) != len(self.top_n):
            raise ValueError("historical backtest top_n must not contain duplicates")
        if self.bear_annual_return_threshold >= self.bull_annual_return_threshold:
            raise ValueError("historical backtest bear threshold must be below bull threshold")
        for name, period in self.periods.items():
            if period.start_date > period.end_date:
                raise ValueError(f"historical backtest period is reversed: {name}")
        return self


class BacktestDiagnosticSettings(BaseModel):
    """Post-hoc alpha diagnostics for one immutable historical backtest run."""

    horizon: PositiveInt = 5
    score_layers: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    bottom_fraction: float = Field(default=0.20, gt=0, le=1)
    minimum_cross_section: PositiveInt = 20
    factor_quantiles: int = Field(default=5, ge=2)
    shap_sample_rows: PositiveInt = 10_000
    prediction_tolerance: float = Field(default=1e-7, gt=0)

    @model_validator(mode="after")
    def validate_score_layers(self) -> BacktestDiagnosticSettings:
        """Require unique ascending fractions in the open unit interval."""

        if tuple(sorted(set(self.score_layers))) != self.score_layers:
            raise ValueError("backtest.diagnostics.score_layers must be unique and ascending")
        if any(value <= 0 or value >= 1 for value in self.score_layers):
            raise ValueError("backtest.diagnostics.score_layers must be between 0 and 1")
        return self


class BacktestSettings(BaseModel):
    """Executable portfolio backtest assumptions."""

    execution: Literal["next_open", "next_vwap"] = "next_open"
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    top_n: tuple[PositiveInt, ...] = (10, 20, 50)
    holding_period_days: PositiveInt = 5
    commission: float = Field(default=0.00025, ge=0)
    stamp_duty: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    benchmark_index_code: str = "000300.SH"
    annualization_days: PositiveInt = 252
    sell_delay_max_days: int = Field(default=20, ge=0)
    historical: HistoricalBacktestSettings = Field(default_factory=HistoricalBacktestSettings)
    diagnostics: BacktestDiagnosticSettings = Field(default_factory=BacktestDiagnosticSettings)

    @model_validator(mode="after")
    def validate_backtest_settings(self) -> BacktestSettings:
        """Require supported execution and non-duplicated Top-N values."""

        if len(set(self.top_n)) != len(self.top_n):
            raise ValueError("backtest.top_n must not contain duplicates")
        return self


class PaperPortfolioSettings(BaseModel):
    """One isolated virtual portfolio and its immutable signal source."""

    portfolio_id: str
    signal_type: Literal["champion", "model", "ensemble"]
    model_id: str | None = None
    component_model_ids: tuple[str, ...] = ()
    top_n: PositiveInt = 20

    @model_validator(mode="after")
    def validate_signal_source(self) -> PaperPortfolioSettings:
        """Require exactly the model identity needed by each signal type."""

        if self.signal_type == "champion":
            if self.model_id not in {None, "champion"} or self.component_model_ids:
                raise ValueError("champion paper portfolio must use model_id=champion only")
        elif self.signal_type == "model":
            if not self.model_id or self.model_id == "champion" or self.component_model_ids:
                raise ValueError("model paper portfolio requires one non-champion model_id")
        elif not self.component_model_ids or self.model_id is not None:
            raise ValueError("ensemble paper portfolio requires component_model_ids only")
        if len(set(self.component_model_ids)) != len(self.component_model_ids):
            raise ValueError("paper portfolio component_model_ids must be unique")
        return self


class PaperTradingSettings(BaseModel):
    """Append-only single-host virtual execution assumptions."""

    enabled: bool = True
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    execution: Literal["next_open"] = "next_open"
    lot_size: PositiveInt = 100
    commission: float = Field(default=0.00025, ge=0)
    stamp_duty: float = Field(default=0.001, ge=0)
    slippage: float = Field(default=0.0005, ge=0)
    price_tolerance: float = Field(default=1e-6, ge=0)
    portfolios: tuple[PaperPortfolioSettings, ...] = ()

    @model_validator(mode="after")
    def validate_portfolios(self) -> PaperTradingSettings:
        """Require stable, isolated portfolio identifiers."""

        identifiers = [portfolio.portfolio_id for portfolio in self.portfolios]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("paper_trading portfolio_id values must be unique")
        if any(not value or "/" in value or "\\" in value for value in identifiers):
            raise ValueError("paper_trading portfolio_id must be a simple non-empty name")
        return self


class LowerAlertThresholdSettings(BaseModel):
    """Alert thresholds where smaller metric values are worse."""

    enabled: bool = True
    warning: float
    critical: float

    @model_validator(mode="after")
    def validate_thresholds(self) -> LowerAlertThresholdSettings:
        if self.critical > self.warning:
            raise ValueError("lower alert critical threshold must not exceed warning")
        return self


class UpperAlertThresholdSettings(BaseModel):
    """Alert thresholds where larger metric values are worse."""

    enabled: bool = True
    warning: float = Field(ge=0)
    critical: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> UpperAlertThresholdSettings:
        if self.warning > self.critical:
            raise ValueError("upper alert warning threshold must not exceed critical")
        return self


class ScoreCollapseAlertSettings(BaseModel):
    """Lower bounds for score dispersion and cross-sectional uniqueness."""

    enabled: bool = True
    score_std_warning: float = Field(default=0.01, ge=0)
    score_std_critical: float = Field(default=0.001, ge=0)
    unique_ratio_warning: float = Field(default=0.90, ge=0, le=1)
    unique_ratio_critical: float = Field(default=0.50, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ScoreCollapseAlertSettings:
        if self.score_std_critical > self.score_std_warning:
            raise ValueError("score collapse std critical threshold must not exceed warning")
        if self.unique_ratio_critical > self.unique_ratio_warning:
            raise ValueError("score collapse unique critical threshold must not exceed warning")
        return self


class FeatureDriftAlertSettings(BaseModel):
    """Upper thresholds for existing PSI, KS, and missingness drift metrics."""

    enabled: bool = True
    psi_warning: float = Field(default=0.10, ge=0)
    psi_critical: float = Field(default=0.25, ge=0)
    ks_warning: float = Field(default=0.10, ge=0, le=1)
    ks_critical: float = Field(default=0.20, ge=0, le=1)
    missing_ratio_warning: float = Field(default=0.10, ge=0, le=1)
    missing_ratio_critical: float = Field(default=0.20, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> FeatureDriftAlertSettings:
        pairs = (
            (self.psi_warning, self.psi_critical),
            (self.ks_warning, self.ks_critical),
            (self.missing_ratio_warning, self.missing_ratio_critical),
        )
        if any(warning > critical for warning, critical in pairs):
            raise ValueError("feature drift warning thresholds must not exceed critical")
        return self


class CoverageAlertSettings(BaseModel):
    """Lower prediction coverage and upper universe-deviation thresholds."""

    enabled: bool = True
    prediction_warning: float = Field(default=0.95, ge=0, le=1)
    prediction_critical: float = Field(default=0.80, ge=0, le=1)
    universe_deviation_warning: float = Field(default=0.15, ge=0)
    universe_deviation_critical: float = Field(default=0.30, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> CoverageAlertSettings:
        if self.prediction_critical > self.prediction_warning:
            raise ValueError("prediction coverage critical threshold must not exceed warning")
        if self.universe_deviation_warning > self.universe_deviation_critical:
            raise ValueError("universe deviation warning must not exceed critical")
        return self


class ConcentrationAlertSettings(BaseModel):
    """Upper portfolio concentration thresholds."""

    enabled: bool = True
    max_weight_warning: float = Field(default=0.10, ge=0, le=1)
    max_weight_critical: float = Field(default=0.20, ge=0, le=1)
    top5_warning: float = Field(default=0.40, ge=0, le=1)
    top5_critical: float = Field(default=0.60, ge=0, le=1)
    industry_warning: float = Field(default=0.30, ge=0, le=1)
    industry_critical: float = Field(default=0.50, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ConcentrationAlertSettings:
        pairs = (
            (self.max_weight_warning, self.max_weight_critical),
            (self.top5_warning, self.top5_critical),
            (self.industry_warning, self.industry_critical),
        )
        if any(warning > critical for warning, critical in pairs):
            raise ValueError("concentration warning thresholds must not exceed critical")
        return self


class ExecutionQualityAlertSettings(BaseModel):
    """Upper rejected and failed execution-ratio thresholds."""

    enabled: bool = True
    rejected_warning: float = Field(default=0.10, ge=0, le=1)
    rejected_critical: float = Field(default=0.25, ge=0, le=1)
    failed_warning: float = Field(default=0.02, ge=0, le=1)
    failed_critical: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ExecutionQualityAlertSettings:
        if (
            self.rejected_warning > self.rejected_critical
            or self.failed_warning > self.failed_critical
        ):
            raise ValueError("execution warning thresholds must not exceed critical")
        return self


class MonitoringAlertSettings(BaseModel):
    """Configuration-driven read-only alert rules."""

    alpha_decay: LowerAlertThresholdSettings = Field(
        default_factory=lambda: LowerAlertThresholdSettings(warning=0.70, critical=0.50)
    )
    rank_ic_decline: LowerAlertThresholdSettings = Field(
        default_factory=lambda: LowerAlertThresholdSettings(warning=-0.02, critical=-0.05)
    )
    score_collapse: ScoreCollapseAlertSettings = Field(default_factory=ScoreCollapseAlertSettings)
    feature_drift: FeatureDriftAlertSettings = Field(default_factory=FeatureDriftAlertSettings)
    universe_coverage: CoverageAlertSettings = Field(default_factory=CoverageAlertSettings)
    drawdown: UpperAlertThresholdSettings = Field(
        default_factory=lambda: UpperAlertThresholdSettings(warning=0.10, critical=0.20)
    )
    concentration: ConcentrationAlertSettings = Field(default_factory=ConcentrationAlertSettings)
    execution_quality: ExecutionQualityAlertSettings = Field(
        default_factory=ExecutionQualityAlertSettings
    )


class MonitoringSettings(BaseModel):
    """Read-only monitoring and alert configuration."""

    alerts: MonitoringAlertSettings = Field(default_factory=MonitoringAlertSettings)


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

    horizons: tuple[PositiveInt, ...] = (5, 10, 20, 60)
    benchmark_index_code: str = "000300.SH"
    quantile_buckets: int = Field(default=5, ge=2)
    skip_unbuyable_entry: bool = True
    allow_limit_up_entry: bool = False
    allow_limit_down_exit: bool = False
    delay_unsellable_exit: bool = False
    max_exit_delay_days: int = Field(default=5, ge=0)
    price_tolerance: float = Field(default=1e-6, ge=0)
    price_adjustment: Literal["open_times_adj_factor"] = "open_times_adj_factor"

    @model_validator(mode="after")
    def validate_horizons(self) -> LabelSettings:
        """Require deterministic, unique label horizons."""

        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("labels.horizons must be unique and ascending")
        return self


class FeatureSettings(BaseModel):
    """Point-in-time feature construction rules."""

    return_windows: tuple[int, ...] = (1, 3, 5, 10, 20, 60, 120)
    short_windows: tuple[int, ...] = (1, 2, 3, 5)
    medium_windows: tuple[int, ...] = (5, 10, 20, 60)
    long_windows: tuple[int, ...] = (20, 60, 120)
    quantile_buckets: int = Field(default=5, ge=2)
    min_period_fraction: float = Field(default=0.6, ge=0.1, le=1.0)
    min_traded_observation_fraction: float = Field(default=0.6, ge=0.1, le=1.0)
    downside_mar: float = 0.0
    benchmark_index_code: str = "000300.SH"
    include_fundamentals: bool = True
    enable_industry_features: bool = False
    enable_unsafe_fina_indicator_features: bool = False

    @model_validator(mode="after")
    def reject_unsafe_research_features(self) -> FeatureSettings:
        """Reject feature families without production-grade PIT guarantees."""

        if self.enable_industry_features:
            raise ValueError(
                "industry-dependent features are disabled because no verified "
                "point-in-time industry source is configured"
            )
        if self.enable_unsafe_fina_indicator_features:
            raise ValueError(
                "direct fina_indicator features are disabled because local "
                "fina_indicator data lacks f_ann_date and is not revision-safe"
            )
        return self


class ModelDriftDiagnosticSettings(BaseModel):
    """Read-only champion drift diagnostics and deterministic sampling rules."""

    label_horizon: int = Field(default=5, gt=0)
    psi_bins: int = Field(default=10, ge=2, le=50)
    score_reference_months: int = Field(default=12, ge=1)
    reference_sample_rows: int = Field(default=200_000, ge=100)
    evaluation_sample_rows_per_month: int = Field(default=50_000, ge=100)
    minimum_daily_cross_section: int = Field(default=20, ge=3)
    response_bucket_counts: tuple[int, ...] = (5, 10)
    psi_warning_threshold: float = Field(default=0.10, ge=0)
    psi_severe_threshold: float = Field(default=0.25, gt=0)

    @model_validator(mode="after")
    def validate_model_drift(self) -> ModelDriftDiagnosticSettings:
        """Require ordered thresholds and unique supported response buckets."""

        if self.psi_warning_threshold >= self.psi_severe_threshold:
            raise ValueError("model drift PSI warning threshold must be below severe threshold")
        if tuple(sorted(set(self.response_bucket_counts))) != self.response_bucket_counts:
            raise ValueError("model drift response bucket counts must be unique and ascending")
        if any(value < 2 for value in self.response_bucket_counts):
            raise ValueError("model drift response bucket counts must be at least 2")
        return self


class WalkForwardPlanSettings(BaseModel):
    """Trading-session boundaries for purged walk-forward experiment plans."""

    annual_sessions: PositiveInt = 252
    minimum_training_years: PositiveInt = 5
    rolling_window_years: PositiveInt = 5
    validation_sessions: PositiveInt = 252
    purge_days: int = Field(default=6, ge=0)
    embargo_days: int = Field(default=6, ge=0)
    evaluation_frequency: Literal["monthly"] = "monthly"


class HorizonExperimentSettings(BaseModel):
    """One independently trained future multi-horizon challenger specification."""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    horizon: Literal[5, 10, 20, 60]
    holding_days: PositiveInt
    execution_rule: Literal["next_open"] = "next_open"

    @model_validator(mode="after")
    def validate_holding_period(self) -> HorizonExperimentSettings:
        """Prevent label and simulated holding horizons from diverging."""

        if self.holding_days != self.horizon:
            raise ValueError(
                "models.horizon_experiments holding_days must equal horizon: "
                f"{self.name} has horizon={self.horizon}, holding_days={self.holding_days}"
            )
        return self


class ChallengerEvaluationSettings(BaseModel):
    """Frozen post-training comparison and manual promotion-gate thresholds."""

    score_layers: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50)
    minimum_cross_section: PositiveInt = 20
    minimum_labelled_days: PositiveInt = 60
    regime_return_threshold: float = Field(default=0.005, ge=0.0)
    minimum_rank_ic: float = 0.0
    minimum_rank_ic_delta: float = 0.0
    minimum_positive_ic_ratio: float = Field(default=0.50, ge=0.0, le=1.0)
    minimum_top10_return_delta: float = 0.0

    @model_validator(mode="after")
    def validate_score_layers(self) -> ChallengerEvaluationSettings:
        """Require the fixed Top-1/5/10/20/50 comparison layers."""

        required = (0.01, 0.05, 0.10, 0.20, 0.50)
        if self.score_layers != required:
            raise ValueError(
                "models.challenger_evaluation.score_layers must be [0.01, 0.05, 0.10, 0.20, 0.50]"
            )
        return self


class ShadowChallengerModelSettings(BaseModel):
    """One fixed candidate model used for prospective shadow scoring."""

    model_id: str = Field(min_length=1)


class ShadowEnsembleSettings(BaseModel):
    """Deterministic multi-horizon shadow ensemble configuration."""

    enabled: bool = True
    fusion_method: Literal["percentile_mean"] = "percentile_mean"


class ShadowPredictionSettings(BaseModel):
    """Prospective-only shadow scoring configuration."""

    enabled: bool = True
    access_policy: Literal["prospective_production", "frozen_oos_evaluation"] = (
        "prospective_production"
    )
    challenger_models: dict[str, ShadowChallengerModelSettings] = Field(default_factory=dict)
    ensemble: ShadowEnsembleSettings = Field(default_factory=ShadowEnsembleSettings)

    @model_validator(mode="after")
    def validate_shadow_models(self) -> ShadowPredictionSettings:
        """Require exactly one unique model for each supported horizon."""

        required = {"h5", "h10", "h20", "h60"}
        configured = set(self.challenger_models)
        if configured != required:
            raise ValueError(
                "models.shadow_predictions.challenger_models must contain exactly "
                f"{sorted(required)}; configured={sorted(configured)}"
            )
        model_ids = [item.model_id for item in self.challenger_models.values()]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("shadow challenger model_id values must be unique")
        return self


class ModelExperimentSettings(BaseModel):
    """Read-only experiment planning configuration."""

    horizon_experiments: tuple[HorizonExperimentSettings, ...] = (
        HorizonExperimentSettings(name="h5", horizon=5, holding_days=5),
        HorizonExperimentSettings(name="h10", horizon=10, holding_days=10),
        HorizonExperimentSettings(name="h20", horizon=20, holding_days=20),
        HorizonExperimentSettings(name="h60", horizon=60, holding_days=60),
    )
    selection_period: HistoricalBacktestPeriodSettings = HistoricalBacktestPeriodSettings(
        start_date="20150101", end_date="20221231"
    )
    final_test_period: HistoricalBacktestPeriodSettings = HistoricalBacktestPeriodSettings(
        start_date="20230101", end_date="20260710"
    )
    challenger_evaluation: ChallengerEvaluationSettings = Field(
        default_factory=ChallengerEvaluationSettings
    )
    shadow_predictions: ShadowPredictionSettings = Field(
        default_factory=lambda: ShadowPredictionSettings(
            challenger_models={
                "h5": ShadowChallengerModelSettings(model_id="configure_h5_candidate"),
                "h10": ShadowChallengerModelSettings(model_id="configure_h10_candidate"),
                "h20": ShadowChallengerModelSettings(model_id="configure_h20_candidate"),
                "h60": ShadowChallengerModelSettings(model_id="configure_h60_candidate"),
            }
        )
    )

    @model_validator(mode="after")
    def validate_unique_experiments(self) -> ModelExperimentSettings:
        """Require unique names and horizons so each target owns one experiment."""

        if not self.horizon_experiments:
            raise ValueError("models.horizon_experiments must not be empty")
        names = [experiment.name for experiment in self.horizon_experiments]
        horizons = [experiment.horizon for experiment in self.horizon_experiments]
        if len(names) != len(set(names)):
            raise ValueError("models.horizon_experiments names must be unique")
        if len(horizons) != len(set(horizons)):
            raise ValueError("models.horizon_experiments horizons must be unique")
        if self.selection_period.start_date > self.selection_period.end_date:
            raise ValueError("models.selection_period is reversed")
        if self.final_test_period.start_date > self.final_test_period.end_date:
            raise ValueError("models.final_test_period is reversed")
        if self.selection_period.end_date >= self.final_test_period.start_date:
            raise ValueError("models selection_period must end before final_test_period starts")
        return self


class DiagnosticSettings(BaseModel):
    """Leakage-controlled feature diagnostics and selection rules."""

    label_horizon: int = Field(default=5, gt=0)
    minimum_coverage: float = Field(default=0.4, ge=0.0, le=1.0)
    minimum_daily_cross_section: int = Field(default=20, ge=3)
    minimum_ic_days: int = Field(default=60, ge=2)
    correlation_threshold: float = Field(default=0.85, gt=0.0, lt=1.0)
    regime_return_threshold: float = Field(default=0.005, ge=0.0)
    model_sample_rows: int = Field(default=500_000, ge=100)
    correlation_sample_rows: int = Field(default=200_000, ge=100)
    candidate_feature_counts: tuple[int, ...] = (30, 50, 70, 100, 130)
    top_fraction: float = Field(default=0.1, gt=0.0, le=0.5)
    annualization_days: int = Field(default=252, ge=1)
    random_seed: int = 42
    lgbm_num_boost_round: int = Field(default=150, ge=1)
    lgbm_learning_rate: float = Field(default=0.05, gt=0.0)
    lgbm_num_leaves: int = Field(default=31, ge=2)
    lgbm_min_data_in_leaf: int = Field(default=100, ge=1)
    lgbm_feature_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    lgbm_bagging_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    lgbm_bagging_freq: int = Field(default=1, ge=0)
    permutation_repeats: int = Field(default=1, ge=1, le=10)
    model_drift: ModelDriftDiagnosticSettings = Field(default_factory=ModelDriftDiagnosticSettings)


class RankerSettings(BaseModel):
    """Fixed LightGBM Ranker baseline experiment settings."""

    label_horizon: int = Field(default=5, gt=0)
    relevance_grades: int = Field(default=5, ge=2, le=31)
    train_start: str = "20100101"
    train_end: str = "20191231"
    validation_start: str = "20200101"
    validation_end: str = "20221231"
    test_start: str = "20230101"
    test_end: str = "20260710"
    recommended_features_path: Path = Path("reports/feature_diagnostics/latest.json")
    robust_features_path: Path = Path("config/feature_sets/robust_features.json")
    n_estimators: int = Field(default=300, ge=1)
    learning_rate: float = Field(default=0.03, gt=0.0)
    num_leaves: int = Field(default=31, ge=2)
    min_child_samples: int = Field(default=200, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    bagging_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    bagging_freq: int = Field(default=1, ge=0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    random_seed: int = 42
    minimum_group_size: int = Field(default=20, ge=2)
    ndcg_at: tuple[int, ...] = (10, 50)
    portfolio_fractions: tuple[float, ...] = (0.05, 0.10)
    walk_forward: WalkForwardPlanSettings = Field(default_factory=WalkForwardPlanSettings)

    @model_validator(mode="after")
    def validate_chronological_splits(self) -> RankerSettings:
        """Require fixed, non-overlapping chronological experiment periods."""

        dates = (
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.test_start,
            self.test_end,
        )
        if any(len(value) != 8 or not value.isdigit() for value in dates):
            raise ValueError("ranker split dates must use YYYYMMDD")
        if not (
            self.train_start
            <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError("ranker train, validation, and test periods must not overlap")
        if any(value <= 0 or value > 1 for value in self.portfolio_fractions):
            raise ValueError("ranker portfolio fractions must be in (0, 1]")
        return self


class ProductionModelSettings(BaseModel):
    """Final production Ranker training settings after validation approval."""

    train_start: str = "20100101"
    train_end: str = "20260710"
    feature_list_path: Path = Path("config/feature_sets/robust_features.json")
    output_dir_name: str = "production"

    @model_validator(mode="after")
    def validate_training_range(self) -> ProductionModelSettings:
        """Require one chronological production training period."""

        if any(
            len(value) != 8 or not value.isdigit() for value in (self.train_start, self.train_end)
        ):
            raise ValueError("production model train dates must use YYYYMMDD")
        if self.train_start > self.train_end:
            raise ValueError("production model train_start must be <= train_end")
        if not self.output_dir_name or "/" in self.output_dir_name:
            raise ValueError("production model output_dir_name must be a simple directory name")
        return self


class ProductionFreshnessSettings(BaseModel):
    """Session-aware raw and processed artifact readiness thresholds."""

    hard_datasets: tuple[str, ...] = (
        "daily",
        "adj_factor",
        "daily_basic",
        "stk_limit",
        "index_daily",
    )
    legitimate_empty_datasets: tuple[str, ...] = ("suspend_d",)
    soft_dataset_max_lag_calendar_days: dict[str, int] = Field(
        default_factory=lambda: {
            "income": 550,
            "balancesheet": 550,
            "cashflow": 550,
            "fina_indicator": 550,
        }
    )
    event_datasets: tuple[str, ...] = ("namechange",)
    snapshot_max_age_days: dict[str, int] = Field(default_factory=lambda: {"stock_basic": 14})
    required_index_codes: tuple[str, ...] = ()
    baseline_sessions: int = Field(default=20, ge=2)
    minimum_baseline_sessions: int = Field(default=5, ge=1)
    moderate_count_ratio_low: float = Field(default=0.80, gt=0)
    moderate_count_ratio_high: float = Field(default=1.20, gt=0)
    severe_count_ratio_low: float = Field(default=0.65, gt=0)
    severe_count_ratio_high: float = Field(default=1.35, gt=0)
    minimum_daily_rows: int = Field(default=1000, ge=1)
    minimum_universe_rows: int = Field(default=1000, ge=1)
    minimum_base_universe_rows: int = Field(default=500, ge=1)
    minimum_model_universe_rows: int = Field(default=100, ge=1)
    required_feature_list_path: Path | None = None
    hard_required_features: tuple[str, ...] = ()
    warning_features: tuple[str, ...] = (
        "ret_1d",
        "market_excess_ret_5d",
        "turnover_rate",
    )
    structurally_sparse_features: tuple[str, ...] = (
        "current_ratio",
        "debt_to_assets",
        "ocf_to_profit",
        "dv_ttm",
    )
    hard_feature_missing_ratio: float = Field(default=0.20, ge=0, le=1)
    warning_feature_missing_ratio: float = Field(default=0.50, ge=0, le=1)
    git_dirty_policy: Literal["ignore", "warning", "fail"] = "warning"

    @model_validator(mode="after")
    def validate_threshold_order(self) -> ProductionFreshnessSettings:
        """Require warning bands to sit inside hard-failure bands."""

        if not (
            self.severe_count_ratio_low
            <= self.moderate_count_ratio_low
            <= self.moderate_count_ratio_high
            <= self.severe_count_ratio_high
        ):
            raise ValueError("production freshness count-ratio thresholds are inconsistent")
        if self.minimum_baseline_sessions > self.baseline_sessions:
            raise ValueError(
                "production freshness minimum_baseline_sessions must not exceed baseline_sessions"
            )
        return self


class ProductionSchedulerSettings(BaseModel):
    """Single-host production timer behavior."""

    enabled: bool = True
    skip_if_already_successful: bool = True
    max_pipeline_attempts: int = Field(default=1, ge=1, le=3)
    retry_backoff_seconds: float = Field(default=30.0, ge=0)


class ProductionSettings(BaseModel):
    """Single-host production orchestration settings."""

    timezone: str = "Asia/Shanghai"
    market_data_ready_time: str = "18:30"
    scheduler: ProductionSchedulerSettings = Field(default_factory=ProductionSchedulerSettings)
    freshness: ProductionFreshnessSettings = Field(default_factory=ProductionFreshnessSettings)

    @model_validator(mode="after")
    def validate_scheduler_clock(self) -> ProductionSettings:
        """Require an available timezone and an HH:MM readiness time."""

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"production.timezone is unknown: {self.timezone}") from error
        try:
            parsed = time.fromisoformat(self.market_data_ready_time)
        except ValueError as error:
            raise ValueError("production.market_data_ready_time must use HH:MM") from error
        if (
            len(self.market_data_ready_time) != 5
            or parsed.strftime("%H:%M") != self.market_data_ready_time
        ):
            raise ValueError("production.market_data_ready_time must use HH:MM")
        return self


class CandidateSelectionSettings(BaseModel):
    """Point-in-time filters applied to model scores before any trading decision."""

    max_candidates: PositiveInt = 50
    require_model_universe: bool = True
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_low_liquidity: bool = True
    exclude_bj_market: bool = True
    exclude_star_market: bool = False
    exclude_chinext_market: bool = False
    min_list_trading_days: int = Field(default=180, ge=0)
    require_daily_row: bool = True
    require_daily_basic_row: bool = True
    require_stk_limit_row: bool = True
    min_total_mv: float | None = Field(default=500_000.0, ge=0)
    min_daily_amount: float | None = Field(default=30_000.0, ge=0)
    min_turnover_rate: float | None = Field(default=None, ge=0)
    max_turnover_rate: float | None = Field(default=None, ge=0)
    require_valid_ohlc: bool = True
    require_valid_price_limits: bool = True
    price_limit_tolerance: float = Field(default=1e-6, ge=0)

    @model_validator(mode="after")
    def validate_turnover_range(self) -> CandidateSelectionSettings:
        """Require an ordered optional turnover-rate interval."""

        if (
            self.min_turnover_rate is not None
            and self.max_turnover_rate is not None
            and self.min_turnover_rate > self.max_turnover_rate
        ):
            raise ValueError("candidate min_turnover_rate must not exceed max_turnover_rate")
        return self


class StrategySettings(BaseModel):
    """Configuration for model-score post-processing without order generation."""

    candidate_selection: CandidateSelectionSettings = Field(
        default_factory=CandidateSelectionSettings
    )


class DailyResearchReportSettings(BaseModel):
    """Thresholds for descriptive same-date candidate risk reporting."""

    top_candidates: PositiveInt = 50
    abnormal_return_abs_pct: float = Field(default=9.0, ge=0)
    volatility_window: PositiveInt = 20
    volatility_min_observations: PositiveInt = 10
    high_volatility_pct: float = Field(default=4.0, ge=0)
    low_liquidity_amount: float = Field(default=50_000.0, ge=0)

    @model_validator(mode="after")
    def validate_volatility_observations(self) -> DailyResearchReportSettings:
        """Keep minimum observations within the configured trailing window."""

        if self.volatility_min_observations > self.volatility_window:
            raise ValueError(
                "research volatility_min_observations must not exceed volatility_window"
            )
        return self


class ExplainabilitySettings(BaseModel):
    """Thresholds for read-only local model-score explanations."""

    top_positive_features: PositiveInt = 5
    top_negative_features: PositiveInt = 3
    strong_percentile: float = Field(default=0.99, gt=0, le=1)
    moderate_percentile: float = Field(default=0.95, gt=0, le=1)
    minimum_history_sessions: PositiveInt = 20
    high_confidence_history_sessions: PositiveInt = 60
    maximum_history_sessions: PositiveInt = 252
    score_tolerance: float = Field(default=1e-7, gt=0)
    contribution_tolerance: float = Field(default=1e-6, gt=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> ExplainabilitySettings:
        """Require ordered signal and same-model history thresholds."""

        if self.moderate_percentile > self.strong_percentile:
            raise ValueError(
                "research explainability moderate_percentile must not exceed strong_percentile"
            )
        if self.minimum_history_sessions > self.high_confidence_history_sessions:
            raise ValueError(
                "research explainability minimum_history_sessions must not exceed "
                "high_confidence_history_sessions"
            )
        if self.high_confidence_history_sessions > self.maximum_history_sessions:
            raise ValueError(
                "research explainability high_confidence_history_sessions must not exceed "
                "maximum_history_sessions"
            )
        return self


class DecisionSupportSettings(BaseModel):
    """Configurable same-session observations for human investment review."""

    gap_feature: str = "gap_mean_1d"
    ma20_feature: str = "ma_ratio_20d"
    amount_ratio_feature: str = "amount_ratio_20d"
    liquidity_feature: str = "amihud_20d"
    short_return_feature: str = "ret_5d"
    volatility_feature: str = "realized_vol_20d"
    maximum_abs_open_gap: float = Field(default=0.03, ge=0)
    minimum_ma20_ratio: float = 0.0
    minimum_amount_ratio: float = Field(default=0.8, ge=0)
    minimum_turnover_rate: float = Field(default=0.5, ge=0)
    maximum_amihud: float = Field(default=1e-5, gt=0)
    excessive_short_return: float = Field(default=0.15, ge=0)
    elevated_volatility: float = Field(default=0.04, ge=0)
    liquidity_decline_amount_ratio: float = Field(default=0.6, ge=0)
    score_tolerance: float = Field(default=1e-7, gt=0)


class ResearchAgentSettings(BaseModel):
    """Provider-neutral, read-only daily research-agent settings."""

    enabled: bool = True
    provider: Literal["openai", "claude", "gemini", "deepseek"] = "openai"
    model: str = Field(default="gpt-4.1-mini", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    timeout_seconds: PositiveInt = 60
    max_retries: int = Field(default=2, ge=0, le=3)
    max_output_tokens: PositiveInt = 4_000
    top_candidates: PositiveInt = 20
    prompt_version: str = Field(default="v2", min_length=1)
    allow_advisory_language: bool = True


class ResearchSettings(BaseModel):
    """Human-readable quantitative research reporting configuration."""

    daily_report: DailyResearchReportSettings = Field(default_factory=DailyResearchReportSettings)
    explainability: ExplainabilitySettings = Field(default_factory=ExplainabilitySettings)
    decision_support: DecisionSupportSettings = Field(default_factory=DecisionSupportSettings)
    agent: ResearchAgentSettings = Field(default_factory=ResearchAgentSettings)


class PromotionReviewSettings(BaseModel):
    """Human-review authorization and approval expiry policy."""

    reviewer_allowlist: tuple[str, ...] = ()
    allow_requester_as_reviewer: bool = False
    review_expire_hours: PositiveInt = 72

    @model_validator(mode="after")
    def validate_reviewers(self) -> PromotionReviewSettings:
        """Require unique, non-empty OS account names."""

        if any(not value.strip() for value in self.reviewer_allowlist):
            raise ValueError("promotion.reviewer_allowlist contains an empty reviewer")
        if len(self.reviewer_allowlist) != len(set(self.reviewer_allowlist)):
            raise ValueError("promotion.reviewer_allowlist contains duplicate reviewers")
        return self


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
    diagnostics: DiagnosticSettings = Field(default_factory=DiagnosticSettings)
    ranker: RankerSettings = Field(default_factory=RankerSettings)
    models: ModelExperimentSettings = Field(default_factory=ModelExperimentSettings)
    production_model: ProductionModelSettings = Field(default_factory=ProductionModelSettings)
    production: ProductionSettings = Field(default_factory=ProductionSettings)
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    promotion: PromotionReviewSettings = Field(default_factory=PromotionReviewSettings)
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
        configured_labels = set(self.labels.horizons)
        planned_horizons = {experiment.horizon for experiment in self.models.horizon_experiments}
        missing_horizons = sorted(planned_horizons - configured_labels)
        if missing_horizons:
            raise ValueError(
                "models.horizon_experiments require horizons missing from labels.horizons: "
                f"{missing_horizons}"
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
