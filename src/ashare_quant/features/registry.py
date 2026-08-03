"""Feature registry metadata for candidate stock-selection features."""

from __future__ import annotations

from dataclasses import dataclass, replace

INDUSTRY_FEATURE_DISABLED_REASON = (
    "disabled: current stock_basic.industry is not a verified point-in-time historical source"
)
FINA_INDICATOR_DISABLED_REASON = (
    "disabled: local fina_indicator lacks f_ann_date, so revisions cannot be assigned "
    "revision-safe availability dates; update_flag is not an availability timestamp"
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Describe one feature's lineage and point-in-time availability."""

    name: str
    family: str
    required_source_columns: tuple[str, ...]
    lookback: int
    minimum_history: int
    availability_lag: str
    description: str
    point_in_time_safe: bool = True
    enabled: bool = True
    disabled_reason: str | None = None
    source_datasets: tuple[str, ...] = ()


def build_feature_registry() -> tuple[FeatureSpec, ...]:
    """Return the candidate feature registry.

    All daily market features are available after the market close of
    `trade_date`. Financial features are available only when their explicit
    announcement date is not later than `trade_date`.
    """

    specs: list[FeatureSpec] = []
    return_windows = (1, 3, 5, 10, 20, 60, 120)
    short_windows = (1, 2, 3, 5)
    medium_windows = (5, 10, 20, 60)
    long_windows = (20, 60, 120)

    for window in return_windows:
        specs.extend(
            [
                market_spec(f"ret_{window}d", "returns_momentum", window, "Adjusted close return."),
                market_spec(
                    f"logret_sum_{window}d",
                    "returns_momentum",
                    window,
                    "Rolling sum of log adjusted returns.",
                ),
                market_spec(
                    f"market_excess_ret_{window}d",
                    "market_relative_momentum",
                    window,
                    "Stock adjusted return minus benchmark return.",
                ),
                market_spec(
                    f"industry_excess_ret_{window}d",
                    "industry_relative_momentum",
                    window,
                    "Stock return minus same-industry average return.",
                ),
            ]
        )
    for window in short_windows:
        specs.extend(
            [
                market_spec(
                    f"reversal_ret_{window}d",
                    "short_term_reversal",
                    window,
                    "Negative recent return.",
                ),
                market_spec(
                    f"gap_mean_{window}d",
                    "gap_behavior",
                    window,
                    "Rolling mean of open-to-previous-close gaps.",
                ),
            ]
        )
    for window in medium_windows:
        specs.extend(
            [
                market_spec(
                    f"ma_ratio_{window}d",
                    "moving_average_relative_position",
                    window,
                    "Adjusted close divided by moving average minus one.",
                ),
                market_spec(
                    f"ma_z_{window}d",
                    "moving_average_relative_position",
                    window,
                    "Adjusted close z-score around moving average.",
                ),
                market_spec(
                    f"trend_slope_{window}d",
                    "trend_strength",
                    window,
                    "Moving-average slope over the window.",
                ),
                market_spec(
                    f"positive_ret_ratio_{window}d",
                    "trend_strength",
                    window,
                    "Share of positive daily returns in the window.",
                ),
                market_spec(
                    f"realized_vol_{window}d",
                    "realized_volatility",
                    window,
                    "Rolling standard deviation of adjusted daily returns.",
                ),
                market_spec(
                    f"downside_vol_{window}d",
                    "downside_volatility",
                    window,
                    "Rolling standard deviation of negative adjusted returns.",
                ),
                market_spec(
                    f"volume_ratio_{window}d",
                    "volume_dynamics",
                    window,
                    "Volume divided by trailing average volume.",
                ),
                market_spec(
                    f"amount_ratio_{window}d",
                    "amount_liquidity",
                    window,
                    "Amount divided by trailing average amount.",
                ),
                market_spec(
                    f"turnover_ratio_{window}d",
                    "turnover_dynamics",
                    window,
                    "Turnover divided by trailing average turnover.",
                ),
                market_spec(
                    f"amount_cv_{window}d",
                    "amount_liquidity",
                    window,
                    "Coefficient of variation of trading amount.",
                ),
            ]
        )
    for window in long_windows:
        specs.extend(
            [
                market_spec(
                    f"dist_high_{window}d",
                    "distance_rolling_high_low",
                    window,
                    "Distance from rolling high adjusted close.",
                ),
                market_spec(
                    f"dist_low_{window}d",
                    "distance_rolling_high_low",
                    window,
                    "Distance from rolling low adjusted close.",
                ),
                market_spec(
                    f"range_pos_{window}d",
                    "distance_rolling_high_low",
                    window,
                    "Position inside rolling high-low range.",
                ),
                market_spec(
                    f"drawdown_{window}d", "drawdown", window, "Current drawdown from rolling high."
                ),
                market_spec(
                    f"max_drawdown_{window}d",
                    "drawdown",
                    window,
                    "Worst daily drawdown inside the rolling window.",
                ),
                market_spec(
                    f"amihud_{window}d",
                    "amihud_illiquidity",
                    window,
                    "Rolling mean of absolute return divided by amount.",
                ),
                market_spec(
                    f"ret_amount_corr_{window}d",
                    "price_volume_correlation",
                    window,
                    "Rolling correlation between return and amount.",
                ),
                market_spec(
                    f"ret_volume_corr_{window}d",
                    "price_volume_correlation",
                    window,
                    "Rolling correlation between return and volume.",
                ),
                market_spec(
                    f"beta_{window}d",
                    "beta_residual_volatility",
                    window,
                    "Rolling beta to benchmark returns.",
                ),
                market_spec(
                    f"residual_vol_{window}d",
                    "beta_residual_volatility",
                    window,
                    "Rolling standard deviation of market-model residual returns.",
                ),
            ]
        )

    candle_names = {
        "intraday_ret": "Close-to-open intraday return.",
        "candle_body_pct": "Candle body divided by intraday range.",
        "upper_shadow_pct": "Upper shadow divided by intraday range.",
        "lower_shadow_pct": "Lower shadow divided by intraday range.",
        "close_location_value": "Close location inside high-low range.",
        "gap_open_ret": "Open versus previous adjusted close.",
        "gap_abs": "Absolute open gap.",
        "turnover_rate": "Daily turnover rate from daily_basic.",
    }
    specs.extend(
        FeatureSpec(
            name=name,
            family="intraday_candle_structure" if "gap" not in name else "gap_behavior",
            required_source_columns=("daily.open", "daily.high", "daily.low", "daily.close"),
            lookback=1,
            minimum_history=1,
            availability_lag="after_close_trade_date",
            description=description,
        )
        for name, description in candle_names.items()
    )

    valuation = {
        "earnings_yield": "Inverse PE.",
        "book_to_market": "Inverse PB.",
        "sales_yield": "Inverse PS.",
        "pe_ttm_inv": "Inverse trailing PE.",
        "ps_ttm_inv": "Inverse trailing PS.",
        "dv_ttm": "Trailing dividend yield.",
        "ln_total_mv": "Log total market capitalization.",
        "ln_circ_mv": "Log circulating market capitalization.",
    }
    specs.extend(
        daily_basic_spec(name, "valuation", description) for name, description in valuation.items()
    )

    fundamental_names = {
        "roe": (
            "Return on equity from point-in-time financial indicators.",
            ("fina_indicator.roe", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "roa": (
            "Return on assets from point-in-time financial indicators.",
            ("fina_indicator.roa", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "grossprofit_margin": (
            "Gross profit margin from announced financial data.",
            ("fina_indicator.grossprofit_margin", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "netprofit_margin": (
            "Net profit margin from announced financial data.",
            ("fina_indicator.netprofit_margin", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "debt_to_assets": (
            "Debt to assets from point-in-time balance sheet.",
            (
                "balancesheet.total_liab",
                "balancesheet.total_assets",
                "financial.availability_date",
            ),
            ("balancesheet",),
        ),
        "current_ratio": (
            "Current assets divided by current liabilities.",
            (
                "balancesheet.total_cur_assets",
                "balancesheet.total_cur_liab",
                "financial.availability_date",
            ),
            ("balancesheet",),
        ),
        "ocf_to_profit": (
            "Operating cash flow divided by net profit.",
            (
                "cashflow.n_cashflow_act",
                "income.n_income",
                "financial.availability_date",
            ),
            ("cashflow", "income"),
        ),
        "revenue_yoy": (
            "Operating revenue year-over-year growth from point-in-time indicators.",
            ("fina_indicator.or_yoy", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "netprofit_yoy": (
            "Net profit growth from point-in-time indicators.",
            ("fina_indicator.netprofit_yoy", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "roe_delta": (
            "Change in announced ROE versus previous announcement.",
            ("fina_indicator.roe", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "revenue_yoy_delta": (
            "Change in operating revenue growth versus previous announcement.",
            ("fina_indicator.or_yoy", "financial.availability_date"),
            ("fina_indicator",),
        ),
        "netprofit_yoy_delta": (
            "Change in net profit growth versus previous announcement.",
            ("fina_indicator.netprofit_yoy", "financial.availability_date"),
            ("fina_indicator",),
        ),
    }
    for name, (description, required_source_columns, source_datasets) in fundamental_names.items():
        specs.append(
            FeatureSpec(
                name=name,
                family=fundamental_family(name),
                required_source_columns=required_source_columns,
                lookback=0,
                minimum_history=0,
                availability_lag="financial_ann_date_lte_trade_date",
                description=description,
                source_datasets=source_datasets,
            )
        )

    rank_targets = tuple(
        spec.name for spec in specs if spec.family not in {"intraday_candle_structure"}
    )
    for base_name in rank_targets[:42]:
        specs.append(
            FeatureSpec(
                name=f"cs_rank_{base_name}",
                family="cross_sectional_percentile_ranks",
                required_source_columns=(base_name,),
                lookback=0,
                minimum_history=0,
                availability_lag=availability_for_base(specs, base_name),
                description=f"Cross-sectional percentile rank of {base_name}.",
            )
        )
    for base_name in rank_targets[:28]:
        specs.append(
            FeatureSpec(
                name=f"ind_rank_{base_name}",
                family="industry_neutral_percentile_ranks",
                required_source_columns=(base_name, "universe.industry"),
                lookback=0,
                minimum_history=0,
                availability_lag=availability_for_base(specs, base_name),
                description=f"Within-industry percentile rank of {base_name}.",
            )
        )
    return tuple(disable_unsafe_feature(spec) for spec in specs)


def disable_unsafe_feature(spec: FeatureSpec) -> FeatureSpec:
    """Mark features that lack production-grade PIT safety as disabled."""

    depends_on_industry = spec.family in {
        "industry_relative_momentum",
        "industry_neutral_percentile_ranks",
    } or any("industry_excess_ret_" in source for source in spec.required_source_columns)
    depends_on_fina_indicator = "fina_indicator" in spec.source_datasets
    if depends_on_industry:
        return replace(
            spec,
            point_in_time_safe=False,
            enabled=False,
            disabled_reason=INDUSTRY_FEATURE_DISABLED_REASON,
        )
    if depends_on_fina_indicator:
        return replace(
            spec,
            point_in_time_safe=False,
            enabled=False,
            disabled_reason=FINA_INDICATOR_DISABLED_REASON,
        )
    return replace(
        spec,
        source_datasets=spec.source_datasets or infer_source_datasets(spec.required_source_columns),
    )


def market_spec(name: str, family: str, window: int, description: str) -> FeatureSpec:
    """Create a daily market feature spec."""

    return FeatureSpec(
        name=name,
        family=family,
        required_source_columns=("daily", "adj_factor"),
        lookback=window,
        minimum_history=max(1, int(window * 0.6)),
        availability_lag="after_close_trade_date",
        description=description,
        source_datasets=("daily", "adj_factor"),
    )


def daily_basic_spec(name: str, family: str, description: str) -> FeatureSpec:
    """Create a daily_basic feature spec."""

    return FeatureSpec(
        name=name,
        family=family,
        required_source_columns=("daily_basic",),
        lookback=1,
        minimum_history=1,
        availability_lag="after_close_trade_date",
        description=description,
        source_datasets=("daily_basic",),
    )


def infer_source_datasets(required_source_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Infer source dataset names from dotted registry source columns."""

    datasets = []
    for source in required_source_columns:
        if "." not in source:
            continue
        dataset = source.split(".", 1)[0]
        if dataset != "financial" and dataset not in datasets:
            datasets.append(dataset)
    return tuple(datasets)


def fundamental_family(name: str) -> str:
    """Map fundamental feature names to broad families."""

    if name in {"roe", "roa", "grossprofit_margin", "netprofit_margin"}:
        return "profitability"
    if name in {
        "revenue_yoy",
        "netprofit_yoy",
        "roe_delta",
        "revenue_yoy_delta",
        "netprofit_yoy_delta",
    }:
        return "growth"
    if name in {"ocf_to_profit"}:
        return "cash_flow_quality"
    return "balance_sheet_quality"


def availability_for_base(specs: list[FeatureSpec], base_name: str) -> str:
    """Return availability metadata for a base feature name."""

    for spec in specs:
        if spec.name == base_name:
            return spec.availability_lag
    return "after_close_trade_date"


ALL_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = build_feature_registry()
FEATURE_REGISTRY: tuple[FeatureSpec, ...] = tuple(
    spec for spec in ALL_FEATURE_REGISTRY if spec.enabled and spec.point_in_time_safe
)
DISABLED_FEATURE_REGISTRY: tuple[FeatureSpec, ...] = tuple(
    spec for spec in ALL_FEATURE_REGISTRY if not spec.enabled
)
