from __future__ import annotations

import time

import pandas as pd
import pytest

from ashare_quant.config import load_settings
from ashare_quant.data.datasets import DatasetSpec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.features import (
    DISABLED_FEATURE_REGISTRY,
    FEATURE_REGISTRY,
    FeatureBuilder,
    build_feature_frame,
)
from ashare_quant.features.fundamentals import (
    build_fundamental_features,
    prepare_fina_indicator,
    validate_financial_registry_sources,
)


def feature_fixture_inputs(days: int = 140) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=days).strftime("%Y%m%d").tolist()
    codes = ("000001.SZ", "000002.SZ", "000003.SZ")
    industries = {"000001.SZ": "Bank", "000002.SZ": "Tech", "000003.SZ": "Tech"}
    daily_rows: list[dict[str, object]] = []
    adj_rows: list[dict[str, object]] = []
    daily_basic_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(dates):
        for code_index, code in enumerate(codes):
            base = 10.0 + code_index * 5.0
            close = base + day_index * (0.10 + code_index * 0.02)
            open_price = close - 0.03
            high = close + 0.20
            low = close - 0.25
            adj_factor = 1.0 + (0.01 if code == "000001.SZ" and day_index >= 30 else 0.0)
            daily_rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "vol": 1000 + day_index * 5 + code_index * 100,
                    "amount": (1000 + day_index * 5 + code_index * 100) * close,
                }
            )
            adj_rows.append({"ts_code": code, "trade_date": trade_date, "adj_factor": adj_factor})
            daily_basic_rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "turnover_rate": 1.0 + code_index * 0.1,
                    "pe": 10.0 + code_index,
                    "pe_ttm": 11.0 + code_index,
                    "pb": 1.5 + code_index * 0.2,
                    "ps": 2.0 + code_index * 0.1,
                    "ps_ttm": 2.1 + code_index * 0.1,
                    "dv_ttm": 1.0 + code_index,
                    "total_mv": 100000 + day_index * 100 + code_index * 1000,
                    "circ_mv": 80000 + day_index * 100 + code_index * 1000,
                }
            )
            universe_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "industry": industries[code],
                    "is_suspended": False,
                    "in_model_universe": True,
                    "in_base_universe": True,
                    "can_buy": True,
                    "can_sell": True,
                }
            )
    index_rows = [
        {
            "ts_code": "000300.SH",
            "trade_date": trade_date,
            "open": 1000.0 + index,
            "high": 1001.0 + index,
            "low": 999.0 + index,
            "close": 1000.0 + index * 1.2,
            "vol": 0.0,
            "amount": 0.0,
        }
        for index, trade_date in enumerate(dates)
    ]
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "ann_date": ["20240201", "20240601", "20240201"],
            "roe": [10.0, 20.0, 15.0],
            "roa": [5.0, 8.0, 7.0],
            "grossprofit_margin": [30.0, 35.0, 40.0],
            "netprofit_margin": [12.0, 14.0, 13.0],
            "or_yoy": [8.0, 12.0, 9.0],
            "netprofit_yoy": [6.0, 10.0, 7.0],
        }
    )
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20240201", "20240201"],
            "revenue": [100.0, 120.0],
            "n_income": [10.0, 12.0],
            "total_profit": [12.0, 15.0],
        }
    )
    balancesheet = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20240201", "20240201"],
            "total_assets": [200.0, 240.0],
            "total_liab": [100.0, 80.0],
            "total_cur_assets": [80.0, 90.0],
            "total_cur_liab": [40.0, 30.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "ann_date": ["20240201", "20240201"],
            "n_cashflow_act": [9.0, 15.0],
        }
    )
    return {
        "trade_cal": pd.DataFrame({"cal_date": dates, "is_open": 1}),
        "daily": pd.DataFrame(daily_rows),
        "adj_factor": pd.DataFrame(adj_rows),
        "daily_basic": pd.DataFrame(daily_basic_rows),
        "index_daily": pd.DataFrame(index_rows),
        "universe": pd.DataFrame(universe_rows),
        "fina_indicator": fina_indicator,
        "income": income,
        "balancesheet": balancesheet,
        "cashflow": cashflow,
    }


def test_feature_registry_count_is_in_target_range() -> None:
    assert 150 <= len(FEATURE_REGISTRY) <= 220
    assert all(spec.availability_lag for spec in FEATURE_REGISTRY)
    assert all(spec.enabled and spec.point_in_time_safe for spec in FEATURE_REGISTRY)


def test_features_are_deterministic_and_have_rank_boundaries() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()

    first = build_feature_frame(inputs, settings, "20240520", "20240520")
    second = build_feature_frame(inputs, settings, "20240520", "20240520")

    pd.testing.assert_frame_equal(first, second)
    rank_columns = [column for column in first.columns if column.startswith("cs_rank_")]
    assert rank_columns
    for column in rank_columns:
        values = first[column].dropna()
        if not values.empty:
            assert values.between(0, 1).all()


def test_market_features_fail_when_benchmark_code_is_absent() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()
    inputs["index_daily"] = inputs["index_daily"].assign(ts_code="000001.SH")

    with pytest.raises(DataValidationError, match="benchmark index 000300.SH is missing"):
        build_feature_frame(inputs, settings, "20240520", "20240520")


def test_features_do_not_depend_on_future_rows() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()
    mutated = {name: frame.copy() for name, frame in inputs.items()}
    future_mask = mutated["daily"]["trade_date"] > "20240520"
    mutated["daily"].loc[future_mask, "close"] = mutated["daily"].loc[future_mask, "close"] * 100.0

    baseline = build_feature_frame(inputs, settings, "20240520", "20240520")
    changed_future = build_feature_frame(mutated, settings, "20240520", "20240520")

    pd.testing.assert_frame_equal(baseline, changed_future)


def test_expected_nan_warmup_behavior() -> None:
    settings = load_settings("config/default.yaml")
    frame = build_feature_frame(feature_fixture_inputs(days=30), settings, "20240110", "20240110")

    assert frame["ret_120d"].isna().all()
    assert frame["realized_vol_60d"].isna().all()
    assert frame["ret_1d"].notna().all()


def test_ret_1d_is_null_after_one_day_suspension_gap() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    suspend_stock_dates(inputs, "000001.SZ", ["20240103"])

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")
    rows = frame.set_index("ts_code")

    assert pd.isna(rows.loc["000001.SZ", "ret_1d"])
    assert pd.notna(rows.loc["000002.SZ", "ret_1d"])


def test_ret_1d_is_null_after_multi_day_suspension_gap() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=10)
    suspend_stock_dates(inputs, "000001.SZ", ["20240103", "20240104", "20240105"])

    frame = build_feature_frame(inputs, settings, "20240108", "20240108")
    row = frame.set_index("ts_code").loc["000001.SZ"]

    assert pd.isna(row["ret_1d"])


def test_rolling_amount_window_respects_calendar_and_minimum_observations() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=10)
    suspend_stock_dates(inputs, "000001.SZ", ["20240103", "20240104", "20240105", "20240108"])

    frame = build_feature_frame(inputs, settings, "20240109", "20240109")
    rows = frame.set_index("ts_code")

    assert pd.isna(rows.loc["000001.SZ", "amount_ratio_5d"])
    assert pd.notna(rows.loc["000002.SZ", "amount_ratio_5d"])


def test_rolling_amount_minimum_valid_observation_threshold_is_configurable() -> None:
    settings = load_settings("config/default.yaml")
    settings = settings.model_copy(
        update={
            "features": settings.features.model_copy(
                update={"min_traded_observation_fraction": 0.2}
            )
        }
    )
    inputs = feature_fixture_inputs(days=10)
    suspend_stock_dates(inputs, "000001.SZ", ["20240103", "20240104", "20240105", "20240108"])

    frame = build_feature_frame(inputs, settings, "20240109", "20240109")
    row = frame.set_index("ts_code").loc["000001.SZ"]

    assert pd.notna(row["amount_ratio_5d"])


def test_resumed_stock_does_not_use_stale_liquidity_before_suspension() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=10)
    suspend_stock_dates(inputs, "000001.SZ", ["20240103", "20240104", "20240105", "20240108"])

    frame = build_feature_frame(inputs, settings, "20240109", "20240109")
    row = frame.set_index("ts_code").loc["000001.SZ"]

    assert pd.isna(row["amount_ratio_5d"])
    assert pd.isna(row["turnover_ratio_5d"])


def test_st_stock_does_not_affect_cross_sectional_rank() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    set_one_day_return_fixture(inputs, "20240103", "20240104")
    mark_ineligible(inputs, "000003.SZ", "20240104")

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")
    rows = frame.set_index("ts_code")

    assert rows.loc["000001.SZ", "cs_rank_ret_1d"] == pytest.approx(0.5)
    assert rows.loc["000002.SZ", "cs_rank_ret_1d"] == pytest.approx(1.0)
    assert pd.isna(rows.loc["000003.SZ", "cs_rank_ret_1d"])


def test_low_liquidity_stock_does_not_affect_cross_sectional_rank() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    set_one_day_return_fixture(inputs, "20240103", "20240104")
    mark_ineligible(inputs, "000003.SZ", "20240104")

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")
    rows = frame.set_index("ts_code")

    assert rows.loc["000002.SZ", "cs_rank_ret_1d"] == pytest.approx(1.0)
    assert pd.isna(rows.loc["000003.SZ", "cs_rank_ret_1d"])


def test_new_stock_does_not_affect_cross_sectional_rank() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    set_one_day_return_fixture(inputs, "20240103", "20240104")
    mark_ineligible(inputs, "000003.SZ", "20240104")

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")
    rows = frame.set_index("ts_code")

    assert rows.loc["000001.SZ", "cs_rank_ret_1d"] == pytest.approx(0.5)
    assert pd.isna(rows.loc["000003.SZ", "cs_rank_ret_1d"])


def test_same_date_same_value_rank_is_deterministic_for_eligible_stocks() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    set_one_day_return_fixture(
        inputs,
        "20240103",
        "20240104",
        current_closes={"000001.SZ": 11.0, "000002.SZ": 11.0, "000003.SZ": 100.0},
    )
    mark_ineligible(inputs, "000003.SZ", "20240104")

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")
    rows = frame.set_index("ts_code")

    assert rows.loc["000001.SZ", "cs_rank_ret_1d"] == pytest.approx(0.75)
    assert rows.loc["000002.SZ", "cs_rank_ret_1d"] == pytest.approx(0.75)
    assert pd.isna(rows.loc["000003.SZ", "cs_rank_ret_1d"])


def test_reversal_ret_2d_matches_negative_two_day_return() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 4)
    set_return_sequence(inputs, "000001.SZ", dates[:3], [0.10, -0.05])

    frame = build_feature_frame(inputs, settings, dates[2], dates[2])
    row = frame.set_index("ts_code").loc["000001.SZ"]

    expected_return = (1.10 * 0.95) - 1.0
    assert row["reversal_ret_2d"] == pytest.approx(-expected_return)


def test_reversal_ret_2d_is_null_during_warmup_and_suspension_gap() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 4)

    warmup = build_feature_frame(inputs, settings, dates[1], dates[1])
    assert pd.isna(warmup.set_index("ts_code").loc["000001.SZ", "reversal_ret_2d"])

    suspend_stock_dates(inputs, "000001.SZ", [dates[1]])
    after_gap = build_feature_frame(inputs, settings, dates[2], dates[2])
    assert pd.isna(after_gap.set_index("ts_code").loc["000001.SZ", "reversal_ret_2d"])


def test_cs_rank_reversal_ret_2d_uses_only_model_universe() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 3)
    set_return_sequence(inputs, "000001.SZ", dates, [0.10, 0.00])
    set_return_sequence(inputs, "000002.SZ", dates, [0.00, -0.10])
    set_return_sequence(inputs, "000003.SZ", dates, [-0.50, -0.50])
    mark_ineligible(inputs, "000003.SZ", dates[2])

    frame = build_feature_frame(inputs, settings, dates[2], dates[2])
    rows = frame.set_index("ts_code")

    assert rows.loc["000001.SZ", "cs_rank_reversal_ret_2d"] == pytest.approx(0.5)
    assert rows.loc["000002.SZ", "cs_rank_reversal_ret_2d"] == pytest.approx(1.0)
    assert pd.isna(rows.loc["000003.SZ", "cs_rank_reversal_ret_2d"])


def test_registered_active_features_include_non_null_reversal_ret_2d_outputs() -> None:
    settings = load_settings("config/default.yaml")
    frame = build_feature_frame(feature_fixture_inputs(), settings, "20240520", "20240524")

    assert {spec.name for spec in FEATURE_REGISTRY} <= set(frame.columns)
    assert frame["reversal_ret_2d"].notna().any()
    assert frame["cs_rank_reversal_ret_2d"].notna().any()


def test_industry_dependent_features_are_not_generated_from_current_industry() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    set_one_day_return_fixture(inputs, "20240103", "20240104")

    frame = build_feature_frame(inputs, settings, "20240104", "20240104")

    assert not any(column.startswith("industry_excess_ret_") for column in frame.columns)
    assert not any(column.startswith("ind_rank_") for column in frame.columns)
    assert "cs_rank_ret_1d" in frame.columns


def test_current_industry_metadata_cannot_enter_production_feature_registry() -> None:
    active_names = {spec.name for spec in FEATURE_REGISTRY}

    assert "industry" not in active_names
    assert not any("universe.industry" in spec.required_source_columns for spec in FEATURE_REGISTRY)
    assert not any("industry_excess_ret_" in spec.name for spec in FEATURE_REGISTRY)
    assert not any(spec.name.startswith("ind_rank_") for spec in FEATURE_REGISTRY)


def test_disabled_industry_features_have_explicit_unsafe_metadata() -> None:
    assert DISABLED_FEATURE_REGISTRY
    assert all(not spec.enabled for spec in DISABLED_FEATURE_REGISTRY)
    assert all(not spec.point_in_time_safe for spec in DISABLED_FEATURE_REGISTRY)
    assert all(spec.disabled_reason for spec in DISABLED_FEATURE_REGISTRY)


def test_unsafe_fina_indicator_features_are_excluded_by_default() -> None:
    disabled_names = {
        spec.name for spec in DISABLED_FEATURE_REGISTRY if "fina_indicator" in spec.source_datasets
    }
    active_names = {spec.name for spec in FEATURE_REGISTRY}

    assert disabled_names == {
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "revenue_yoy",
        "netprofit_yoy",
        "roe_delta",
        "revenue_yoy_delta",
        "netprofit_yoy_delta",
    }
    assert active_names.isdisjoint(disabled_names)


def test_statement_derived_financial_features_remain_enabled() -> None:
    active = {spec.name: spec for spec in FEATURE_REGISTRY}

    assert {"debt_to_assets", "current_ratio", "ocf_to_profit"}.issubset(active)
    assert active["debt_to_assets"].source_datasets == ("balancesheet",)
    assert active["current_ratio"].source_datasets == ("balancesheet",)
    assert active["ocf_to_profit"].source_datasets == ("cashflow", "income")


def test_update_flag_does_not_make_fina_indicator_revision_safe() -> None:
    disabled = {
        spec.name: spec
        for spec in DISABLED_FEATURE_REGISTRY
        if "fina_indicator" in spec.source_datasets
    }

    assert disabled["revenue_yoy"].point_in_time_safe is False
    assert "update_flag is not an availability timestamp" in disabled["revenue_yoy"].disabled_reason


def test_model_feature_enumeration_excludes_unsafe_fina_indicator_features() -> None:
    frame = build_feature_frame(
        feature_fixture_inputs(),
        load_settings("config/default.yaml"),
        "20240520",
        "20240524",
    )

    assert "revenue_yoy" not in frame.columns
    assert "roe" not in frame.columns
    assert "debt_to_assets" in frame.columns
    assert all("fina_indicator" not in spec.source_datasets for spec in FEATURE_REGISTRY)


def test_downside_vol_all_positive_returns_is_zero_after_warmup() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    set_return_sequence(inputs, "000001.SZ", dates, [0.01, 0.02, 0.03, 0.01, 0.02])

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]

    assert value == pytest.approx(0.0)


def test_downside_vol_mixed_returns_matches_downside_deviation() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    returns = [0.01, -0.02, 0.03, -0.01, 0.0]
    set_return_sequence(inputs, "000001.SZ", dates, returns)

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]
    expected = ((0.0**2 + (-0.02) ** 2 + 0.0**2 + (-0.01) ** 2 + 0.0**2) / 5) ** 0.5

    assert value == pytest.approx(expected)


def test_downside_vol_all_negative_returns_matches_downside_deviation() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    returns = [-0.01, -0.02, -0.03, -0.04, -0.05]
    set_return_sequence(inputs, "000001.SZ", dates, returns)

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]
    expected = sum(ret**2 for ret in returns) / len(returns)

    assert value == pytest.approx(expected**0.5)


def test_downside_vol_missing_returns_reduce_coverage_but_nonnegative_count_as_zero() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    set_return_sequence(inputs, "000001.SZ", dates, [0.01, -0.02, -0.03, -0.04, -0.05])
    suspend_stock_dates(inputs, "000001.SZ", [dates[3]])

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]
    expected = (0.0**2 + (-0.02) ** 2 + (-0.05) ** 2) / 3

    assert value == pytest.approx(expected**0.5)


def test_downside_vol_insufficient_valid_returns_is_null() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    set_return_sequence(inputs, "000001.SZ", dates, [0.01, -0.02, -0.03, -0.04, -0.05])
    suspend_stock_dates(inputs, "000001.SZ", [dates[2], dates[3]])

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]

    assert pd.isna(value)


def test_downside_vol_old_all_null_behavior_is_prevented() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=8)
    dates = first_trade_dates(inputs, 6)
    set_return_sequence(inputs, "000001.SZ", dates, [0.01, 0.01, 0.01, 0.01, 0.01])

    frame = build_feature_frame(inputs, settings, dates[-1], dates[-1])
    value = frame.set_index("ts_code").loc["000001.SZ", "downside_vol_5d"]

    assert pd.notna(value)
    assert value == pytest.approx(0.0)


def test_point_in_time_financial_join_uses_announcement_date() -> None:
    inputs = feature_fixture_inputs()

    before = build_fundamental_features(
        pd.DataFrame({"trade_date": ["20240520"], "ts_code": ["000001.SZ"]}),
        inputs["fina_indicator"],
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )
    after = build_fundamental_features(
        pd.DataFrame({"trade_date": ["20240610"], "ts_code": ["000001.SZ"]}),
        inputs["fina_indicator"],
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )
    before_roe = before.set_index("ts_code").loc["000001.SZ", "roe"]
    after_roe = after.set_index("ts_code").loc["000001.SZ", "roe"]

    assert before_roe == 10.0
    assert after_roe == 20.0


def test_financial_f_ann_date_later_than_ann_date_is_not_visible_before_f_ann_date() -> None:
    base = financial_base(["20240229", "20240301"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240131"],
            "f_ann_date": ["20240301"],
            "roe": [10.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )
    rows = frame.set_index("trade_date")

    assert pd.isna(rows.loc["20240229", "roe"])
    assert rows.loc["20240301", "roe"] == 10.0


def test_corrected_statement_is_visible_only_on_or_after_own_f_ann_date() -> None:
    base = financial_base(["20240430", "20240501"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "ann_date": ["20240201", "20240201"],
            "f_ann_date": ["20240201", "20240501"],
            "update_flag": ["0", "1"],
            "roe": [10.0, 20.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )
    rows = frame.set_index("trade_date")

    assert rows.loc["20240430", "roe"] == 10.0
    assert rows.loc["20240501", "roe"] == 20.0


def test_financial_ann_date_fallback_is_used_only_when_f_ann_date_missing() -> None:
    base = financial_base(["20240201"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": [""],
            "roe": [11.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )

    assert frame.set_index("trade_date").loc["20240201", "roe"] == 11.0


def test_revenue_yoy_uses_operating_revenue_yoy_or_yoy() -> None:
    base = financial_base(["20240201"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "or_yoy": [8.5],
            "tr_yoy": [99.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )

    assert frame.set_index("trade_date").loc["20240201", "revenue_yoy"] == 8.5


def test_tr_yoy_is_not_used_as_revenue_yoy() -> None:
    prepared = prepare_fina_indicator(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "end_date": ["20231231"],
                "ann_date": ["20240201"],
                "f_ann_date": ["20240201"],
                "tr_yoy": [99.0],
            }
        )
    )

    assert "revenue_yoy" not in prepared.columns


def test_missing_or_yoy_results_in_null_revenue_yoy_without_fallback() -> None:
    base = financial_base(["20240201"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "tr_yoy": [99.0],
            "netprofit_yoy": [5.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )

    assert pd.isna(frame.set_index("trade_date").loc["20240201", "revenue_yoy"])


def test_financial_feature_registry_sources_match_known_schema() -> None:
    required_columns = tuple(
        source
        for spec in FEATURE_REGISTRY
        if spec.availability_lag == "financial_ann_date_lte_trade_date"
        for source in spec.required_source_columns
    )

    assert validate_financial_registry_sources(required_columns) == []


def test_generated_revenue_yoy_is_not_systematically_null_with_valid_or_yoy() -> None:
    inputs = feature_fixture_inputs()
    base = financial_base(["20240520", "20240521", "20240522"])
    frame = build_fundamental_features(
        base,
        inputs["fina_indicator"],
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )

    assert frame["revenue_yoy"].notna().any()


def test_ocf_to_profit_uses_same_end_date_and_later_component_availability() -> None:
    base = financial_base(["20240215", "20240301"])
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "n_income": [10.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20230930", "20231231"],
            "ann_date": ["20240201", "20240201"],
            "f_ann_date": ["20240201", "20240301"],
            "n_cashflow_act": [100.0, 20.0],
        }
    )

    frame = build_fundamental_features(
        base, empty_financial_frame(), income, empty_financial_frame(), cashflow
    )
    rows = frame.set_index("trade_date")

    assert pd.isna(rows.loc["20240215", "ocf_to_profit"])
    assert rows.loc["20240301", "ocf_to_profit"] == 2.0


def test_mismatched_income_and_cashflow_periods_do_not_produce_ratio() -> None:
    base = financial_base(["20240301"])
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "n_income": [10.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20230930"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "n_cashflow_act": [20.0],
        }
    )

    frame = build_fundamental_features(
        base, empty_financial_frame(), income, empty_financial_frame(), cashflow
    )

    assert pd.isna(frame.set_index("trade_date").loc["20240301", "ocf_to_profit"])


def test_duplicate_revised_financial_records_are_selected_deterministically() -> None:
    base = financial_base(["20240301"])
    income = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "end_date": ["20231231", "20231231"],
            "ann_date": ["20240201", "20240201"],
            "f_ann_date": ["20240201", "20240201"],
            "update_flag": ["0", "1"],
            "n_income": [10.0, 12.0],
        }
    )
    cashflow = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "n_cashflow_act": [24.0],
        }
    )

    frame = build_fundamental_features(
        base, empty_financial_frame(), income, empty_financial_frame(), cashflow
    )

    assert frame.set_index("trade_date").loc["20240301", "ocf_to_profit"] == 2.0


def test_future_financial_record_does_not_appear_in_features() -> None:
    base = financial_base(["20240131"])
    fina_indicator = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "end_date": ["20231231"],
            "ann_date": ["20240201"],
            "f_ann_date": ["20240201"],
            "roe": [10.0],
        }
    )

    frame = build_fundamental_features(
        base,
        fina_indicator,
        empty_financial_frame(),
        empty_financial_frame(),
        empty_financial_frame(),
    )

    assert pd.isna(frame.set_index("trade_date").loc["20240131", "roe"])


def test_feature_computation_time_and_missing_stats_on_fixture() -> None:
    settings = load_settings("config/default.yaml")
    started = time.perf_counter()
    frame = build_feature_frame(feature_fixture_inputs(), settings, "20240520", "20240524")
    elapsed = time.perf_counter() - started
    feature_columns = [
        column for column in frame.columns if column not in {"trade_date", "ts_code"}
    ]
    missing_stats = frame[feature_columns].isna().mean().sort_values(ascending=False)

    assert elapsed < 5.0
    assert len(feature_columns) == len(FEATURE_REGISTRY)
    assert missing_stats.notna().all()


def test_feature_builder_reads_only_required_date_windows() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()
    raw_store = FakeRawStore(inputs)
    universe_store = FakeUniverseStore(inputs["universe"])
    feature_store = FakeFeatureStore()
    builder = FeatureBuilder(raw_store, universe_store, feature_store, settings)

    frame = builder.preview("20240520", "20240520")

    assert not frame.empty
    assert raw_store.calls["daily"] == [("20221117", "20240520")]
    assert raw_store.calls["adj_factor"] == [("20221117", "20240520")]
    assert raw_store.calls["daily_basic"] == [("20221117", "20240520")]
    assert raw_store.calls["index_daily"] == [("20221117", "20240520")]
    assert raw_store.calls["trade_cal"] == [("20221117", "20240520")]
    assert raw_store.calls["fina_indicator"] == [(None, "20240520")]
    assert universe_store.calls == [("20221117", "20240520")]


def test_feature_builder_month_chunks_match_one_shot_output() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs(days=150)
    raw_store = FakeRawStore(inputs)
    universe_store = FakeUniverseStore(inputs["universe"])
    feature_store = FakeFeatureStore()
    builder = FeatureBuilder(raw_store, universe_store, feature_store, settings)

    result = builder.build("20240530", "20240605")
    chunked = feature_store.frame()
    one_shot = build_feature_frame(inputs, settings, "20240530", "20240605")

    assert result.rows_built == len(one_shot)
    pd.testing.assert_frame_equal(
        chunked.sort_values(["trade_date", "ts_code"]).reset_index(drop=True),
        one_shot.sort_values(["trade_date", "ts_code"]).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


class FakeRawStore:
    def __init__(self, inputs: dict[str, pd.DataFrame]) -> None:
        self.inputs = inputs
        self.calls: dict[str, list[tuple[str | None, str | None]]] = {}

    def read_dataset(
        self,
        spec: DatasetSpec,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        self.calls.setdefault(spec.name, []).append((start_date, end_date))
        frame = self.inputs[spec.name].copy()
        if spec.date_column is not None and spec.date_column in frame.columns:
            frame[spec.date_column] = frame[spec.date_column].astype(str)
            if start_date is not None:
                frame = frame[frame[spec.date_column] >= start_date]
            if end_date is not None:
                frame = frame[frame[spec.date_column] <= end_date]
        return frame.reset_index(drop=True)


def suspend_stock_dates(
    inputs: dict[str, pd.DataFrame],
    ts_code: str,
    trade_dates: list[str],
) -> None:
    """Remove traded observations and mark universe rows as suspended."""

    date_set = set(trade_dates)
    for name in ("daily", "adj_factor", "daily_basic"):
        frame = inputs[name]
        mask = (frame["ts_code"] == ts_code) & frame["trade_date"].isin(date_set)
        inputs[name] = frame.loc[~mask].reset_index(drop=True)
    universe = inputs["universe"]
    universe_mask = (universe["ts_code"] == ts_code) & universe["trade_date"].isin(date_set)
    inputs["universe"].loc[universe_mask, ["is_suspended", "can_buy", "can_sell"]] = [
        True,
        False,
        False,
    ]


def set_one_day_return_fixture(
    inputs: dict[str, pd.DataFrame],
    previous_date: str,
    current_date: str,
    current_closes: dict[str, float] | None = None,
) -> None:
    """Set deterministic adjacent-day closes for rank tests."""

    closes = current_closes or {
        "000001.SZ": 11.0,
        "000002.SZ": 12.0,
        "000003.SZ": 100.0,
    }
    daily = inputs["daily"]
    for code in ("000001.SZ", "000002.SZ", "000003.SZ"):
        previous_mask = (daily["ts_code"] == code) & (daily["trade_date"] == previous_date)
        current_mask = (daily["ts_code"] == code) & (daily["trade_date"] == current_date)
        daily.loc[previous_mask, ["open", "high", "low", "close"]] = [10.0, 10.0, 10.0, 10.0]
        close = closes[code]
        daily.loc[current_mask, ["open", "high", "low", "close"]] = [
            close,
            close,
            close,
            close,
        ]


def mark_ineligible(inputs: dict[str, pd.DataFrame], ts_code: str, trade_date: str) -> None:
    """Mark one stock/date outside the model universe."""

    universe = inputs["universe"]
    mask = (universe["ts_code"] == ts_code) & (universe["trade_date"] == trade_date)
    inputs["universe"].loc[mask, "in_model_universe"] = False


def first_trade_dates(inputs: dict[str, pd.DataFrame], count: int) -> list[str]:
    """Return the first fixture trading dates."""

    return inputs["trade_cal"]["cal_date"].astype(str).head(count).tolist()


def set_return_sequence(
    inputs: dict[str, pd.DataFrame],
    ts_code: str,
    dates: list[str],
    returns: list[float],
) -> None:
    """Set closes so adjacent trading-day returns equal the supplied values."""

    if len(dates) != len(returns) + 1:
        raise ValueError("dates must contain one more value than returns")
    close = 10.0
    daily = inputs["daily"]
    first_mask = (daily["ts_code"] == ts_code) & (daily["trade_date"] == dates[0])
    daily.loc[first_mask, ["open", "high", "low", "close"]] = [close, close, close, close]
    for trade_date, ret in zip(dates[1:], returns, strict=True):
        close *= 1.0 + ret
        mask = (daily["ts_code"] == ts_code) & (daily["trade_date"] == trade_date)
        daily.loc[mask, ["open", "high", "low", "close"]] = [close, close, close, close]


def financial_base(trade_dates: list[str]) -> pd.DataFrame:
    """Return a one-stock feature key frame for fundamental tests."""

    return pd.DataFrame({"trade_date": trade_dates, "ts_code": "000001.SZ"})


def empty_financial_frame() -> pd.DataFrame:
    """Return an empty financial frame with required date metadata columns."""

    return pd.DataFrame(columns=["ts_code", "end_date", "ann_date", "f_ann_date"])


class FakeUniverseStore:
    def __init__(self, universe: pd.DataFrame) -> None:
        self.universe = universe
        self.calls: list[tuple[str | None, str | None]] = []

    def read(self, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        self.calls.append((start_date, end_date))
        frame = self.universe.copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        if start_date is not None:
            frame = frame[frame["trade_date"] >= start_date]
        if end_date is not None:
            frame = frame[frame["trade_date"] <= end_date]
        return frame.reset_index(drop=True)


class FakeFeatureStore:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def write(self, frame: pd.DataFrame) -> int:
        self.frames.append(frame.copy())
        return len(frame)

    def frame(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        return pd.concat(self.frames, ignore_index=True)
