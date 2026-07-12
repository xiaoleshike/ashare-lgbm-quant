from __future__ import annotations

import time

import pandas as pd
import pytest

from ashare_quant.config import load_settings
from ashare_quant.data.datasets import DatasetSpec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.features import FEATURE_REGISTRY, FeatureBuilder, build_feature_frame


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
            "revenue_yoy": [8.0, 12.0, 9.0],
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
    assert 160 <= len(FEATURE_REGISTRY) <= 220
    assert all(spec.availability_lag for spec in FEATURE_REGISTRY)


def test_features_are_deterministic_and_have_rank_boundaries() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()

    first = build_feature_frame(inputs, settings, "20240520", "20240520")
    second = build_feature_frame(inputs, settings, "20240520", "20240520")

    pd.testing.assert_frame_equal(first, second)
    rank_columns = [
        column for column in first.columns if column.startswith(("cs_rank_", "ind_rank_"))
    ]
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


def test_point_in_time_financial_join_uses_announcement_date() -> None:
    settings = load_settings("config/default.yaml")
    inputs = feature_fixture_inputs()

    before = build_feature_frame(inputs, settings, "20240520", "20240520")
    after = build_feature_frame(inputs, settings, "20240610", "20240610")
    before_roe = before.set_index("ts_code").loc["000001.SZ", "roe"]
    after_roe = after.set_index("ts_code").loc["000001.SZ", "roe"]

    assert before_roe == 10.0
    assert after_roe == 20.0


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
