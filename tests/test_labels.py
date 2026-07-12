from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.config.settings import LabelSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.labels import LabelBuilder, LabelStore, build_label_frame
from ashare_quant.labels.builder import build_label_frame_iterative
from ashare_quant.labels.validation import LabelValidator, validate_label_frame
from ashare_quant.universe import UniverseStore


def label_fixture_inputs() -> dict[str, pd.DataFrame]:
    trade_dates = [
        "20240102",
        "20240103",
        "20240104",
        "20240105",
        "20240108",
        "20240109",
        "20240110",
        "20240111",
        "20240112",
        "20240115",
        "20240116",
        "20240117",
        "20240118",
        "20240119",
        "20240122",
    ]
    codes = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
    ]
    base_open = {
        "000001.SZ": 10.0,
        "000002.SZ": 20.0,
        "000003.SZ": 30.0,
        "000004.SZ": 40.0,
        "000005.SZ": 50.0,
        "000006.SZ": 60.0,
        "000007.SZ": 70.0,
    }
    daily_rows: list[dict[str, object]] = []
    adj_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(trade_dates):
        for code in codes:
            if code == "000002.SZ" and trade_date == "20240103":
                continue
            if code == "000003.SZ" and trade_date == "20240108":
                continue
            open_price = base_open[code] + day_index
            daily_rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": open_price + 0.5,
                    "low": open_price - 0.5,
                    "close": open_price + 0.1,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            )
            adj_factor = 1.0
            if code == "000001.SZ" and trade_date == "20240108":
                adj_factor = 2.0
            adj_rows.append({"ts_code": code, "trade_date": trade_date, "adj_factor": adj_factor})

            is_limit_up = code == "000005.SZ" and trade_date == "20240103"
            is_limit_down = code == "000006.SZ" and trade_date == "20240108"
            is_suspended = code == "000004.SZ" and trade_date == "20240103"
            can_buy = not is_suspended and not is_limit_up
            can_sell = not is_suspended and not is_limit_down
            universe_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "name": code,
                    "market": "主板",
                    "exchange": "SZSE",
                    "industry": "",
                    "list_date": "20200101",
                    "delist_date": None,
                    "list_days": 1000,
                    "is_listed": True,
                    "is_new_stock": False,
                    "is_st": False,
                    "is_suspended": is_suspended,
                    "is_low_liquidity": False,
                    "is_limit_up": is_limit_up,
                    "is_limit_down": is_limit_down,
                    "can_buy": can_buy,
                    "can_sell": can_sell,
                    "in_base_universe": True,
                    "in_model_universe": True,
                    "exclude_reason": "",
                }
            )

    index_rows = [
        {
            "ts_code": "000300.SH",
            "trade_date": trade_date,
            "open": 1000.0 + day_index * 10.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "vol": 0.0,
            "amount": 0.0,
        }
        for day_index, trade_date in enumerate(trade_dates)
        if trade_date != "20240110"
    ]
    universe_rows.extend(
        [
            {
                "trade_date": "20240103",
                "ts_code": "000002.SZ",
                "name": "000002.SZ",
                "market": "主板",
                "exchange": "SZSE",
                "industry": "",
                "list_date": "20200101",
                "delist_date": None,
                "list_days": 1000,
                "is_listed": True,
                "is_new_stock": False,
                "is_st": False,
                "is_suspended": False,
                "is_low_liquidity": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "can_buy": True,
                "can_sell": True,
                "in_base_universe": True,
                "in_model_universe": True,
                "exclude_reason": "",
            },
            {
                "trade_date": "20240108",
                "ts_code": "000003.SZ",
                "name": "000003.SZ",
                "market": "主板",
                "exchange": "SZSE",
                "industry": "",
                "list_date": "20200101",
                "delist_date": None,
                "list_days": 1000,
                "is_listed": True,
                "is_new_stock": False,
                "is_st": False,
                "is_suspended": False,
                "is_low_liquidity": False,
                "is_limit_up": False,
                "is_limit_down": False,
                "can_buy": True,
                "can_sell": True,
                "in_base_universe": True,
                "in_model_universe": True,
                "exclude_reason": "",
            },
        ]
    )
    return {
        "trade_cal": pd.DataFrame(
            {
                "exchange": ["SSE"] * len(trade_dates),
                "cal_date": trade_dates,
                "is_open": [1] * len(trade_dates),
            }
        ),
        "daily": pd.DataFrame(daily_rows),
        "adj_factor": pd.DataFrame(adj_rows),
        "index_daily": pd.DataFrame(index_rows),
        "universe": pd.DataFrame(universe_rows),
    }


def default_label_settings() -> LabelSettings:
    return LabelSettings(
        horizons=(3, 5, 10),
        benchmark_index_code="000300.SH",
        quantile_buckets=5,
        skip_unbuyable_entry=True,
        delay_unsellable_exit=False,
    )


def test_label_builder_covers_core_availability_rules_and_adjusted_prices() -> None:
    frame = build_label_frame(
        label_fixture_inputs(),
        default_label_settings(),
        "20240102",
        "20240102",
        (3, 5, 10),
    )
    rows = frame.set_index(["ts_code", "horizon"])

    normal_3d = rows.loc[("000001.SZ", 3)]
    assert bool(normal_3d["is_label_available"])
    assert normal_3d["entry_date"] == "20240103"
    assert normal_3d["exit_date"] == "20240108"
    assert float(normal_3d["entry_price"]) == 11.0
    assert float(normal_3d["exit_price"]) == 28.0
    assert float(normal_3d["stock_forward_ret"]) == 28.0 / 11.0 - 1.0

    assert rows.loc[("000001.SZ", 5)]["exit_date"] == "20240110"
    assert rows.loc[("000001.SZ", 10)]["exit_date"] == "20240117"

    missing_entry = rows.loc[("000002.SZ", 3)]
    assert not bool(missing_entry["is_label_available"])
    assert missing_entry["label_unavailable_reason"] == "missing_entry_price"

    missing_exit = rows.loc[("000003.SZ", 3)]
    assert not bool(missing_exit["is_label_available"])
    assert missing_exit["label_unavailable_reason"] == "missing_exit_price"

    suspended_entry = rows.loc[("000004.SZ", 3)]
    assert not bool(suspended_entry["is_label_available"])
    assert suspended_entry["label_unavailable_reason"] == "entry_not_buyable"

    limit_up_entry = rows.loc[("000005.SZ", 3)]
    assert not bool(limit_up_entry["is_label_available"])
    assert limit_up_entry["label_unavailable_reason"] == "entry_not_buyable"

    limit_down_exit = rows.loc[("000006.SZ", 3)]
    assert not bool(limit_down_exit["is_label_available"])
    assert limit_down_exit["label_unavailable_reason"] == "exit_not_sellable"

    benchmark_missing = rows.loc[("000007.SZ", 5)]
    assert not bool(benchmark_missing["is_label_available"])
    assert benchmark_missing["label_unavailable_reason"] == "missing_benchmark_price"

    assert validate_label_frame(frame, quantile_buckets=5).ok


def test_vectorized_labels_match_iterative_reference() -> None:
    inputs = label_fixture_inputs()
    settings = default_label_settings()

    vectorized = build_label_frame(inputs, settings, "20240102", "20240105", (3, 5, 10))
    iterative = build_label_frame_iterative(inputs, settings, "20240102", "20240105", (3, 5, 10))

    pd.testing.assert_frame_equal(
        normalize_missing_values(vectorized),
        normalize_missing_values(iterative),
        check_dtype=False,
    )


def normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize pandas missing-value sentinels for reference comparisons."""

    return frame.astype(object).where(pd.notna(frame), None)


def test_label_builder_fails_when_benchmark_code_is_absent() -> None:
    inputs = label_fixture_inputs()
    inputs["index_daily"] = inputs["index_daily"].assign(ts_code="000001.SH")

    with pytest.raises(DataValidationError, match="benchmark index 000300.SH is missing"):
        build_label_frame(inputs, default_label_settings(), "20240102", "20240102", (3,))


def test_label_builder_marks_insufficient_future_calendar_unavailable() -> None:
    frame = build_label_frame(
        label_fixture_inputs(),
        default_label_settings(),
        "20240119",
        "20240119",
        (10,),
    )

    assert not frame.empty
    assert set(frame["label_unavailable_reason"]) == {"insufficient_future_calendar"}
    assert not frame["is_label_available"].any()
    assert validate_label_frame(frame, quantile_buckets=5).ok


def test_rank_percentile_and_quantile_generation() -> None:
    frame = build_label_frame(
        label_fixture_inputs(),
        default_label_settings(),
        "20240102",
        "20240102",
        (3,),
    )
    available = frame[frame["is_label_available"].astype(bool)]

    assert not available.empty
    assert available["future_rank_pct"].between(0, 1).all()
    assert available["future_quantile"].between(0, 4).all()
    assert int(available["future_quantile"].max()) == 4


def test_label_store_builder_is_idempotent_with_fixture_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")
    raw_store = ParquetDataStore(tmp_path / "raw")
    inputs = label_fixture_inputs()
    for name in ("trade_cal", "daily", "adj_factor", "index_daily"):
        raw_store.write(get_dataset_spec(name), inputs[name])
    universe_store = UniverseStore(tmp_path / "processed")
    universe_store.write(inputs["universe"])
    label_store = LabelStore(tmp_path / "processed")
    builder = LabelBuilder(raw_store, universe_store, label_store, settings)

    first = builder.build("20240102", "20240102", (3,))
    second = builder.build("20240102", "20240102", (3,))
    stored = label_store.read("20240102", "20240102", 3)

    assert first.validation.ok
    assert second.validation.ok
    assert len(stored) == 7
    assert not stored.duplicated(subset=["trade_date", "ts_code", "horizon"]).any()


def test_label_validator_detects_row_count_mismatch_against_base_universe(tmp_path) -> None:
    inputs = label_fixture_inputs()
    universe_store = UniverseStore(tmp_path / "processed")
    universe_store.write(inputs["universe"])
    label_store = LabelStore(tmp_path / "processed")
    labels = build_label_frame(inputs, default_label_settings(), "20240102", "20240102", (3,))
    label_store.write(labels.iloc[:-1].copy())

    validator = LabelValidator(label_store, quantile_buckets=5, universe_store=universe_store)
    result = validator.validate("20240102", "20240102")

    assert not result.ok
    assert any("label row count must match in_base_universe" in error for error in result.errors)
