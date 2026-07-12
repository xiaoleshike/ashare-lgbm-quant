from __future__ import annotations

import pandas as pd

from ashare_quant.config.settings import UniverseSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.universe import UniverseBuilder, UniverseStore, build_universe_frame
from ashare_quant.universe.builder import year_date_ranges
from ashare_quant.universe.validation import validate_universe_frame


def fixture_inputs() -> dict[str, pd.DataFrame]:
    trade_dates = ["20240102", "20240103", "20240104", "20240105", "20240108"]
    codes = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
        "000008.SZ",
    ]
    names = [
        "Ping An",
        "New Tech",
        "*ST Risk",
        "Halt Co",
        "Limit Up",
        "Limit Down",
        "Thin Co",
        "Old Gone",
    ]
    list_dates = [
        "20230101",
        "20240105",
        "20230101",
        "20230101",
        "20230101",
        "20230101",
        "20230101",
        "20230101",
    ]
    delist_dates = [None, None, None, None, None, None, None, "20240104"]
    stock_basic = pd.DataFrame(
        {
            "ts_code": codes,
            "symbol": [code[:6] for code in codes],
            "name": names,
            "market": ["主板"] * len(codes),
            "industry": ["Bank"] * len(codes),
            "list_date": list_dates,
            "delist_date": delist_dates,
        }
    )

    daily_rows: list[dict[str, object]] = []
    for trade_date in ("20240104", "20240105"):
        for code in codes:
            if code == "000002.SZ" and trade_date == "20240104":
                continue
            close = 10.0
            amount = 1000.0
            if code == "000005.SZ" and trade_date == "20240105":
                close = 11.0
            if code == "000006.SZ" and trade_date == "20240105":
                close = 9.0
            if code == "000007.SZ":
                amount = 10.0
            daily_rows.append(
                {
                    "ts_code": code,
                    "trade_date": trade_date,
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "vol": 100.0,
                    "amount": amount,
                }
            )
    daily = pd.DataFrame(daily_rows)
    trade_cal = pd.DataFrame(
        {
            "exchange": ["SSE"] * len(trade_dates),
            "cal_date": trade_dates,
            "is_open": [1] * len(trade_dates),
        }
    )
    suspend_d = pd.DataFrame(
        {"ts_code": ["000004.SZ"], "trade_date": ["20240105"], "suspend_type": ["S"]}
    )
    stk_limit = pd.DataFrame(
        {
            "ts_code": codes,
            "trade_date": ["20240105"] * len(codes),
            "up_limit": [11.0] * len(codes),
            "down_limit": [9.0] * len(codes),
        }
    )
    return {
        "stock_basic": stock_basic,
        "trade_cal": trade_cal,
        "daily": daily,
        "daily_basic": pd.DataFrame(columns=["ts_code", "trade_date"]),
        "suspend_d": suspend_d,
        "stk_limit": stk_limit,
    }


def test_universe_builder_covers_core_membership_and_tradability_rules() -> None:
    settings = UniverseSettings(
        min_list_trading_days=3,
        liquidity_window_days=2,
        min_avg_amount=100.0,
        require_full_liquidity_window=True,
    )

    frame = build_universe_frame(fixture_inputs(), settings, "20240105", "20240105")
    rows = frame.set_index("ts_code")

    normal = rows.loc["000001.SZ"]
    assert bool(normal["in_base_universe"])
    assert bool(normal["in_model_universe"])
    assert bool(normal["can_buy"])
    assert bool(normal["can_sell"])

    new_stock = rows.loc["000002.SZ"]
    assert bool(new_stock["is_new_stock"])
    assert not bool(new_stock["in_model_universe"])
    assert "new_stock" in str(new_stock["exclude_reason"])

    st_stock = rows.loc["000003.SZ"]
    assert bool(st_stock["is_st"])
    assert not bool(st_stock["in_model_universe"])
    assert "st" in str(st_stock["exclude_reason"])

    suspended = rows.loc["000004.SZ"]
    assert bool(suspended["is_suspended"])
    assert not bool(suspended["can_buy"])
    assert not bool(suspended["can_sell"])
    assert not bool(suspended["in_model_universe"])

    limit_up = rows.loc["000005.SZ"]
    assert bool(limit_up["is_limit_up"])
    assert not bool(limit_up["can_buy"])
    assert bool(limit_up["can_sell"])

    limit_down = rows.loc["000006.SZ"]
    assert bool(limit_down["is_limit_down"])
    assert bool(limit_down["can_buy"])
    assert not bool(limit_down["can_sell"])

    low_liquidity = rows.loc["000007.SZ"]
    assert bool(low_liquidity["is_low_liquidity"])
    assert not bool(low_liquidity["in_model_universe"])
    assert "low_liquidity" in str(low_liquidity["exclude_reason"])

    delisted = rows.loc["000008.SZ"]
    assert not bool(delisted["is_listed"])
    assert not bool(delisted["in_base_universe"])
    assert not bool(delisted["in_model_universe"])
    assert "not_listed" in str(delisted["exclude_reason"])

    assert validate_universe_frame(frame).ok


def test_universe_validation_detects_duplicates() -> None:
    settings = UniverseSettings(
        min_list_trading_days=3, liquidity_window_days=1, min_avg_amount=0.0
    )
    frame = build_universe_frame(fixture_inputs(), settings, "20240105", "20240105")
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    result = validate_universe_frame(duplicated)

    assert not result.ok
    assert any("duplicate universe rows" in error for error in result.errors)


def test_year_date_ranges_splits_trade_dates_by_calendar_year() -> None:
    ranges = year_date_ranges(["20231229", "20240102", "20240103", "20250102"])

    assert ranges == [
        ("20231229", "20231229"),
        ("20240102", "20240103"),
        ("20250102", "20250102"),
    ]


def test_universe_store_is_idempotent_and_cli_builder_uses_raw_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")
    raw_store = ParquetDataStore(tmp_path / "raw")
    for name, frame in fixture_inputs().items():
        raw_store.write(get_dataset_spec(name), frame)
    universe_store = UniverseStore(tmp_path / "processed")
    builder = UniverseBuilder(raw_store, universe_store, settings)

    first = builder.build("20240105", "20240105")
    second = builder.build("20240105", "20240105")
    stored = universe_store.read("20240105", "20240105")

    assert first.validation.ok
    assert second.validation.ok
    assert len(stored) == len(fixture_inputs()["stock_basic"])
    assert not stored.duplicated(subset=["trade_date", "ts_code"]).any()


def historical_st_fixture_inputs() -> dict[str, pd.DataFrame]:
    trade_dates = ["20240102", "20240103", "20240104", "20240105"]
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"]
    stock_basic = pd.DataFrame(
        {
            "ts_code": codes,
            "symbol": [code[:6] for code in codes],
            "name": ["*ST Future", "Recovered Co", "Open End", "Normal Now", "Overlap Co"],
            "market": ["主板"] * len(codes),
            "industry": ["Test"] * len(codes),
            "list_date": ["20200101"] * len(codes),
            "delist_date": [None] * len(codes),
        }
    )
    daily = pd.DataFrame(
        [
            {
                "ts_code": code,
                "trade_date": trade_date,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "vol": 100.0,
                "amount": 1000.0,
            }
            for trade_date in trade_dates
            for code in codes
        ]
    )
    trade_cal = pd.DataFrame(
        {"exchange": ["SSE"] * len(trade_dates), "cal_date": trade_dates, "is_open": [1] * 4}
    )
    namechange = pd.DataFrame(
        {
            "ts_code": [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
                "000005.SZ",
            ],
            "name": [
                "*ST Future",
                "SST Recovered",
                "S*ST Open",
                "退 Old",
                "ST Overlap",
                "ST Overlap",
            ],
            "start_date": ["20240104", "20240102", "20240103", "20240102", "20240103", "20240103"],
            "end_date": ["", "20240103", "", "20240102", "20240104", "20240104"],
            "ann_date": ["20240104", "20240102", "20240103", "20240102", "20240103", "20240103"],
        }
    )
    return {
        "stock_basic": stock_basic,
        "trade_cal": trade_cal,
        "daily": daily,
        "daily_basic": pd.DataFrame(columns=["ts_code", "trade_date"]),
        "suspend_d": pd.DataFrame(columns=["ts_code", "trade_date"]),
        "stk_limit": pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"]),
        "namechange": namechange,
    }


def test_universe_uses_historical_namechange_for_st_intervals() -> None:
    settings = UniverseSettings(
        min_list_trading_days=0,
        liquidity_window_days=1,
        min_avg_amount=0.0,
        require_full_liquidity_window=True,
    )

    frame = build_universe_frame(
        historical_st_fixture_inputs(), settings, "20240102", "20240105"
    )
    rows = frame.set_index(["trade_date", "ts_code"])

    assert not bool(rows.loc[("20240103", "000001.SZ"), "is_st"])
    assert bool(rows.loc[("20240104", "000001.SZ"), "is_st"])
    assert bool(rows.loc[("20240103", "000001.SZ"), "in_model_universe"])
    assert "st" in str(rows.loc[("20240104", "000001.SZ"), "exclude_reason"])

    assert bool(rows.loc[("20240103", "000002.SZ"), "is_st"])
    assert not bool(rows.loc[("20240104", "000002.SZ"), "is_st"])
    assert not bool(rows.loc[("20240103", "000002.SZ"), "in_model_universe"])
    assert bool(rows.loc[("20240104", "000002.SZ"), "in_model_universe"])

    assert bool(rows.loc[("20240105", "000003.SZ"), "is_st"])
    assert bool(rows.loc[("20240102", "000004.SZ"), "is_st"])
    assert bool(rows.loc[("20240103", "000005.SZ"), "is_st"])
    assert not frame.duplicated(subset=["trade_date", "ts_code"]).any()
