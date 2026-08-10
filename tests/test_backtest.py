from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.backtest.runner import build_predictions_frame
from ashare_quant.cli import parse_top_n
from ashare_quant.config.settings import BacktestSettings


def test_backtest_buy_uses_next_open_and_not_signal_day_open() -> None:
    inputs = make_inputs(entry_open=20.0, signal_day_open=10.0)

    result = simulate_portfolio(inputs, top_n=1, settings=make_settings())
    buy = result.trades[(result.trades["side"] == "buy") & (result.trades["status"] == "filled")]

    assert buy.iloc[0]["trade_date"] == "20240103"
    assert buy.iloc[0]["price"] == pytest.approx(20.0)


def test_suspended_stock_cannot_be_bought() -> None:
    inputs = make_inputs(entry_can_buy=False)

    result = simulate_portfolio(inputs, top_n=1, settings=make_settings())

    assert result.trades[result.trades["status"] == "filled"].empty
    rejected = result.trades[result.trades["status"] == "rejected"].iloc[0]
    assert rejected["side"] == "buy"
    assert rejected["reason"] == "not_buyable"


def test_limit_up_buy_rejection_uses_entry_date_constraint() -> None:
    inputs = make_inputs(entry_can_buy=False)

    result = simulate_portfolio(inputs, top_n=1, settings=make_settings())

    assert "not_buyable" in result.trades["reason"].tolist()


def test_limit_down_sell_rejection_delays_exit_until_sellable() -> None:
    inputs = make_inputs(exit_can_sell=False, delayed_exit_can_sell=True)

    result = simulate_portfolio(inputs, top_n=1, settings=make_settings())
    sells = result.trades[result.trades["side"] == "sell"]

    assert sells.iloc[0]["status"] == "rejected"
    assert sells.iloc[0]["trade_date"] == "20240105"
    assert sells.iloc[1]["status"] == "filled"
    assert sells.iloc[1]["trade_date"] == "20240108"


def test_persistently_untradeable_position_keeps_value_after_max_delay() -> None:
    inputs = make_inputs(exit_can_sell=False, delayed_exit_can_sell=False)
    settings = make_settings().model_copy(update={"sell_delay_max_days": 1})

    result = simulate_portfolio(inputs, top_n=1, settings=settings)
    assert "written_off" not in result.trades["status"].tolist()
    assert result.metrics["written_off_positions"] == 0
    final_date = result.daily_returns["trade_date"].astype(str).max()
    final_holding = result.holdings[result.holdings["trade_date"].astype(str).eq(final_date)]
    assert len(final_holding) == 1
    assert final_holding.iloc[0]["market_value"] > 0
    assert result.accounting_summary["unresolved_positions"] == 1


def test_costs_are_deducted_from_cash_and_reported() -> None:
    settings = make_settings(commission=0.001, stamp_duty=0.002, slippage=0.003)
    inputs = make_inputs(entry_open=10.0, exit_open=11.0)

    result = simulate_portfolio(inputs, top_n=1, settings=settings)
    buy = result.trades[(result.trades["side"] == "buy") & (result.trades["status"] == "filled")]
    sell = result.trades[(result.trades["side"] == "sell") & (result.trades["status"] == "filled")]

    buy_cost = buy.iloc[0]["gross_value"] * (settings.commission + settings.slippage)
    sell_cost = sell.iloc[0]["gross_value"] * (
        settings.commission + settings.stamp_duty + settings.slippage
    )
    assert buy.iloc[0]["cost"] == pytest.approx(buy_cost)
    assert sell.iloc[0]["cost"] == pytest.approx(sell_cost)
    assert result.daily_returns["cost"].sum() == pytest.approx(buy_cost + sell_cost)


def test_portfolio_accounting_equity_equals_cash_plus_holdings() -> None:
    result = simulate_portfolio(make_inputs(), top_n=1, settings=make_settings())

    assert (result.daily_returns["cash"] + result.daily_returns["holdings_value"]).equals(
        result.daily_returns["equity"]
    )


def test_parse_top_n_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        parse_top_n("10,10")


def test_predictions_frame_records_pre_execution_rankings() -> None:
    signals = pd.DataFrame(
        [
            {"trade_date": "20240102", "ts_code": "000003.SZ", "score": 0.5},
            {"trade_date": "20240102", "ts_code": "000001.SZ", "score": 0.9},
            {"trade_date": "20240102", "ts_code": "000002.SZ", "score": 0.5},
            {"trade_date": "20240103", "ts_code": "000004.SZ", "score": -0.1},
        ]
    )

    predictions = build_predictions_frame(signals, (2,))

    assert predictions.columns.tolist() == [
        "trade_date",
        "ts_code",
        "prediction_score",
        "rank",
        "selected_flag",
    ]
    day = predictions[predictions["trade_date"] == "20240102"]
    assert day["ts_code"].tolist() == ["000001.SZ", "000002.SZ", "000003.SZ"]
    assert day["rank"].tolist() == [1, 2, 3]
    assert day["selected_flag"].tolist() == [True, True, False]


def make_inputs(
    *,
    signal_day_open: float = 10.0,
    entry_open: float = 20.0,
    exit_open: float = 22.0,
    entry_can_buy: bool = True,
    exit_can_sell: bool = True,
    delayed_exit_can_sell: bool = True,
) -> BacktestInputs:
    calendar = ("20240102", "20240103", "20240104", "20240105", "20240108")
    signals = pd.DataFrame([{"trade_date": "20240102", "ts_code": "000001.SZ", "score": 1.0}])
    prices = pd.DataFrame(
        [
            {
                "trade_date": "20240102",
                "ts_code": "000001.SZ",
                "open": signal_day_open,
                "close": signal_day_open,
                "can_buy": True,
                "can_sell": True,
            },
            {
                "trade_date": "20240103",
                "ts_code": "000001.SZ",
                "open": entry_open,
                "close": entry_open,
                "can_buy": entry_can_buy,
                "can_sell": True,
            },
            {
                "trade_date": "20240104",
                "ts_code": "000001.SZ",
                "open": 21.0,
                "close": 21.0,
                "can_buy": True,
                "can_sell": True,
            },
            {
                "trade_date": "20240105",
                "ts_code": "000001.SZ",
                "open": exit_open,
                "close": exit_open,
                "can_buy": True,
                "can_sell": exit_can_sell,
            },
            {
                "trade_date": "20240108",
                "ts_code": "000001.SZ",
                "open": exit_open + 1.0,
                "close": exit_open + 1.0,
                "can_buy": True,
                "can_sell": delayed_exit_can_sell,
            },
        ]
    )
    benchmark = pd.DataFrame(
        {"trade_date": list(calendar), "close": [100.0, 101.0, 102.0, 103.0, 104.0]}
    )
    return BacktestInputs(signals=signals, prices=prices, calendar=calendar, benchmark=benchmark)


def make_settings(
    *, commission: float = 0.00025, stamp_duty: float = 0.001, slippage: float = 0.0005
) -> BacktestSettings:
    return BacktestSettings(
        initial_cash=1000.0,
        top_n=(1,),
        holding_period_days=2,
        commission=commission,
        stamp_duty=stamp_duty,
        slippage=slippage,
        sell_delay_max_days=2,
    )
