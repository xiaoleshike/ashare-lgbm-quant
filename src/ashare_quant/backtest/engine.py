"""Executable next-open portfolio simulation from daily ranking scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.config.settings import BacktestSettings

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class BacktestInputs:
    """Prepared inputs for one Top-N backtest."""

    signals: DataFrame
    prices: DataFrame
    calendar: tuple[str, ...]
    benchmark: DataFrame


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete backtest outputs for persistence."""

    top_n: int
    daily_returns: DataFrame
    trades: DataFrame
    holdings: DataFrame
    metrics: dict[str, float | int | None]


@dataclass(slots=True)
class Position:
    """One open stock position."""

    ts_code: str
    shares: float
    entry_date: str
    target_exit_date: str
    delayed_exit_days: int = 0


def simulate_portfolio(
    inputs: BacktestInputs,
    *,
    top_n: int,
    settings: BacktestSettings,
) -> BacktestResult:
    """Simulate equal-weight Top-N signals with next-open execution.

    Signals are observed after close on `trade_date`; buys are attempted at the
    next trading day's open. Exits are attempted at the open after the configured
    holding period and delayed only when a stock cannot be sold.
    """

    calendar = list(inputs.calendar)
    signal_by_entry = _signals_by_entry_date(inputs.signals, calendar, top_n)
    price_map = _price_map(inputs.prices)
    benchmark_returns = _benchmark_returns(inputs.benchmark)
    positions: dict[str, Position] = {}
    cash = settings.initial_cash
    previous_equity = settings.initial_cash
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []

    simulation_dates = [
        date
        for date in calendar
        if date >= inputs.signals["trade_date"].astype(str).min() and date <= max(calendar)
    ]
    for current_date in simulation_dates:
        day_cost = 0.0
        day_gross_pnl = 0.0

        for code in list(positions):
            position = positions[code]
            if current_date < position.target_exit_date:
                continue
            price = price_map.get((current_date, code))
            if price is None or not _can_sell(price):
                position.delayed_exit_days += 1
                trade_rows.append(_rejected_trade(current_date, code, "sell", "not_sellable"))
                if position.delayed_exit_days <= settings.sell_delay_max_days:
                    continue
                trade_rows.append(_written_off_position(current_date, position, top_n))
                del positions[code]
                continue
            gross_value = position.shares * price.open
            costs = gross_value * (settings.commission + settings.stamp_duty + settings.slippage)
            cash += gross_value - costs
            day_cost += costs
            day_gross_pnl += gross_value - _market_value(position, price_map, current_date)
            trade_rows.append(
                {
                    "trade_date": current_date,
                    "ts_code": code,
                    "side": "sell",
                    "status": "filled",
                    "shares": position.shares,
                    "price": price.open,
                    "gross_value": gross_value,
                    "commission": gross_value * settings.commission,
                    "stamp_duty": gross_value * settings.stamp_duty,
                    "slippage": gross_value * settings.slippage,
                    "cost": costs,
                    "reason": "",
                    "top_n": top_n,
                }
            )
            del positions[code]

        candidates = [
            code for code in signal_by_entry.get(current_date, []) if code not in positions
        ]
        executable = [
            code
            for code in candidates
            if (price := price_map.get((current_date, code))) is not None and _can_buy(price)
        ]
        for code in candidates:
            price = price_map.get((current_date, code))
            if price is None or not _can_buy(price):
                trade_rows.append(
                    _rejected_trade(current_date, code, "buy", "not_buyable", top_n=top_n)
                )
        if executable and cash > 0:
            allocation = cash / len(executable)
            for code in executable:
                price = price_map[(current_date, code)]
                gross_notional = allocation / (1.0 + settings.commission + settings.slippage)
                shares = gross_notional / price.open
                costs = gross_notional * (settings.commission + settings.slippage)
                total_cash_used = gross_notional + costs
                if shares <= 0 or total_cash_used > cash * (1.0 + 1e-9):
                    continue
                exit_date = _calendar_offset(calendar, current_date, settings.holding_period_days)
                positions[code] = Position(code, shares, current_date, exit_date)
                cash -= total_cash_used
                day_cost += costs
                trade_rows.append(
                    {
                        "trade_date": current_date,
                        "ts_code": code,
                        "side": "buy",
                        "status": "filled",
                        "shares": shares,
                        "price": price.open,
                        "gross_value": gross_notional,
                        "commission": gross_notional * settings.commission,
                        "stamp_duty": 0.0,
                        "slippage": gross_notional * settings.slippage,
                        "cost": costs,
                        "reason": "",
                        "top_n": top_n,
                    }
                )

        holdings_value = 0.0
        for position in positions.values():
            market_value = _market_value(position, price_map, current_date)
            holdings_value += market_value
            holding_rows.append(
                {
                    "trade_date": current_date,
                    "ts_code": position.ts_code,
                    "shares": position.shares,
                    "market_value": market_value,
                    "entry_date": position.entry_date,
                    "target_exit_date": position.target_exit_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "top_n": top_n,
                }
            )
        equity = cash + holdings_value
        net_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        gross_return = (equity + day_cost) / previous_equity - 1.0 if previous_equity > 0 else 0.0
        turnover = _daily_turnover(trade_rows, current_date, top_n, previous_equity)
        daily_rows.append(
            {
                "trade_date": current_date,
                "top_n": top_n,
                "cash": cash,
                "holdings_value": holdings_value,
                "equity": equity,
                "turnover": turnover,
                "gross_return": gross_return,
                "net_return": net_return,
                "cost": day_cost,
                "benchmark_return": benchmark_returns.get(current_date, 0.0),
            }
        )
        previous_equity = equity

    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    holdings = pd.DataFrame(holding_rows)
    return BacktestResult(
        top_n=top_n,
        daily_returns=daily,
        trades=trades,
        holdings=holdings,
        metrics=calculate_metrics(daily, trades, settings),
    )


@dataclass(frozen=True, slots=True)
class PriceRow:
    """Per-stock execution and valuation fields for one date."""

    open: float
    close: float
    can_buy: bool
    can_sell: bool


def calculate_metrics(
    daily: DataFrame, trades: DataFrame, settings: BacktestSettings
) -> dict[str, float | int | None]:
    """Compute compact daily-return based backtest metrics."""

    if daily.empty:
        return {}
    returns = daily["net_return"].astype(float)
    benchmark = daily["benchmark_return"].astype(float)
    equity = daily["equity"].astype(float)
    days = len(daily)
    total_return = equity.iloc[-1] / settings.initial_cash - 1.0
    annual_return = (1.0 + total_return) ** (settings.annualization_days / days) - 1.0
    annual_vol = returns.std(ddof=0) * np.sqrt(settings.annualization_days)
    sharpe = annual_return / annual_vol if annual_vol > 0 else None
    drawdown = equity / equity.cummax() - 1.0
    filled = trades[trades["status"] == "filled"] if not trades.empty else trades
    closures = (
        trades[trades["status"].isin(["filled", "written_off"])] if not trades.empty else trades
    )
    buys = closures[closures["side"] == "buy"] if not closures.empty else closures
    sells = closures[closures["side"] == "sell"] if not closures.empty else closures
    closed_pnl = _closed_trade_pnl(closures)
    profitable = [value for value in closed_pnl if value > 0]
    losing = [value for value in closed_pnl if value < 0]
    return {
        "days": int(days),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe": None if sharpe is None else float(sharpe),
        "maximum_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "trade_win_rate": (
            None
            if not closed_pnl
            else float(sum(value > 0 for value in closed_pnl) / len(closed_pnl))
        ),
        "profit_loss_ratio": (
            None
            if not profitable or not losing
            else float(np.mean(profitable) / abs(np.mean(losing)))
        ),
        "average_turnover": float(daily["turnover"].astype(float).mean()),
        "average_holding_period": _average_holding_period(buys, sells),
        "excess_return_vs_benchmark": float(returns.sum() - benchmark.sum()),
        "filled_trades": int(len(filled)),
        "written_off_positions": (
            int((trades["status"] == "written_off").sum()) if not trades.empty else 0
        ),
        "rejected_trades": int((trades["status"] == "rejected").sum()) if not trades.empty else 0,
    }


def _closed_trade_pnl(filled: DataFrame) -> list[float]:
    """Pair the engine's non-overlapping per-symbol positions into net trade P/L."""

    if filled.empty:
        return []
    cost_basis: dict[str, float] = {}
    closed: list[float] = []
    for row in filled.itertuples(index=False):
        typed = cast(Any, row)
        code = str(typed.ts_code)
        if str(typed.side) == "buy":
            cost_basis[code] = float(typed.gross_value) + float(typed.cost)
            continue
        basis = cost_basis.pop(code, None)
        if basis is not None:
            closed.append(float(typed.gross_value) - float(typed.cost) - basis)
    return closed


def _average_holding_period(buys: DataFrame, sells: DataFrame) -> float | None:
    if buys.empty or sells.empty:
        return None
    buy_dates = buys.groupby("ts_code")["trade_date"].min()
    sell_dates = sells.groupby("ts_code")["trade_date"].max()
    common = buy_dates.index.intersection(sell_dates.index)
    if len(common) == 0:
        return None
    return float(len(common))


def _signals_by_entry_date(
    signals: DataFrame, calendar: list[str], top_n: int
) -> dict[str, list[str]]:
    by_entry: dict[str, list[str]] = {}
    for trade_date, group in signals.groupby("trade_date", sort=True):
        entry_date = _calendar_offset(calendar, str(trade_date), 1)
        selected = (
            group.sort_values(["score", "ts_code"], ascending=[False, True])
            .head(top_n)["ts_code"]
            .astype(str)
            .tolist()
        )
        by_entry.setdefault(entry_date, []).extend(selected)
    return by_entry


def _calendar_offset(calendar: list[str], date: str, offset: int) -> str:
    index = calendar.index(date)
    target = min(index + offset, len(calendar) - 1)
    return calendar[target]


def _price_map(prices: DataFrame) -> dict[tuple[str, str], PriceRow]:
    mapping: dict[tuple[str, str], PriceRow] = {}
    for row in prices.itertuples(index=False):
        typed_row = cast(Any, row)
        mapping[(str(typed_row.trade_date), str(typed_row.ts_code))] = PriceRow(
            open=float(typed_row.open),
            close=float(typed_row.close),
            can_buy=bool(typed_row.can_buy),
            can_sell=bool(typed_row.can_sell),
        )
    return mapping


def _benchmark_returns(benchmark: DataFrame) -> dict[str, float]:
    if benchmark.empty:
        return {}
    frame = benchmark.sort_values("trade_date").copy()
    frame["benchmark_return"] = frame["close"].astype(float).pct_change().fillna(0.0)
    return dict(zip(frame["trade_date"].astype(str), frame["benchmark_return"], strict=False))


def _can_buy(price: PriceRow) -> bool:
    return bool(price.can_buy and np.isfinite(price.open) and price.open > 0)


def _can_sell(price: PriceRow) -> bool:
    return bool(price.can_sell and np.isfinite(price.open) and price.open > 0)


def _market_value(position: Position, prices: dict[tuple[str, str], PriceRow], date: str) -> float:
    price = prices.get((date, position.ts_code))
    if price is None or not np.isfinite(price.close):
        return 0.0
    return position.shares * price.close


def _daily_turnover(
    trade_rows: list[dict[str, object]], date: str, top_n: int, previous_equity: float
) -> float:
    if previous_equity <= 0:
        return 0.0
    traded = [
        float(cast(float, row["gross_value"]))
        for row in trade_rows
        if row.get("trade_date") == date
        and row.get("top_n") == top_n
        and row.get("status") == "filled"
    ]
    return sum(traded) / previous_equity


def _rejected_trade(
    date: str, code: str, side: str, reason: str, *, top_n: int | None = None
) -> dict[str, object]:
    return {
        "trade_date": date,
        "ts_code": code,
        "side": side,
        "status": "rejected",
        "shares": 0.0,
        "price": np.nan,
        "gross_value": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "slippage": 0.0,
        "cost": 0.0,
        "reason": reason,
        "top_n": top_n,
    }


def _written_off_position(date: str, position: Position, top_n: int) -> dict[str, object]:
    """Close a persistently untradeable position at zero after the configured delay."""

    return {
        "trade_date": date,
        "ts_code": position.ts_code,
        "side": "sell",
        "status": "written_off",
        "shares": position.shares,
        "price": 0.0,
        "gross_value": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "slippage": 0.0,
        "cost": 0.0,
        "reason": "untradeable_after_max_sell_delay",
        "top_n": top_n,
    }
