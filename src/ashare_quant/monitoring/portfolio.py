"""Portfolio measurements over append-only paper-trading ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.schemas import PortfolioMetrics
from ashare_quant.paper_trading.storage import read_ledger

type DataFrame = pd.DataFrame


def build_portfolio_metrics(
    *,
    as_of: str,
    paper_root: Path,
    portfolio_ids: tuple[str, ...],
) -> tuple[DataFrame, dict[str, str]]:
    """Measure each configured portfolio independently through the requested date."""

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for portfolio_id in portfolio_ids:
        root = paper_root / portfolio_id
        account = _load_account(root / "account.json", portfolio_id)
        initial_cash = float(account["initial_cash"])
        ledgers = {
            name: _through_as_of(read_ledger(root / f"{name}.parquet"), as_of, portfolio_id, name)
            for name in ("orders", "trades", "positions", "equity_curve")
        }
        hashes[portfolio_id] = _ledger_hash(account, ledgers)
        metric = _portfolio_metric(
            as_of=as_of,
            portfolio_id=portfolio_id,
            initial_cash=initial_cash,
            trades=ledgers["trades"],
            positions=ledgers["positions"],
            equity_curve=ledgers["equity_curve"],
        )
        rows.append(metric.to_dict())
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("portfolio_id", kind="mergesort").reset_index(drop=True)
    return frame, hashes


def _portfolio_metric(
    *,
    as_of: str,
    portfolio_id: str,
    initial_cash: float,
    trades: DataFrame,
    positions: DataFrame,
    equity_curve: DataFrame,
) -> PortfolioMetrics:
    history = (
        equity_curve.sort_values(["as_of", "equity_id"], kind="mergesort")
        if not equity_curve.empty
        else equity_curve
    )
    latest = history.iloc[-1] if not history.empty else None
    equity = initial_cash if latest is None else float(latest["equity"])
    cash = initial_cash if latest is None else float(latest["cash"])
    nav = equity / initial_cash
    daily_return = 0.0 if latest is None else float(latest["daily_return"])
    if history.empty:
        drawdown = 0.0
        max_drawdown = 0.0
        previous_equity = initial_cash
    else:
        equities = pd.to_numeric(history["equity"], errors="coerce")
        drawdowns = equities / equities.cummax() - 1.0
        drawdown = float(drawdowns.iloc[-1])
        max_drawdown = float(drawdowns.min())
        previous_equity = float(equities.iloc[-2]) if len(equities) > 1 else initial_cash
    current_positions = _latest_positions(positions)
    market_values = (
        pd.to_numeric(current_positions["market_value"], errors="coerce").fillna(0.0)
        if not current_positions.empty
        else pd.Series(dtype=float)
    )
    weights = market_values / equity if equity > 0 else market_values * 0.0
    today_trades = trades.loc[trades["as_of"].astype(str) == as_of] if not trades.empty else trades
    filled = (
        today_trades.loc[today_trades["status"].astype(str) == "filled"]
        if not today_trades.empty
        else today_trades
    )
    if not filled.empty:
        missing_trade_columns = sorted({"cost", "gross_value"} - set(filled.columns))
        if missing_trade_columns:
            raise DataValidationError(
                f"paper trades ledger lacks metric columns: {missing_trade_columns}"
            )
    costs = (
        float(pd.to_numeric(filled["cost"], errors="coerce").fillna(0.0).sum())
        if not filled.empty
        else 0.0
    )
    turnover = (
        float(pd.to_numeric(filled["gross_value"], errors="coerce").fillna(0.0).sum())
        / previous_equity
        if previous_equity > 0 and not filled.empty
        else 0.0
    )
    execution_count = len(today_trades)
    rejected_ratio = (
        float(today_trades["status"].astype(str).eq("rejected").mean())
        if execution_count and "status" in today_trades
        else 0.0
    )
    failed_ratio = (
        float(today_trades["status"].astype(str).eq("failed").mean())
        if execution_count and "status" in today_trades
        else 0.0
    )
    return PortfolioMetrics(
        as_of=as_of,
        portfolio_id=portfolio_id,
        nav=nav,
        daily_return=daily_return,
        cumulative_return=nav - 1.0,
        drawdown=drawdown,
        max_drawdown=max_drawdown,
        turnover=turnover,
        transaction_cost_ratio=costs / previous_equity if previous_equity > 0 else 0.0,
        position_count=len(current_positions),
        max_position_weight=float(weights.max()) if not weights.empty else 0.0,
        top5_concentration=float(weights.nlargest(5).sum()) if not weights.empty else 0.0,
        industry_concentration=None,
        cash_ratio=cash / equity if equity > 0 else 0.0,
        rejected_order_ratio=rejected_ratio,
        failed_execution_ratio=failed_ratio,
    )


def _latest_positions(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return frame
    latest = (
        frame.sort_values(["as_of", "event_id"], kind="mergesort")
        .groupby("ts_code", sort=False)
        .tail(1)
    )
    return latest.loc[pd.to_numeric(latest["shares"], errors="coerce").fillna(0) > 0].copy()


def _through_as_of(
    frame: DataFrame,
    as_of: str,
    portfolio_id: str,
    ledger_name: str,
) -> DataFrame:
    if frame.empty:
        return frame
    required = {"as_of", "portfolio_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"paper {ledger_name} ledger lacks required columns: {portfolio_id}: {missing}"
        )
    if set(frame["portfolio_id"].astype(str)) != {portfolio_id}:
        raise DataValidationError(f"paper {ledger_name} ledger crosses portfolio boundaries")
    return frame.loc[frame["as_of"].astype(str) <= as_of].copy()


def _load_account(path: Path, portfolio_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"paper account does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read paper account: {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("portfolio_id") != portfolio_id:
        raise DataValidationError(f"paper account identity mismatch: {path}")
    if not isinstance(payload.get("initial_cash"), int | float):
        raise DataValidationError(f"paper account lacks initial_cash: {path}")
    return payload


def _ledger_hash(account: dict[str, Any], ledgers: dict[str, DataFrame]) -> str:
    payload = {
        "account": account,
        "ledgers": {name: _records(frame) for name, frame in sorted(ledgers.items())},
    }
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _records(frame: DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    normalized = frame.replace({np.nan: None})
    return cast(list[dict[str, Any]], normalized.to_dict("records"))
