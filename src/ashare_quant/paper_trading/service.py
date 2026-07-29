"""Replayable paper-account lifecycle and next-open virtual execution."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import AppSettings, PaperPortfolioSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.paper_trading.signals import PaperSignalProvider
from ashare_quant.paper_trading.storage import append_ledger, payload_sha256, read_ledger
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

type DataFrame = pd.DataFrame

ORDER_COLUMNS = (
    "order_id",
    "run_id",
    "as_of",
    "execution_rule",
    "portfolio_id",
    "ts_code",
    "rank",
    "prediction_score",
    "target_weight",
    "model_id",
    "source_signal_manifest_hash",
    "environment",
    "created_at",
)
TRADE_COLUMNS = (
    "trade_id",
    "order_id",
    "run_id",
    "as_of",
    "signal_date",
    "portfolio_id",
    "ts_code",
    "side",
    "status",
    "shares",
    "price",
    "gross_value",
    "commission",
    "stamp_duty",
    "slippage",
    "cost",
    "cash_delta",
    "realized_pnl",
    "reason",
    "source_signal_manifest_hash",
    "environment",
)
POSITION_COLUMNS = (
    "event_id",
    "run_id",
    "as_of",
    "portfolio_id",
    "ts_code",
    "shares",
    "average_cost",
    "entry_date",
    "last_mark_price",
    "market_value",
    "source_signal_manifest_hash",
)
EQUITY_COLUMNS = (
    "equity_id",
    "run_id",
    "as_of",
    "portfolio_id",
    "cash",
    "holdings_value",
    "equity",
    "nav",
    "daily_return",
    "drawdown",
    "turnover",
    "win_rate",
    "source_signal_manifest_hash",
)


@dataclass(frozen=True, slots=True)
class PaperTradingInitResult:
    """Result of idempotent paper-account initialization."""

    account_count: int
    created_count: int
    root: Path


@dataclass(frozen=True, slots=True)
class PaperTradingRebalanceResult:
    """Result of creating immutable T+1 target orders."""

    as_of: str
    execution_rule: str
    portfolio_count: int
    orders_written: int
    root: Path


@dataclass(frozen=True, slots=True)
class PaperTradingExecutionResult:
    """Result of one virtual next-open execution session."""

    as_of: str
    portfolio_count: int
    trades_written: int
    equity_rows_written: int
    root: Path


@dataclass(frozen=True, slots=True)
class PaperTradingReportResult:
    """Published daily paper-account report."""

    as_of: str
    report_path: Path
    summary_path: Path
    portfolio_count: int


@dataclass(frozen=True, slots=True)
class PaperTradingDailyResult:
    """Combined production-pipeline paper-trading result."""

    as_of: str
    rebalance: PaperTradingRebalanceResult
    execution: PaperTradingExecutionResult
    report: PaperTradingReportResult


@dataclass(slots=True)
class _Position:
    shares: int
    average_cost: float
    entry_date: str
    last_mark_price: float


@dataclass(frozen=True, slots=True)
class _Price:
    open: float
    close: float
    can_buy: bool
    can_sell: bool


class PaperTradingService:
    """Operate isolated virtual accounts without a broker or external side effects."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        registry: ModelRegistry,
        raw_root: Path,
        processed_root: Path,
        reports_root: Path,
        paper_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.registry = registry
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.paper_root = paper_root or settings.paths.paper_trading
        self.signal_provider = PaperSignalProvider(
            registry=registry,
            processed_root=processed_root,
            reports_root=reports_root,
        )

    def init(self) -> PaperTradingInitResult:
        """Create immutable account definitions for every configured portfolio."""

        if not self.settings.paper_trading.portfolios:
            raise DataValidationError("paper_trading.portfolios is empty")
        self.paper_root.mkdir(parents=True, exist_ok=True)
        created = 0
        accounts: list[dict[str, Any]] = []
        for portfolio in self.settings.paper_trading.portfolios:
            directory = self._portfolio_root(portfolio.portfolio_id)
            account_path = directory / "account.json"
            payload = self._account_payload(portfolio)
            if account_path.exists():
                existing = _load_json(account_path, "paper account")
                if _account_contract(existing) != _account_contract(payload):
                    raise DataValidationError(
                        f"paper account definition changed and cannot be overwritten: "
                        f"{portfolio.portfolio_id}"
                    )
                payload = existing
            else:
                directory.mkdir(parents=True, exist_ok=True)
                atomic_write_json(account_path, payload)
                created += 1
            accounts.append(payload)
        manifest = {
            "schema_version": 1,
            "artifact_name": "paper_trading_accounts",
            "run_id": "paper_accounts_" + payload_sha256(accounts)[:20],
            "as_of": None,
            "portfolio_id": None,
            "source_signal_manifest_hash": None,
            "storage_semantics": "logical_append_only_atomic_parquet_ledgers",
            "broker_connected": False,
            "real_orders_generated": False,
            "config_hash": config_hash(self.config_path),
            "portfolios": [
                {
                    "portfolio_id": account["portfolio_id"],
                    "account_hash": payload_sha256(account),
                    "account_path": str(
                        self._portfolio_root(str(account["portfolio_id"])) / "account.json"
                    ),
                }
                for account in accounts
            ],
        }
        manifest_path = self.paper_root / "manifest.json"
        if manifest_path.exists():
            existing_manifest = _load_json(manifest_path, "paper manifest")
            existing_ids = {
                str(item["portfolio_id"])
                for item in existing_manifest.get("portfolios", [])
                if isinstance(item, dict) and "portfolio_id" in item
            }
            configured_ids = {
                portfolio.portfolio_id for portfolio in self.settings.paper_trading.portfolios
            }
            if existing_ids != configured_ids:
                raise DataValidationError(
                    "paper-trading portfolio set changed and cannot overwrite existing manifest"
                )
        else:
            atomic_write_json(manifest_path, manifest)
        return PaperTradingInitResult(len(accounts), created, self.paper_root)

    def rebalance(
        self,
        as_of: str,
        *,
        production_summary_path: Path | None = None,
    ) -> PaperTradingRebalanceResult:
        """Create T+1 target-weight orders from immutable same-date model rankings."""

        self.init()
        summary = production_summary_path or (self.reports_root / as_of / "production_summary.json")
        execution_rule = "next_open"
        total_written = 0
        for portfolio in self.settings.paper_trading.portfolios:
            signal = self.signal_provider.load(portfolio, as_of, summary)
            selected = signal.ranking.head(portfolio.top_n)
            positions = self._current_positions(portfolio.portfolio_id, _date_after(as_of))
            target_codes = set(selected["ts_code"].astype(str))
            all_codes = sorted(target_codes | set(positions))
            selected_by_code = selected.set_index("ts_code")
            run_id = (
                "rebalance_"
                + payload_sha256(
                    {
                        "portfolio_id": portfolio.portfolio_id,
                        "as_of": as_of,
                        "execution_rule": execution_rule,
                        "source": signal.source_signal_manifest_hash,
                    }
                )[:20]
            )
            created_at = f"{as_of}T00:00:00+00:00"
            rows: list[dict[str, Any]] = []
            for code in all_codes:
                is_target = code in target_codes
                record = cast(Any, selected_by_code.loc[code]) if is_target else None
                order_id = payload_sha256(
                    [
                        portfolio.portfolio_id,
                        as_of,
                        execution_rule,
                        code,
                        signal.source_signal_manifest_hash,
                    ]
                )
                rows.append(
                    {
                        "order_id": order_id,
                        "run_id": run_id,
                        "as_of": as_of,
                        "execution_rule": execution_rule,
                        "portfolio_id": portfolio.portfolio_id,
                        "ts_code": code,
                        "rank": int(record["rank"]) if record is not None else 0,
                        "prediction_score": (
                            float(record["prediction_score"]) if record is not None else np.nan
                        ),
                        "target_weight": 1.0 / portfolio.top_n if is_target else 0.0,
                        "model_id": signal.model_id,
                        "source_signal_manifest_hash": signal.source_signal_manifest_hash,
                        "environment": "paper",
                        "created_at": created_at,
                    }
                )
            frame = pd.DataFrame(rows, columns=ORDER_COLUMNS)
            total_written += append_ledger(
                self._portfolio_root(portfolio.portfolio_id) / "orders.parquet",
                frame,
                unique_columns=("order_id",),
                sort_columns=("as_of", "rank", "ts_code"),
            )
        return PaperTradingRebalanceResult(
            as_of,
            execution_rule,
            len(self.settings.paper_trading.portfolios),
            total_written,
            self.paper_root,
        )

    def execute(self, as_of: str) -> PaperTradingExecutionResult:
        """Execute pending orders for exactly one authoritative trading-session open."""

        self.init()
        total_trades = 0
        total_equity = 0
        prices = self._execution_prices(as_of)
        previous_session = self._previous_open_session(as_of)
        for portfolio in self.settings.paper_trading.portfolios:
            root = self._portfolio_root(portfolio.portfolio_id)
            orders = read_ledger(root / "orders.parquet")
            if orders.empty:
                continue
            due = orders.loc[
                orders["execution_rule"].astype(str).eq("next_open")
                & orders["as_of"].astype(str).eq(previous_session)
            ].copy()
            if due.empty:
                continue
            trades = read_ledger(root / "trades.parquet")
            handled = set(trades["order_id"].astype(str)) if not trades.empty else set()
            due = due.loc[~due["order_id"].astype(str).isin(handled)]
            if due.empty:
                continue
            written_trades, written_equity = self._execute_portfolio(portfolio, as_of, due, prices)
            total_trades += written_trades
            total_equity += written_equity
        return PaperTradingExecutionResult(
            as_of,
            len(self.settings.paper_trading.portfolios),
            total_trades,
            total_equity,
            self.paper_root,
        )

    def report(self, as_of: str) -> PaperTradingReportResult:
        """Publish a deterministic human-readable snapshot of all virtual accounts."""

        self.init()
        portfolios = [
            self._portfolio_report(portfolio, as_of)
            for portfolio in self.settings.paper_trading.portfolios
        ]
        output_dir = self.reports_root / "paper_trading_daily" / as_of
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.md"
        summary = {
            "schema_version": 1,
            "artifact_name": "paper_trading_daily_report",
            "as_of": as_of,
            "portfolio_count": len(portfolios),
            "portfolios": portfolios,
            "constraints": {
                "broker_connected": False,
                "real_orders_generated": False,
                "execution": "next_open",
            },
        }
        atomic_write_json(summary_path, summary)
        _atomic_write_text(report_path, _render_report(summary))
        return PaperTradingReportResult(as_of, report_path, summary_path, len(portfolios))

    def run_daily(
        self,
        as_of: str,
        *,
        production_summary_path: Path | None = None,
    ) -> PaperTradingDailyResult:
        """Execute prior T+1 orders, create today's targets, then publish a report."""

        execution = self.execute(as_of)
        rebalance = self.rebalance(as_of, production_summary_path=production_summary_path)
        report = self.report(as_of)
        return PaperTradingDailyResult(as_of, rebalance, execution, report)

    def _execute_portfolio(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        due: DataFrame,
        prices: dict[str, _Price],
    ) -> tuple[int, int]:
        root = self._portfolio_root(portfolio.portfolio_id)
        positions = self._current_positions(portfolio.portfolio_id, as_of)
        cash = self._current_cash(portfolio.portfolio_id, as_of)
        source_hash = payload_sha256(sorted(set(due["source_signal_manifest_hash"].astype(str))))
        run_id = (
            "execute_"
            + payload_sha256([portfolio.portfolio_id, as_of, sorted(due["order_id"].astype(str))])[
                :20
            ]
        )
        open_value = sum(
            position.shares
            * (
                prices[code].open
                if code in prices and math.isfinite(prices[code].open)
                else position.last_mark_price
            )
            for code, position in positions.items()
        )
        target_equity = cash + open_value
        trade_rows: list[dict[str, Any]] = []
        touched: set[str] = set()

        due = due.sort_values(["target_weight", "rank", "ts_code"], kind="mergesort")
        for raw_row in due.itertuples(index=False):
            row = cast(Any, raw_row)
            code = str(row.ts_code)
            target_weight = float(row.target_weight)
            position = positions.get(code, _Position(0, 0.0, "", 0.0))
            price = prices.get(code)
            desired = _desired_shares(
                target_equity,
                target_weight,
                position.shares,
                price,
                self.settings.paper_trading.lot_size,
            )
            if desired >= position.shares:
                continue
            shares = position.shares - desired
            touched.add(code)
            if position.entry_date >= as_of:
                trade_rows.append(
                    self._trade_row(row, run_id, as_of, "sell", "rejected", 0, np.nan, "t_plus_one")
                )
                continue
            if price is None or not price.can_sell:
                trade_rows.append(
                    self._trade_row(
                        row, run_id, as_of, "sell", "rejected", 0, np.nan, "not_sellable"
                    )
                )
                continue
            gross = shares * price.open
            commission = gross * self.settings.paper_trading.commission
            stamp = gross * self.settings.paper_trading.stamp_duty
            slippage = gross * self.settings.paper_trading.slippage
            cost = commission + stamp + slippage
            proceeds = gross - cost
            cash += proceeds
            realized = proceeds - position.average_cost * shares
            position.shares = desired
            if desired == 0:
                position.average_cost = 0.0
                position.entry_date = ""
            position.last_mark_price = price.close
            positions[code] = position
            trade_rows.append(
                self._trade_row(
                    row,
                    run_id,
                    as_of,
                    "sell",
                    "filled",
                    shares,
                    price.open,
                    "",
                    commission=commission,
                    stamp_duty=stamp,
                    slippage=slippage,
                    cash_delta=proceeds,
                    realized_pnl=realized,
                )
            )

        due = due.sort_values(["rank", "ts_code"], kind="mergesort")
        handled_orders = {str(row["order_id"]) for row in trade_rows}
        for raw_row in due.itertuples(index=False):
            row = cast(Any, raw_row)
            code = str(row.ts_code)
            target_weight = float(row.target_weight)
            position = positions.get(code, _Position(0, 0.0, "", 0.0))
            price = prices.get(code)
            desired = _desired_shares(
                target_equity,
                target_weight,
                position.shares,
                price,
                self.settings.paper_trading.lot_size,
            )
            if desired <= position.shares:
                if str(row.order_id) not in handled_orders:
                    trade_rows.append(
                        self._trade_row(row, run_id, as_of, "none", "noop", 0, np.nan, "")
                    )
                continue
            touched.add(code)
            if price is None or not price.can_buy:
                trade_rows.append(
                    self._trade_row(row, run_id, as_of, "buy", "rejected", 0, np.nan, "not_buyable")
                )
                continue
            requested = desired - position.shares
            per_lot_cash = (
                self.settings.paper_trading.lot_size
                * price.open
                * (
                    1
                    + self.settings.paper_trading.commission
                    + self.settings.paper_trading.slippage
                )
            )
            affordable_lots = int(cash // per_lot_cash)
            shares = min(
                requested,
                affordable_lots * self.settings.paper_trading.lot_size,
            )
            if shares <= 0:
                trade_rows.append(
                    self._trade_row(
                        row, run_id, as_of, "buy", "rejected", 0, np.nan, "insufficient_cash"
                    )
                )
                continue
            gross = shares * price.open
            commission = gross * self.settings.paper_trading.commission
            slippage = gross * self.settings.paper_trading.slippage
            cost = commission + slippage
            spent = gross + cost
            old_cost = position.average_cost * position.shares
            position.shares += shares
            position.average_cost = (old_cost + spent) / position.shares
            position.entry_date = as_of if not position.entry_date else position.entry_date
            position.last_mark_price = price.close
            positions[code] = position
            cash -= spent
            trade_rows.append(
                self._trade_row(
                    row,
                    run_id,
                    as_of,
                    "buy",
                    "filled",
                    shares,
                    price.open,
                    "",
                    commission=commission,
                    slippage=slippage,
                    cash_delta=-spent,
                )
            )

        trade_frame = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
        position_rows = self._position_rows(
            portfolio.portfolio_id, as_of, run_id, source_hash, positions, prices, touched
        )
        append_ledger(
            root / "positions.parquet",
            position_rows,
            unique_columns=("event_id",),
            sort_columns=("as_of", "ts_code"),
        )
        equity_row = self._equity_row(
            portfolio.portfolio_id,
            as_of,
            run_id,
            source_hash,
            cash,
            positions,
            prices,
            trade_frame,
        )
        equity_count = append_ledger(
            root / "equity_curve.parquet",
            equity_row,
            unique_columns=("equity_id",),
            sort_columns=("as_of",),
        )
        # Trade confirmation is the commit marker. If an earlier ledger write fails,
        # the deterministic position/equity events can be replayed safely.
        trade_count = append_ledger(
            root / "trades.parquet",
            trade_frame,
            unique_columns=("trade_id",),
            sort_columns=("as_of", "trade_id"),
        )
        return trade_count, equity_count

    def _trade_row(
        self,
        order: object,
        run_id: str,
        as_of: str,
        side: str,
        status: str,
        shares: int,
        price: float,
        reason: str,
        *,
        commission: float = 0.0,
        stamp_duty: float = 0.0,
        slippage: float = 0.0,
        cash_delta: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> dict[str, Any]:
        typed_order = cast(Any, order)
        gross = shares * price if shares and math.isfinite(price) else 0.0
        return {
            "trade_id": payload_sha256([str(typed_order.order_id), as_of]),
            "order_id": str(typed_order.order_id),
            "run_id": run_id,
            "as_of": as_of,
            "signal_date": str(typed_order.as_of),
            "portfolio_id": str(typed_order.portfolio_id),
            "ts_code": str(typed_order.ts_code),
            "side": side,
            "status": status,
            "shares": shares,
            "price": price,
            "gross_value": gross,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "slippage": slippage,
            "cost": commission + stamp_duty + slippage,
            "cash_delta": cash_delta,
            "realized_pnl": realized_pnl,
            "reason": reason,
            "source_signal_manifest_hash": str(typed_order.source_signal_manifest_hash),
            "environment": "paper",
        }

    def _position_rows(
        self,
        portfolio_id: str,
        as_of: str,
        run_id: str,
        source_hash: str,
        positions: dict[str, _Position],
        prices: dict[str, _Price],
        touched: set[str],
    ) -> DataFrame:
        rows: list[dict[str, Any]] = []
        for code in sorted(set(positions) | touched):
            position = positions.get(code, _Position(0, 0.0, "", 0.0))
            if code in prices and math.isfinite(prices[code].close):
                position.last_mark_price = prices[code].close
            rows.append(
                {
                    "event_id": payload_sha256([run_id, code]),
                    "run_id": run_id,
                    "as_of": as_of,
                    "portfolio_id": portfolio_id,
                    "ts_code": code,
                    "shares": position.shares,
                    "average_cost": position.average_cost,
                    "entry_date": position.entry_date,
                    "last_mark_price": position.last_mark_price,
                    "market_value": position.shares * position.last_mark_price,
                    "source_signal_manifest_hash": source_hash,
                }
            )
        return pd.DataFrame(rows, columns=POSITION_COLUMNS)

    def _equity_row(
        self,
        portfolio_id: str,
        as_of: str,
        run_id: str,
        source_hash: str,
        cash: float,
        positions: dict[str, _Position],
        prices: dict[str, _Price],
        trades: DataFrame,
    ) -> DataFrame:
        holdings = sum(
            position.shares
            * (
                prices[code].close
                if code in prices and math.isfinite(prices[code].close)
                else position.last_mark_price
            )
            for code, position in positions.items()
        )
        equity = cash + holdings
        history = read_ledger(self._portfolio_root(portfolio_id) / "equity_curve.parquet")
        if not history.empty:
            history = history.loc[history["as_of"].astype(str) < as_of]
        previous = (
            float(history.sort_values("as_of").iloc[-1]["equity"])
            if not history.empty
            else self.settings.paper_trading.initial_cash
        )
        previous_peak = max(float(history["equity"].max()), equity) if not history.empty else equity
        filled = trades.loc[trades["status"] == "filled"] if not trades.empty else trades
        turnover = (
            float(filled["gross_value"].sum()) / previous
            if previous > 0 and not filled.empty
            else 0.0
        )
        existing_trades = read_ledger(self._portfolio_root(portfolio_id) / "trades.parquet")
        all_trades = pd.concat([existing_trades, trades], ignore_index=True)
        closed = all_trades.loc[
            (all_trades.get("status") == "filled") & (all_trades.get("side") == "sell")
        ]
        win_rate = (
            float((closed["realized_pnl"].astype(float) > 0).mean()) if not closed.empty else np.nan
        )
        return pd.DataFrame(
            [
                {
                    "equity_id": payload_sha256([portfolio_id, as_of, source_hash]),
                    "run_id": run_id,
                    "as_of": as_of,
                    "portfolio_id": portfolio_id,
                    "cash": cash,
                    "holdings_value": holdings,
                    "equity": equity,
                    "nav": equity / self.settings.paper_trading.initial_cash,
                    "daily_return": equity / previous - 1.0 if previous > 0 else 0.0,
                    "drawdown": equity / previous_peak - 1.0 if previous_peak > 0 else 0.0,
                    "turnover": turnover,
                    "win_rate": win_rate,
                    "source_signal_manifest_hash": source_hash,
                }
            ],
            columns=EQUITY_COLUMNS,
        )

    def _execution_prices(self, as_of: str) -> dict[str, _Price]:
        daily = self.raw_root / "daily" / "**" / "*.parquet"
        limits = self.raw_root / "stk_limit" / "**" / "*.parquet"
        universe = self.processed_root / "universe_daily" / "**" / "*.parquet"
        query = f"""
            SELECT CAST(d.ts_code AS VARCHAR) AS ts_code,
                   CAST(d.open AS DOUBLE) AS open,
                   CAST(d.close AS DOUBLE) AS close,
                   CAST(l.up_limit AS DOUBLE) AS up_limit,
                   CAST(l.down_limit AS DOUBLE) AS down_limit,
                   COALESCE(CAST(u.is_suspended AS BOOLEAN), FALSE) AS is_suspended,
                   COALESCE(CAST(u.is_st AS BOOLEAN), FALSE) AS is_st
            FROM read_parquet('{daily.as_posix()}', hive_partitioning=false) d
            LEFT JOIN read_parquet('{limits.as_posix()}', hive_partitioning=false) l
              ON CAST(d.trade_date AS VARCHAR)=CAST(l.trade_date AS VARCHAR)
             AND CAST(d.ts_code AS VARCHAR)=CAST(l.ts_code AS VARCHAR)
            LEFT JOIN read_parquet('{universe.as_posix()}', hive_partitioning=false) u
              ON CAST(d.trade_date AS VARCHAR)=CAST(u.trade_date AS VARCHAR)
             AND CAST(d.ts_code AS VARCHAR)=CAST(u.ts_code AS VARCHAR)
            WHERE CAST(d.trade_date AS VARCHAR)=?
            ORDER BY d.ts_code
        """  # noqa: S608 -- configured local Parquet paths
        with duckdb.connect() as connection:
            frame = connection.execute(query, [as_of]).fetch_df()
        if frame.empty:
            raise DataValidationError(f"paper execution has no daily prices for {as_of}")
        tolerance = self.settings.paper_trading.price_tolerance
        prices: dict[str, _Price] = {}
        for raw_row in frame.itertuples(index=False):
            row = cast(Any, raw_row)
            prices[str(row.ts_code)] = _Price(
                float(row.open),
                float(row.close),
                bool(
                    not row.is_suspended
                    and not row.is_st
                    and row.open > 0
                    and (pd.isna(row.up_limit) or row.open < row.up_limit - tolerance)
                ),
                bool(
                    not row.is_suspended
                    and row.open > 0
                    and (pd.isna(row.down_limit) or row.open > row.down_limit + tolerance)
                ),
            )
        return prices

    def _previous_open_session(self, as_of: str) -> str:
        calendar = self.raw_root / "trade_cal" / "**" / "*.parquet"
        query = f"""
            SELECT MAX(CAST(cal_date AS VARCHAR))
            FROM read_parquet('{calendar.as_posix()}', hive_partitioning=false)
            WHERE CAST(is_open AS INTEGER)=1 AND CAST(cal_date AS VARCHAR)<?
        """  # noqa: S608 -- configured local Parquet path
        with duckdb.connect() as connection:
            result = connection.execute(query, [as_of]).fetchone()
        value = result[0] if result is not None else None
        if value is None:
            raise DataValidationError(f"no previous trading session exists before {as_of}")
        return str(value)

    def _current_positions(self, portfolio_id: str, before_or_on: str) -> dict[str, _Position]:
        frame = read_ledger(self._portfolio_root(portfolio_id) / "positions.parquet")
        if frame.empty:
            return {}
        eligible = frame.loc[frame["as_of"].astype(str) < before_or_on].copy()
        if eligible.empty:
            return {}
        latest = (
            eligible.sort_values(["as_of", "event_id"], kind="mergesort")
            .groupby("ts_code", sort=False)
            .tail(1)
        )
        positions: dict[str, _Position] = {}
        for raw_row in latest.itertuples(index=False):
            row = cast(Any, raw_row)
            shares = int(row.shares)
            if shares <= 0:
                continue
            positions[str(row.ts_code)] = _Position(
                shares,
                float(row.average_cost),
                str(row.entry_date),
                float(row.last_mark_price),
            )
        return positions

    def _current_cash(self, portfolio_id: str, before_or_on: str) -> float:
        trades = read_ledger(self._portfolio_root(portfolio_id) / "trades.parquet")
        if trades.empty:
            return self.settings.paper_trading.initial_cash
        eligible = trades.loc[trades["as_of"].astype(str) < before_or_on]
        return self.settings.paper_trading.initial_cash + float(eligible["cash_delta"].sum())

    def _portfolio_report(self, portfolio: PaperPortfolioSettings, as_of: str) -> dict[str, Any]:
        root = self._portfolio_root(portfolio.portfolio_id)
        equity = read_ledger(root / "equity_curve.parquet")
        equity = equity.loc[equity["as_of"].astype(str) <= as_of] if not equity.empty else equity
        latest = equity.sort_values("as_of").iloc[-1] if not equity.empty else None
        positions = self._current_positions(portfolio.portfolio_id, _date_after(as_of))
        orders = read_ledger(root / "orders.parquet")
        today_orders = (
            orders.loc[orders["as_of"].astype(str) == as_of] if not orders.empty else orders
        )
        trades = read_ledger(root / "trades.parquet")
        today_trades = (
            trades.loc[trades["as_of"].astype(str) == as_of] if not trades.empty else trades
        )
        return {
            "portfolio_id": portfolio.portfolio_id,
            "signal_type": portfolio.signal_type,
            "top_n": portfolio.top_n,
            "cash": (
                self.settings.paper_trading.initial_cash
                if latest is None
                else float(latest["cash"])
            ),
            "equity": (
                self.settings.paper_trading.initial_cash
                if latest is None
                else float(latest["equity"])
            ),
            "nav": 1.0 if latest is None else float(latest["nav"]),
            "daily_return": 0.0 if latest is None else float(latest["daily_return"]),
            "drawdown": 0.0 if latest is None else float(latest["drawdown"]),
            "turnover": 0.0 if latest is None else float(latest["turnover"]),
            "win_rate": (
                None if latest is None or pd.isna(latest["win_rate"]) else float(latest["win_rate"])
            ),
            "position_count": len(positions),
            "signal_orders": len(today_orders),
            "executions": int((today_trades["status"] == "filled").sum())
            if not today_trades.empty
            else 0,
            "rejections": int((today_trades["status"] == "rejected").sum())
            if not today_trades.empty
            else 0,
        }

    def _account_payload(self, portfolio: PaperPortfolioSettings) -> dict[str, Any]:
        git = current_git_info()
        payload = {
            "schema_version": 1,
            "artifact_name": "paper_trading_account",
            "portfolio_id": portfolio.portfolio_id,
            "signal": portfolio.model_dump(mode="json"),
            "initial_cash": self.settings.paper_trading.initial_cash,
            "execution": self.settings.paper_trading.execution,
            "lot_size": self.settings.paper_trading.lot_size,
            "commission": self.settings.paper_trading.commission,
            "stamp_duty": self.settings.paper_trading.stamp_duty,
            "slippage": self.settings.paper_trading.slippage,
            "environment": "paper",
            "broker_connected": False,
            "real_orders_generated": False,
            "git_commit": git["commit"],
            "config_hash": config_hash(self.config_path),
        }
        payload["run_id"] = "paper_account_" + payload_sha256(payload)[:20]
        payload["as_of"] = None
        payload["source_signal_manifest_hash"] = None
        return payload

    def _portfolio_root(self, portfolio_id: str) -> Path:
        return self.paper_root / portfolio_id


def _round_lot(shares: float, lot_size: int) -> int:
    if not math.isfinite(shares) or shares <= 0:
        return 0
    return int(shares // lot_size) * lot_size


def _desired_shares(
    equity: float,
    target_weight: float,
    current_shares: int,
    price: _Price | None,
    lot_size: int,
) -> int:
    if price is not None and price.open > 0:
        return _round_lot(equity * target_weight / price.open, lot_size)
    if target_weight <= 0:
        return 0
    return current_shares + lot_size


def _account_contract(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "portfolio_id",
            "signal",
            "initial_cash",
            "execution",
            "lot_size",
            "commission",
            "stamp_duty",
            "slippage",
            "environment",
            "broker_connected",
            "real_orders_generated",
        )
    }


def _date_after(value: str) -> str:
    return f"{value}~"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Paper Trading Daily Report",
        "",
        f"Date: {summary['as_of']}",
        "",
        "This is a broker-free virtual execution report. It contains no real orders.",
        "",
        "| Portfolio | NAV | Daily return | Drawdown | Turnover | Positions | Filled | Rejected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for portfolio in summary["portfolios"]:
        lines.append(
            "| {portfolio_id} | {nav:.6f} | {daily_return:.4%} | {drawdown:.4%} | "
            "{turnover:.4%} | {position_count} | {executions} | {rejections} |".format(**portfolio)
        )
    return "\n".join(lines) + "\n"
