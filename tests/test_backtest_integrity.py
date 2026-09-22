from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ashare_quant.backtest.costs import ExecutionCostPolicy
from ashare_quant.backtest.data import load_execution_prices
from ashare_quant.backtest.engine import BacktestInputs, calculate_metrics, simulate_portfolio
from ashare_quant.backtest.invalidation import BacktestInvalidationService
from ashare_quant.backtest.provenance import (
    require_oos_evaluation,
    resolve_model_evaluation_boundary,
)
from ashare_quant.config.settings import (
    BacktestSettings,
    ExecutionCostPolicySettings,
    ExecutionCostScheduleEntry,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver


def test_evidence_model_manifest_is_required(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="BACKTEST_MODEL_PROVENANCE_REQUIRED"):
        resolve_model_evaluation_boundary(tmp_path)


def test_in_sample_overlap_is_rejected_and_exact_next_date_is_allowed(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "artifact_name": "lightgbm_ranker_baseline",
        "experiment_id": "model-a",
        "train_start": "20230101",
        "train_end": "20240131",
        "validation_start": "20230101",
        "validation_end": "20240131",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    boundary = resolve_model_evaluation_boundary(tmp_path)

    with pytest.raises(DataValidationError, match="BACKTEST_IN_SAMPLE_OVERLAP"):
        require_oos_evaluation(
            boundary,
            model_dir=tmp_path,
            evaluation_start="20230101",
            evaluation_end="20240201",
        )
    require_oos_evaluation(
        boundary,
        model_dir=tmp_path,
        evaluation_start="20240201",
        evaluation_end="20240202",
    )


def test_suspended_quote_carries_last_close_without_equity_collapse() -> None:
    inputs = _inputs(
        [
            _price("20240102", 10.0, 10.0),
            _price("20240103", 10.0, 10.0),
            _price("20240104", np.nan, np.nan, can_sell=False, suspended=True),
            _price("20240105", 10.5, 10.5),
            _price("20240108", 10.5, 10.5),
        ],
        calendar=("20240102", "20240103", "20240104", "20240105", "20240108"),
    )
    settings = _settings(holding_period_days=3)

    result = simulate_portfolio(inputs, top_n=1, settings=settings, purpose="executable_validation")

    stale = result.holdings[result.holdings["trade_date"] == "20240104"].iloc[0]
    assert stale["valuation_status"] == "STALE_SUSPENDED"
    assert stale["market_value"] == pytest.approx(
        result.holdings[result.holdings["trade_date"] == "20240103"].iloc[0]["market_value"]
    )
    assert result.daily_returns["net_return"].min() > -0.1
    assert result.accounting_summary["stale_valuation_days"] == 1


def test_920305_suspension_uses_last_valid_close_and_resumes_current_valuation() -> None:
    dates = (
        "20250424",
        "20250425",
        "20250428",
        "20250429",
        "20250430",
        "20250506",
        "20250507",
    )
    inputs = _inputs(
        [
            _price("20250424", 27.25, 27.25),
            _price("20250425", 27.25, 27.25),
            _price("20250428", 19.08, 19.08),
            _price("20250429", 14.09, 14.09),
            _price("20250430", np.nan, np.nan, can_sell=False, suspended=True),
            _price("20250506", 12.79, 12.79),
            _price("20250507", 14.18, 14.18),
        ],
        calendar=dates,
    )

    result = simulate_portfolio(
        inputs,
        top_n=1,
        settings=_settings(holding_period_days=6),
        purpose="executable_validation",
    )

    suspended = result.holdings[result.holdings["trade_date"] == "20250430"].iloc[0]
    resumed = result.holdings[result.holdings["trade_date"] == "20250506"].iloc[0]
    shares = float(suspended["shares"])
    assert suspended["valuation_status"] == "STALE_SUSPENDED"
    assert suspended["market_value"] == pytest.approx(shares * 14.09)
    assert resumed["valuation_status"] == "CURRENT"
    assert resumed["market_value"] == pytest.approx(shares * 12.79)


def test_execution_price_loader_joins_raw_alias_to_canonical_universe(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    for dataset in ("daily", "stk_limit"):
        (raw_root / dataset / "year=2025" / "month=04").mkdir(parents=True)
    (processed_root / "universe_daily" / "year=2025" / "month=04").mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["20250430"],
            "ts_code": ["839680.BJ"],
            "open": [7.0],
            "close": [7.1],
        }
    ).to_parquet(raw_root / "daily" / "year=2025" / "month=04" / "data.parquet")
    pd.DataFrame(
        {
            "trade_date": ["20250430"],
            "ts_code": ["839680.BJ"],
            "up_limit": [12.33],
            "down_limit": [6.65],
        }
    ).to_parquet(raw_root / "stk_limit" / "year=2025" / "month=04" / "data.parquet")
    pd.DataFrame(
        {
            "trade_date": ["20250430"],
            "ts_code": ["920680.BJ"],
            "is_suspended": [False],
            "is_st": [False],
            "is_listed": [True],
            "delist_date": [None],
        }
    ).to_parquet(processed_root / "universe_daily" / "year=2025" / "month=04" / "data.parquet")

    prices = load_execution_prices(
        raw_root,
        processed_root,
        "20250430",
        "20250430",
        1e-6,
        identity_resolver=SecurityIdentityResolver.from_path(
            Path("config/security_identity/bse_code_aliases.json")
        ),
        ts_codes={"920680.BJ"},
    )

    assert prices.loc[0, "ts_code"] == "920680.BJ"
    assert prices.loc[0, "close"] == pytest.approx(7.1)
    assert bool(prices.loc[0, "can_buy"])


def test_execution_price_loader_applies_authoritative_listing_suspension(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    for dataset in ("daily", "stk_limit"):
        (raw_root / dataset / "year=2019" / "month=05").mkdir(parents=True)
    universe_dir = processed_root / "universe_daily" / "year=2019" / "month=05"
    universe_dir.mkdir(parents=True)
    pd.DataFrame(columns=["trade_date", "ts_code", "open", "close"]).to_parquet(
        raw_root / "daily" / "year=2019" / "month=05" / "data.parquet"
    )
    pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"]).to_parquet(
        raw_root / "stk_limit" / "year=2019" / "month=05" / "data.parquet"
    )
    pd.DataFrame(
        {
            "trade_date": ["20190513"],
            "ts_code": ["300028.SZ"],
            "is_suspended": [False],
            "is_st": [False],
            "is_listed": [True],
            "delist_date": ["20200803"],
        }
    ).to_parquet(universe_dir / "data.parquet")

    prices = load_execution_prices(
        raw_root,
        processed_root,
        "20190513",
        "20190513",
        1e-6,
        lifecycle_resolver=SecurityLifecycleResolver.from_path(
            Path("config/security_identity/security_lifecycle_events.json")
        ),
        ts_codes={"300028.SZ"},
    )

    row = prices.iloc[0]
    assert bool(row["is_suspended"])
    assert not bool(row["can_buy"])
    assert not bool(row["can_sell"])
    assert pd.isna(row["close"])


def test_execution_price_loader_normalizes_delist_date_as_terminal_boundary(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    for dataset in ("daily", "stk_limit"):
        (raw_root / dataset / "year=2024" / "month=01").mkdir(parents=True)
    universe_dir = processed_root / "universe_daily" / "year=2024" / "month=01"
    universe_dir.mkdir(parents=True)
    pd.DataFrame(columns=["trade_date", "ts_code", "open", "close"]).to_parquet(
        raw_root / "daily" / "year=2024" / "month=01" / "data.parquet"
    )
    pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"]).to_parquet(
        raw_root / "stk_limit" / "year=2024" / "month=01" / "data.parquet"
    )
    pd.DataFrame(
        {
            "trade_date": ["20240110"],
            "ts_code": ["000001.SZ"],
            "is_suspended": [False],
            "is_st": [False],
            # Legacy processed snapshots used an inclusive delist boundary.
            "is_listed": [True],
            "delist_date": ["20240110"],
        }
    ).to_parquet(universe_dir / "data.parquet")

    prices = load_execution_prices(
        raw_root,
        processed_root,
        "20240110",
        "20240110",
        1e-6,
        ts_codes={"000001.SZ"},
    )

    row = prices.iloc[0]
    assert not bool(row["is_listed"])
    assert not bool(row["can_buy"])
    assert not bool(row["can_sell"])
    assert row["delist_date"] == "20240110"


def test_unexplained_missing_quote_fails_evidence_mode() -> None:
    prices = [
        _price("20240102", 10.0, 10.0),
        _price("20240103", 10.0, 10.0),
        _price("20240105", 10.0, 10.0),
    ]
    inputs = _inputs(
        prices,
        calendar=("20240102", "20240103", "20240104", "20240105"),
    )
    with pytest.raises(DataValidationError, match="BACKTEST_MARKET_DATA_INCOMPLETE"):
        simulate_portfolio(
            inputs,
            top_n=1,
            settings=_settings(holding_period_days=2),
            purpose="oos_evidence",
        )


def test_sell_delay_never_implies_zero_value_and_fails_closed() -> None:
    inputs = _inputs(
        [
            _price("20240102", 10.0, 10.0),
            _price("20240103", 10.0, 10.0),
            _price("20240104", 10.0, 10.0, can_sell=False),
        ],
        calendar=("20240102", "20240103", "20240104"),
    )
    with pytest.raises(DataValidationError, match="BACKTEST_UNRESOLVED_POSITION"):
        simulate_portfolio(
            inputs,
            top_n=1,
            settings=_settings(holding_period_days=1, sell_delay_max_days=0),
            purpose="executable_validation",
        )


def test_long_suspension_carries_past_alert_and_sells_on_authoritative_resume() -> None:
    spring_festival = {"20170127", "20170130", "20170131", "20170201", "20170202"}
    suspension_dates = tuple(
        value.strftime("%Y%m%d")
        for value in pd.bdate_range("2017-01-03", "2017-03-28")
        if value.strftime("%Y%m%d") not in spring_festival
    )
    dates = ("20161229", "20161230", *suspension_dates, "20170329")
    assert len(suspension_dates) == 56
    prices = [_price(dates[0], 10.0, 10.0), _price(dates[1], 10.0, 10.0)]
    prices.extend(
        _price(date, np.nan, np.nan, can_sell=False, suspended=True) for date in dates[2:58]
    )
    prices.append(_price(dates[58], 9.0, 9.0))

    result = simulate_portfolio(
        _inputs(prices, calendar=dates),
        top_n=1,
        settings=_settings(holding_period_days=5, sell_delay_max_days=20),
        purpose="executable_validation",
        delayed_exit_policy="carry_to_calendar_end",
    )

    sell = result.trades[
        (result.trades["side"] == "sell") & (result.trades["status"] == "filled")
    ].iloc[0]
    stale = result.holdings[result.holdings["valuation_status"] == "STALE_SUSPENDED"]
    assert sell["trade_date"] == "20170329"
    assert sell["delayed_exit_days"] == 52
    assert bool(sell["sell_delay_breached"])
    assert result.accounting_summary["sell_delay_breaches"] == 1
    assert result.accounting_summary["maximum_delayed_exit_days"] == 52
    assert result.accounting_summary["resolved_after_sell_delay_breach"] == 1
    assert result.accounting_summary["unresolved_positions"] == 0
    assert np.allclose(stale["market_value"], stale["shares"] * 10.0)
    assert len(result.trades[result.trades["side"] == "buy"]) == 1


def test_long_suspension_unresolved_at_governed_cutoff_fails_closed() -> None:
    dates = tuple(value.strftime("%Y%m%d") for value in pd.bdate_range("2024-01-02", periods=30))
    prices = [_price(dates[0], 10.0, 10.0), _price(dates[1], 10.0, 10.0)]
    prices.extend(
        _price(date, np.nan, np.nan, can_sell=False, suspended=True) for date in dates[2:]
    )

    with pytest.raises(DataValidationError, match="open positions remain at governed cutoff"):
        simulate_portfolio(
            _inputs(prices, calendar=dates),
            top_n=1,
            settings=_settings(holding_period_days=2, sell_delay_max_days=5),
            purpose="executable_validation",
            delayed_exit_policy="carry_to_calendar_end",
        )


def test_terminal_writeoff_requires_explicit_delisting_state() -> None:
    terminal = _price(
        "20240104",
        np.nan,
        np.nan,
        can_sell=False,
        listed=False,
        delist_date="20240104",
    )
    result = simulate_portfolio(
        _inputs(
            [_price("20240102", 10.0, 10.0), _price("20240103", 10.0, 10.0), terminal],
            calendar=("20240102", "20240103", "20240104"),
        ),
        top_n=1,
        settings=_settings(holding_period_days=1),
        purpose="executable_validation",
    )
    terminal_trades = result.trades[result.trades["status"] == "terminal_writeoff"]
    assert len(terminal_trades) == 1
    assert terminal_trades.iloc[0]["reason"] == "verified_terminal_security"


def test_metrics_use_sessions_compounding_and_daily_sharpe() -> None:
    daily = pd.DataFrame(
        {
            "net_return": [0.10, -0.05, 0.02],
            "benchmark_return": [0.05, -0.02, 0.01],
            "equity": [1100.0, 1045.0, 1065.9],
            "turnover": [0.2, 0.4, 0.0],
        }
    )
    trades = pd.DataFrame(
        [
            _trade("p1", "buy", "20240102", 100.0, 1.0),
            _trade("p1", "sell", "20240109", 110.0, 1.0, holding_sessions=5),
            _trade("p2", "buy", "20240110", 100.0, 1.0),
            _trade("p2", "sell", "20240117", 90.0, 1.0, holding_sessions=5),
        ]
    )
    settings = _settings()

    metrics = calculate_metrics(daily, trades, settings)

    benchmark_total = (1.05 * 0.98 * 1.01) - 1.0
    total = 1065.9 / 1000.0 - 1.0
    expected_sharpe = (
        np.mean([0.10, -0.05, 0.02]) / np.std([0.10, -0.05, 0.02], ddof=1) * np.sqrt(252)
    )
    assert metrics["benchmark_total_return"] == pytest.approx(benchmark_total)
    assert metrics["cumulative_excess_return"] == pytest.approx(
        (1.0 + total) / (1.0 + benchmark_total) - 1.0
    )
    assert metrics["sharpe"] == pytest.approx(expected_sharpe)
    assert metrics["average_holding_period_sessions"] == pytest.approx(5.0)
    assert metrics["daily_win_rate"] == pytest.approx(2 / 3)
    assert metrics["trade_win_rate"] == pytest.approx(0.5)
    assert metrics["average_two_way_turnover"] == pytest.approx(0.2)
    assert metrics["maximum_drawdown"] == pytest.approx(1045.0 / 1100.0 - 1.0)


def test_effective_dated_stamp_duty_transition_and_buy_side() -> None:
    policy = ExecutionCostPolicy(ExecutionCostPolicySettings())

    before = policy.calculate("20230825", "sell", 100_000.0)
    after = policy.calculate("20230828", "sell", 100_000.0)
    buy = policy.calculate("20230828", "buy", 100_000.0)

    assert before.stamp_duty == pytest.approx(100.0)
    assert after.stamp_duty == pytest.approx(50.0)
    assert buy.stamp_duty == 0.0
    assert len(policy.policy_hash) == 64


def test_minimum_commission_is_applied_when_configured() -> None:
    policy = ExecutionCostPolicy(
        ExecutionCostPolicySettings(
            schedules=(
                ExecutionCostScheduleEntry(
                    effective_from="19000101",
                    commission_rate=0.00025,
                    minimum_commission=5.0,
                ),
            )
        )
    )
    assert policy.calculate("20240101", "buy", 1000.0).commission == pytest.approx(5.0)


def test_unsupported_execution_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BacktestSettings(execution="next_vwap")  # type: ignore[arg-type]


def test_backtest_invalidation_is_append_only_and_idempotent(tmp_path: Path) -> None:
    backtest = tmp_path / "backtests" / "legacy-run"
    backtest.mkdir(parents=True)
    original = b'{"schema_version":1,"artifact_name":"ranker_executable_backtest"}\n'
    (backtest / "manifest.json").write_bytes(original)
    service = BacktestInvalidationService(
        backtests_root=tmp_path / "backtests", reports_root=tmp_path / "reports"
    )

    first = service.create(
        backtest_id="legacy-run",
        reason_codes=("IN_SAMPLE_MODEL_EVALUATION",),
        reviewed_by="operator",
        note="reviewed fixture",
    )
    second = service.create(
        backtest_id="legacy-run",
        reason_codes=("IN_SAMPLE_MODEL_EVALUATION",),
        reviewed_by="operator",
        note="reviewed fixture",
    )

    assert first.invalidation_id == second.invalidation_id
    assert second.idempotent is True
    assert (first.output_dir / "manifest.json").is_file()
    assert (backtest / "manifest.json").read_bytes() == original


def _settings(**updates: object) -> BacktestSettings:
    values: dict[str, object] = {
        "initial_cash": 1000.0,
        "top_n": (1,),
        "holding_period_days": 2,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "slippage": 0.0,
        "sell_delay_max_days": 2,
    }
    values.update(updates)
    return BacktestSettings.model_validate(values)


def _inputs(prices: list[dict[str, object]], *, calendar: tuple[str, ...]) -> BacktestInputs:
    return BacktestInputs(
        signals=pd.DataFrame([{"trade_date": calendar[0], "ts_code": "000001.SZ", "score": 1.0}]),
        prices=pd.DataFrame(prices),
        calendar=calendar,
        benchmark=pd.DataFrame({"trade_date": calendar, "close": [100.0] * len(calendar)}),
    )


def _price(
    date: str,
    open_price: float,
    close: float,
    *,
    can_sell: bool = True,
    suspended: bool = False,
    listed: bool = True,
    delist_date: str | None = None,
) -> dict[str, object]:
    return {
        "trade_date": date,
        "ts_code": "000001.SZ",
        "open": open_price,
        "close": close,
        "can_buy": not suspended and listed and np.isfinite(open_price),
        "can_sell": can_sell,
        "is_suspended": suspended,
        "is_listed": listed,
        "delist_date": delist_date,
    }


def _trade(
    position_id: str,
    side: str,
    date: str,
    gross: float,
    cost: float,
    *,
    holding_sessions: int | None = None,
) -> dict[str, object]:
    return {
        "position_id": position_id,
        "ts_code": position_id,
        "trade_date": date,
        "side": side,
        "status": "filled",
        "gross_value": gross,
        "cost": cost,
        "holding_sessions": holding_sessions,
    }
