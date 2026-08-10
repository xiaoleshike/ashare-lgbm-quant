from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ashare_quant.backtest.costs import ExecutionCostPolicy
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
