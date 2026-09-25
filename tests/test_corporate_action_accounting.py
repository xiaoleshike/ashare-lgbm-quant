from __future__ import annotations

import pandas as pd
import pytest

from ashare_quant.backtest.corporate_actions import (
    CORPORATE_ACTION_EXECUTION_POLICY_VERSION,
    CorporateActionExecutionPolicy,
    default_corporate_action_execution_policy,
)
from ashare_quant.backtest.engine import (
    ACCOUNTING_SCHEMA_VERSION,
    BacktestInputs,
    simulate_portfolio,
)
from ashare_quant.config.settings import BacktestSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity_transition import SecurityIdentityTransition


@pytest.mark.parametrize("ratio", [1.0, 2.0, 0.5])
def test_supported_share_conversion_preserves_position_and_mark_value(ratio: float) -> None:
    result = _run(
        transitions=(_transition("AAA.SZ", "BBB.SZ", "20240104", ratio),),
        prices=[
            _price("20240103", "AAA.SZ", 20.0, 20.0),
            _price("20240104", "BBB.SZ", 20.0 / ratio, 20.0 / ratio),
            _price("20240105", "BBB.SZ", 21.0 / ratio, 21.0 / ratio),
        ],
        holding_days=2,
    )

    action = result.corporate_actions.iloc[0]
    buy = result.trades.query("side == 'buy' and status == 'filled'").iloc[0]
    sell = result.trades.query("side == 'sell' and status == 'filled'").iloc[0]
    assert action["old_shares"] == pytest.approx(100.0)
    assert action["new_shares"] == pytest.approx(100.0 * ratio)
    assert action["old_last_valid_close"] == pytest.approx(20.0)
    assert action["adjusted_last_valid_close"] == pytest.approx(20.0 / ratio)
    assert action["pre_action_mark_value"] == pytest.approx(2000.0)
    assert action["post_action_mark_value"] == pytest.approx(2000.0)
    assert action["cash_delta"] == 0.0
    assert action["transaction_cost"] == 0.0
    assert action["turnover_contribution"] == 0.0
    assert buy["position_id"] == sell["position_id"] == action["position_id"]
    assert result.holdings.query("trade_date == '20240104'").iloc[0]["ts_code"] == "BBB.SZ"
    assert (
        result.holdings.query("trade_date == '20240104'").iloc[0]["original_entry_ts_code"]
        == "AAA.SZ"
    )
    assert result.metrics["filled_trades"] == 2
    assert result.metrics["corporate_action_transformations"] == 1
    assert result.metrics["unsupported_transition_crossings"] == 0
    assert result.accounting_summary["accounting_schema_version"] == ACCOUNTING_SCHEMA_VERSION
    assert result.corporate_action_policy["version"] == CORPORATE_ACTION_EXECUTION_POLICY_VERSION
    assert len(str(result.execution_provenance["corporate_action_ledger_hash"])) == 64


def test_transition_happens_before_same_day_exit_without_synthetic_trade() -> None:
    result = _run(
        transitions=(_transition("AAA.SZ", "BBB.SZ", "20240104", 1.0),),
        prices=[
            _price("20240103", "AAA.SZ", 20.0, 20.0),
            _price("20240104", "BBB.SZ", 22.0, 22.0),
        ],
        holding_days=1,
    )

    sell = result.trades.query("side == 'sell' and status == 'filled'").iloc[0]
    assert sell["trade_date"] == "20240104"
    assert sell["ts_code"] == "BBB.SZ"
    assert len(result.corporate_actions) == 1
    assert len(result.trades.query("status == 'filled'")) == 2


def test_suspended_successor_uses_adjusted_stale_mark() -> None:
    result = _run(
        transitions=(_transition("AAA.SZ", "BBB.SZ", "20240104", 2.0),),
        prices=[
            _price("20240103", "AAA.SZ", 20.0, 20.0),
            _price("20240104", "BBB.SZ", float("nan"), float("nan"), suspended=True),
            _price("20240105", "BBB.SZ", 10.0, 10.0),
            _price("20240108", "BBB.SZ", 10.0, 10.0),
        ],
        holding_days=3,
    )

    holding = result.holdings.query("trade_date == '20240104'").iloc[0]
    assert holding["valuation_status"] == "STALE_SUSPENDED"
    assert holding["shares"] == pytest.approx(200.0)
    assert holding["last_valid_close"] == pytest.approx(10.0)
    assert holding["market_value"] == pytest.approx(2000.0)


@pytest.mark.parametrize(
    ("ratio", "transition_type", "continuity_type"),
    [
        (None, "CODE_CHANGE_CONTINUITY", "SAME_LISTED_ENTITY"),
        (1.0, "RESTRUCTURING_CODE_CHANGE", "SAME_LISTED_ENTITY"),
        (1.0, "SHARE_CONVERSION", "SUCCESSOR_ENTITY"),
    ],
)
def test_unknown_ratio_and_unsupported_semantics_fail_closed(
    ratio: float | None,
    transition_type: str,
    continuity_type: str,
) -> None:
    transition = _transition(
        "AAA.SZ",
        "BBB.SZ",
        "20240104",
        ratio,
        transition_type=transition_type,
        continuity_type=continuity_type,
    )
    with pytest.raises(DataValidationError, match="CORPORATE_ACTION_EXECUTION_UNSUPPORTED"):
        _run(
            transitions=(transition,),
            prices=[
                _price("20240103", "AAA.SZ", 20.0, 20.0),
                _price("20240104", "BBB.SZ", 20.0, 20.0),
                _price("20240105", "BBB.SZ", 20.0, 20.0),
            ],
            holding_days=2,
        )


def test_successor_position_collision_fails_closed() -> None:
    signals = pd.DataFrame(
        [
            {"trade_date": "20240102", "ts_code": "AAA.SZ", "score": 2.0},
            {"trade_date": "20240102", "ts_code": "BBB.SZ", "score": 1.0},
        ]
    )
    with pytest.raises(DataValidationError, match="CORPORATE_ACTION_POSITION_COLLISION"):
        _run(
            transitions=(_transition("AAA.SZ", "BBB.SZ", "20240104", 1.0),),
            prices=[
                _price("20240103", "AAA.SZ", 20.0, 20.0),
                _price("20240103", "BBB.SZ", 10.0, 10.0),
                _price("20240104", "BBB.SZ", 10.0, 10.0),
            ],
            holding_days=2,
            signals=signals,
            top_n=2,
        )


def test_supported_monotonic_chain_is_applied_atomically() -> None:
    result = _run(
        transitions=(
            _transition("AAA.SZ", "BBB.SZ", "20240104", 2.0),
            _transition("BBB.SZ", "CCC.SZ", "20240105", 0.5),
        ),
        prices=[
            _price("20240103", "AAA.SZ", 20.0, 20.0),
            _price("20240104", "BBB.SZ", 10.0, 10.0),
            _price("20240105", "CCC.SZ", 20.0, 20.0),
            _price("20240108", "CCC.SZ", 20.0, 20.0),
        ],
        holding_days=3,
    )

    assert result.corporate_actions["predecessor_ts_code"].tolist() == ["AAA.SZ", "BBB.SZ"]
    assert result.corporate_actions["position_id"].nunique() == 1
    assert result.holdings.query("trade_date == '20240105'").iloc[0]["ts_code"] == "CCC.SZ"


def test_same_day_transition_chain_fails_closed_before_partial_application() -> None:
    with pytest.raises(DataValidationError, match="not strictly increasing"):
        _run(
            transitions=(
                _transition("AAA.SZ", "BBB.SZ", "20240104", 2.0),
                _transition("BBB.SZ", "CCC.SZ", "20240104", 0.5),
            ),
            prices=[
                _price("20240103", "AAA.SZ", 20.0, 20.0),
                _price("20240104", "CCC.SZ", 20.0, 20.0),
            ],
            holding_days=2,
        )


def test_transition_does_not_block_when_position_closes_before_effective_date() -> None:
    result = _run(
        transitions=(_transition("AAA.SZ", "BBB.SZ", "20240108", None),),
        prices=[
            _price("20240103", "AAA.SZ", 20.0, 20.0),
            _price("20240104", "AAA.SZ", 20.0, 20.0),
        ],
        holding_days=1,
    )
    assert result.corporate_actions.empty
    assert len(result.trades.query("status == 'filled'")) == 2


def test_policy_identity_is_content_defined_and_restructuring_is_not_auto_supported() -> None:
    policy = default_corporate_action_execution_policy()
    assert policy.policy_hash == default_corporate_action_execution_policy().policy_hash
    assert len(policy.policy_hash) == 64
    assert not policy.classify(
        _transition(
            "AAA.SZ",
            "BBB.SZ",
            "20240104",
            1.0,
            transition_type="RESTRUCTURING_CODE_CHANGE",
        )
    ).supported


def test_unreviewed_execution_policy_is_rejected_by_evidence_grade_simulation() -> None:
    inputs = BacktestInputs(
        signals=pd.DataFrame([{"trade_date": "20240102", "ts_code": "AAA.SZ", "score": 1.0}]),
        prices=pd.DataFrame([_price("20240103", "AAA.SZ", 20.0, 20.0)]),
        calendar=("20240102", "20240103"),
        benchmark=pd.DataFrame({"trade_date": ("20240102", "20240103"), "close": (100.0, 100.0)}),
        identity_transition_version="fixture-v1",
        identity_transition_hash="b" * 64,
        corporate_action_policy=CorporateActionExecutionPolicy(version="unreviewed-v2"),
    )
    settings = BacktestSettings.model_validate(
        {
            "initial_cash": 2000.0,
            "top_n": (1,),
            "holding_period_days": 1,
            "commission": 0.0,
            "stamp_duty": 0.0,
            "slippage": 0.0,
        }
    )
    with pytest.raises(DataValidationError, match="CORPORATE_ACTION_EXECUTION_POLICY_UNSUPPORTED"):
        simulate_portfolio(
            inputs,
            top_n=1,
            settings=settings,
            purpose="executable_validation",
        )


def _run(
    *,
    transitions: tuple[SecurityIdentityTransition, ...],
    prices: list[dict[str, object]],
    holding_days: int,
    signals: pd.DataFrame | None = None,
    top_n: int = 1,
):
    calendar = ("20240102", "20240103", "20240104", "20240105", "20240108")
    inputs = BacktestInputs(
        signals=(
            signals
            if signals is not None
            else pd.DataFrame([{"trade_date": "20240102", "ts_code": "AAA.SZ", "score": 1.0}])
        ),
        prices=pd.DataFrame(prices),
        calendar=calendar,
        benchmark=pd.DataFrame({"trade_date": calendar, "close": [100.0] * len(calendar)}),
        identity_transitions=transitions,
        identity_transition_version="fixture-v1",
        identity_transition_hash="b" * 64,
    )
    settings = BacktestSettings.model_validate(
        {
            "initial_cash": 2000.0,
            "top_n": (top_n,),
            "holding_period_days": holding_days,
            "commission": 0.0,
            "stamp_duty": 0.0,
            "slippage": 0.0,
            "sell_delay_max_days": 2,
        }
    )
    return simulate_portfolio(
        inputs,
        top_n=top_n,
        settings=settings,
        purpose="executable_validation",
    )


def _transition(
    predecessor: str,
    successor: str,
    effective_date: str,
    ratio: float | None,
    *,
    transition_type: str = "CODE_CHANGE_CONTINUITY",
    continuity_type: str = "SAME_LISTED_ENTITY",
) -> SecurityIdentityTransition:
    return SecurityIdentityTransition(
        predecessor_ts_code=predecessor,
        successor_ts_code=successor,
        predecessor_name="old",
        successor_name="new",
        transition_type=transition_type,
        effective_date=effective_date,
        continuity_type=continuity_type,
        share_conversion_ratio=ratio,
        evidence_package_id="evidence-fixture",
        evidence_package_hash="a" * 64,
    )


def _price(
    date: str,
    code: str,
    open_price: float,
    close_price: float,
    *,
    suspended: bool = False,
) -> dict[str, object]:
    return {
        "trade_date": date,
        "ts_code": code,
        "open": open_price,
        "close": close_price,
        "can_buy": not suspended,
        "can_sell": not suspended,
        "is_suspended": suspended,
        "is_listed": True,
        "delist_date": None,
    }
