"""Effective-dated execution-cost resolution for executable simulations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from ashare_quant.config.settings import BacktestSettings, ExecutionCostPolicySettings
from ashare_quant.data.exceptions import DataValidationError

TradeSide = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class ResolvedExecutionCosts:
    """Rates applicable to one trade date and side."""

    effective_from: str
    commission_rate: float
    minimum_commission: float
    stamp_duty_rate: float
    transfer_fee_rate: float
    slippage_rate: float


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """Deterministic monetary costs for one gross trade notional."""

    commission: float
    stamp_duty: float
    transfer_fee: float
    slippage: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee + self.slippage


class ExecutionCostPolicy:
    """Resolve immutable policy settings by effective trade date."""

    def __init__(self, policy: ExecutionCostPolicySettings) -> None:
        self.policy = policy
        self.identity = policy.model_dump(mode="json")
        encoded = json.dumps(self.identity, sort_keys=True, separators=(",", ":"))
        self.policy_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def from_backtest_settings(cls, settings: BacktestSettings) -> ExecutionCostPolicy:
        return cls(settings.execution_costs)

    def resolve(self, trade_date: str, side: TradeSide) -> ResolvedExecutionCosts:
        if len(trade_date) != 8 or not trade_date.isdigit():
            raise DataValidationError(f"BACKTEST_COST_POLICY_INVALID: invalid date {trade_date}")
        applicable = [item for item in self.policy.schedules if item.effective_from <= trade_date]
        if not applicable:
            raise DataValidationError(
                f"BACKTEST_COST_POLICY_INVALID: no cost regime for {trade_date}"
            )
        selected = applicable[-1]
        return ResolvedExecutionCosts(
            effective_from=selected.effective_from,
            commission_rate=selected.commission_rate,
            minimum_commission=selected.minimum_commission,
            stamp_duty_rate=selected.stamp_duty_sell if side == "sell" else 0.0,
            transfer_fee_rate=selected.transfer_fee_rate,
            slippage_rate=selected.slippage_rate,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete schedule and its deterministic identity."""

        return {**self.identity, "cost_policy_hash": self.policy_hash}

    def calculate(self, trade_date: str, side: TradeSide, gross_value: float) -> TradeCosts:
        if not (gross_value >= 0):
            raise DataValidationError("BACKTEST_COST_POLICY_INVALID: negative gross value")
        rates = self.resolve(trade_date, side)
        commission = (
            max(gross_value * rates.commission_rate, rates.minimum_commission)
            if gross_value > 0
            else 0.0
        )
        return TradeCosts(
            commission=commission,
            stamp_duty=gross_value * rates.stamp_duty_rate,
            transfer_fee=gross_value * rates.transfer_fee_rate,
            slippage=gross_value * rates.slippage_rate,
        )

    def maximum_affordable_gross(self, trade_date: str, side: TradeSide, cash: float) -> float:
        """Return the largest buy notional whose notional plus costs fits cash."""

        if cash <= 0:
            return 0.0
        low, high = 0.0, cash
        for _ in range(64):
            middle = (low + high) / 2.0
            if middle + self.calculate(trade_date, side, middle).total <= cash:
                low = middle
            else:
                high = middle
        return low
