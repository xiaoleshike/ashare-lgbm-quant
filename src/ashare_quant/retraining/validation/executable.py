"""Executable next-open validation without modifying paper-trading state."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_quant.backtest.data import load_benchmark, load_calendar, load_execution_prices
from ashare_quant.backtest.engine import BacktestInputs, simulate_portfolio
from ashare_quant.backtest.executable_validation import (
    REQUIRED_TOP_N,
    _fully_executable_signal_dates,
    _signals,
)
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.validation.schemas import (
    CandidateValidationContext,
    ExecutableValidationEvidence,
    OfflineValidationRun,
)


class RetrainingExecutableValidator:
    """Apply existing portfolio mechanics to the frozen OOS evaluation scores."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        settings: AppSettings,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.settings = settings

    def evaluate(
        self,
        context: CandidateValidationContext,
        offline: OfflineValidationRun,
    ) -> ExecutableValidationEvidence:
        horizon = context.artifact.horizon
        execution = self.settings.backtest.model_copy(
            update={
                "execution": "next_open",
                "holding_period_days": horizon,
                "top_n": REQUIRED_TOP_N,
            }
        )
        requested_dates = tuple(sorted(offline.predictions["trade_date"].astype(str).unique()))
        calendar = load_calendar(
            self.raw_root,
            requested_dates[0],
            requested_dates[-1],
            horizon + execution.sell_delay_max_days,
        )
        if not calendar:
            raise DataValidationError("VALIDATION_FAILED: executable calendar is empty")
        prices = load_execution_prices(
            self.raw_root,
            self.processed_root,
            calendar[0],
            calendar[-1],
            self.settings.universe.price_tolerance,
        )
        maximum_price_date = str(prices["trade_date"].astype(str).max())
        calendar = [date for date in calendar if date <= maximum_price_date]
        executable_dates = _fully_executable_signal_dates(
            requested_dates, calendar, horizon, execution.sell_delay_max_days
        )
        if executable_dates != requested_dates:
            missing = sorted(set(requested_dates) - set(executable_dates))
            raise DataValidationError(
                f"VALIDATION_FAILED: incomplete executable exits for signal dates: {missing}"
            )
        benchmark = load_benchmark(
            self.raw_root,
            execution.benchmark_index_code,
            calendar[0],
            calendar[-1],
        )
        inputs = BacktestInputs(
            signals=_signals(offline.predictions),
            prices=prices,
            calendar=tuple(calendar),
            benchmark=benchmark,
        )
        results = tuple(
            simulate_portfolio(inputs, top_n=top_n, settings=execution) for top_n in REQUIRED_TOP_N
        )
        for result in results:
            if (
                not result.holdings.empty
                and result.holdings["trade_date"].astype(str).eq(calendar[-1]).any()
            ):
                raise DataValidationError(
                    f"VALIDATION_FAILED: unresolved holdings for Top{result.top_n}"
                )
        metrics = {str(result.top_n): result.metrics for result in results}
        return ExecutableValidationEvidence(
            model_id=context.model.model_id,
            horizon=horizon,
            holding_period=horizon,
            signal_dates=len(requested_dates),
            minimum_signal_date=requested_dates[0],
            maximum_signal_date=requested_dates[-1],
            top_n=REQUIRED_TOP_N,
            execution_config=execution.model_dump(mode="json"),
            metrics=metrics,
        )


def prediction_keys(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Return deterministic score keys for focused executable tests."""

    return tuple(
        frame.loc[:, ["trade_date", "ts_code"]]
        .astype(str)
        .sort_values(["trade_date", "ts_code"], kind="mergesort")
        .itertuples(index=False, name=None)
    )
