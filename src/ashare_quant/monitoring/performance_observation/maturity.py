"""Authoritative trading-session maturity calculations."""

from __future__ import annotations

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.performance_observation.schemas import SUPPORTED_HORIZONS

type DataFrame = pd.DataFrame


def open_sessions(trade_cal: DataFrame) -> tuple[str, ...]:
    """Return sorted unique open sessions from the authoritative calendar."""

    required = {"cal_date", "is_open"}
    if trade_cal.empty or not required.issubset(trade_cal.columns):
        raise DataValidationError("trade_cal with cal_date and is_open is required")
    calendar = trade_cal.loc[pd.to_numeric(trade_cal["is_open"], errors="coerce").eq(1), "cal_date"]
    sessions = tuple(sorted(calendar.astype(str).drop_duplicates()))
    if not sessions:
        raise DataValidationError("trade_cal contains no open trading sessions")
    return sessions


def maturity_dates(
    sessions: tuple[str, ...],
    signal_date: str,
    horizon: int,
) -> tuple[str, str]:
    """Return T+1 entry and H-session exit without calendar-day approximation."""

    if horizon not in SUPPORTED_HORIZONS:
        raise DataValidationError(f"unsupported observation horizon: {horizon}")
    try:
        signal_position = sessions.index(signal_date)
    except ValueError as error:
        raise DataValidationError(
            f"shadow signal date is not an open trade_cal session: {signal_date}"
        ) from error
    entry_position = signal_position + 1
    exit_position = entry_position + horizon
    if exit_position >= len(sessions):
        raise DataValidationError(
            f"trade_cal cannot determine maturity for signal={signal_date} horizon={horizon}"
        )
    return sessions[entry_position], sessions[exit_position]


def require_observation_session(sessions: tuple[str, ...], as_of: str) -> None:
    """Require a real open session as the observation cutoff."""

    if as_of not in sessions:
        raise DataValidationError(f"observation as_of is not an open trade_cal session: {as_of}")
