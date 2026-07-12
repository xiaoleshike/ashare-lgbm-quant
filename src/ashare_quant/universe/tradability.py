"""Tradability flags derived from raw daily, suspension, and limit-price data."""

from __future__ import annotations

import pandas as pd

from ashare_quant.config.settings import UniverseSettings

type DataFrame = pd.DataFrame


def add_tradability_flags(frame: DataFrame, settings: UniverseSettings) -> DataFrame:
    """Add daily tradability flags without changing universe membership."""

    working = frame.copy()
    up_limit = numeric_column(working, "up_limit")
    down_limit = numeric_column(working, "down_limit")
    close = numeric_column(working, "close")

    has_limit_prices = up_limit.notna() & down_limit.notna() & (up_limit > 0) & (down_limit > 0)
    no_price_limit = up_limit.eq(99999.99) & down_limit.eq(0)
    usable_limit = has_limit_prices & ~no_price_limit

    working["is_limit_up"] = bool_series(
        usable_limit & close.notna() & (close >= up_limit - settings.price_tolerance)
    )
    working["is_limit_down"] = bool_series(
        usable_limit & close.notna() & (close <= down_limit + settings.price_tolerance)
    )

    has_price_data = working["has_price_data"].astype(bool)
    is_listed = working["is_listed"].astype(bool)
    is_suspended = working["is_suspended"].astype(bool)
    can_trade = is_listed & has_price_data & ~is_suspended

    working["can_buy"] = can_trade
    working["can_sell"] = can_trade
    if settings.mark_limit_up_not_buyable:
        working.loc[working["is_limit_up"], "can_buy"] = False
    if settings.mark_limit_down_not_sellable:
        working.loc[working["is_limit_down"], "can_sell"] = False
    working["can_buy"] = working["can_buy"].astype(bool)
    working["can_sell"] = working["can_sell"].astype(bool)
    return working


def numeric_column(frame: DataFrame, column: str) -> pd.Series:
    """Return a numeric series for a possibly missing column."""

    values = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


def bool_series(values: pd.Series) -> pd.Series:
    """Return a boolean series with missing values treated as false."""

    return values.fillna(False).astype(bool)
