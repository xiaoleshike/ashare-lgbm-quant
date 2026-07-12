"""Point-in-time financial statement and indicator features."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

type DataFrame = pd.DataFrame


def build_fundamental_features(
    base: DataFrame,
    fina_indicator: DataFrame,
    income: DataFrame,
    balancesheet: DataFrame,
    cashflow: DataFrame,
) -> DataFrame:
    """Attach financial features using only announcements available by trade_date."""

    working = base[["trade_date", "ts_code"]].copy()
    frames = [
        prepare_fina_indicator(fina_indicator),
        prepare_income(income),
        prepare_balancesheet(balancesheet),
        prepare_cashflow(cashflow),
    ]
    for frame in frames:
        if frame.empty:
            continue
        working = point_in_time_join(working, frame)
    return derive_fundamental_ratios(working)


def prepare_fina_indicator(frame: DataFrame) -> DataFrame:
    """Normalize financial indicator features keyed by announcement date."""

    columns = [
        "ts_code",
        "ann_date",
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "revenue_yoy",
        "netprofit_yoy",
    ]
    return prepare_announced_frame(frame, columns)


def prepare_income(frame: DataFrame) -> DataFrame:
    """Normalize income statement fields keyed by announcement date."""

    columns = ["ts_code", "ann_date", "revenue", "n_income", "total_profit"]
    return prepare_announced_frame(frame, columns)


def prepare_balancesheet(frame: DataFrame) -> DataFrame:
    """Normalize balance-sheet fields keyed by announcement date."""

    columns = [
        "ts_code",
        "ann_date",
        "total_assets",
        "total_liab",
        "total_cur_assets",
        "total_cur_liab",
    ]
    return prepare_announced_frame(frame, columns)


def prepare_cashflow(frame: DataFrame) -> DataFrame:
    """Normalize cash-flow fields keyed by announcement date."""

    columns = ["ts_code", "ann_date", "n_cashflow_act"]
    return prepare_announced_frame(frame, columns)


def prepare_announced_frame(frame: DataFrame, columns: list[str]) -> DataFrame:
    """Return a cleaned announcement-date frame with feature deltas."""

    if frame.empty or not {"ts_code", "ann_date"}.issubset(frame.columns):
        return pd.DataFrame(columns=["ts_code", "ann_date"])
    keep = [column for column in columns if column in frame.columns]
    if len(keep) <= 2:
        return pd.DataFrame(columns=["ts_code", "ann_date"])
    working = frame[keep].copy()
    working["ts_code"] = working["ts_code"].astype(str)
    working["ann_date"] = working["ann_date"].astype(str)
    working = working[working["ann_date"].str.len() == 8]
    value_columns = [column for column in keep if column not in {"ts_code", "ann_date"}]
    for column in value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.sort_values(["ts_code", "ann_date"]).drop_duplicates(
        subset=["ts_code", "ann_date"], keep="last"
    )
    for column in ("roe", "revenue_yoy", "netprofit_yoy"):
        if column in working.columns:
            working[f"{column}_delta"] = working.groupby("ts_code", sort=False)[column].diff()
    return working


def point_in_time_join(base: DataFrame, announced: DataFrame) -> DataFrame:
    """As-of join announcement rows where ann_date is not later than trade_date."""

    if announced.empty:
        return base
    base_work = base.copy()
    base_work = base_work.drop(
        columns=[column for column in base_work.columns if column.startswith("ann_date")],
        errors="ignore",
    )
    base_work["trade_date"] = base_work["trade_date"].astype(str)
    base_work["ts_code"] = base_work["ts_code"].astype(str)
    base_work["_trade_key"] = pd.to_numeric(base_work["trade_date"], errors="coerce")
    announced_work = announced.copy()
    announced_work["ann_date"] = announced_work["ann_date"].astype(str)
    announced_work["ts_code"] = announced_work["ts_code"].astype(str)
    announced_work["_ann_key"] = pd.to_numeric(announced_work["ann_date"], errors="coerce")
    base_work = base_work.dropna(subset=["_trade_key"]).sort_values(["ts_code", "_trade_key"])
    announced_work = announced_work.dropna(subset=["_ann_key"]).sort_values(["ts_code", "_ann_key"])
    if base_work.empty or announced_work.empty:
        return base.drop(columns=["_trade_key"], errors="ignore")

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("base_work", base_work)
        connection.register("announced_work", announced_work)
        merged = connection.execute(
            """
            SELECT b.*, a.* EXCLUDE (ts_code, ann_date, _ann_key)
            FROM base_work AS b
            ASOF LEFT JOIN announced_work AS a
              ON b.ts_code = a.ts_code
             AND b._trade_key >= a._ann_key
            ORDER BY b.ts_code, b._trade_key
            """
        ).fetch_df()
    finally:
        connection.close()
    return merged.drop(columns=["_trade_key"], errors="ignore")


def derive_fundamental_ratios(frame: DataFrame) -> DataFrame:
    """Derive quality ratios from point-in-time financial fields."""

    working = frame.copy()
    working["debt_to_assets"] = ratio(working, "total_liab", "total_assets")
    working["current_ratio"] = ratio(working, "total_cur_assets", "total_cur_liab")
    working["ocf_to_profit"] = ratio(working, "n_cashflow_act", "n_income")
    for column in (
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "revenue_yoy",
        "netprofit_yoy",
        "roe_delta",
        "revenue_yoy_delta",
        "netprofit_yoy_delta",
    ):
        if column not in working.columns:
            working[column] = np.nan
    keep = [
        "trade_date",
        "ts_code",
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "current_ratio",
        "ocf_to_profit",
        "revenue_yoy",
        "netprofit_yoy",
        "roe_delta",
        "revenue_yoy_delta",
        "netprofit_yoy_delta",
    ]
    return working[keep].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")


def ratio(frame: DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Return numerator / denominator when both columns exist and denominator is non-zero."""

    if numerator not in frame.columns or denominator not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    num = pd.to_numeric(frame[numerator], errors="coerce")
    den = pd.to_numeric(frame[denominator], errors="coerce").replace(0, np.nan)
    return num / den
