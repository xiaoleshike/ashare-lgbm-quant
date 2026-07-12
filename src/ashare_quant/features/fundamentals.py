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
    income_prepared = prepare_income(income)
    cashflow_prepared = prepare_cashflow(cashflow)
    frames = [
        prepare_fina_indicator(fina_indicator),
        income_prepared,
        prepare_balancesheet(balancesheet),
        cashflow_prepared,
        prepare_ocf_to_profit(income_prepared, cashflow_prepared),
    ]
    for frame in frames:
        if frame.empty:
            continue
        working = point_in_time_join(working, frame)
    return derive_fundamental_ratios(working)


def prepare_fina_indicator(frame: DataFrame) -> DataFrame:
    """Normalize financial indicator features keyed by point-in-time availability date."""

    columns = [
        "ts_code",
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "revenue_yoy",
        "netprofit_yoy",
    ]
    return prepare_announced_frame(frame, columns, "fina_indicator")


def prepare_income(frame: DataFrame) -> DataFrame:
    """Normalize income statement fields keyed by point-in-time availability date."""

    columns = ["ts_code", "revenue", "n_income", "total_profit"]
    return prepare_announced_frame(frame, columns, "income")


def prepare_balancesheet(frame: DataFrame) -> DataFrame:
    """Normalize balance-sheet fields keyed by point-in-time availability date."""

    columns = [
        "total_assets",
        "total_liab",
        "total_cur_assets",
        "total_cur_liab",
    ]
    return prepare_announced_frame(frame, columns, "balancesheet")


def prepare_cashflow(frame: DataFrame) -> DataFrame:
    """Normalize cash-flow fields keyed by point-in-time availability date."""

    columns = ["ts_code", "n_cashflow_act"]
    return prepare_announced_frame(frame, columns, "cashflow")


def prepare_announced_frame(frame: DataFrame, columns: list[str], source_table: str) -> DataFrame:
    """Return records keyed by their point-in-time financial availability date.

    `f_ann_date` is the preferred availability date because corrected statements
    can have an original `ann_date` but only become knowable when `f_ann_date`
    arrives. `ann_date` is used only when `f_ann_date` is missing.
    """

    required = {"ts_code", "ann_date"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["ts_code", "availability_date"])
    metadata = [
        "ts_code",
        "end_date",
        "ann_date",
        "f_ann_date",
        "report_type",
        "update_flag",
    ]
    keep = [column for column in metadata if column in frame.columns]
    keep.extend(column for column in columns if column in frame.columns and column not in keep)
    value_columns = [column for column in keep if column not in set(metadata)]
    if not value_columns:
        return pd.DataFrame(columns=["ts_code", "availability_date"])
    working = frame[keep].copy()
    working["source_table"] = source_table
    working["ts_code"] = working["ts_code"].astype(str)
    for column in ("end_date", "ann_date", "f_ann_date", "report_type", "update_flag"):
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str)
    f_ann_date = valid_date_or_empty(working["f_ann_date"])
    ann_date = valid_date_or_empty(working["ann_date"])
    working["availability_date"] = f_ann_date.mask(f_ann_date.eq(""), ann_date)
    working = working[working["availability_date"].str.len() == 8]
    for column in value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["_availability_key"] = pd.to_numeric(working["availability_date"], errors="coerce")
    working["_end_key"] = pd.to_numeric(valid_date_or_empty(working["end_date"]), errors="coerce")
    working["_ann_key"] = pd.to_numeric(valid_date_or_empty(working["ann_date"]), errors="coerce")
    working["_report_type_key"] = pd.to_numeric(working["report_type"], errors="coerce")
    working["_update_flag_key"] = pd.to_numeric(working["update_flag"], errors="coerce")
    working = working.sort_values(
        [
            "ts_code",
            "_availability_key",
            "_end_key",
            "_report_type_key",
            "_update_flag_key",
            "_ann_key",
        ],
        na_position="first",
    ).drop_duplicates(
        subset=["ts_code", "end_date", "availability_date"],
        keep="last",
    )
    for column in ("roe", "revenue_yoy", "netprofit_yoy"):
        if column in working.columns:
            working[f"{column}_delta"] = working.groupby("ts_code", sort=False)[column].diff()
    return working.drop(
        columns=[
            "_availability_key",
            "_end_key",
            "_ann_key",
            "_report_type_key",
            "_update_flag_key",
        ],
        errors="ignore",
    )


def point_in_time_join(base: DataFrame, announced: DataFrame) -> DataFrame:
    """As-of join rows where financial availability_date is not later than trade_date."""

    if announced.empty:
        return base
    base_work = base.copy()
    base_work = base_work.drop(
        columns=[
            column
            for column in base_work.columns
            if column.startswith(("ann_date", "availability_date", "f_ann_date"))
            or column in {"end_date", "report_type", "update_flag", "source_table"}
        ],
        errors="ignore",
    )
    base_work["trade_date"] = base_work["trade_date"].astype(str)
    base_work["ts_code"] = base_work["ts_code"].astype(str)
    base_work["_trade_key"] = pd.to_numeric(base_work["trade_date"], errors="coerce")
    announced_work = announced.copy()
    for column in (
        "end_date",
        "ann_date",
        "f_ann_date",
        "report_type",
        "update_flag",
        "source_table",
    ):
        if column not in announced_work.columns:
            announced_work[column] = ""
    announced_work["availability_date"] = announced_work["availability_date"].astype(str)
    announced_work["ts_code"] = announced_work["ts_code"].astype(str)
    announced_work["_availability_key"] = pd.to_numeric(
        announced_work["availability_date"], errors="coerce"
    )
    announced_work["_end_key"] = pd.to_numeric(
        valid_date_or_empty(announced_work["end_date"]), errors="coerce"
    )
    base_work = base_work.dropna(subset=["_trade_key"]).sort_values(["ts_code", "_trade_key"])
    announced_work = announced_work.dropna(subset=["_availability_key"]).sort_values(
        ["ts_code", "_availability_key", "_end_key"], na_position="first"
    )
    if base_work.empty or announced_work.empty:
        return base.drop(columns=["_trade_key"], errors="ignore")

    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("base_work", base_work)
        connection.register("announced_work", announced_work)
        merged = connection.execute(
            """
            SELECT b.*, a.* EXCLUDE (
                ts_code,
                end_date,
                ann_date,
                f_ann_date,
                report_type,
                update_flag,
                source_table,
                availability_date,
                _availability_key,
                _end_key
            )
            FROM base_work AS b
            ASOF LEFT JOIN announced_work AS a
              ON b.ts_code = a.ts_code
             AND b._trade_key >= a._availability_key
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
        "ocf_to_profit",
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


def prepare_ocf_to_profit(income: DataFrame, cashflow: DataFrame) -> DataFrame:
    """Build OCF/profit records aligned by the same report period.

    The ratio is available only after both source statements are available, so
    its availability date is the later of income and cash-flow availability.
    """

    required_income = {"ts_code", "end_date", "availability_date", "n_income"}
    required_cashflow = {"ts_code", "end_date", "availability_date", "n_cashflow_act"}
    if (
        income.empty
        or cashflow.empty
        or not required_income.issubset(income.columns)
        or not required_cashflow.issubset(cashflow.columns)
    ):
        return pd.DataFrame(columns=["ts_code", "end_date", "availability_date", "ocf_to_profit"])
    income_work = income[["ts_code", "end_date", "availability_date", "n_income"]].copy()
    cashflow_work = cashflow[["ts_code", "end_date", "availability_date", "n_cashflow_act"]].copy()
    merged = income_work.merge(
        cashflow_work,
        on=["ts_code", "end_date"],
        how="inner",
        suffixes=("_income", "_cashflow"),
    )
    if merged.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "availability_date", "ocf_to_profit"])
    income_key = pd.to_numeric(merged["availability_date_income"], errors="coerce")
    cashflow_key = pd.to_numeric(merged["availability_date_cashflow"], errors="coerce")
    merged["availability_date"] = np.maximum(income_key, cashflow_key).astype("Int64").astype(str)
    merged["ocf_to_profit"] = (
        pd.to_numeric(merged["n_cashflow_act"], errors="coerce")
        / pd.to_numeric(merged["n_income"], errors="coerce").replace(0, np.nan)
    )
    return merged[
        [
            "ts_code",
            "end_date",
            "availability_date",
            "ocf_to_profit",
        ]
    ].sort_values(["ts_code", "availability_date", "end_date"])


def valid_date_or_empty(values: pd.Series) -> pd.Series:
    """Return YYYYMMDD date strings, replacing invalid dates with empty strings."""

    text = values.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return text.where(text.str.fullmatch(r"\d{8}"), "")
