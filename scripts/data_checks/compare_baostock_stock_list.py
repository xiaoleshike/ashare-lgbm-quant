#!/usr/bin/env python3
"""Compare baostock daily stock lists with locally stored Parquet data.

This script is intentionally read-only. It does not modify local data because the
baostock and Tushare coverage rules may differ, especially around suspensions,
new listings, delistings, and Beijing Stock Exchange securities.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ashare_quant.config import load_settings

BaostockStatus = Literal["trading", "all"]
CheckMode = Literal["list", "quotes"]


@dataclass(frozen=True)
class DailyComparison:
    trade_date: str
    baostock_count: int
    local_count: int
    common_count: int
    only_baostock_count: int
    only_local_count: int
    only_baostock_sample: list[str]
    only_local_sample: list[str]


@dataclass(frozen=True)
class QuoteComparison:
    trade_date: str
    baostock_rows: int
    local_rows: int
    common_rows: int
    only_baostock_count: int
    only_local_count: int
    mismatched_rows: int
    mismatch_counts: dict[str, int]
    only_baostock_sample: list[str]
    only_local_sample: list[str]
    mismatch_sample: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baostock stock lists against local daily Parquet data."
    )
    parser.add_argument("--config", default="config/default.yaml", help="Project YAML config path.")
    parser.add_argument(
        "--check",
        choices=("list", "quotes"),
        default="list",
        help="Compare only stock-code lists, or full daily quote fields.",
    )
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", help="Single date, YYYYMMDD or YYYY-MM-DD.")
    date_group.add_argument("--start-date", help="Start date, YYYYMMDD or YYYY-MM-DD.")
    date_group.add_argument(
        "--recent-trading-days",
        type=int,
        help="Use the most recent N local trading dates from the local dataset.",
    )
    parser.add_argument("--end-date", help="End date for --start-date, YYYYMMDD or YYYY-MM-DD.")
    parser.add_argument(
        "--sample-by-month",
        action="store_true",
        help="Randomly sample trading dates by calendar month from the resolved date range.",
    )
    parser.add_argument(
        "--local-dataset",
        default="daily",
        choices=("daily",),
        help="Local dataset used as the per-day stock universe.",
    )
    parser.add_argument(
        "--baostock-status",
        choices=("trading", "all"),
        default="trading",
        help="Use only baostock tradeStatus=1 rows, or all baostock rows.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=20, help="Number of differing codes to print per side."
    )
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="Output format."
    )
    parser.add_argument("--price-tolerance", type=float, default=0.001)
    parser.add_argument("--pct-tolerance", type=float, default=0.01)
    parser.add_argument("--volume-tolerance", type=float, default=1.0)
    parser.add_argument("--amount-tolerance", type=float, default=1000.0)
    parser.add_argument("--trading-days-per-month", type=int, default=5)
    parser.add_argument("--stocks-per-day", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--baostock-retries", type=int, default=5)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    return parser.parse_args()


def normalize_date(value: str) -> str:
    return value.replace("-", "")


def baostock_day(value: str) -> str:
    normalized = normalize_date(value)
    return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"


def normalize_baostock_code(code: str) -> str | None:
    if "." not in code:
        return None
    exchange, symbol = code.split(".", maxsplit=1)
    if len(symbol) != 6 or not symbol.isdigit():
        return None
    suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(exchange.lower())
    if suffix is None:
        return None
    return f"{symbol}.{suffix}"


def is_a_share_stock_code(code: str) -> bool:
    if len(code) != 9 or code[6] != ".":
        return False
    symbol = code[:6]
    exchange = code[7:]
    if exchange == "SH":
        return symbol.startswith(("600", "601", "603", "605", "688", "689"))
    if exchange == "SZ":
        return symbol.startswith(("000", "001", "002", "003", "300", "301"))
    if exchange == "BJ":
        return symbol.startswith(("4", "8", "920"))
    return False


def local_month_file(parquet_root: Path, dataset: str, trade_date: str) -> Path:
    return (
        parquet_root
        / dataset
        / f"year={trade_date[:4]}"
        / f"month={trade_date[4:6]}"
        / "data.parquet"
    )


def local_codes_for_date(parquet_root: Path, dataset: str, trade_date: str) -> set[str]:
    path = local_month_file(parquet_root, dataset, trade_date)
    if not path.exists():
        return set()
    frame = pd.read_parquet(path, columns=["ts_code", "trade_date"])
    day_rows = frame.loc[frame["trade_date"].astype(str) == trade_date, "ts_code"]
    return {code for code in day_rows.dropna().astype(str) if is_a_share_stock_code(code)}


def local_trade_dates(parquet_root: Path, dataset: str) -> list[str]:
    dataset_root = parquet_root / dataset
    files = sorted(dataset_root.glob("year=*/month=*/data.parquet"))
    dates: set[str] = set()
    for file in files:
        frame = pd.read_parquet(file, columns=["trade_date"])
        dates.update(frame["trade_date"].dropna().astype(str).unique())
    return sorted(dates)


def date_range_from_local(parquet_root: Path, dataset: str, start: str, end: str) -> list[str]:
    start_date = normalize_date(start)
    end_date = normalize_date(end)
    return [
        date for date in local_trade_dates(parquet_root, dataset) if start_date <= date <= end_date
    ]


def sample_monthly_dates(dates: list[str], days_per_month: int, seed: int) -> list[str]:
    if days_per_month <= 0:
        raise SystemExit("--trading-days-per-month must be positive")
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not security.
    by_month: dict[str, list[str]] = {}
    for date in dates:
        by_month.setdefault(date[:6], []).append(date)
    sampled: list[str] = []
    for month in sorted(by_month):
        month_dates = sorted(by_month[month])
        count = min(days_per_month, len(month_dates))
        sampled.extend(sorted(rng.sample(month_dates, count)))
    return sampled


def resolve_dates(args: argparse.Namespace, parquet_root: Path) -> list[str]:
    if args.date:
        dates = [normalize_date(args.date)]
    elif args.recent_trading_days:
        local_dates = local_trade_dates(parquet_root, args.local_dataset)
        dates = local_dates[-args.recent_trading_days :]
    else:
        if not args.end_date:
            raise SystemExit("--end-date is required when --start-date is used")
        dates = date_range_from_local(
            parquet_root, args.local_dataset, args.start_date, args.end_date
        )
    if args.sample_by_month:
        return sample_monthly_dates(dates, args.trading_days_per_month, args.seed)
    return dates


def baostock_call(operation: str, retries: int, sleep_seconds: float, call: object) -> object:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = call()  # type: ignore[operator]
        except Exception as error:  # noqa: BLE001 - baostock raises transport exceptions.
            last_error = error
            if attempt == retries:
                break
            time.sleep(sleep_seconds)
            continue
        error_code = getattr(result, "error_code", "0")
        if error_code == "0":
            return result
        last_error = RuntimeError(getattr(result, "error_msg", "unknown baostock error"))
        if attempt < retries:
            time.sleep(sleep_seconds)
    raise RuntimeError(f"baostock {operation} failed after {retries} attempts: {last_error}")


def login_baostock(retries: int, sleep_seconds: float) -> None:
    import baostock as bs  # type: ignore[import-untyped]

    baostock_call("login", retries, sleep_seconds, bs.login)


def fetch_baostock_codes(
    day: str, status: BaostockStatus, retries: int, sleep_seconds: float
) -> set[str]:
    try:
        import baostock as bs  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        message = "baostock is not installed. Run: .venv/bin/python -m pip install baostock"
        raise SystemExit(message) from error

    result = baostock_call(
        f"query_all_stock {day}",
        retries,
        sleep_seconds,
        lambda: bs.query_all_stock(day=baostock_day(day)),
    )

    codes: set[str] = set()
    fields = list(result.fields)
    while result.next():
        row = dict(zip(fields, result.get_row_data(), strict=False))
        if status == "trading" and row.get("tradeStatus") not in {"1", "1.0"}:
            continue
        code = normalize_baostock_code(row.get("code", ""))
        if code is not None and is_a_share_stock_code(code):
            codes.add(code)
    return codes


def baostock_code(ts_code: str) -> str:
    symbol, exchange = ts_code.split(".", maxsplit=1)
    return f"{exchange.lower()}.{symbol}"


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def local_quotes_for_date(parquet_root: Path, dataset: str, trade_date: str) -> pd.DataFrame:
    path = local_month_file(parquet_root, dataset, trade_date)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame = frame.loc[frame["trade_date"].astype(str) == trade_date].copy()
    frame = frame.loc[frame["ts_code"].astype(str).map(is_a_share_stock_code)].copy()
    if frame.empty:
        return frame
    frame["ts_code"] = frame["ts_code"].astype(str)
    return frame.drop_duplicates(subset=["ts_code"], keep="last").set_index("ts_code")


def fetch_baostock_quote(
    code: str, day: str, retries: int, sleep_seconds: float
) -> dict[str, float | str] | None:
    import baostock as bs  # type: ignore[import-untyped]

    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    result = baostock_call(
        f"query_history_k_data_plus {code} {day}",
        retries,
        sleep_seconds,
        lambda: bs.query_history_k_data_plus(
            baostock_code(code),
            fields,
            start_date=baostock_day(day),
            end_date=baostock_day(day),
            frequency="d",
        ),
    )
    if not result.next():
        return None
    row = dict(zip(result.fields, result.get_row_data(), strict=False))
    normalized_code = normalize_baostock_code(row.get("code", ""))
    if normalized_code is None:
        return None
    return {
        "ts_code": normalized_code,
        "open": to_float(row.get("open")),
        "high": to_float(row.get("high")),
        "low": to_float(row.get("low")),
        "close": to_float(row.get("close")),
        "pre_close": to_float(row.get("preclose")),
        "vol_shares": to_float(row.get("volume")),
        "amount_yuan": to_float(row.get("amount")),
        "pct_chg": to_float(row.get("pctChg")),
    }


def fetch_baostock_quotes(
    day: str, codes: Iterable[str], retries: int, sleep_seconds: float
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for code in sorted(codes):
        row = fetch_baostock_quote(code, day, retries, sleep_seconds)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["ts_code"], keep="last").set_index("ts_code")


def local_value(row: pd.Series, field: str) -> float | None:
    if field == "vol_shares":
        value = to_float(row.get("vol"))
        return None if value is None else value * 100.0
    if field == "amount_yuan":
        value = to_float(row.get("amount"))
        return None if value is None else value * 1000.0
    return to_float(row.get(field))


def values_match(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def compare_quotes_day(
    parquet_root: Path,
    dataset: str,
    trade_date: str,
    status: BaostockStatus,
    sample_size: int,
    price_tolerance: float,
    pct_tolerance: float,
    volume_tolerance: float,
    amount_tolerance: float,
    stocks_per_day: int | None = None,
    seed: int = 0,
    retries: int = 5,
    sleep_seconds: float = 5.0,
) -> QuoteComparison:
    baostock_codes = fetch_baostock_codes(trade_date, status, retries, sleep_seconds)
    local_quotes = local_quotes_for_date(parquet_root, dataset, trade_date)
    local_codes_for_sampling = (
        set(local_quotes.index.astype(str)) if not local_quotes.empty else set()
    )
    query_codes = sorted(baostock_codes & local_codes_for_sampling)
    if stocks_per_day is not None:
        if stocks_per_day <= 0:
            raise SystemExit("--stocks-per-day must be positive")
        count = min(stocks_per_day, len(query_codes))
        rng = random.Random(f"{seed}:{trade_date}")  # noqa: S311 - deterministic sampling.
        query_codes = sorted(rng.sample(query_codes, count))
    baostock_quotes = fetch_baostock_quotes(trade_date, query_codes, retries, sleep_seconds)
    baostock_quote_codes = (
        set(baostock_quotes.index.astype(str)) if not baostock_quotes.empty else set()
    )
    local_codes = set(local_quotes.index.astype(str)) if not local_quotes.empty else set()
    common_codes = sorted(baostock_quote_codes & local_codes)
    only_baostock = sorted(baostock_quote_codes - local_codes)
    only_local = sorted(local_codes - baostock_quote_codes)
    if stocks_per_day is not None:
        only_local = []
    tolerances = {
        "open": price_tolerance,
        "high": price_tolerance,
        "low": price_tolerance,
        "close": price_tolerance,
        "pre_close": price_tolerance,
        "pct_chg": pct_tolerance,
        "vol_shares": volume_tolerance,
        "amount_yuan": amount_tolerance,
    }
    mismatch_counts = {field: 0 for field in tolerances}
    mismatch_sample: list[dict[str, object]] = []
    mismatched_rows = 0
    for code in common_codes:
        local_row = local_quotes.loc[code]
        baostock_row = baostock_quotes.loc[code]
        row_mismatches: dict[str, dict[str, float | None]] = {}
        for field, tolerance in tolerances.items():
            local_field = local_value(local_row, field)
            baostock_field = to_float(baostock_row.get(field))
            if not values_match(local_field, baostock_field, tolerance):
                mismatch_counts[field] += 1
                row_mismatches[field] = {"local": local_field, "baostock": baostock_field}
        if row_mismatches:
            mismatched_rows += 1
            if len(mismatch_sample) < sample_size:
                mismatch_sample.append({"ts_code": code, "fields": row_mismatches})
    return QuoteComparison(
        trade_date=trade_date,
        baostock_rows=len(baostock_quote_codes),
        local_rows=len(local_codes),
        common_rows=len(common_codes),
        only_baostock_count=len(only_baostock),
        only_local_count=len(only_local),
        mismatched_rows=mismatched_rows,
        mismatch_counts={field: count for field, count in mismatch_counts.items() if count},
        only_baostock_sample=only_baostock[:sample_size],
        only_local_sample=only_local[:sample_size],
        mismatch_sample=mismatch_sample,
    )


def compare_day(
    parquet_root: Path, dataset: str, trade_date: str, status: BaostockStatus, sample_size: int
) -> DailyComparison:
    baostock_codes = fetch_baostock_codes(trade_date, status, 5, 5.0)
    local_codes = local_codes_for_date(parquet_root, dataset, trade_date)
    only_baostock = sorted(baostock_codes - local_codes)
    only_local = sorted(local_codes - baostock_codes)
    return DailyComparison(
        trade_date=trade_date,
        baostock_count=len(baostock_codes),
        local_count=len(local_codes),
        common_count=len(baostock_codes & local_codes),
        only_baostock_count=len(only_baostock),
        only_local_count=len(only_local),
        only_baostock_sample=only_baostock[:sample_size],
        only_local_sample=only_local[:sample_size],
    )


def print_table(results: list[DailyComparison]) -> None:
    print("trade_date  baostock  local  common  only_baostock  only_local  samples")
    for result in results:
        sample = (
            f"bs_only={','.join(result.only_baostock_sample) or '-'}; "
            f"local_only={','.join(result.only_local_sample) or '-'}"
        )
        print(
            f"{result.trade_date}  {result.baostock_count:8d}  {result.local_count:5d}  "
            f"{result.common_count:6d}  {result.only_baostock_count:13d}  "
            f"{result.only_local_count:10d}  {sample}"
        )


def print_quote_table(results: list[QuoteComparison]) -> None:
    print(
        "trade_date  bs_rows  local  common  only_bs  only_local  "
        "mismatch_rows  mismatch_fields  samples"
    )
    for result in results:
        fields = ",".join(f"{key}:{value}" for key, value in result.mismatch_counts.items()) or "-"
        sample = (
            f"bs_only={','.join(result.only_baostock_sample) or '-'}; "
            f"local_only={','.join(result.only_local_sample) or '-'}; "
            f"mismatch={json.dumps(result.mismatch_sample[:2], ensure_ascii=False)}"
        )
        print(
            f"{result.trade_date}  {result.baostock_rows:7d}  {result.local_rows:5d}  "
            f"{result.common_rows:6d}  {result.only_baostock_count:7d}  "
            f"{result.only_local_count:10d}  {result.mismatched_rows:13d}  {fields}  {sample}"
        )


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    parquet_root = settings.paths.parquet_store
    dates = resolve_dates(args, parquet_root)
    if not dates:
        raise SystemExit("No local trading dates found for the requested range.")

    try:
        import baostock as bs  # type: ignore[import-untyped]
    except ModuleNotFoundError as error:
        message = "baostock is not installed. Run: .venv/bin/python -m pip install baostock"
        raise SystemExit(message) from error

    try:
        login_baostock(args.baostock_retries, args.retry_sleep_seconds)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    try:
        if args.check == "quotes":
            quote_results = [
                compare_quotes_day(
                    parquet_root,
                    args.local_dataset,
                    date,
                    args.baostock_status,
                    args.sample_size,
                    args.price_tolerance,
                    args.pct_tolerance,
                    args.volume_tolerance,
                    args.amount_tolerance,
                    args.stocks_per_day if args.sample_by_month else None,
                    args.seed,
                    args.baostock_retries,
                    args.retry_sleep_seconds,
                )
                for date in dates
            ]
            if args.format == "json":
                print(
                    json.dumps(
                        [asdict(result) for result in quote_results],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print_quote_table(quote_results)
        else:
            results = [
                compare_day(
                    parquet_root, args.local_dataset, date, args.baostock_status, args.sample_size
                )
                for date in dates
            ]
            if args.format == "json":
                print(
                    json.dumps(
                        [asdict(result) for result in results],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print_table(results)
    finally:
        bs.logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
