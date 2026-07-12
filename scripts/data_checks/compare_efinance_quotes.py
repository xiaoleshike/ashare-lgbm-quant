#!/usr/bin/env python3
"""Compare efinance daily A-share quotes with local Tushare daily Parquet data.

Read-only checker. It compares raw daily OHLCV fields and never modifies stored data.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ashare_quant.config import load_settings

type EfinanceRawResult = pd.DataFrame | dict[str, pd.DataFrame]


@dataclass(frozen=True)
class EfinanceQuoteComparison:
    trade_date: str
    efinance_rows: int
    local_rows: int
    common_rows: int
    only_efinance_count: int
    only_local_count: int
    mismatched_rows: int
    mismatch_counts: dict[str, int]
    only_efinance_sample: list[str]
    only_local_sample: list[str]
    mismatch_sample: list[dict[str, object]]
    fetch_error_count: int
    fetch_error_sample: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare efinance quotes against local daily data."
    )
    parser.add_argument("--config", default="config/default.yaml")
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date")
    date_group.add_argument("--start-date")
    date_group.add_argument("--recent-trading-days", type=int)
    parser.add_argument("--end-date")
    parser.add_argument("--local-dataset", default="daily", choices=("daily",))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--price-tolerance", type=float, default=0.001)
    parser.add_argument("--pct-tolerance", type=float, default=0.01)
    parser.add_argument("--volume-tolerance", type=float, default=1.0)
    parser.add_argument("--amount-tolerance", type=float, default=1000.0)
    parser.add_argument("--request-timeout", type=float, default=10.0)
    return parser.parse_args()


def normalize_date(value: str) -> str:
    return value.replace("-", "")


def efinance_date(value: str) -> str:
    normalized = normalize_date(value)
    return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"


def local_month_file(parquet_root: Path, dataset: str, trade_date: str) -> Path:
    return (
        parquet_root
        / dataset
        / f"year={trade_date[:4]}"
        / f"month={trade_date[4:6]}"
        / "data.parquet"
    )


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


def local_trade_dates(parquet_root: Path, dataset: str) -> list[str]:
    files = sorted((parquet_root / dataset).glob("year=*/month=*/data.parquet"))
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


def resolve_dates(args: argparse.Namespace, parquet_root: Path) -> list[str]:
    if args.date:
        return [normalize_date(args.date)]
    if args.recent_trading_days:
        dates = local_trade_dates(parquet_root, args.local_dataset)
        return dates[-args.recent_trading_days :]
    if not args.end_date:
        raise SystemExit("--end-date is required when --start-date is used")
    return date_range_from_local(parquet_root, args.local_dataset, args.start_date, args.end_date)


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


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def values_match(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tolerance


def local_value(row: pd.Series, field: str) -> float | None:
    if field == "amount_yuan":
        value = to_float(row.get("amount"))
        return None if value is None else value * 1000.0
    if field == "vol":
        return to_float(row.get("vol"))
    return to_float(row.get(field))


def ts_code_to_ef_code(ts_code: str) -> str:
    return ts_code[:6]


def normalize_ef_code(code: object) -> str | None:
    value = str(code).strip()
    if len(value) != 6 or not value.isdigit():
        return None
    if value.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{value}.SH"
    if value.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{value}.SZ"
    if value.startswith(("4", "8", "920")):
        return f"{value}.BJ"
    return None


def call_efinance(
    codes: list[str],
    start: str,
    end: str,
    retries: int,
    sleep_seconds: float,
    request_timeout: float,
) -> EfinanceRawResult:
    import efinance as ef
    from efinance.common import getter

    original_get = getter.session.get

    def get_with_timeout(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("timeout", request_timeout)
        return original_get(*args, **kwargs)

    getter.session.get = get_with_timeout  # type: ignore[method-assign]
    query: str | list[str] = codes[0] if len(codes) == 1 else codes
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return ef.stock.get_quote_history(
                query,
                beg=start,
                end=end,
                klt=101,
                fqt=0,
                suppress_error=True,
            )
        except Exception as error:  # noqa: BLE001 - network library raises varied exceptions.
            last_error = error
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"efinance failed after {retries} attempts: {last_error}")


def normalize_efinance_frame(raw: EfinanceRawResult, trade_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if isinstance(raw, dict):
        frames = [
            frame for frame in raw.values() if isinstance(frame, pd.DataFrame) and not frame.empty
        ]
    elif isinstance(raw, pd.DataFrame) and not raw.empty:
        frames = [raw]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    if "日期" not in frame.columns or "股票代码" not in frame.columns:
        return pd.DataFrame()
    frame = frame.loc[frame["日期"].astype(str) == efinance_date(trade_date)].copy()
    if frame.empty:
        return frame
    frame["ts_code"] = frame["股票代码"].map(normalize_ef_code)
    frame = frame.loc[frame["ts_code"].notna()].copy()
    return frame.drop_duplicates(subset=["ts_code"], keep="last").set_index("ts_code")


def fetch_efinance_quotes(
    local_codes: list[str],
    trade_date: str,
    batch_size: int,
    retries: int,
    sleep_seconds: float,
    request_timeout: float,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    start = normalize_date(trade_date)
    end = normalize_date(trade_date)
    for idx in range(0, len(local_codes), batch_size):
        batch_local = local_codes[idx : idx + batch_size]
        batch = [ts_code_to_ef_code(code) for code in batch_local]
        try:
            raw = call_efinance(batch, start, end, retries, sleep_seconds, request_timeout)
        except RuntimeError:
            failed.extend(batch_local)
            continue
        frame = normalize_efinance_frame(raw, trade_date)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), failed
    return pd.concat(frames).drop_duplicates(keep="last"), failed


def compare_day(
    parquet_root: Path,
    dataset: str,
    trade_date: str,
    batch_size: int,
    retries: int,
    sleep_seconds: float,
    sample_size: int,
    price_tolerance: float,
    pct_tolerance: float,
    volume_tolerance: float,
    amount_tolerance: float,
    request_timeout: float,
    max_stocks: int | None,
) -> EfinanceQuoteComparison:
    local_quotes = local_quotes_for_date(parquet_root, dataset, trade_date)
    local_codes = sorted(set(local_quotes.index.astype(str))) if not local_quotes.empty else []
    if max_stocks is not None:
        local_codes = local_codes[:max_stocks]
    ef_quotes, fetch_errors = fetch_efinance_quotes(
        local_codes, trade_date, batch_size, retries, sleep_seconds, request_timeout
    )
    ef_codes = set(ef_quotes.index.astype(str)) if not ef_quotes.empty else set()
    local_code_set = set(local_codes)
    common_codes = sorted(ef_codes & local_code_set)
    only_ef = sorted(ef_codes - local_code_set)
    only_local = sorted(local_code_set - ef_codes)
    field_map = {
        "open": ("开盘", price_tolerance),
        "high": ("最高", price_tolerance),
        "low": ("最低", price_tolerance),
        "close": ("收盘", price_tolerance),
        "pct_chg": ("涨跌幅", pct_tolerance),
        "vol": ("成交量", volume_tolerance),
        "amount_yuan": ("成交额", amount_tolerance),
    }
    mismatch_counts = {field: 0 for field in field_map}
    mismatch_sample: list[dict[str, object]] = []
    mismatched_rows = 0
    for code in common_codes:
        local_row = local_quotes.loc[code]
        ef_row = ef_quotes.loc[code]
        row_mismatches: dict[str, dict[str, float | None]] = {}
        for local_field, (ef_field, tolerance) in field_map.items():
            left = local_value(local_row, local_field)
            right = to_float(ef_row.get(ef_field))
            if not values_match(left, right, tolerance):
                mismatch_counts[local_field] += 1
                row_mismatches[local_field] = {"local": left, "efinance": right}
        if row_mismatches:
            mismatched_rows += 1
            if len(mismatch_sample) < sample_size:
                mismatch_sample.append({"ts_code": code, "fields": row_mismatches})
    return EfinanceQuoteComparison(
        trade_date=trade_date,
        efinance_rows=len(ef_codes),
        local_rows=len(local_code_set),
        common_rows=len(common_codes),
        only_efinance_count=len(only_ef),
        only_local_count=len(only_local),
        mismatched_rows=mismatched_rows,
        mismatch_counts={field: count for field, count in mismatch_counts.items() if count},
        only_efinance_sample=only_ef[:sample_size],
        only_local_sample=only_local[:sample_size],
        mismatch_sample=mismatch_sample,
        fetch_error_count=len(fetch_errors),
        fetch_error_sample=fetch_errors[:sample_size],
    )


def print_table(results: list[EfinanceQuoteComparison]) -> None:
    print(
        "trade_date  ef_rows  local  common  only_ef  only_local  "
        "mismatch_rows  fetch_errors  mismatch_fields  samples"
    )
    for result in results:
        fields = ",".join(f"{key}:{value}" for key, value in result.mismatch_counts.items()) or "-"
        sample = (
            f"ef_only={','.join(result.only_efinance_sample) or '-'}; "
            f"local_only={','.join(result.only_local_sample) or '-'}; "
            f"mismatch={json.dumps(result.mismatch_sample[:2], ensure_ascii=False)}"
        )
        print(
            f"{result.trade_date}  {result.efinance_rows:7d}  {result.local_rows:5d}  "
            f"{result.common_rows:6d}  {result.only_efinance_count:7d}  "
            f"{result.only_local_count:10d}  {result.mismatched_rows:13d}  "
            f"{result.fetch_error_count:12d}  {fields}  {sample}"
        )


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    parquet_root = settings.paths.parquet_store
    dates = resolve_dates(args, parquet_root)
    if not dates:
        raise SystemExit("No local trading dates found for the requested range.")
    results = [
        compare_day(
            parquet_root,
            args.local_dataset,
            date,
            args.batch_size,
            args.retries,
            args.retry_sleep_seconds,
            args.sample_size,
            args.price_tolerance,
            args.pct_tolerance,
            args.volume_tolerance,
            args.amount_tolerance,
            args.request_timeout,
            args.max_stocks,
        )
        for date in dates
    ]
    if args.format == "json":
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
