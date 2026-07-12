"""Validation checks for canonical raw data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from ashare_quant.data.datasets import DATASET_SPECS, DatasetSpec, get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore

type DataFrame = pd.DataFrame
ValidationStatus = Literal["valid", "invalid", "missing", "empty", "skipped_optional"]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Validation outcome for one dataset."""

    dataset: str
    ok: bool
    status: ValidationStatus
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


class DataValidator:
    """Run schema, uniqueness, duplicate, and calendar coverage checks."""

    def __init__(self, store: ParquetDataStore) -> None:
        self._store = store

    def validate_dataset(self, spec: DatasetSpec) -> ValidationResult:
        """Validate one locally stored dataset."""

        status = self._store.status(spec)
        if not status.exists:
            if spec.optional:
                return ValidationResult(
                    spec.name,
                    ok=True,
                    status="skipped_optional",
                    warnings=("optional dataset is not downloaded; skipped validation",),
                )
            return ValidationResult(
                spec.name,
                ok=False,
                status="missing",
                errors=("required dataset is not downloaded",),
            )

        frame = self._store.read_dataset(spec)
        errors: list[str] = []
        warnings: list[str] = []
        if frame.empty:
            if spec.optional:
                return ValidationResult(
                    spec.name,
                    ok=True,
                    status="skipped_optional",
                    warnings=("optional dataset is empty; skipped validation",),
                )
            return ValidationResult(
                spec.name,
                ok=False,
                status="empty",
                errors=("required dataset is empty",),
            )

        missing = [column for column in spec.required_columns if column not in frame.columns]
        if missing:
            errors.append(f"missing required columns: {missing}")

        pk_columns = list(spec.primary_key)
        missing_pk = [column for column in pk_columns if column not in frame.columns]
        if missing_pk:
            errors.append(f"missing primary-key columns: {missing_pk}")
        elif frame.duplicated(subset=pk_columns).any():
            duplicate_count = int(frame.duplicated(subset=pk_columns).sum())
            errors.append(f"primary key is not unique; duplicate rows={duplicate_count}")

        if spec.uses_trade_calendar and not spec.allow_empty_trading_days:
            warnings.extend(self._missing_trading_days(spec, frame))

        if spec.name == "daily":
            warnings.extend(self._daily_ohlc_quality_warnings(frame))
        if spec.name == "stk_limit":
            limit_errors, limit_warnings = self._stk_limit_quality_messages(frame)
            errors.extend(limit_errors)
            warnings.extend(limit_warnings)

        return ValidationResult(
            spec.name,
            ok=not errors,
            status="invalid" if errors else "valid",
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _daily_ohlc_quality_warnings(frame: DataFrame) -> list[str]:
        required = {"ts_code", "trade_date", "open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            return []
        prices = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        bad_mask = (
            (prices["high"] < prices[["open", "close"]].max(axis=1))
            | (prices["low"] > prices[["open", "close"]].min(axis=1))
            | (prices["high"] < prices["low"])
        )
        if not bool(bad_mask.any()):
            return []
        dates = frame.loc[bad_mask, "trade_date"].astype(str)
        pre_2020 = int((dates < "20200101").sum())
        post_2020 = int((dates >= "20200101").sum())
        samples = frame.loc[bad_mask, ["ts_code", "trade_date", "open", "high", "low", "close"]]
        sample_text = samples.head(5).to_dict("records")
        warnings: list[str] = []
        if pre_2020:
            warnings.append(
                "daily has "
                f"{pre_2020} pre-2020 invalid OHLC rows; cleaning rule: exclude rows; "
                f"samples={sample_text}"
            )
        if post_2020:
            warnings.append(
                "daily has "
                f"{post_2020} post-2020 invalid OHLC rows; cleaning rule: mark rows unavailable; "
                f"samples={sample_text}"
            )
        return warnings

    def _stk_limit_quality_messages(self, frame: DataFrame) -> tuple[list[str], list[str]]:
        required = {"ts_code", "trade_date", "up_limit", "down_limit"}
        if not required.issubset(frame.columns):
            return [], []

        working = frame[["ts_code", "trade_date", "up_limit", "down_limit"]].copy()
        working["ts_code"] = working["ts_code"].astype(str)
        working["trade_date"] = working["trade_date"].astype(str)
        prices = working[["up_limit", "down_limit"]].apply(pd.to_numeric, errors="coerce")
        up_limit = prices["up_limit"]
        down_limit = prices["down_limit"]

        null_mask = up_limit.isna() | down_limit.isna()
        no_limit_mask = up_limit.eq(99999.99) & down_limit.eq(0)
        zero_limit_mask = up_limit.eq(0) & down_limit.eq(0)
        bad_order_mask = up_limit.lt(down_limit) & ~no_limit_mask
        unexpected_nonpositive_mask = (
            ((up_limit <= 0) | (down_limit <= 0)) & ~no_limit_mask & ~zero_limit_mask
        )

        errors: list[str] = []
        warnings: list[str] = []
        if bool(null_mask.any()):
            errors.append(f"stk_limit has {int(null_mask.sum())} rows with null limit prices")
        if bool(bad_order_mask.any()):
            errors.append(
                f"stk_limit has {int(bad_order_mask.sum())} rows where up_limit < down_limit"
            )
        if bool(unexpected_nonpositive_mask.any()):
            samples = working.loc[unexpected_nonpositive_mask].head(5).to_dict("records")
            errors.append(
                "stk_limit has "
                f"{int(unexpected_nonpositive_mask.sum())} unexpected non-positive limit rows; "
                f"samples={samples}"
            )

        if bool(no_limit_mask.any()):
            samples = working.loc[no_limit_mask].head(5).to_dict("records")
            warnings.append(
                "stk_limit has "
                f"{int(no_limit_mask.sum())} no-price-limit sentinel rows "
                "(up_limit=99999.99, down_limit=0); cleaning rule: treat as no limit, "
                f"not as executable prices; samples={samples}"
            )

        if bool(zero_limit_mask.any()):
            zero_rows = working.loc[zero_limit_mask, ["ts_code", "trade_date"]].drop_duplicates()
            suspend = self._store.read_dataset(get_dataset_spec("suspend_d"))
            unmatched_count: int | None = None
            if not suspend.empty and {"ts_code", "trade_date"}.issubset(suspend.columns):
                suspend_keys = suspend[["ts_code", "trade_date"]].copy()
                suspend_keys["ts_code"] = suspend_keys["ts_code"].astype(str)
                suspend_keys["trade_date"] = suspend_keys["trade_date"].astype(str)
                merged = zero_rows.merge(
                    suspend_keys.drop_duplicates(),
                    on=["ts_code", "trade_date"],
                    how="left",
                    indicator=True,
                )
                unmatched_count = int((merged["_merge"] == "left_only").sum())
            samples = working.loc[zero_limit_mask].head(5).to_dict("records")
            if unmatched_count is None:
                warnings.append(
                    "stk_limit has "
                    f"{int(zero_limit_mask.sum())} zero-limit rows (up_limit=0, down_limit=0); "
                    "suspend_d is unavailable, so suspension status was not confirmed; "
                    f"samples={samples}"
                )
            elif unmatched_count == 0:
                warnings.append(
                    "stk_limit has "
                    f"{int(zero_limit_mask.sum())} suspended zero-limit rows; cleaning rule: "
                    "treat as no executable limit price and exclude from tradable universe; "
                    f"samples={samples}"
                )
            else:
                errors.append(
                    "stk_limit has "
                    f"{unmatched_count} zero-limit rows without matching suspend_d records; "
                    f"samples={samples}"
                )

        return errors, warnings

    def validate_all(self, dataset_names: tuple[str, ...] | None = None) -> list[ValidationResult]:
        """Validate selected datasets, or all configured datasets."""

        specs = (
            [get_dataset_spec(name) for name in dataset_names]
            if dataset_names
            else DATASET_SPECS.values()
        )
        return [self.validate_dataset(spec) for spec in specs]

    def _missing_trading_days(self, spec: DatasetSpec, frame: DataFrame) -> list[str]:
        if spec.date_column is None or spec.date_column not in frame.columns:
            return []
        calendar = self._store.read_dataset(get_dataset_spec("trade_cal"))
        if (
            calendar.empty
            or "cal_date" not in calendar.columns
            or "is_open" not in calendar.columns
        ):
            return ["trade_cal is unavailable; cannot check trading-day coverage"]

        dates = frame[spec.date_column].astype(str)
        calendar_dates = calendar["cal_date"].astype(str)
        min_date = str(dates.min())
        max_date = str(calendar_dates.max())
        open_days = calendar.loc[
            (calendar["is_open"].astype(int) == 1)
            & (calendar_dates >= min_date)
            & (calendar_dates <= max_date),
            "cal_date",
        ].astype(str)
        present_days = set(dates.unique())
        missing_days = sorted(day for day in open_days.unique() if day not in present_days)
        if not missing_days:
            return []
        preview = ", ".join(missing_days[:5])
        return [f"missing {len(missing_days)} open trading days from trade_cal; first={preview}"]
