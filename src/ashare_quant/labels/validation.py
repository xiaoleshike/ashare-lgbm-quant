"""Validation checks for executable forward-return labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from ashare_quant.labels.storage import LabelStore
from ashare_quant.universe.storage import UniverseStore

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class LabelValidationResult:
    """Validation result for stored or in-memory label rows."""

    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


class LabelValidator:
    """Run consistency checks required by the executable label layer."""

    def __init__(
        self,
        store: LabelStore,
        quantile_buckets: int,
        universe_store: UniverseStore | None = None,
    ) -> None:
        self._store = store
        self._quantile_buckets = quantile_buckets
        self._universe_store = universe_store

    def validate(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> LabelValidationResult:
        """Validate stored labels over an optional date range."""

        frame = self._store.read(start_date, end_date)
        result = validate_label_frame(frame, self._quantile_buckets)
        if self._universe_store is None or frame.empty:
            return result
        row_count_errors = validate_label_row_counts(
            self._store, self._universe_store, start_date, end_date
        )
        if not row_count_errors:
            return result
        return LabelValidationResult(
            ok=False,
            errors=(*result.errors, *row_count_errors),
            warnings=result.warnings,
        )


def validate_label_row_counts(
    label_store: LabelStore,
    universe_store: UniverseStore,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, ...]:
    """Validate label row counts against eligible base-universe rows."""

    label_files = sorted(label_store.dataset_dir.glob("**/*.parquet"))
    universe_files = sorted(universe_store.dataset_dir.glob("**/*.parquet"))
    if not label_files or not universe_files:
        return ()

    label_path = str(label_store.dataset_dir / "**/*.parquet")
    universe_path = str(universe_store.dataset_dir / "**/*.parquet")
    start_filter = start_date or "00000000"
    end_filter = end_date or "99999999"
    connection = duckdb.connect(":memory:")
    try:
        mismatch_row = connection.execute(
            """
            WITH labels AS (
              SELECT trade_date, CAST(horizon AS BIGINT) AS horizon, COUNT(*) AS label_rows
              FROM read_parquet(?)
              WHERE trade_date >= ? AND trade_date <= ?
              GROUP BY trade_date, horizon
            ),
            horizons AS (
              SELECT DISTINCT horizon FROM labels
            ),
            universe AS (
              SELECT trade_date, COUNT(*) AS base_rows
              FROM read_parquet(?)
              WHERE trade_date >= ? AND trade_date <= ? AND in_base_universe
              GROUP BY trade_date
            ),
            expected AS (
              SELECT universe.trade_date, horizons.horizon, universe.base_rows
              FROM universe CROSS JOIN horizons
            ),
            mismatches AS (
              SELECT
                expected.trade_date,
                expected.horizon,
                expected.base_rows,
                COALESCE(labels.label_rows, 0) AS label_rows
              FROM expected
              LEFT JOIN labels
                ON expected.trade_date = labels.trade_date
               AND expected.horizon = labels.horizon
              WHERE COALESCE(labels.label_rows, 0) <> expected.base_rows
            )
            SELECT COUNT(*) FROM mismatches
            """,
            [label_path, start_filter, end_filter, universe_path, start_filter, end_filter],
        ).fetchone()
    finally:
        connection.close()
    mismatch_count = int(mismatch_row[0]) if mismatch_row is not None else 0
    if mismatch_count == 0:
        return ()
    return (
        "label row count must match in_base_universe count for every trade_date and horizon; "
        f"mismatch groups={mismatch_count}",
    )


def validate_label_frame(frame: DataFrame, quantile_buckets: int) -> LabelValidationResult:
    """Validate one label frame."""

    errors: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        warnings.append("labels are empty or not built")
        return LabelValidationResult(ok=True, warnings=tuple(warnings))

    if frame.duplicated(subset=["trade_date", "ts_code", "horizon"]).any():
        duplicate_count = int(frame.duplicated(subset=["trade_date", "ts_code", "horizon"]).sum())
        errors.append(f"duplicate label rows={duplicate_count}")

    trade_date = frame["trade_date"].astype(str)
    entry_date = frame["entry_date"].astype(str)
    exit_date = frame["exit_date"].astype(str)
    available = frame["is_label_available"].astype(bool)
    has_entry_date = entry_date.ne("") & frame["entry_date"].notna()

    if ((entry_date <= trade_date) & has_entry_date).any():
        errors.append("entry_date must be greater than trade_date")
    if ((exit_date <= entry_date) & available).any():
        errors.append("exit_date must be greater than entry_date for available labels")

    unavailable = ~available
    if frame.loc[unavailable, "stock_forward_ret"].notna().any():
        errors.append("stock_forward_ret must be null when is_label_available=false")
    if frame.loc[unavailable, "future_excess_ret"].notna().any():
        errors.append("future_excess_ret must be null when is_label_available=false")

    rank = pd.to_numeric(frame.loc[available, "future_rank_pct"], errors="coerce")
    if rank.isna().any() or ((rank < 0) | (rank > 1)).any():
        errors.append("future_rank_pct must be between 0 and 1 when available")

    quantile = pd.to_numeric(frame.loc[available, "future_quantile"], errors="coerce")
    if quantile.isna().any() or ((quantile < 0) | (quantile >= quantile_buckets)).any():
        errors.append("future_quantile must be in the configured bucket range when available")

    tail_available = (
        frame.groupby(["horizon", "trade_date"])["is_label_available"]
        .any()
        .reset_index()
        .sort_values(["horizon", "trade_date"])
    )
    for horizon, group in tail_available.groupby("horizon", sort=True):
        dates_with_available = group.loc[group["is_label_available"].astype(bool), "trade_date"]
        if dates_with_available.empty:
            continue
        last_available_date = str(dates_with_available.max())
        later_available = group.loc[
            (group["trade_date"].astype(str) > last_available_date)
            & group["is_label_available"].astype(bool)
        ]
        if not later_available.empty:
            errors.append(f"horizon {horizon} has available labels after its last available date")

    return LabelValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
