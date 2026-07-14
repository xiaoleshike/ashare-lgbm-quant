"""DuckDB-backed inputs for feature diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


class DiagnosticDataLoader:
    """Join features, labels, and model-universe membership with column pruning."""

    def __init__(self, processed_root: Path, feature_names: Sequence[str], horizon: int) -> None:
        self.processed_root = processed_root
        self.feature_names = tuple(feature_names)
        self.horizon = horizon
        self.feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
        self.label_glob = processed_root / "labels_forward" / "**" / "*.parquet"
        self.universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        for name, path in (
            ("features_daily", self.feature_glob),
            ("labels_forward", self.label_glob),
            ("universe_daily", self.universe_glob),
        ):
            if not list(path.parent.parent.glob("**/*.parquet")):
                raise DataValidationError(f"{name} is required for feature diagnostics")

    def available_months(self, start_date: str, end_date: str) -> tuple[str, ...]:
        """Return feature months intersecting an inclusive date range."""

        query = f"""
            SELECT DISTINCT substr(CAST(trade_date AS VARCHAR), 1, 6) AS month
            FROM read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false)
            WHERE CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
            ORDER BY month
        """  # noqa: S608 -- paths are local configuration, not user SQL
        with duckdb.connect() as connection:
            rows = connection.execute(query, [start_date, end_date]).fetchall()
        return tuple(str(row[0]) for row in rows)

    def iter_period(
        self, start_date: str, end_date: str, *, available_only: bool = True
    ) -> Iterator[DataFrame]:
        """Yield joined monthly frames without loading full history at once."""

        for month in self.available_months(start_date, end_date):
            month_start = max(start_date, f"{month}01")
            month_end = min(end_date, f"{month}31")
            yield self.load(month_start, month_end, available_only=available_only)

    def load(
        self,
        start_date: str,
        end_date: str,
        *,
        available_only: bool = True,
        max_rows: int | None = None,
    ) -> DataFrame:
        """Load a joined date range, optionally with deterministic hash sampling."""

        selected = ",\n".join(f'f."{name}"' for name in self.feature_names)
        availability = "AND CAST(l.is_label_available AS BOOLEAN)" if available_only else ""
        limit = (
            "" if max_rows is None else f"ORDER BY hash(f.trade_date, f.ts_code) LIMIT {max_rows:d}"
        )
        query = f"""
            SELECT
                CAST(f.trade_date AS VARCHAR) AS trade_date,
                CAST(f.ts_code AS VARCHAR) AS ts_code,
                {selected},
                CAST(l.future_excess_ret AS DOUBLE) AS target,
                CAST(l.benchmark_forward_ret AS DOUBLE) AS benchmark_forward_ret
            FROM read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) AS f
            INNER JOIN read_parquet('{self.label_glob.as_posix()}', hive_partitioning=false) AS l
                ON CAST(f.trade_date AS VARCHAR) = CAST(l.trade_date AS VARCHAR)
               AND CAST(f.ts_code AS VARCHAR) = CAST(l.ts_code AS VARCHAR)
               AND CAST(l.horizon AS INTEGER) = ?
            INNER JOIN read_parquet('{self.universe_glob.as_posix()}', hive_partitioning=false) AS u
                ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR)
               AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)
            WHERE CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
              AND CAST(u.in_model_universe AS BOOLEAN)
              {availability}
            {limit}
        """  # noqa: S608 -- identifiers come from the static feature registry
        with duckdb.connect() as connection:
            return connection.execute(query, [self.horizon, start_date, end_date]).fetch_df()
