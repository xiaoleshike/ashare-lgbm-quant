"""DuckDB data loading and per-date relevance construction for Ranker experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


@dataclass(slots=True)
class RankerDataset:
    """One chronologically bounded, date-grouped ranking dataset."""

    frame: DataFrame
    feature_names: tuple[str, ...]

    @property
    def features(self) -> DataFrame:
        """Return the float32 model matrix in configured feature order."""

        return self.frame.loc[:, list(self.feature_names)]

    @property
    def relevance(self) -> np.ndarray:
        """Return integer per-date relevance grades required by lambdarank."""

        return self.frame["relevance"].to_numpy(dtype=np.int32, copy=False)

    @property
    def groups(self) -> list[int]:
        """Return contiguous trade-date group sizes."""

        return self.frame.groupby("trade_date", sort=False).size().astype(int).tolist()


class RankerDataLoader:
    """Load eligible 5-day labels and selected features with DuckDB column pruning."""

    def __init__(self, processed_root: Path, horizon: int, minimum_group_size: int) -> None:
        self.processed_root = processed_root
        self.horizon = horizon
        self.minimum_group_size = minimum_group_size
        self.feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
        self.label_glob = processed_root / "labels_forward" / "**" / "*.parquet"
        self.universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
        self._validate_inputs()

    def load(
        self,
        start_date: str,
        end_date: str,
        feature_names: Sequence[str],
        relevance_grades: int,
    ) -> RankerDataset:
        """Read one period and construct deterministic within-date relevance grades."""

        selected = ",\n".join(f'f."{name}"' for name in feature_names)
        query = f"""
            SELECT
                CAST(f.trade_date AS VARCHAR) AS trade_date,
                CAST(f.ts_code AS VARCHAR) AS ts_code,
                {selected},
                CAST(l.future_excess_ret AS DOUBLE) AS future_excess_ret_5d
            FROM read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) AS f
            INNER JOIN read_parquet('{self.label_glob.as_posix()}', hive_partitioning=false) AS l
                ON CAST(f.trade_date AS VARCHAR) = CAST(l.trade_date AS VARCHAR)
               AND CAST(f.ts_code AS VARCHAR) = CAST(l.ts_code AS VARCHAR)
               AND CAST(l.horizon AS INTEGER) = ?
            INNER JOIN read_parquet('{self.universe_glob.as_posix()}', hive_partitioning=false) AS u
                ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR)
               AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)
            WHERE CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
              AND CAST(l.is_label_available AS BOOLEAN)
              AND CAST(u.in_model_universe AS BOOLEAN)
              AND isfinite(CAST(l.future_excess_ret AS DOUBLE))
            ORDER BY f.trade_date, f.ts_code
        """  # noqa: S608 -- feature identifiers are validated against the static registry
        with duckdb.connect() as connection:
            frame = connection.execute(query, [self.horizon, start_date, end_date]).fetch_df()
        if frame.empty:
            raise DataValidationError(f"ranker data is empty for {start_date}..{end_date}")
        group_sizes = frame.groupby("trade_date")["ts_code"].transform("size")
        frame = frame.loc[group_sizes >= self.minimum_group_size].reset_index(drop=True)
        if frame.empty:
            raise DataValidationError(
                f"no ranker groups meet minimum_group_size={self.minimum_group_size}"
            )
        for feature in feature_names:
            values = pd.to_numeric(frame[feature], errors="coerce")
            frame[feature] = values.replace([np.inf, -np.inf], np.nan).astype("float32")
        percentile = frame.groupby("trade_date", sort=False)["future_excess_ret_5d"].rank(
            method="average", pct=True
        )
        relevance = np.ceil(percentile * relevance_grades) - 1
        frame["relevance"] = relevance.clip(0, relevance_grades - 1).astype("int32")
        return RankerDataset(frame=frame, feature_names=tuple(feature_names))

    def _validate_inputs(self) -> None:
        for name, directory in (
            ("features_daily", self.processed_root / "features_daily"),
            ("labels_forward", self.processed_root / "labels_forward"),
            ("universe_daily", self.processed_root / "universe_daily"),
        ):
            if not list(directory.glob("**/*.parquet")):
                raise DataValidationError(f"{name} is required for Ranker experiments")
