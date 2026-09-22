"""DuckDB data loading and per-date relevance construction for Ranker experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
    sample_selection_by_date: DataFrame = field(default_factory=pd.DataFrame)

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


@dataclass(slots=True)
class RankerPredictionDataset:
    """Label-free signal-date universe and its observable feature coverage."""

    frame: DataFrame
    feature_names: tuple[str, ...]
    coverage_by_date: DataFrame

    @property
    def features(self) -> DataFrame:
        """Return the float32 scoring matrix in governed feature order."""

        return self.frame.loc[:, list(self.feature_names)]


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
            audit = connection.execute(
                f"""
                SELECT CAST(u.trade_date AS VARCHAR) AS trade_date,
                       count(*) AS expected_model_universe_rows,
                       count(f.ts_code) AS feature_rows_present,
                       count(l.ts_code) AS label_rows_present,
                       count(*) FILTER (
                         WHERE coalesce(CAST(l.is_label_available AS BOOLEAN),false)
                           AND isfinite(CAST(l.future_excess_ret AS DOUBLE))
                       ) AS selected_labeled_rows,
                       count(*) FILTER (WHERE l.ts_code IS NULL) AS missing_label_rows,
                       count(*) FILTER (WHERE l.ts_code IS NOT NULL
                         AND NOT coalesce(CAST(l.is_label_available AS BOOLEAN),false))
                         AS unavailable_label_rows
                FROM read_parquet('{self.universe_glob.as_posix()}', hive_partitioning=false) u
                LEFT JOIN read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) f
                  ON CAST(f.trade_date AS VARCHAR)=CAST(u.trade_date AS VARCHAR)
                 AND CAST(f.ts_code AS VARCHAR)=CAST(u.ts_code AS VARCHAR)
                LEFT JOIN read_parquet('{self.label_glob.as_posix()}', hive_partitioning=false) l
                  ON CAST(l.trade_date AS VARCHAR)=CAST(u.trade_date AS VARCHAR)
                 AND CAST(l.ts_code AS VARCHAR)=CAST(u.ts_code AS VARCHAR)
                 AND CAST(l.horizon AS INTEGER)=?
                WHERE CAST(u.trade_date AS VARCHAR) BETWEEN ? AND ?
                  AND CAST(u.in_model_universe AS BOOLEAN)
                GROUP BY 1 ORDER BY 1
                """,  # noqa: S608 -- configured local Parquet paths
                [self.horizon, start_date, end_date],
            ).fetch_df()
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
        return RankerDataset(
            frame=frame,
            feature_names=tuple(feature_names),
            sample_selection_by_date=audit,
        )

    def load_prediction_universe(
        self,
        start_date: str,
        end_date: str,
        feature_names: Sequence[str],
    ) -> RankerPredictionDataset:
        """Load signal-date candidates without consulting forward labels."""

        selected = ",\n".join(f'f."{name}"' for name in feature_names)
        query = f"""
            SELECT
                CAST(u.trade_date AS VARCHAR) AS trade_date,
                CAST(u.ts_code AS VARCHAR) AS ts_code,
                f.ts_code IS NOT NULL AS feature_row_present,
                {selected}
            FROM read_parquet('{self.universe_glob.as_posix()}', hive_partitioning=false) AS u
            LEFT JOIN read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) AS f
                ON CAST(u.trade_date AS VARCHAR) = CAST(f.trade_date AS VARCHAR)
               AND CAST(u.ts_code AS VARCHAR) = CAST(f.ts_code AS VARCHAR)
            WHERE CAST(u.trade_date AS VARCHAR) BETWEEN ? AND ?
              AND CAST(u.in_model_universe AS BOOLEAN)
            ORDER BY u.trade_date, u.ts_code
        """  # noqa: S608 -- feature identifiers are validated against the static registry
        with duckdb.connect() as connection:
            expected = connection.execute(query, [start_date, end_date]).fetch_df()
        if expected.empty:
            raise DataValidationError(
                f"ranker prediction universe is empty for {start_date}..{end_date}"
            )
        _require_unique_keys(expected, "prediction universe")
        expected["feature_row_present"] = expected["feature_row_present"].fillna(False).astype(bool)
        coverage = (
            expected.groupby("trade_date", sort=True)
            .agg(
                expected_universe_rows=("ts_code", "size"),
                feature_rows_present=("feature_row_present", "sum"),
            )
            .reset_index()
        )
        frame = expected.loc[expected["feature_row_present"]].drop(columns=["feature_row_present"])
        if frame.empty:
            raise DataValidationError("walk-forward prediction universe has no feature rows")
        for feature in feature_names:
            values = pd.to_numeric(frame[feature], errors="coerce")
            frame[feature] = values.replace([np.inf, -np.inf], np.nan).astype("float32")
        return RankerPredictionDataset(
            frame=frame.reset_index(drop=True),
            feature_names=tuple(feature_names),
            coverage_by_date=coverage,
        )

    def attach_evaluation_labels(
        self,
        predictions: DataFrame,
        relevance_grades: int,
        *,
        trade_calendar: Sequence[str],
        maturity_cutoff: str,
    ) -> DataFrame:
        """Left-join labels after prediction keys and scores have been frozen."""

        required = {"trade_date", "ts_code", "prediction_score"}
        if not required.issubset(predictions.columns):
            raise DataValidationError(
                f"evaluation predictions are missing columns: {sorted(required - set(predictions))}"
            )
        _require_unique_keys(predictions, "evaluation predictions")
        scores = pd.to_numeric(predictions["prediction_score"], errors="coerce")
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            raise DataValidationError("walk-forward predictions contain non-finite scores")
        start_date = str(predictions["trade_date"].astype(str).min())
        end_date = str(predictions["trade_date"].astype(str).max())
        query = f"""
            SELECT
                CAST(trade_date AS VARCHAR) AS trade_date,
                CAST(ts_code AS VARCHAR) AS ts_code,
                CAST(entry_date AS VARCHAR) AS entry_date,
                CAST(exit_date AS VARCHAR) AS exit_date,
                CAST(future_excess_ret AS DOUBLE) AS future_excess_ret_5d,
                CAST(is_label_available AS BOOLEAN) AS is_label_available,
                CAST(label_unavailable_reason AS VARCHAR) AS label_unavailable_reason,
                TRUE AS label_row_exists
            FROM read_parquet('{self.label_glob.as_posix()}', hive_partitioning=false)
            WHERE CAST(horizon AS INTEGER) = ?
              AND CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
            ORDER BY trade_date, ts_code
        """  # noqa: S608 -- local configured Parquet path
        with duckdb.connect() as connection:
            labels = connection.execute(query, [self.horizon, start_date, end_date]).fetch_df()
        if not labels.empty:
            _require_unique_keys(labels, "evaluation labels")
        merged = predictions.merge(
            labels,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        label_exists = merged["label_row_exists"].fillna(False).astype(bool)
        calendar = tuple(str(value) for value in trade_calendar)
        if (
            not calendar
            or calendar != tuple(sorted(set(calendar)))
            or maturity_cutoff not in calendar
        ):
            raise DataValidationError("EVALUATION_LABEL_CALENDAR_INVALID")
        calendar_index = {date: index for index, date in enumerate(calendar)}
        mature: list[bool] = []
        expected_exits: list[str | None] = []
        for position, row in enumerate(merged.itertuples(index=False)):
            if not bool(label_exists.iloc[position]):
                mature.append(False)
                expected_exits.append(None)
                continue
            signal = str(row.trade_date)
            entry = str(row.entry_date)
            exit_date = str(row.exit_date)
            signal_index = calendar_index.get(signal)
            expected_entry_index = None if signal_index is None else signal_index + 1
            expected_exit_index = (
                None if expected_entry_index is None else expected_entry_index + self.horizon
            )
            expected_entry = (
                None
                if expected_entry_index is None or expected_entry_index >= len(calendar)
                else calendar[expected_entry_index]
            )
            expected_exit = (
                None
                if expected_exit_index is None or expected_exit_index >= len(calendar)
                else calendar[expected_exit_index]
            )
            valid = (
                expected_entry is not None
                and expected_exit is not None
                and entry == expected_entry
                and exit_date in calendar_index
                and exit_date >= expected_exit
                and exit_date <= maturity_cutoff
            )
            mature.append(valid)
            expected_exits.append(expected_exit)
        merged["expected_exit_date"] = expected_exits
        merged["is_label_mature"] = pd.Series(mature, index=merged.index, dtype=bool)
        available = merged["is_label_available"].fillna(False).astype(bool)
        returns = pd.to_numeric(merged["future_excess_ret_5d"], errors="coerce")
        if (available & ~merged["is_label_mature"]).any():
            raise DataValidationError("EVALUATION_LABEL_AVAILABLE_BEFORE_MATURITY")
        merged["is_label_available"] = available & merged["is_label_mature"] & np.isfinite(returns)
        merged["future_excess_ret_5d"] = returns
        merged.loc[~merged["is_label_available"], "future_excess_ret_5d"] = np.nan
        reason = merged["label_unavailable_reason"].fillna("").astype(str)
        reason = reason.mask(~label_exists, "missing_label_row")
        reason = reason.mask(
            label_exists & ~merged["is_label_available"] & reason.eq(""), "invalid_label"
        )
        merged["label_unavailable_reason"] = reason
        merged["relevance"] = pd.Series(pd.NA, index=merged.index, dtype="Int32")
        metric_rows = merged["is_label_available"]
        if metric_rows.any():
            percentile = (
                merged.loc[metric_rows]
                .groupby("trade_date", sort=False)["future_excess_ret_5d"]
                .rank(method="average", pct=True)
            )
            relevance = (np.ceil(percentile * relevance_grades) - 1).clip(0, relevance_grades - 1)
            merged.loc[metric_rows, "relevance"] = relevance.astype("int32")
        return merged

    def _validate_inputs(self) -> None:
        for name, directory in (
            ("features_daily", self.processed_root / "features_daily"),
            ("labels_forward", self.processed_root / "labels_forward"),
            ("universe_daily", self.processed_root / "universe_daily"),
        ):
            if not list(directory.glob("**/*.parquet")):
                raise DataValidationError(f"{name} is required for Ranker experiments")


def _require_unique_keys(frame: DataFrame, description: str) -> None:
    duplicates = frame.duplicated(subset=["trade_date", "ts_code"], keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, ["trade_date", "ts_code"]].head(5).to_dict("records")
        raise DataValidationError(f"{description} has duplicate security keys: {sample}")
