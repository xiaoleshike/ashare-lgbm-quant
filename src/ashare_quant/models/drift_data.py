"""DuckDB-backed loading for immutable model drift diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import (
    feature_list_hash,
    load_json_object,
    parse_feature_array,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class DriftModelContext:
    """Frozen model and historical prediction identity used by one diagnostic run."""

    model: RegisteredModel
    feature_names: tuple[str, ...]
    feature_hash: str
    backtest_dir: Path
    backtest_manifest: dict[str, Any]
    prediction_path: Path


class DriftDataLoader:
    """Read generated features, predictions, universes, and matured labels without writes."""

    def __init__(
        self,
        *,
        processed_root: Path,
        reports_root: Path,
        models_root: Path,
    ) -> None:
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.models_root = models_root
        self.feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
        self.universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
        self.label_glob = processed_root / "labels_forward" / "**" / "*.parquet"
        for name in ("features_daily", "universe_daily", "labels_forward"):
            if not list((processed_root / name).glob("**/*.parquet")):
                raise DataValidationError(f"{name} is required for model drift diagnostics")

    def resolve_context(
        self,
        model_id: str,
        start_date: str,
        end_date: str,
    ) -> DriftModelContext:
        """Resolve a registered model and one encompassing immutable prediction artifact."""

        records = [
            record
            for record in ModelRegistry(self.models_root).list_models()
            if record.model_id == model_id
        ]
        if not records:
            raise DataValidationError(f"model_id is not registered: {model_id}")
        model = records[0]
        artifact = Path(model.artifact_path)
        payload = load_json_object(artifact / "feature_list.json")
        feature_names = parse_feature_array(payload, "features", artifact / "feature_list.json")
        digest = feature_list_hash(feature_names)
        if digest != model.feature_hash:
            raise DataValidationError(
                "registered model feature hash differs from feature_list.json"
            )
        backtest_dir, manifest = self._find_backtest(model_id, start_date, end_date)
        prediction_path = backtest_dir / "predictions.parquet"
        if not prediction_path.exists():
            raise DataValidationError(
                f"historical prediction artifact is missing: {prediction_path}; "
                "rerun backtest historical"
            )
        if str(manifest.get("feature_hash")) != digest:
            raise DataValidationError("historical predictions use a different feature hash")
        return DriftModelContext(
            model=model,
            feature_names=feature_names,
            feature_hash=digest,
            backtest_dir=backtest_dir,
            backtest_manifest=manifest,
            prediction_path=prediction_path,
        )

    def load_predictions(
        self, context: DriftModelContext, start_date: str, end_date: str
    ) -> DataFrame:
        """Load frozen predictions and derive within-date percentiles without rescoring."""

        query = f"""
            SELECT CAST(trade_date AS VARCHAR) AS trade_date,
                   CAST(ts_code AS VARCHAR) AS ts_code,
                   CAST(prediction_score AS DOUBLE) AS prediction_score,
                   CAST(rank AS BIGINT) AS rank,
                   COUNT(*) OVER (PARTITION BY trade_date) AS cross_section_size
            FROM read_parquet('{context.prediction_path.as_posix()}', hive_partitioning=false)
            WHERE CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
            ORDER BY trade_date, rank, ts_code
        """  # noqa: S608 -- immutable local artifact and parameterized dates
        frame = self._query(query, [start_date, end_date], "historical predictions")
        if frame.empty:
            raise DataValidationError(f"no historical predictions for {start_date}..{end_date}")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise DataValidationError("historical predictions contain duplicate keys")
        if not np.isfinite(frame["prediction_score"]).all():
            raise DataValidationError("historical predictions contain non-finite scores")
        frame["score_percentile"] = 1.0 - (
            (frame["rank"].astype(float) - 1.0) / frame["cross_section_size"].astype(float)
        )
        return frame

    def load_feature_samples(
        self,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
        *,
        reference_rows: int,
        evaluation_rows_per_month: int,
    ) -> tuple[DataFrame, DataFrame]:
        """Load deterministic training-reference and monthly evaluation feature samples."""

        columns = _feature_columns(context.feature_names)
        prediction_file = context.prediction_path.as_posix()
        feature_files = self.feature_glob.as_posix()
        training_start = context.model.training_date_range["start"]
        training_end = context.model.training_date_range["end"]
        reference_query = f"""
            SELECT CAST(f.trade_date AS VARCHAR) AS trade_date,
                   CAST(f.ts_code AS VARCHAR) AS ts_code,
                   {columns}
            FROM read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) AS f
            INNER JOIN read_parquet('{self.universe_glob.as_posix()}', hive_partitioning=false) AS u
              ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR)
             AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)
            WHERE CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
              AND CAST(u.in_model_universe AS BOOLEAN)
            ORDER BY hash(CAST(f.trade_date AS VARCHAR) || CAST(f.ts_code AS VARCHAR)),
                     f.trade_date, f.ts_code
            LIMIT ?
        """  # noqa: S608 -- validated feature identifiers and local artifacts
        reference = self._query(
            reference_query,
            [training_start, training_end, reference_rows],
            "training feature reference",
        )
        evaluation_query = f"""
            WITH joined AS (
                SELECT CAST(p.trade_date AS VARCHAR) AS trade_date,
                       CAST(p.ts_code AS VARCHAR) AS ts_code,
                       substr(CAST(p.trade_date AS VARCHAR), 1, 6) AS month,
                       {columns},
                       row_number() OVER (
                           PARTITION BY substr(CAST(p.trade_date AS VARCHAR), 1, 6)
                           ORDER BY hash(
                               CAST(p.trade_date AS VARCHAR) || CAST(p.ts_code AS VARCHAR)
                           ),
                                    p.trade_date, p.ts_code
                       ) AS sample_rank
                FROM read_parquet('{prediction_file}', hive_partitioning=false) AS p
                INNER JOIN read_parquet('{feature_files}', hive_partitioning=false) AS f
                  ON CAST(p.trade_date AS VARCHAR) = CAST(f.trade_date AS VARCHAR)
                 AND CAST(p.ts_code AS VARCHAR) = CAST(f.ts_code AS VARCHAR)
                WHERE CAST(p.trade_date AS VARCHAR) BETWEEN ? AND ?
            )
            SELECT * EXCLUDE (sample_rank)
            FROM joined
            WHERE sample_rank <= ?
            ORDER BY trade_date, ts_code
        """  # noqa: S608 -- validated feature identifiers and local artifacts
        evaluation = self._query(
            evaluation_query,
            [start_date, end_date, evaluation_rows_per_month],
            "evaluation feature sample",
        )
        if reference.empty or evaluation.empty:
            raise DataValidationError("feature drift samples must be non-empty")
        return reference, evaluation

    def load_feature_coverage(
        self,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
    ) -> tuple[DataFrame, DataFrame]:
        """Calculate exact training and monthly finite-feature coverage."""

        training_start = context.model.training_date_range["start"]
        training_end = context.model.training_date_range["end"]
        reference = self._coverage_query(
            context,
            training_start,
            training_end,
            evaluation=False,
        )
        evaluation = self._coverage_query(context, start_date, end_date, evaluation=True)
        return reference, evaluation

    def load_response_samples(
        self,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
        *,
        horizon: int,
        reference_rows: int,
        evaluation_rows_per_month: int,
    ) -> tuple[DataFrame, DataFrame]:
        """Load post-hoc matured labels joined to same-date features for response analysis."""

        columns = _feature_columns(context.feature_names)
        training_start = context.model.training_date_range["start"]
        training_end = context.model.training_date_range["end"]
        reference_query = self._response_query(
            context,
            columns,
            use_predictions=False,
            monthly_limit=False,
        )
        reference = self._query(
            reference_query,
            [horizon, training_start, training_end, reference_rows],
            "training feature-response reference",
        )
        evaluation_query = self._response_query(
            context,
            columns,
            use_predictions=True,
            monthly_limit=True,
        )
        evaluation = self._query(
            evaluation_query,
            [horizon, start_date, end_date, evaluation_rows_per_month],
            "evaluation feature-response sample",
        )
        for name, frame in (("reference", reference), ("evaluation", evaluation)):
            if frame.empty:
                raise DataValidationError(f"{name} feature-response sample is empty")
            invalid = frame["exit_date"].astype(str) <= frame["trade_date"].astype(str)
            if invalid.any():
                raise DataValidationError(f"{name} response sample contains non-future labels")
        return reference, evaluation

    def _coverage_query(
        self,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
        *,
        evaluation: bool,
    ) -> DataFrame:
        group = "substr(CAST(f.trade_date AS VARCHAR), 1, 6)" if evaluation else "'reference'"
        joins = (
            f"INNER JOIN read_parquet('{context.prediction_path.as_posix()}', "
            "hive_partitioning=false) AS p "
            "ON CAST(f.trade_date AS VARCHAR) = CAST(p.trade_date AS VARCHAR) "
            "AND CAST(f.ts_code AS VARCHAR) = CAST(p.ts_code AS VARCHAR)"
            if evaluation
            else f"INNER JOIN read_parquet('{self.universe_glob.as_posix()}', "
            "hive_partitioning=false) AS u "
            "ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR) "
            "AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)"
        )
        universe_filter = "" if evaluation else "AND CAST(u.in_model_universe AS BOOLEAN)"
        aggregates = ",\n".join(
            f'SUM(CASE WHEN f."{feature}" IS NOT NULL '
            f'AND isfinite(CAST(f."{feature}" AS DOUBLE)) THEN 1 ELSE 0 END) '
            f'AS "{feature}"'
            for feature in context.feature_names
        )
        query = f"""
            SELECT {group} AS month, COUNT(*) AS rows, {aggregates}
            FROM read_parquet('{self.feature_glob.as_posix()}', hive_partitioning=false) AS f
            {joins}
            WHERE CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
              {universe_filter}
            GROUP BY month
            ORDER BY month
        """  # noqa: S608 -- validated feature identifiers and local artifacts
        wide = self._query(query, [start_date, end_date], "feature coverage")
        records: list[dict[str, Any]] = []
        for row in wide.to_dict(orient="records"):
            total = int(row["rows"])
            for feature in context.feature_names:
                valid = int(row[feature])
                records.append(
                    {
                        "month": str(row["month"]),
                        "feature": feature,
                        "rows": total,
                        "valid_rows": valid,
                        "coverage": valid / total if total else 0.0,
                        "missing_ratio": 1.0 - valid / total if total else 1.0,
                    }
                )
        result = pd.DataFrame.from_records(records)
        return result.drop(columns="month") if not evaluation else result

    def _response_query(
        self,
        context: DriftModelContext,
        columns: str,
        *,
        use_predictions: bool,
        monthly_limit: bool,
    ) -> str:
        eligibility_join = (
            f"INNER JOIN read_parquet('{context.prediction_path.as_posix()}', "
            "hive_partitioning=false) AS p "
            "ON CAST(f.trade_date AS VARCHAR) = CAST(p.trade_date AS VARCHAR) "
            "AND CAST(f.ts_code AS VARCHAR) = CAST(p.ts_code AS VARCHAR)"
            if use_predictions
            else f"INNER JOIN read_parquet('{self.universe_glob.as_posix()}', "
            "hive_partitioning=false) AS u "
            "ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR) "
            "AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)"
        )
        universe_filter = "" if use_predictions else "AND CAST(u.in_model_universe AS BOOLEAN)"
        feature_files = self.feature_glob.as_posix()
        label_files = self.label_glob.as_posix()
        rank_expression = (
            "row_number() OVER (PARTITION BY substr(CAST(f.trade_date AS VARCHAR), 1, 6) "
            "ORDER BY hash(CAST(f.trade_date AS VARCHAR) || CAST(f.ts_code AS VARCHAR)), "
            "f.trade_date, f.ts_code)"
            if monthly_limit
            else "row_number() OVER (ORDER BY hash(CAST(f.trade_date AS VARCHAR) || "
            "CAST(f.ts_code AS VARCHAR)), f.trade_date, f.ts_code)"
        )
        return f"""
            WITH joined AS (
                SELECT CAST(f.trade_date AS VARCHAR) AS trade_date,
                       CAST(f.ts_code AS VARCHAR) AS ts_code,
                       substr(CAST(f.trade_date AS VARCHAR), 1, 6) AS month,
                       CAST(l.exit_date AS VARCHAR) AS exit_date,
                       CAST(l.future_excess_ret AS DOUBLE) AS future_excess_ret,
                       {columns},
                       {rank_expression} AS sample_rank
                FROM read_parquet('{feature_files}', hive_partitioning=false) AS f
                {eligibility_join}
                INNER JOIN read_parquet('{label_files}', hive_partitioning=false) AS l
                  ON CAST(f.trade_date AS VARCHAR) = CAST(l.trade_date AS VARCHAR)
                 AND CAST(f.ts_code AS VARCHAR) = CAST(l.ts_code AS VARCHAR)
                WHERE CAST(l.horizon AS INTEGER) = ?
                  AND CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
                  AND CAST(l.is_label_available AS BOOLEAN)
                  AND l.future_excess_ret IS NOT NULL
                  AND isfinite(CAST(l.future_excess_ret AS DOUBLE))
                  {universe_filter}
            )
            SELECT * EXCLUDE (sample_rank)
            FROM joined
            WHERE sample_rank <= ?
            ORDER BY trade_date, ts_code
        """  # noqa: S608 -- validated feature identifiers and local artifacts

    def _find_backtest(
        self,
        model_id: str,
        start_date: str,
        end_date: str,
    ) -> tuple[Path, dict[str, Any]]:
        candidates: list[tuple[str, str, Path, dict[str, Any]]] = []
        for path in sorted((self.reports_root / "backtest").glob("*/manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("model_id") != model_id:
                continue
            available_start = str(payload.get("requested_start_date", ""))
            available_end = str(payload.get("effective_end_date", ""))
            if (
                payload.get("out_of_sample") is True
                and available_start <= start_date
                and available_end >= end_date
            ):
                candidates.append((available_start, available_end, path.parent, payload))
        if not candidates:
            raise DataValidationError(
                "no immutable OOS historical predictions cover "
                f"model_id={model_id} range={start_date}..{end_date}"
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
        _, _, directory, manifest = candidates[0]
        return directory, manifest

    @staticmethod
    def _query(query: str, parameters: list[object], description: str) -> DataFrame:
        try:
            with duckdb.connect() as connection:
                return connection.execute(query, parameters).fetch_df()
        except duckdb.Error as error:
            raise DataValidationError(f"cannot load {description}: {error}") from error


def _feature_columns(features: tuple[str, ...]) -> str:
    for feature in features:
        if not feature.replace("_", "").isalnum():
            raise DataValidationError(f"unsafe feature identifier in model artifact: {feature}")
    return ",\n".join(f'f."{feature}"' for feature in features)
