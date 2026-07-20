"""Frozen-model and single-factor attribution for historical diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.research.explainability.contributions import (
    ExplainableModel,
    compute_tree_contributions,
)

type DataFrame = pd.DataFrame


def model_feature_importance(
    model: lgb.Booster,
    feature_names: tuple[str, ...],
) -> list[dict[str, float | int | str]]:
    """Return gain and split importance without fitting or mutating the model."""

    model_names = tuple(model.feature_name())
    if model_names and model_names != feature_names:
        raise DataValidationError("model feature names differ from frozen feature_list.json")
    gains = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    splits = np.asarray(model.feature_importance(importance_type="split"), dtype=int)
    if len(gains) != len(feature_names) or len(splits) != len(feature_names):
        raise DataValidationError("model feature importance length differs from feature list")
    return [
        {"feature": feature, "gain": float(gain), "split": int(split)}
        for feature, gain, split in zip(feature_names, gains, splits, strict=True)
    ]


def shap_importance(
    model: lgb.Booster,
    sample: DataFrame,
    feature_names: tuple[str, ...],
    *,
    prediction_tolerance: float,
) -> tuple[list[dict[str, float | str]], str, float]:
    """Compute deterministic mean absolute TreeSHAP after score reproducibility checks."""

    matrix = _numeric_matrix(sample, feature_names)
    predicted = np.asarray(model.predict(matrix), dtype=float)
    expected = sample["prediction_score"].to_numpy(dtype=float)
    maximum_error = float(np.max(np.abs(predicted - expected))) if len(expected) else 0.0
    if maximum_error > prediction_tolerance:
        raise DataValidationError(
            "diagnostic feature sample does not reproduce frozen predictions: "
            f"max_abs_error={maximum_error:.12g} tolerance={prediction_tolerance:.12g}"
        )
    contributions = compute_tree_contributions(cast(ExplainableModel, model), matrix)
    additive = contributions.base_values + contributions.values.sum(axis=1)
    additive_error = float(np.max(np.abs(additive - expected))) if len(expected) else 0.0
    if additive_error > prediction_tolerance:
        raise DataValidationError(
            "diagnostic SHAP contributions are not additive to frozen predictions: "
            f"max_abs_error={additive_error:.12g} tolerance={prediction_tolerance:.12g}"
        )
    rows: list[dict[str, float | str]] = [
        {
            "feature": feature,
            "mean_abs_shap": float(cast(float, np.mean(np.abs(contributions.values[:, index])))),
            "mean_shap": float(cast(float, np.mean(contributions.values[:, index]))),
        }
        for index, feature in enumerate(feature_names)
    ]
    rows.sort(key=lambda row: (-float(row["mean_abs_shap"]), str(row["feature"])))
    return rows, contributions.method, max(maximum_error, additive_error)


def load_attribution_sample(
    processed_root: Path,
    prediction_path: Path,
    feature_names: tuple[str, ...],
    maximum_rows: int,
) -> DataFrame:
    """Load a deterministic bounded same-date feature sample for frozen predictions."""

    selected = ", ".join(f'f."{name}"' for name in feature_names)
    feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
    query = f"""
        SELECT CAST(p.trade_date AS VARCHAR) AS trade_date,
               CAST(p.ts_code AS VARCHAR) AS ts_code,
               CAST(p.prediction_score AS DOUBLE) AS prediction_score,
               {selected}
        FROM read_parquet('{prediction_path.as_posix()}', hive_partitioning=false) AS p
        INNER JOIN read_parquet('{feature_glob.as_posix()}', hive_partitioning=false) AS f
          ON CAST(p.trade_date AS VARCHAR) = CAST(f.trade_date AS VARCHAR)
         AND CAST(p.ts_code AS VARCHAR) = CAST(f.ts_code AS VARCHAR)
        ORDER BY hash(CAST(p.trade_date AS VARCHAR) || CAST(p.ts_code AS VARCHAR)),
                 p.trade_date, p.ts_code
        LIMIT ?
    """  # noqa: S608 -- local artifact paths and validated feature identifiers
    try:
        with duckdb.connect() as connection:
            frame = connection.execute(query, [maximum_rows]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot load feature attribution sample: {error}") from error
    if frame.empty:
        raise DataValidationError("feature attribution sample is empty")
    return frame


def single_factor_group_returns(
    processed_root: Path,
    prediction_path: Path,
    labels_path: Path,
    feature_names: tuple[str, ...],
    *,
    horizon: int,
    quantiles: int,
) -> list[dict[str, Any]]:
    """Compute daily cross-sectional feature-quantile forward excess returns."""

    feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
    records: list[dict[str, Any]] = []
    for feature in feature_names:
        query = f"""
            SELECT CAST(p.trade_date AS VARCHAR) AS trade_date,
                   CAST(f."{feature}" AS DOUBLE) AS feature_value,
                   CAST(l.future_excess_ret AS DOUBLE) AS future_excess_ret
            FROM read_parquet('{prediction_path.as_posix()}', hive_partitioning=false) AS p
            INNER JOIN read_parquet('{feature_glob.as_posix()}', hive_partitioning=false) AS f
              ON CAST(p.trade_date AS VARCHAR) = CAST(f.trade_date AS VARCHAR)
             AND CAST(p.ts_code AS VARCHAR) = CAST(f.ts_code AS VARCHAR)
            INNER JOIN read_parquet('{labels_path.as_posix()}', hive_partitioning=false) AS l
              ON CAST(p.trade_date AS VARCHAR) = CAST(l.trade_date AS VARCHAR)
             AND CAST(p.ts_code AS VARCHAR) = CAST(l.ts_code AS VARCHAR)
            WHERE CAST(l.horizon AS INTEGER) = ?
              AND CAST(l.is_label_available AS BOOLEAN)
              AND f."{feature}" IS NOT NULL
              AND isfinite(CAST(f."{feature}" AS DOUBLE))
              AND l.future_excess_ret IS NOT NULL
        """  # noqa: S608 -- local artifacts and validated model feature identifiers
        try:
            with duckdb.connect() as connection:
                frame = connection.execute(query, [horizon]).fetch_df()
        except duckdb.Error as error:
            raise DataValidationError(f"cannot attribute feature {feature}: {error}") from error
        records.extend(_factor_quantile_summary(frame, feature, quantiles))
    return records


def _factor_quantile_summary(
    frame: DataFrame,
    feature: str,
    quantiles: int,
) -> list[dict[str, Any]]:
    if frame.empty:
        return [
            {
                "feature": feature,
                "quantile": None,
                "observations": 0,
                "mean_forward_excess_return": None,
                "top_minus_bottom": None,
            }
        ]
    ranked = frame.copy()
    percentile = ranked.groupby("trade_date", sort=False)["feature_value"].rank(
        method="average", pct=True
    )
    ranked["quantile"] = np.minimum(
        np.floor((percentile.to_numpy(dtype=float) - np.finfo(float).eps) * quantiles),
        quantiles - 1,
    ).astype(int)
    grouped = (
        ranked.groupby("quantile", sort=True)["future_excess_ret"]
        .agg(["size", "mean"])
        .reset_index()
    )
    means = dict(zip(grouped["quantile"], grouped["mean"], strict=True))
    spread = (
        float(means[quantiles - 1] - means[0]) if 0 in means and quantiles - 1 in means else None
    )
    records: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "feature": feature,
                "quantile": int(cast(int, row["quantile"])),
                "observations": int(cast(int, row["size"])),
                "mean_forward_excess_return": float(cast(float, row["mean"])),
                "top_minus_bottom": spread,
            }
        )
    return records


def _numeric_matrix(frame: DataFrame, feature_names: tuple[str, ...]) -> DataFrame:
    matrix = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    return cast(DataFrame, matrix.replace([np.inf, -np.inf], np.nan).astype("float32"))
