"""Read-only production scoring from a registered champion Ranker."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.utils.manifest import (
    atomic_write_json,
    config_hash,
    current_git_info,
    read_manifest,
)

type DataFrame = pd.DataFrame

_SAFE_FEATURE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PredictionModel(Protocol):
    """Minimal prediction interface implemented by a LightGBM Booster."""

    def predict(self, data: DataFrame) -> NDArray[np.float64]:
        """Return one score per input row."""


class ReadinessProvider(Protocol):
    """Read-only readiness contract used before production scoring."""

    def check_all(self, as_of: str) -> tuple[GateResult, ...]:
        """Return production gate results for one session."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Published production prediction result."""

    as_of: str
    model_id: str
    feature_count: int
    universe_size: int
    prediction_count: int
    output_dir: Path
    predictions: DataFrame


@dataclass(frozen=True, slots=True)
class RegisteredPredictionBatch:
    """Scores and validated input identity for one registered model and date range."""

    model_id: str
    feature_names: tuple[str, ...]
    feature_hash: str
    universe_rows: int
    predictions: DataFrame


class ProductionInferenceEngine:
    """Validate, score, and publish one completed A-share session."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        config_path: Path,
        freshness: ReadinessProvider,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.config_path = config_path
        self.freshness = freshness
        self._model_loader = model_loader or _load_lightgbm_model

    def predict(self, as_of: str) -> InferenceResult:
        """Score only model-universe rows from one explicitly requested session."""

        started = monotonic()
        readiness = self.freshness.check_all(as_of)
        _require_readiness(readiness)
        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no champion is registered for model_type=lightgbm_ranker")
        batch = score_registered_model_range(
            champion,
            processed_root=self.processed_root,
            start_date=as_of,
            end_date=as_of,
            model_loader=self._model_loader,
        )
        predictions = batch.predictions.drop(columns="rank")
        output_dir = self._publish(
            as_of,
            champion,
            batch.feature_names,
            batch.feature_hash,
            predictions,
            universe_size=batch.universe_rows,
            readiness=readiness,
            elapsed_seconds=monotonic() - started,
        )
        return InferenceResult(
            as_of=as_of,
            model_id=champion.model_id,
            feature_count=len(batch.feature_names),
            universe_size=batch.universe_rows,
            prediction_count=len(predictions),
            output_dir=output_dir,
            predictions=predictions,
        )

    def _publish(
        self,
        as_of: str,
        champion: RegisteredModel,
        feature_names: tuple[str, ...],
        digest: str,
        predictions: DataFrame,
        *,
        universe_size: int,
        readiness: tuple[GateResult, ...],
        elapsed_seconds: float,
    ) -> Path:
        completed_at = datetime.now(UTC).isoformat()
        git_info = current_git_info()
        current_config_hash = config_hash(self.config_path)
        output_dir = self.reports_root / as_of
        ranking = predictions.loc[:, ["ts_code", "prediction_score"]].copy()
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
        summary = {
            "as_of": as_of,
            "model_id": champion.model_id,
            "feature_count": len(feature_names),
            "universe_size": universe_size,
            "prediction_count": len(predictions),
            "generation_time": completed_at,
            "elapsed_seconds": elapsed_seconds,
            "git_commit": git_info["commit"],
            "config_hash": current_config_hash,
        }
        model_manifest = _load_json(Path(champion.artifact_path) / "manifest.json", "model")
        manifest = {
            "schema_version": 1,
            "artifact_name": "production_predictions",
            "as_of": as_of,
            "model_id": champion.model_id,
            "model_type": champion.model_type,
            "feature_hash": digest,
            "feature_count": len(feature_names),
            "prediction_count": len(predictions),
            "generation_time": completed_at,
            "elapsed_seconds": elapsed_seconds,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "config_path": str(self.config_path),
            "config_hash": current_config_hash,
            "input_artifact_manifests": {
                "model": model_manifest,
                "features_daily": read_manifest(self.processed_root / "features_daily"),
                "universe_daily": read_manifest(self.processed_root / "universe_daily"),
            },
            "readiness": [result.to_dict() for result in readiness],
        }

        self.reports_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.reports_root) as temporary:
            staging = Path(temporary)
            predictions.to_parquet(staging / "predictions.parquet", index=False)
            ranking.to_csv(staging / "ranking.csv", index=False)
            atomic_write_json(staging / "summary.json", summary)
            atomic_write_json(staging / "manifest.json", manifest)
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("predictions.parquet", "ranking.csv", "summary.json"):
                os.replace(staging / filename, output_dir / filename)
            # The manifest is the completion marker and is always published last.
            os.replace(staging / "manifest.json", output_dir / "manifest.json")
        return output_dir


def _load_lightgbm_model(path: Path) -> PredictionModel:
    return cast(PredictionModel, lgb.Booster(model_file=str(path)))


def score_registered_model_range(
    model_record: RegisteredModel,
    *,
    processed_root: Path,
    start_date: str,
    end_date: str,
    allowed_ranges: Sequence[tuple[str, str]] | None = None,
    model_loader: Callable[[Path], PredictionModel] | None = None,
) -> RegisteredPredictionBatch:
    """Score validated model-universe rows without reading labels or future prices."""

    if start_date > end_date:
        raise DataValidationError("inference date range is reversed")
    artifact = Path(model_record.artifact_path)
    feature_names, digest = load_registered_feature_list(artifact, model_record)
    model_path = artifact / "model.txt"
    if not model_path.is_file():
        raise DataValidationError(f"registered model.txt does not exist: {model_path}")
    features, universe = _load_range_frames(
        processed_root,
        start_date,
        end_date,
        feature_names,
    )
    if allowed_ranges is not None:
        _validate_allowed_ranges(allowed_ranges, start_date, end_date)
        feature_mask = _date_range_mask(features["trade_date"], allowed_ranges)
        universe_mask = _date_range_mask(universe["trade_date"], allowed_ranges)
        features = features.loc[feature_mask].reset_index(drop=True)
        universe = universe.loc[universe_mask].reset_index(drop=True)
    feature_dates = set(features["trade_date"].astype(str))
    universe_dates = set(universe["trade_date"].astype(str))
    if feature_dates != universe_dates:
        raise DataValidationError(
            "feature and universe artifacts contain different inference dates: "
            f"features_only={sorted(feature_dates - universe_dates)} "
            f"universe_only={sorted(universe_dates - feature_dates)}"
        )
    if not feature_dates:
        raise DataValidationError(f"inference inputs are empty for {start_date}..{end_date}")
    eligible_parts = [
        _validate_and_filter_inputs(
            features.loc[features["trade_date"].astype(str) == trade_date].copy(),
            universe.loc[universe["trade_date"].astype(str) == trade_date].copy(),
            trade_date,
        )
        for trade_date in sorted(feature_dates)
    ]
    eligible = pd.concat(eligible_parts, ignore_index=True)
    matrix = eligible.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).astype("float32")
    loader = model_loader or _load_lightgbm_model
    model = loader(model_path)
    scores = np.asarray(model.predict(matrix), dtype=float)
    if scores.ndim != 1 or len(scores) != len(eligible):
        raise DataValidationError(
            "model returned an invalid prediction shape: "
            f"expected={len(eligible)} actual={scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise DataValidationError("model returned non-finite prediction scores")
    predictions = eligible.loc[:, ["trade_date", "ts_code"]].copy()
    predictions["prediction_score"] = scores
    predictions["model_id"] = model_record.model_id
    predictions = predictions.sort_values(
        ["trade_date", "prediction_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    predictions["rank"] = (predictions.groupby("trade_date", sort=False).cumcount() + 1).astype(int)
    return RegisteredPredictionBatch(
        model_id=model_record.model_id,
        feature_names=feature_names,
        feature_hash=digest,
        universe_rows=len(universe),
        predictions=predictions,
    )


def _require_readiness(results: tuple[GateResult, ...]) -> None:
    failures = [
        f"{result.gate}: {failure}" for result in results for failure in result.hard_failures
    ]
    if failures:
        raise DataValidationError("production readiness failed: " + "; ".join(failures))


def load_registered_feature_list(
    artifact: Path, model_record: RegisteredModel
) -> tuple[tuple[str, ...], str]:
    """Validate one registered model's ordered feature identity."""

    payload = _load_json(artifact / "feature_list.json", "feature list")
    raw_features = payload.get("features")
    if (
        not isinstance(raw_features, list)
        or not raw_features
        or not all(isinstance(item, str) for item in raw_features)
    ):
        raise DataValidationError("registered feature_list.json lacks a non-empty `features` array")
    features = tuple(str(item) for item in raw_features)
    if len(features) != len(set(features)):
        raise DataValidationError("registered feature_list.json contains duplicate feature names")
    unsafe = [name for name in features if _SAFE_FEATURE_NAME.fullmatch(name) is None]
    if unsafe:
        raise DataValidationError(
            f"registered model contains invalid feature identifiers: {unsafe}"
        )
    digest = feature_list_hash(features)
    declared = payload.get("feature_hash")
    model_manifest = _load_json(artifact / "manifest.json", "model manifest")
    manifest_hash = model_manifest.get("feature_list_hash")
    mismatches: list[str] = []
    if digest != model_record.feature_hash:
        mismatches.append("registered feature hash")
    if declared != digest:
        mismatches.append("feature_list.json feature_hash")
    if manifest_hash != digest:
        mismatches.append("model manifest feature_list_hash")
    if model_record.feature_count != len(features):
        mismatches.append("registered feature_count")
    if mismatches:
        raise DataValidationError(
            f"registered model feature identity mismatch ({', '.join(mismatches)}); "
            f"computed={digest}"
        )
    return features, digest


def _load_range_frames(
    processed_root: Path,
    start_date: str,
    end_date: str,
    feature_names: tuple[str, ...],
) -> tuple[DataFrame, DataFrame]:
    feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
    universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
    if not list((processed_root / "features_daily").glob("**/*.parquet")):
        raise DataValidationError("features_daily artifact does not exist")
    if not list((processed_root / "universe_daily").glob("**/*.parquet")):
        raise DataValidationError("universe_daily artifact does not exist")
    selected = ", ".join(f'"{name}"' for name in feature_names)
    feature_query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               {selected}
        FROM read_parquet('{feature_glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
        ORDER BY trade_date, ts_code
    """  # noqa: S608 -- validated identifiers and configured local Parquet path
    universe_query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(in_model_universe AS BOOLEAN) AS in_model_universe
        FROM read_parquet('{universe_glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
        ORDER BY trade_date, ts_code
    """  # noqa: S608 -- configured local Parquet path
    try:
        with duckdb.connect() as connection:
            available = set(
                connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{feature_glob.as_posix()}')"  # noqa: S608
                )
                .fetch_df()["column_name"]
                .astype(str)
            )
            missing = sorted(set(feature_names) - available)
            if missing:
                raise DataValidationError(f"features_daily lacks model features: {missing}")
            features = connection.execute(feature_query, [start_date, end_date]).fetch_df()
            universe = connection.execute(universe_query, [start_date, end_date]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(
            f"cannot load inference inputs for {start_date}..{end_date}: {error}"
        ) from error
    return features, universe


def _validate_and_filter_inputs(features: DataFrame, universe: DataFrame, as_of: str) -> DataFrame:
    if features.empty:
        raise DataValidationError(f"features_daily has no rows for as-of date {as_of}")
    if universe.empty:
        raise DataValidationError(f"universe_daily has no rows for as-of date {as_of}")
    keys = ["trade_date", "ts_code"]
    if features.duplicated(keys).any():
        raise DataValidationError(f"features_daily contains duplicate keys for {as_of}")
    if universe.duplicated(keys).any():
        raise DataValidationError(f"universe_daily contains duplicate keys for {as_of}")
    feature_keys = pd.MultiIndex.from_frame(features[keys])
    universe_keys = pd.MultiIndex.from_frame(universe[keys])
    if len(features) != len(universe) or set(feature_keys) != set(universe_keys):
        raise DataValidationError(
            f"feature rows do not match universe rows for {as_of}: "
            f"features={len(features)} universe={len(universe)}"
        )
    joined = features.merge(universe, on=keys, how="inner", validate="one_to_one")
    eligible = joined.loc[joined["in_model_universe"].fillna(False).astype(bool)].copy()
    if eligible.empty:
        raise DataValidationError(f"in_model_universe has no prediction rows for {as_of}")
    return eligible.sort_values("ts_code", kind="mergesort").reset_index(drop=True)


def _validate_allowed_ranges(
    ranges: Sequence[tuple[str, str]], start_date: str, end_date: str
) -> None:
    if not ranges:
        raise DataValidationError("inference allowed date ranges are empty")
    for range_start, range_end in ranges:
        if range_start > range_end:
            raise DataValidationError("inference allowed date range is reversed")
        if range_start < start_date or range_end > end_date:
            raise DataValidationError("inference allowed date range exceeds query boundaries")


def _date_range_mask(dates: pd.Series[Any], ranges: Sequence[tuple[str, str]]) -> pd.Series[Any]:
    values = dates.astype(str)
    mask = pd.Series(False, index=dates.index)
    for range_start, range_end in ranges:
        mask |= values.between(range_start, range_end)
    return mask


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} JSON must contain an object: {path}")
    return payload
