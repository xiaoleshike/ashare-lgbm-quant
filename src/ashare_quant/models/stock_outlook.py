"""Single-stock relative outlook from a registered horizon-matched Ranker."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry, RegisteredModel

type DataFrame = pd.DataFrame

_SAFE_FEATURE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OutlookModel(Protocol):
    """Prediction interface implemented by a persisted LightGBM Booster."""

    def predict(self, data: DataFrame) -> NDArray[np.float64]:
        """Return one cross-sectional score per row."""


@dataclass(frozen=True, slots=True)
class StockOutlookResult:
    """One model-relative stock outlook without an absolute return forecast."""

    as_of: str
    ts_code: str
    model_id: str
    model_status: str
    target: str
    horizon_trading_days: int
    entry_date: str
    exit_date: str
    prediction_score: float
    rank: int
    universe_size: int
    score_percentile: float
    relative_outlook: str
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


class StockOutlookPredictor:
    """Score one stock within its complete same-date model universe."""

    def __init__(
        self,
        *,
        raw_root: Path,
        processed_root: Path,
        models_root: Path,
        model_loader: Callable[[Path], OutlookModel] | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.models_root = models_root
        self._model_loader = model_loader or _load_model

    def predict(
        self,
        *,
        model_id: str,
        ts_code: str,
        as_of: str,
        horizon: int = 10,
    ) -> StockOutlookResult:
        """Return a horizon-matched relative-strength outlook for one stock."""

        _validate_date(as_of)
        if horizon < 1:
            raise DataValidationError("horizon must be a positive number of trading days")
        model_record = _registered_model(self.models_root, model_id)
        artifact = Path(model_record.artifact_path)
        manifest = _load_json(artifact / "manifest.json", "model manifest")
        model_horizon = _required_horizon(manifest)
        if model_horizon != horizon:
            raise DataValidationError(
                "model horizon does not match requested outlook: "
                f"model_id={model_id} model_horizon={model_horizon} requested_horizon={horizon}; "
                f"train or register a horizon={horizon} Ranker"
            )
        target = str(manifest.get("target", f"future_excess_ret_{model_horizon}d"))
        expected_target = f"future_excess_ret_{model_horizon}d"
        if target != expected_target:
            raise DataValidationError(
                f"model target is not the required executable excess-return target: {target}"
            )
        feature_names = _load_features(artifact, model_record)
        eligible = _load_eligible_features(
            self.processed_root,
            as_of,
            feature_names,
        )
        normalized_code = ts_code.strip().upper()
        if normalized_code not in set(eligible["ts_code"].astype(str)):
            raise DataValidationError(
                f"stock is not in_model_universe on {as_of}: {normalized_code}"
            )
        matrix = eligible.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan).astype("float32")
        model_path = artifact / "model.txt"
        if not model_path.is_file():
            raise DataValidationError(f"registered model.txt does not exist: {model_path}")
        scores = np.asarray(self._model_loader(model_path).predict(matrix), dtype=float)
        if scores.ndim != 1 or len(scores) != len(eligible) or not np.isfinite(scores).all():
            raise DataValidationError(
                f"model returned invalid scores: expected={len(eligible)} actual={scores.shape}"
            )
        ranked = eligible.loc[:, ["trade_date", "ts_code"]].copy()
        ranked["prediction_score"] = scores
        ranked = ranked.sort_values(
            ["prediction_score", "ts_code"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        selected = ranked.loc[ranked["ts_code"] == normalized_code].iloc[0]
        rank = int(selected["rank"])
        percentile = 1.0 - (rank - 1) / len(ranked)
        entry_date, exit_date = _execution_dates(self.raw_root, as_of, horizon)
        return StockOutlookResult(
            as_of=as_of,
            ts_code=normalized_code,
            model_id=model_id,
            model_status=model_record.status,
            target=target,
            horizon_trading_days=horizon,
            entry_date=entry_date,
            exit_date=exit_date,
            prediction_score=float(selected["prediction_score"]),
            rank=rank,
            universe_size=len(ranked),
            score_percentile=percentile,
            relative_outlook=_relative_outlook(percentile),
            interpretation=(
                "Cross-sectional relative-strength ranking versus the eligible A-share universe; "
                "not an absolute price path, target price, or guaranteed return."
            ),
        )


def _load_model(path: Path) -> OutlookModel:
    return cast(OutlookModel, lgb.Booster(model_file=str(path)))


def _registered_model(models_root: Path, model_id: str) -> RegisteredModel:
    matches = [
        record for record in ModelRegistry(models_root).list_models() if record.model_id == model_id
    ]
    if not matches:
        raise DataValidationError(f"model_id is not registered: {model_id}")
    return matches[0]


def _required_horizon(manifest: dict[str, object]) -> int:
    value = manifest.get("label_horizon")
    if not isinstance(value, int) or value < 1:
        raise DataValidationError("model manifest lacks a positive integer label_horizon")
    return value


def _load_features(artifact: Path, model: RegisteredModel) -> tuple[str, ...]:
    payload = _load_json(artifact / "feature_list.json", "feature list")
    raw = payload.get("features")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError("feature_list.json lacks a non-empty features array")
    features = tuple(str(item) for item in raw)
    if len(features) != len(set(features)):
        raise DataValidationError("feature_list.json contains duplicate feature names")
    invalid = [name for name in features if _SAFE_FEATURE_NAME.fullmatch(name) is None]
    if invalid:
        raise DataValidationError(f"model contains invalid feature identifiers: {invalid}")
    digest = feature_list_hash(features)
    if digest != model.feature_hash or payload.get("feature_hash") != digest:
        raise DataValidationError("registered model and feature_list.json feature hashes differ")
    return features


def _load_eligible_features(
    processed_root: Path,
    as_of: str,
    feature_names: tuple[str, ...],
) -> DataFrame:
    feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
    universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
    if not list((processed_root / "features_daily").glob("**/*.parquet")):
        raise DataValidationError("features_daily artifact does not exist")
    if not list((processed_root / "universe_daily").glob("**/*.parquet")):
        raise DataValidationError("universe_daily artifact does not exist")
    selected = ", ".join(f'f."{name}"' for name in feature_names)
    query = f"""
        SELECT CAST(f.trade_date AS VARCHAR) AS trade_date,
               CAST(f.ts_code AS VARCHAR) AS ts_code,
               {selected}
        FROM read_parquet('{feature_glob.as_posix()}', hive_partitioning=false) AS f
        INNER JOIN read_parquet('{universe_glob.as_posix()}', hive_partitioning=false) AS u
          ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR)
         AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)
        WHERE CAST(f.trade_date AS VARCHAR) = ?
          AND CAST(u.in_model_universe AS BOOLEAN)
        ORDER BY f.ts_code
    """  # noqa: S608 -- validated feature names and configured local Parquet artifacts
    try:
        with duckdb.connect() as connection:
            frame = connection.execute(query, [as_of]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot load stock outlook features: {error}") from error
    if frame.empty:
        raise DataValidationError(f"no eligible model features exist for {as_of}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("stock outlook inputs contain duplicate keys")
    return frame


def _execution_dates(raw_root: Path, as_of: str, horizon: int) -> tuple[str, str]:
    calendar_glob = raw_root / "trade_cal" / "**" / "*.parquet"
    if not list((raw_root / "trade_cal").glob("**/*.parquet")):
        raise DataValidationError("trade_cal is required to resolve the 10-day outlook window")
    query = f"""
        SELECT CAST(cal_date AS VARCHAR) AS trade_date
        FROM read_parquet('{calendar_glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(is_open AS INTEGER) = 1
          AND CAST(cal_date AS VARCHAR) > ?
        ORDER BY cal_date
        LIMIT ?
    """  # noqa: S608 -- configured local Parquet artifact and parameterized values
    try:
        with duckdb.connect() as connection:
            dates = connection.execute(query, [as_of, horizon + 1]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot resolve future trading dates: {error}") from error
    if len(dates) < horizon + 1:
        raise DataValidationError(
            f"trade_cal lacks {horizon + 1} future open sessions after {as_of}"
        )
    values = dates["trade_date"].astype(str).tolist()
    return values[0], values[horizon]


def _relative_outlook(percentile: float) -> str:
    if percentile >= 0.90:
        return "very_strong_relative"
    if percentile >= 0.70:
        return "strong_relative"
    if percentile > 0.30:
        return "neutral_relative"
    if percentile > 0.10:
        return "weak_relative"
    return "very_weak_relative"


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"as_of must use YYYYMMDD: {value}")


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain a JSON object: {path}")
    return payload
