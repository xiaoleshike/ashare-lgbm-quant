"""Read-only orchestration for daily LightGBM candidate explanations."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import ExplainabilitySettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.research.explainability.contributions import (
    ContributionMatrix,
    ExplainableModel,
    compute_tree_contributions,
)
from ashare_quant.research.explainability.descriptions import describe_feature
from ashare_quant.research.explainability.history import (
    current_score_percentiles,
    historical_percentile,
    history_assessment,
    load_same_model_history,
)
from ashare_quant.research.explainability.rendering import build_payload, render_markdown
from ashare_quant.research.explainability.schemas import (
    ExplainabilityResult,
    ExplanationConfidence,
    FeatureContribution,
    SignalStrength,
    StockExplanation,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type ContributionProvider = Callable[[ExplainableModel, DataFrame], ContributionMatrix]

_SAFE_FEATURE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExplainabilityEngine:
    """Validate and explain unchanged candidate scores for one session."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        settings: ExplainabilitySettings,
        model_loader: Callable[[Path], ExplainableModel] | None = None,
        contribution_provider: ContributionProvider | None = None,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self._model_loader = model_loader or _load_model
        self._contribution_provider = contribution_provider or compute_tree_contributions

    def explain(self, as_of: str) -> ExplainabilityResult:
        """Generate JSON and Markdown without modifying scores or candidate ranks."""

        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no champion is registered for model_type=lightgbm_ranker")
        features, digest = _load_feature_identity(champion)
        report_dir = self.reports_root / as_of
        predictions = _load_predictions(report_dir / "predictions.parquet", as_of, champion)
        candidates = _load_candidates(report_dir / "candidates.csv", as_of, champion)
        _validate_prediction_manifest(report_dir / "manifest.json", champion, digest)
        model_ranked = _rank_predictions(predictions)
        selected = _join_candidates(
            candidates, model_ranked, tolerance=self.settings.score_tolerance
        )
        feature_frame = _load_candidate_features(
            self.processed_root, as_of, features, tuple(selected["ts_code"].astype(str))
        )
        selected = selected.merge(
            feature_frame,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        if selected.loc[:, list(features)].isna().all(axis=1).any():
            missing = selected.loc[
                selected.loc[:, list(features)].isna().all(axis=1), "ts_code"
            ].tolist()
            raise DataValidationError(f"candidate feature rows are missing: {missing}")

        matrix = (
            selected.loc[:, list(features)]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
        model = self._model_loader(Path(champion.artifact_path) / "model.txt")
        rescored = np.asarray(model.predict(matrix), dtype=float)
        expected_scores = selected["prediction_score"].to_numpy(dtype=float)
        _validate_scores(rescored, expected_scores, self.settings.score_tolerance)
        contribution_matrix = self._contribution_provider(model, matrix)
        _validate_additivity(
            contribution_matrix,
            expected_scores,
            self.settings.contribution_tolerance,
        )

        percentiles = current_score_percentiles(predictions)
        history, history_sessions = load_same_model_history(
            self.reports_root,
            as_of=as_of,
            model_id=champion.model_id,
            maximum_sessions=self.settings.maximum_history_sessions,
        )
        history_status, confidence = history_assessment(history_sessions, self.settings)
        explanations = tuple(
            self._stock_explanation(
                selected.iloc[index],
                matrix.iloc[index],
                features,
                contribution_matrix,
                index,
                percentiles,
                history,
                history_status,
                confidence,
            )
            for index in range(len(selected))
        )
        payload = build_payload(
            as_of=as_of,
            model_id=champion.model_id,
            feature_hash=digest,
            feature_count=len(features),
            method=contribution_matrix.method,
            history_sessions=history_sessions,
            explanations=explanations,
        )
        json_path = report_dir / "explanations.json"
        markdown_path = report_dir / "explanations.md"
        _publish(json_path, markdown_path, payload, render_markdown(payload))
        return ExplainabilityResult(
            as_of=as_of,
            model_id=champion.model_id,
            candidate_count=len(explanations),
            method=contribution_matrix.method,
            json_path=str(json_path),
            markdown_path=str(markdown_path),
        )

    def _stock_explanation(
        self,
        row: pd.Series[Any],
        feature_values: pd.Series[Any],
        features: tuple[str, ...],
        matrix: ContributionMatrix,
        index: int,
        percentiles: dict[str, float],
        history: np.ndarray,
        history_status: str,
        confidence: ExplanationConfidence,
    ) -> StockExplanation:
        contributions = matrix.values[index]
        positive_indices = sorted(
            (item for item in range(len(features)) if contributions[item] > 0),
            key=lambda item: (-contributions[item], features[item]),
        )[: self.settings.top_positive_features]
        negative_indices = sorted(
            (item for item in range(len(features)) if contributions[item] < 0),
            key=lambda item: (contributions[item], features[item]),
        )[: self.settings.top_negative_features]
        ts_code = str(row["ts_code"])
        score = float(row["prediction_score"])
        score_percentile = percentiles[ts_code]
        return StockExplanation(
            ts_code=ts_code,
            model_rank=int(row["model_rank"]),
            candidate_rank=int(row["candidate_rank"]),
            prediction_score=score,
            score_percentile=score_percentile,
            historical_score_percentile=historical_percentile(score, history),
            history_status=history_status,
            signal_strength=_signal_strength(score_percentile, self.settings),
            confidence=confidence,
            base_value=float(matrix.base_values[index]),
            positive_contributions=tuple(
                _feature_contribution(
                    features[item], float(feature_values.iloc[item]), contributions[item]
                )
                for item in positive_indices
            ),
            negative_contributions=tuple(
                _feature_contribution(
                    features[item], float(feature_values.iloc[item]), contributions[item]
                )
                for item in negative_indices
            ),
        )


def _load_model(path: Path) -> ExplainableModel:
    if not path.is_file():
        raise DataValidationError(f"champion model.txt does not exist: {path}")
    return cast(ExplainableModel, lgb.Booster(model_file=str(path)))


def _load_feature_identity(champion: RegisteredModel) -> tuple[tuple[str, ...], str]:
    artifact = Path(champion.artifact_path)
    payload = _load_json(artifact / "feature_list.json", "feature list")
    raw = payload.get("features")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError("champion feature_list.json lacks a non-empty `features` array")
    features = tuple(str(item) for item in raw)
    if len(features) != len(set(features)):
        raise DataValidationError("champion feature_list.json contains duplicate features")
    unsafe = [feature for feature in features if _SAFE_FEATURE_NAME.fullmatch(feature) is None]
    if unsafe:
        raise DataValidationError(f"champion contains invalid feature identifiers: {unsafe}")
    digest = feature_list_hash(features)
    manifest = _load_json(artifact / "manifest.json", "model manifest")
    mismatches = []
    if payload.get("feature_hash") != digest:
        mismatches.append("feature_list.json")
    if manifest.get("feature_list_hash") != digest:
        mismatches.append("model manifest")
    if champion.feature_hash != digest or champion.feature_count != len(features):
        mismatches.append("model registry")
    if mismatches:
        raise DataValidationError(f"champion feature identity mismatch: {', '.join(mismatches)}")
    return features, digest


def _load_predictions(path: Path, as_of: str, champion: RegisteredModel) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"predictions are missing: {path}")
    frame = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    _require_columns(frame, required, "predictions")
    if frame.empty:
        raise DataValidationError("predictions are empty")
    if set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("prediction dates do not match --as-of")
    if set(frame["model_id"].astype(str)) != {champion.model_id}:
        raise DataValidationError("prediction model_id does not match champion")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("predictions contain duplicate keys")
    scores = pd.to_numeric(frame["prediction_score"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise DataValidationError("predictions contain non-finite scores")
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["prediction_score"] = scores.astype(float)
    return frame


def _load_candidates(path: Path, as_of: str, champion: RegisteredModel) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"research candidates are missing: {path}")
    frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str, "model_id": str})
    required = {"rank", "trade_date", "ts_code", "prediction_score", "model_id"}
    _require_columns(frame, required, "candidates")
    if frame.empty:
        raise DataValidationError("research candidates are empty")
    if set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("candidate dates do not match --as-of")
    if set(frame["model_id"].astype(str)) != {champion.model_id}:
        raise DataValidationError("candidate model_id does not match champion")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("candidates contain duplicate keys")
    return frame.rename(columns={"rank": "candidate_rank"})


def _validate_prediction_manifest(
    path: Path,
    champion: RegisteredModel,
    feature_hash: str,
) -> None:
    manifest = _load_json(path, "prediction manifest")
    if manifest.get("model_id") != champion.model_id:
        raise DataValidationError("prediction manifest model_id does not match champion")
    if manifest.get("feature_hash") != feature_hash:
        raise DataValidationError("prediction manifest feature_hash does not match champion")


def _rank_predictions(predictions: DataFrame) -> DataFrame:
    ranked = predictions.sort_values(
        ["prediction_score", "ts_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["model_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _join_candidates(
    candidates: DataFrame,
    predictions: DataFrame,
    *,
    tolerance: float,
) -> DataFrame:
    candidate_columns = ["trade_date", "ts_code", "candidate_rank", "prediction_score"]
    joined = candidates.loc[:, candidate_columns].merge(
        predictions.loc[:, ["trade_date", "ts_code", "prediction_score", "model_rank"]],
        on=["trade_date", "ts_code"],
        how="left",
        suffixes=("_candidate", "_prediction"),
        validate="one_to_one",
    )
    if joined["model_rank"].isna().any():
        missing = joined.loc[joined["model_rank"].isna(), "ts_code"].tolist()
        raise DataValidationError(f"candidates are absent from predictions: {missing}")
    candidate_scores = joined["prediction_score_candidate"].to_numpy(dtype=float)
    prediction_scores = joined["prediction_score_prediction"].to_numpy(dtype=float)
    maximum_error = float(np.max(np.abs(candidate_scores - prediction_scores), initial=0.0))
    if maximum_error >= tolerance:
        raise DataValidationError(
            "candidate scores differ from immutable predictions: "
            f"max_abs_error={maximum_error:.12g} tolerance={tolerance:.12g}"
        )
    return (
        joined.rename(columns={"prediction_score_prediction": "prediction_score"})
        .drop(columns=["prediction_score_candidate"])
        .sort_values(["candidate_rank", "ts_code"], kind="mergesort")
        .reset_index(drop=True)
    )


def _load_candidate_features(
    processed_root: Path,
    as_of: str,
    features: tuple[str, ...],
    codes: tuple[str, ...],
) -> DataFrame:
    root = processed_root / "features_daily"
    files = list(root.glob("**/*.parquet"))
    if not files:
        raise DataValidationError("features_daily artifact does not exist")
    glob = root / "**" / "*.parquet"
    selected = ", ".join(f'CAST("{feature}" AS DOUBLE) AS "{feature}"' for feature in features)
    placeholders = ", ".join("?" for _ in codes)
    query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               {selected}
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(trade_date AS VARCHAR) = ?
          AND CAST(ts_code AS VARCHAR) IN ({placeholders})
        ORDER BY ts_code
    """  # noqa: S608 -- validated feature identifiers and parameterized values
    try:
        with duckdb.connect() as connection:
            available = set(
                connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{glob.as_posix()}')"  # noqa: S608
                )
                .fetch_df()["column_name"]
                .astype(str)
            )
            missing_columns = sorted(set(features) - available)
            if missing_columns:
                raise DataValidationError(f"features_daily lacks model features: {missing_columns}")
            frame = connection.execute(query, [as_of, *codes]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(
            f"cannot load explanation features for {as_of}: {error}"
        ) from error
    if frame.empty:
        raise DataValidationError(f"features_daily has no candidate rows for {as_of}")
    if set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("feature dates do not match --as-of")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("candidate feature rows contain duplicate keys")
    if set(frame["ts_code"].astype(str)) != set(codes):
        missing_codes = sorted(set(codes) - set(frame["ts_code"].astype(str)))
        raise DataValidationError(f"candidate feature rows are missing: {missing_codes}")
    return frame


def _validate_scores(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> None:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise DataValidationError(
            f"recomputed model scores have invalid shape or values: {actual.shape}"
        )
    maximum_error = float(np.max(np.abs(actual - expected), initial=0.0))
    if maximum_error >= tolerance:
        raise DataValidationError(
            "recomputed model scores differ from immutable predictions: "
            f"max_abs_error={maximum_error:.12g} tolerance={tolerance:.12g}"
        )


def _validate_additivity(
    matrix: ContributionMatrix,
    expected: np.ndarray,
    tolerance: float,
) -> None:
    reconstructed = matrix.base_values + matrix.values.sum(axis=1)
    maximum_error = float(np.max(np.abs(reconstructed - expected), initial=0.0))
    if maximum_error >= tolerance:
        raise DataValidationError(
            "feature contributions do not reconstruct immutable prediction scores: "
            f"max_abs_error={maximum_error:.12g} tolerance={tolerance:.12g}"
        )


def _signal_strength(percentile: float, settings: ExplainabilitySettings) -> SignalStrength:
    if percentile >= settings.strong_percentile:
        return "strong"
    if percentile >= settings.moderate_percentile:
        return "moderate"
    return "weak"


def _feature_contribution(feature: str, value: float, shap_value: float) -> FeatureContribution:
    numeric = None if np.isnan(value) else value
    return FeatureContribution(
        feature=feature,
        value=numeric,
        shap=float(shap_value),
        description=describe_feature(feature),
    )


def _require_columns(frame: DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{name} lack required columns: {missing}")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{name} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {name}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{name} must contain a JSON object")
    return payload


def _publish(
    json_path: Path,
    markdown_path: Path,
    payload: dict[str, Any],
    markdown: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=json_path.parent) as temporary:
        staging = Path(temporary)
        staged_markdown = staging / markdown_path.name
        staged_markdown.write_text(markdown, encoding="utf-8")
        atomic_write_json(staging / json_path.name, payload)
        os.replace(staged_markdown, markdown_path)
        os.replace(staging / json_path.name, json_path)
