"""Prospective shadow scoring orchestration without pipeline integration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import PredictionModel
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.ensemble import (
    build_percentile_ensemble,
    ensemble_model_id,
)
from ashare_quant.models.shadow.schemas import (
    MODEL_ROLES,
    ShadowContext,
    ShadowPredictionResult,
)
from ashare_quant.models.shadow.scoring import add_score_percentile, score_challenger
from ashare_quant.models.shadow.storage import (
    canonical_payload_hash,
    logical_prediction_hash,
    publish_shadow_bundle,
    read_complete_manifest,
)
from ashare_quant.models.shadow.validation import ShadowReadinessValidator
from ashare_quant.utils.manifest import config_hash, current_git_info

type DataFrame = pd.DataFrame

PREDICTION_COLUMNS = (
    "trade_date",
    "ts_code",
    "model_id",
    "model_role",
    "native_horizon",
    "prediction_score",
    "rank",
    "score_percentile",
    "production_run_id",
    "shadow_run_id",
    "prediction_hash",
    "feature_hash",
    "universe_hash",
    "access_policy",
    "generated_at",
)


class ShadowPredictionService:
    """Create an all-or-nothing prospective Champion/challenger/ensemble bundle."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        runs_root: Path = Path("runs"),
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.model_loader = model_loader
        self.readiness = ShadowReadinessValidator(
            settings=settings,
            config_path=config_path,
            registry=registry,
            processed_root=processed_root,
            reports_root=reports_root,
            runs_root=runs_root,
        )

    def predict(self, as_of: str) -> ShadowPredictionResult:
        """Score four candidates and publish one immutable six-model bundle."""

        context = self.readiness.require_ready(as_of)
        shadow_run_id = self._shadow_run_id(context)
        output_dir = self.reports_root / "shadow_predictions" / as_of
        existing = read_complete_manifest(output_dir)
        if existing is not None:
            if existing.get("shadow_run_id") != shadow_run_id:
                raise DataValidationError(
                    "shadow output exists with a different logical input identity"
                )
            return ShadowPredictionResult(
                as_of=as_of,
                production_run_id=context.production_run_id,
                shadow_run_id=shadow_run_id,
                prediction_rows=int(existing.get("prediction_rows", 0)),
                model_count=len(existing.get("models", [])),
                output_dir=output_dir,
                idempotent=True,
            )
        if output_dir.exists():
            raise DataValidationError(
                f"incomplete shadow output directory exists and is not a valid artifact: "
                f"{output_dir}"
            )

        frames: dict[str, DataFrame] = {}
        champion = add_score_percentile(context.champion_predictions)
        frames["champion"] = self._decorate(
            champion,
            context=context,
            shadow_run_id=shadow_run_id,
            model_id=context.champion_model_id,
            model_role="champion",
            native_horizon=5,
        )
        challenger_frames: dict[int, DataFrame] = {}
        component_hash_by_horizon: dict[int, str] = {}
        component_hashes: list[str] = []
        component_ids: list[str] = []
        for horizon in sorted(context.challenger_models):
            model = context.challenger_models[horizon]
            scored = add_score_percentile(
                score_challenger(
                    model,
                    processed_root=self.processed_root,
                    as_of=as_of,
                    model_loader=self.model_loader,
                )
            )
            self._require_same_keys(context.champion_predictions, scored, f"h{horizon}")
            decorated = self._decorate(
                scored,
                context=context,
                shadow_run_id=shadow_run_id,
                model_id=model.model_id,
                model_role=f"challenger_h{horizon}",
                native_horizon=horizon,
            )
            component_hash = logical_prediction_hash(decorated)
            frames[f"h{horizon}"] = decorated
            challenger_frames[horizon] = decorated
            component_hash_by_horizon[horizon] = component_hash
            component_hashes.append(component_hash)
            component_ids.append(model.model_id)

        fusion_method = self.settings.models.shadow_predictions.ensemble.fusion_method
        ensemble_scores = build_percentile_ensemble(challenger_frames)
        ensemble_id = ensemble_model_id(component_ids, component_hashes, fusion_method)
        ensemble_scores = _rank(ensemble_scores)
        ensemble_scores = add_score_percentile(ensemble_scores)
        ensemble = self._decorate(
            ensemble_scores,
            context=context,
            shadow_run_id=shadow_run_id,
            model_id=ensemble_id,
            model_role="multi_horizon_ensemble",
            native_horizon=None,
        )
        ensemble_hash = logical_prediction_hash(ensemble)
        frames["ensemble"] = ensemble

        champion_hash = logical_prediction_hash(frames["champion"])
        predictions = pd.concat(
            [frames[name] for name in ("champion", "h5", "h10", "h20", "h60", "ensemble")],
            ignore_index=True,
        ).loc[:, list(PREDICTION_COLUMNS)]
        predictions["native_horizon"] = predictions["native_horizon"].astype("Int64")
        bundle_hash = logical_prediction_hash(predictions)
        predictions["prediction_hash"] = bundle_hash
        _validate_bundle(
            predictions,
            context,
            expected_prediction_hash=bundle_hash,
        )
        manifest = self._manifest(
            context=context,
            shadow_run_id=shadow_run_id,
            prediction_hash=bundle_hash,
            predictions=predictions,
            model_hashes={
                "champion": champion_hash,
                **{
                    f"h{horizon}": component_hash_by_horizon[horizon]
                    for horizon in sorted(challenger_frames)
                },
                "ensemble": ensemble_hash,
            },
            ensemble_id=ensemble_id,
            component_ids=component_ids,
            component_hashes=component_hashes,
            fusion_method=fusion_method,
        )
        published = publish_shadow_bundle(
            output_dir=output_dir,
            predictions=predictions,
            manifest_without_file_hash=manifest,
        )
        return ShadowPredictionResult(
            as_of=as_of,
            production_run_id=context.production_run_id,
            shadow_run_id=shadow_run_id,
            prediction_rows=len(predictions),
            model_count=len(published["models"]),
            output_dir=output_dir,
        )

    def validate(self, as_of: str) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
        """Expose read-only pre-scoring readiness for the CLI."""

        result, _ = self.readiness.validate(as_of)
        return result.ready, result.hard_failures, result.checks

    def status(self, as_of: str) -> dict[str, Any]:
        """Return publication status without scoring."""

        output_dir = self.reports_root / "shadow_predictions" / as_of
        manifest = read_complete_manifest(output_dir)
        if manifest is not None:
            return {
                "status": "complete",
                "as_of": as_of,
                "shadow_run_id": manifest.get("shadow_run_id"),
                "prediction_rows": manifest.get("prediction_rows"),
                "output": str(output_dir),
            }
        return {
            "status": "incomplete" if output_dir.exists() else "missing",
            "as_of": as_of,
            "shadow_run_id": None,
            "prediction_rows": 0,
            "output": str(output_dir),
        }

    def _shadow_run_id(self, context: ShadowContext) -> str:
        settings = self.settings.models.shadow_predictions
        identity_hash = canonical_payload_hash(
            {
                "production_run_id": context.production_run_id,
                "as_of": context.as_of,
                "champion_prediction_hash": context.champion_prediction_hash,
                "challenger_model_ids": sorted(
                    model.model_id for model in context.challenger_models.values()
                ),
                "component_model_ids": sorted(
                    model.model_id for model in context.challenger_models.values()
                ),
                "challenger_manifest_hashes": sorted(context.challenger_manifest_hashes.values()),
                "feature_hash": context.feature_hash,
                "universe_hash": context.universe_hash,
                "fusion_method": settings.ensemble.fusion_method,
                "config_hash": config_hash(self.config_path),
            }
        )
        return f"shadow_{context.as_of}_{identity_hash[:16]}"

    def _decorate(
        self,
        frame: DataFrame,
        *,
        context: ShadowContext,
        shadow_run_id: str,
        model_id: str,
        model_role: str,
        native_horizon: int | None,
    ) -> DataFrame:
        if model_role not in MODEL_ROLES:
            raise DataValidationError(f"invalid shadow model_role: {model_role}")
        result = frame.loc[
            :, ["trade_date", "ts_code", "prediction_score", "rank", "score_percentile"]
        ].copy()
        result["model_id"] = model_id
        result["model_role"] = model_role
        result["native_horizon"] = native_horizon
        result["production_run_id"] = context.production_run_id
        result["shadow_run_id"] = shadow_run_id
        result["prediction_hash"] = ""
        result["feature_hash"] = context.feature_hash
        result["universe_hash"] = context.universe_hash
        result["access_policy"] = "prospective_production"
        result["generated_at"] = context.generated_at
        return result

    @staticmethod
    def _require_same_keys(left: DataFrame, right: DataFrame, name: str) -> None:
        columns = ["trade_date", "ts_code"]
        left_keys = set(left[columns].astype(str).itertuples(index=False, name=None))
        right_keys = set(right[columns].astype(str).itertuples(index=False, name=None))
        if left_keys != right_keys:
            raise DataValidationError(f"shadow {name} keys differ from Champion")

    def _manifest(
        self,
        *,
        context: ShadowContext,
        shadow_run_id: str,
        prediction_hash: str,
        predictions: DataFrame,
        model_hashes: dict[str, str],
        ensemble_id: str,
        component_ids: list[str],
        component_hashes: list[str],
        fusion_method: str,
    ) -> dict[str, Any]:
        git = current_git_info()
        model_records: list[dict[str, Any]] = [
            {
                "model_id": context.champion_model_id,
                "model_role": "champion",
                "prediction_hash": model_hashes["champion"],
                "feature_hash": context.feature_hash,
                "universe_hash": context.universe_hash,
                "source_models": [],
                "fusion_method": None,
                "access_policy": "prospective_production",
            }
        ]
        for horizon in sorted(context.challenger_models):
            model_records.append(
                {
                    "model_id": context.challenger_models[horizon].model_id,
                    "model_role": f"challenger_h{horizon}",
                    "prediction_hash": model_hashes[f"h{horizon}"],
                    "feature_hash": context.feature_hash,
                    "universe_hash": context.universe_hash,
                    "source_models": [],
                    "fusion_method": None,
                    "access_policy": "prospective_production",
                }
            )
        model_records.append(
            {
                "model_id": ensemble_id,
                "model_role": "multi_horizon_ensemble",
                "prediction_hash": model_hashes["ensemble"],
                "feature_hash": context.feature_hash,
                "universe_hash": context.universe_hash,
                "source_models": sorted(component_ids),
                "source_prediction_hashes": sorted(component_hashes),
                "fusion_method": fusion_method,
                "access_policy": "prospective_production",
            }
        )
        return {
            "schema_version": 1,
            "artifact_name": "shadow_prediction_bundle",
            "production_run_id": context.production_run_id,
            "shadow_run_id": shadow_run_id,
            "prediction_hash": prediction_hash,
            "feature_hash": context.feature_hash,
            "universe_hash": context.universe_hash,
            "prediction_rows": len(predictions),
            "generated_at": context.generated_at,
            "execution_generated_at": datetime.now(UTC).isoformat(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_hash": config_hash(self.config_path),
            "models": model_records,
            "readiness": context.readiness.to_dict(),
            "contracts": {
                "champion_recomputed": False,
                "labels_loaded": False,
                "future_data_loaded": False,
                "registry_modified": False,
                "candidate_selection_called": False,
                "paper_trading_called": False,
                "promotion_called": False,
                "hash_scope": "per-model rows exclude prediction_hash; bundle hash covers all rows",
            },
        }


def _rank(frame: DataFrame) -> DataFrame:
    result = frame.sort_values(
        ["trade_date", "prediction_score", "ts_code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = result.groupby("trade_date", sort=False).cumcount() + 1
    return result


def _validate_bundle(
    frame: DataFrame,
    context: ShadowContext,
    *,
    expected_prediction_hash: str,
) -> None:
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise DataValidationError("shadow prediction schema is not canonical")
    if frame.empty or frame.duplicated(["trade_date", "model_id", "ts_code"]).any():
        raise DataValidationError("shadow predictions are empty or duplicated")
    if set(frame["trade_date"].astype(str)) != {context.as_of}:
        raise DataValidationError("shadow predictions contain future or different dates")
    if not set(frame["model_role"].astype(str)).issubset(MODEL_ROLES):
        raise DataValidationError("shadow predictions contain invalid model_role")
    if set(frame["production_run_id"].astype(str)) != {context.production_run_id}:
        raise DataValidationError("shadow rows lack consistent production_run_id")
    if set(frame["feature_hash"].astype(str)) != {context.feature_hash}:
        raise DataValidationError("shadow rows contain feature hash mismatch")
    if set(frame["universe_hash"].astype(str)) != {context.universe_hash}:
        raise DataValidationError("shadow rows contain universe hash mismatch")
    if set(frame["access_policy"].astype(str)) != {"prospective_production"}:
        raise DataValidationError("shadow rows contain prohibited access policy")
    if set(frame["prediction_hash"].astype(str)) != {expected_prediction_hash}:
        raise DataValidationError("shadow rows contain inconsistent prediction hash")
