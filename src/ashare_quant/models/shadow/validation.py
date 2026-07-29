"""Pre-scoring readiness validation for prospective shadow predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import load_registered_feature_list
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.model_loader import load_shadow_challengers
from ashare_quant.models.shadow.schemas import ReadinessResult, ShadowContext
from ashare_quant.models.shadow.scoring import load_champion_reference
from ashare_quant.models.shadow.storage import (
    canonical_payload_hash,
    file_sha256,
)
from ashare_quant.orchestration.publication import validate_production_publication
from ashare_quant.universe.storage import UniverseStore
from ashare_quant.utils.manifest import parquet_artifact_statistics, read_manifest


class ShadowReadinessValidator:
    """Bind one successful production run to candidate-only scoring inputs."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        runs_root: Path = Path("runs"),
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.runs_root = runs_root

    def validate(self, as_of: str) -> tuple[ReadinessResult, ShadowContext | None]:
        """Return structured readiness and a context only when every hard check passes."""

        try:
            context, checks = self._validate(as_of)
        except (DataValidationError, OSError, ValueError) as error:
            return ReadinessResult(False, (str(error),), {}), None
        return ReadinessResult(True, (), checks), context

    def require_ready(self, as_of: str) -> ShadowContext:
        """Return a validated context or fail before any model scoring occurs."""

        result, context = self.validate(as_of)
        if not result.ready or context is None:
            raise DataValidationError("shadow readiness failed: " + "; ".join(result.hard_failures))
        return context

    def _validate(self, as_of: str) -> tuple[ShadowContext, dict[str, Any]]:
        if len(as_of) != 8 or not as_of.isdigit():
            raise DataValidationError(f"shadow as_of must use YYYYMMDD: {as_of}")
        shadow_settings = self.settings.models.shadow_predictions
        if shadow_settings.access_policy != "prospective_production":
            raise DataValidationError("frozen_oos_evaluation is prohibited for shadow scoring")

        summary = validate_production_publication(
            reports_root=self.reports_root,
            runs_root=self.runs_root,
            as_of=as_of,
        )
        production_run_id = str(summary["run_id"])
        report_dir = self.reports_root / as_of
        prediction_path = report_dir / "predictions.parquet"
        ranking_path = report_dir / "ranking.csv"
        prediction_manifest_path = report_dir / "manifest.json"
        prediction_manifest = _load_json(prediction_manifest_path, "production prediction manifest")
        if prediction_manifest.get("artifact_name") != "production_predictions":
            raise DataValidationError("invalid production prediction manifest identity")
        if prediction_manifest.get("as_of") != as_of:
            raise DataValidationError("production prediction manifest date mismatch")
        champion_model_id = str(summary["model_id"])
        if prediction_manifest.get("model_id") != champion_model_id:
            raise DataValidationError("Champion model_id differs across production artifacts")
        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None or champion.model_id != champion_model_id:
            raise DataValidationError("production prediction model is not the current Champion")
        champion_feature_hash = str(prediction_manifest.get("feature_hash") or "")
        if not champion_feature_hash or champion.feature_hash != champion_feature_hash:
            raise DataValidationError("Champion feature hash mismatch")

        observation_path = Path(str(summary.get("observation_log_path") or ""))
        observation = _load_json(observation_path, "production observation")
        fingerprints = observation.get("source_fingerprints")
        if not isinstance(fingerprints, dict):
            raise DataValidationError("production observation lacks source fingerprints")
        champion_file_hash = file_sha256(prediction_path)
        if fingerprints.get("predictions") != champion_file_hash:
            raise DataValidationError("Champion prediction artifact hash mismatch")
        champion_predictions = load_champion_reference(
            predictions_path=prediction_path,
            ranking_path=ranking_path,
            as_of=as_of,
            model_id=champion_model_id,
        )
        champion_prediction_hash = canonical_payload_hash(
            champion_predictions.sort_values(["trade_date", "ts_code"], kind="mergesort").to_dict(
                "records"
            )
        )

        feature_manifest = _required_processed_manifest(
            self.processed_root / "features_daily", "features_daily", as_of
        )
        universe_manifest = _required_processed_manifest(
            self.processed_root / "universe_daily", "universe_daily", as_of
        )
        _validate_embedded_manifests(
            prediction_manifest,
            feature_manifest=feature_manifest,
            universe_manifest=universe_manifest,
        )
        universe = UniverseStore(self.processed_root).read(as_of, as_of)
        eligible = universe.loc[
            universe["in_model_universe"].fillna(False).astype(bool),
            ["trade_date", "ts_code"],
        ].copy()
        eligible = eligible.sort_values(["trade_date", "ts_code"], kind="mergesort")
        if eligible.empty or eligible.duplicated(["trade_date", "ts_code"]).any():
            raise DataValidationError("current model universe is empty or duplicated")
        champion_keys = set(
            champion_predictions[["trade_date", "ts_code"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        universe_keys = set(eligible.astype(str).itertuples(index=False, name=None))
        if champion_keys != universe_keys:
            raise DataValidationError("Champion prediction keys differ from current model universe")
        universe_hash = canonical_payload_hash(eligible.astype(str).to_dict("records"))

        challengers, challenger_manifests = load_shadow_challengers(self.registry, shadow_settings)
        feature_statistics = parquet_artifact_statistics(self.processed_root / "features_daily")
        for horizon, model in challengers.items():
            features, digest = load_registered_feature_list(Path(model.artifact_path), model)
            if digest != champion_feature_hash:
                raise DataValidationError(
                    f"challenger feature hash differs from Champion: horizon={horizon}"
                )
            missing = sorted(set(features) - set(feature_statistics.column_names))
            if missing:
                raise DataValidationError(
                    f"challenger required features are absent: horizon={horizon} missing={missing}"
                )
        manifest_hashes = {
            horizon: canonical_payload_hash(manifest)
            for horizon, manifest in challenger_manifests.items()
        }
        checks = {
            "production_manifest_exists": True,
            "production_status": "success",
            "production_run_id": production_run_id,
            "champion_prediction_logical_hash": champion_prediction_hash,
            "champion_prediction_hash": champion_file_hash,
            "champion_prediction_rows": len(champion_predictions),
            "feature_manifest_max_date": _manifest_max_date(feature_manifest),
            "universe_manifest_max_date": _manifest_max_date(universe_manifest),
            "universe_hash": universe_hash,
            "challenger_model_ids": {
                str(horizon): model.model_id for horizon, model in challengers.items()
            },
            "challenger_artifacts_valid": len(challenger_manifests) == 4,
            "access_policy": shadow_settings.access_policy,
            "historical_evaluation_sources_used": False,
            "labels_loaded": False,
        }
        context = ShadowContext(
            as_of=as_of,
            production_run_id=production_run_id,
            champion_model_id=champion_model_id,
            champion_feature_hash=champion_feature_hash,
            champion_prediction_hash=champion_prediction_hash,
            champion_prediction_file_hash=champion_file_hash,
            feature_hash=champion_feature_hash,
            universe_hash=universe_hash,
            generated_at=str(
                prediction_manifest.get("generation_time") or summary.get("completed_time") or ""
            ),
            champion_predictions=champion_predictions,
            challenger_models=challengers,
            challenger_manifest_hashes=manifest_hashes,
            readiness=ReadinessResult(True, (), checks),
        )
        if not context.generated_at:
            raise DataValidationError("production prediction generation time is missing")
        return context, checks


def _required_processed_manifest(
    artifact_dir: Path, artifact_name: str, as_of: str
) -> dict[str, Any]:
    manifest = read_manifest(artifact_dir)
    if manifest is None:
        raise DataValidationError(f"{artifact_name} manifest does not exist")
    if manifest.get("artifact_name") != artifact_name:
        raise DataValidationError(f"invalid {artifact_name} manifest identity")
    if _manifest_max_date(manifest) < as_of:
        raise DataValidationError(f"{artifact_name} does not cover shadow as_of={as_of}")
    return manifest


def _manifest_max_date(manifest: dict[str, Any]) -> str:
    canonical = manifest.get("canonical_artifact")
    if isinstance(canonical, dict) and canonical.get("max_date") is not None:
        return str(canonical["max_date"])
    return str(manifest.get("requested_end_date") or "")


def _validate_embedded_manifests(
    prediction_manifest: dict[str, Any],
    *,
    feature_manifest: dict[str, Any],
    universe_manifest: dict[str, Any],
) -> None:
    inputs = prediction_manifest.get("input_artifact_manifests")
    if not isinstance(inputs, dict):
        raise DataValidationError("production prediction lacks processed input manifests")
    for name, current in (
        ("features_daily", feature_manifest),
        ("universe_daily", universe_manifest),
    ):
        embedded = inputs.get(name)
        if not isinstance(embedded, dict):
            raise DataValidationError(f"production prediction lacks embedded {name} manifest")
        if canonical_payload_hash(embedded) != canonical_payload_hash(current):
            raise DataValidationError(f"production prediction {name} manifest hash mismatch")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain an object")
    return payload
