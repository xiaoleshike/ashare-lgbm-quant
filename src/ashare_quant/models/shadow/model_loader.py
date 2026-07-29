"""Candidate-only model resolution for shadow scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.config.settings import ShadowPredictionSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import load_registered_feature_list
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.configuration import configured_model_ids


def load_shadow_challengers(
    registry: ModelRegistry,
    settings: ShadowPredictionSettings,
) -> tuple[dict[int, RegisteredModel], dict[int, dict[str, Any]]]:
    """Resolve exactly four candidate artifacts without resolving the Champion."""

    records = {record.model_id: record for record in registry.list_models()}
    models: dict[int, RegisteredModel] = {}
    manifests: dict[int, dict[str, Any]] = {}
    for horizon, model_id in configured_model_ids(settings).items():
        model = records.get(model_id)
        if model is None:
            raise DataValidationError(f"shadow challenger is not registered: {model_id}")
        if model.status != "candidate":
            raise DataValidationError(
                f"shadow challenger must have candidate status: {model_id}={model.status}"
            )
        artifact = Path(model.artifact_path)
        _reject_historical_source_path(artifact)
        for filename in ("model.txt", "feature_list.json", "manifest.json"):
            if not (artifact / filename).is_file():
                raise DataValidationError(
                    f"shadow challenger artifact is missing {filename}: {model_id}"
                )
        manifest = _load_json(artifact / "manifest.json")
        if manifest.get("artifact_name") != "lightgbm_ranker_challenger":
            raise DataValidationError(f"invalid shadow challenger artifact: {model_id}")
        if int(manifest.get("horizon", -1)) != horizon:
            raise DataValidationError(
                f"shadow challenger horizon mismatch: {model_id} "
                f"expected={horizon} actual={manifest.get('horizon')}"
            )
        if manifest.get("access_policy") == "frozen_oos_evaluation":
            raise DataValidationError(f"frozen_oos_evaluation source is prohibited: {model_id}")
        _, feature_hash = load_registered_feature_list(artifact, model)
        if feature_hash != model.feature_hash:
            raise DataValidationError(f"shadow challenger feature hash mismatch: {model_id}")
        models[horizon] = model
        manifests[horizon] = manifest
    return models, manifests


def _reject_historical_source_path(path: Path) -> None:
    normalized = path.as_posix()
    prohibited = ("reports/challenger_predictions", "reports/ensemble_evaluation")
    if any(value in normalized for value in prohibited):
        raise DataValidationError(f"historical evaluation artifact is prohibited: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid challenger manifest: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"challenger manifest must contain an object: {path}")
    return payload
