"""Validated access to fixed shadow model configuration."""

from __future__ import annotations

from ashare_quant.config.settings import ShadowPredictionSettings
from ashare_quant.data.exceptions import DataValidationError

HORIZON_KEYS: dict[int, str] = {5: "h5", 10: "h10", 20: "h20", 60: "h60"}


def configured_model_ids(settings: ShadowPredictionSettings) -> dict[int, str]:
    """Return configured challenger IDs keyed by their native horizon."""

    if not settings.enabled:
        raise DataValidationError("models.shadow_predictions is disabled")
    if settings.access_policy != "prospective_production":
        raise DataValidationError(
            "shadow prediction access_policy must be prospective_production; "
            "frozen_oos_evaluation is prohibited"
        )
    if not settings.ensemble.enabled:
        raise DataValidationError("multi-horizon shadow ensemble must be enabled")
    return {
        horizon: settings.challenger_models[key].model_id for horizon, key in HORIZON_KEYS.items()
    }
