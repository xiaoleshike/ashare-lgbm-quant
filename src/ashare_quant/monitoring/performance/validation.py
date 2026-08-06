"""Validation for immutable performance-observation inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.schemas import MODEL_ORIGINS, MODEL_ROLES
from ashare_quant.monitoring.performance_observation.schemas import (
    OBSERVATION_COLUMNS,
    OBSERVATION_KEY,
    SUPPORTED_HORIZONS,
)
from ashare_quant.monitoring.performance_observation.storage import (
    normalize_observation_lineage,
)

type DataFrame = pd.DataFrame

IMMATURE_STATUSES: frozenset[str] = frozenset(
    {"immature", "insufficient_future_calendar", "insufficient_future_trading_dates"}
)


def validate_observation_frame(frame: DataFrame, as_of: str) -> None:
    """Validate schema, maturity, identity, and lineage without other data sources."""

    frame = normalize_observation_lineage(frame)
    missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"performance observations lack required columns: {missing}")
    if frame.empty:
        return
    if frame.duplicated(list(OBSERVATION_KEY)).any():
        raise DataValidationError("performance observation identities are duplicated")
    if frame["observation_id"].duplicated().any():
        raise DataValidationError("performance observation_id is duplicated")
    if not set(frame["model_role"].astype(str)).issubset(MODEL_ROLES):
        raise DataValidationError("performance observations contain unsupported model_role")
    if not set(frame["model_origin"].astype(str)).issubset(MODEL_ORIGINS):
        raise DataValidationError("performance observations contain unsupported model_origin")
    horizons = set(pd.to_numeric(frame["horizon"], errors="coerce").dropna().astype(int).unique())
    if not horizons or not horizons.issubset(SUPPORTED_HORIZONS):
        raise DataValidationError("performance observations contain unsupported horizon")
    statuses = frame["label_status"].fillna("").astype(str)
    immature = sorted(set(statuses) & IMMATURE_STATUSES)
    if immature:
        raise DataValidationError(f"immature performance observations are prohibited: {immature}")
    if statuses.eq("").any():
        raise DataValidationError("performance observations contain empty label_status")
    if (frame["observation_as_of"].astype(str) > as_of).any():
        raise DataValidationError("performance observations include a future observation_as_of")
    available = statuses.eq("available")
    returns = pd.to_numeric(frame["future_excess_ret"], errors="coerce")
    if returns.loc[available].isna().any():
        raise DataValidationError("available observations contain null future_excess_ret")
    if returns.loc[~available].notna().any():
        raise DataValidationError("unavailable observations contain future_excess_ret")
    for column in (
        "production_run_id",
        "shadow_run_id",
        "prediction_hash",
        "feature_hash",
        "universe_hash",
    ):
        if frame[column].fillna("").astype(str).eq("").any():
            raise DataValidationError(f"performance observations contain empty {column}")
    retrained = frame["model_origin"].astype(str).eq("retrained_challenger")
    for column in (
        "parent_model_id",
        "training_request_id",
        "training_run_id",
        "validation_run_id",
    ):
        if frame.loc[retrained, column].fillna("").astype(str).eq("").any():
            raise DataValidationError(f"retrained observations contain empty {column}")


def validate_model_lineage(
    frame: DataFrame,
    model_lineage: dict[str, dict[str, Any]],
) -> None:
    """Require one stable model role and feature/universe identity per model."""

    for model_id, group in frame.groupby("model_id", sort=True):
        identity = model_lineage.get(str(model_id))
        if identity is None:
            raise DataValidationError(f"observation manifest lacks model lineage: {model_id}")
        for column in ("model_role", "model_origin", "feature_hash", "universe_hash"):
            values = set(group[column].astype(str))
            expected = str(identity.get(column, ""))
            if len(values) != 1 or values != {expected}:
                raise DataValidationError(
                    f"performance observation {column} lineage mismatch: model={model_id}"
                )
        role = str(identity["model_role"])
        if role == "multi_horizon_ensemble":
            sources = identity.get("source_models")
            if not isinstance(sources, list) or not sources:
                raise DataValidationError("ensemble observation lineage lacks source_models")
            if identity.get("fusion_method") != "percentile_mean":
                raise DataValidationError("ensemble observation lineage has invalid fusion_method")


def validate_source_manifest(manifest: dict[str, Any], directory_name: str) -> None:
    """Validate an observation manifest's prospective-only contract."""

    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != "performance_observation"
    ):
        raise DataValidationError("invalid performance observation artifact identity")
    if str(manifest.get("observation_as_of")) != directory_name:
        raise DataValidationError("performance observation directory date mismatch")
    if manifest.get("access_policy") != "prospective_production":
        raise DataValidationError("performance observation is not prospective_production")
    contracts = manifest.get("contracts")
    if not isinstance(contracts, dict):
        raise DataValidationError("performance observation lacks isolation contracts")
    required_false = (
        "historical_predictions_used",
        "inference_called",
        "backtest_called",
        "paper_trading_called",
        "registry_modified",
    )
    if any(contracts.get(key) is not False for key in required_false):
        raise DataValidationError("performance observation violates read-only source contract")
    if contracts.get("labels_used_only_after_maturity") is not True:
        raise DataValidationError("performance observation maturity contract is not asserted")
