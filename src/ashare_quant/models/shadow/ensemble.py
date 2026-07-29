"""Deterministic percentile-rank shadow ensemble."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash

type DataFrame = pd.DataFrame


def ensemble_model_id(
    component_model_ids: list[str],
    prediction_hashes: list[str],
    fusion_method: str,
) -> str:
    """Return the required environment-independent ensemble model ID."""

    digest = canonical_payload_hash(
        {
            "component_model_ids": sorted(component_model_ids),
            "prediction_hashes": sorted(prediction_hashes),
            "fusion_method": fusion_method,
        }
    )
    return f"ensemble_{digest}"


def build_percentile_ensemble(
    components: Mapping[int, DataFrame],
) -> DataFrame:
    """Average per-model score percentiles after exact key validation."""

    if tuple(sorted(components)) != (5, 10, 20, 60):
        raise DataValidationError("shadow ensemble requires horizons 5, 10, 20, and 60")
    keys = ["trade_date", "ts_code"]
    reference: set[tuple[str, str]] | None = None
    combined: DataFrame | None = None
    for horizon in sorted(components):
        frame = components[horizon]
        current = set(frame.loc[:, keys].astype(str).itertuples(index=False, name=None))
        if reference is None:
            reference = current
        elif current != reference:
            raise DataValidationError(f"shadow ensemble stock keys differ for horizon={horizon}")
        selected = frame.loc[:, [*keys, "score_percentile"]].rename(
            columns={"score_percentile": f"h{horizon}_percentile"}
        )
        combined = (
            selected
            if combined is None
            else combined.merge(selected, on=keys, how="inner", validate="one_to_one")
        )
    assert combined is not None
    columns = [f"h{horizon}_percentile" for horizon in sorted(components)]
    combined["prediction_score"] = combined[columns].mean(axis=1)
    return (
        combined.loc[:, [*keys, "prediction_score"]]
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
