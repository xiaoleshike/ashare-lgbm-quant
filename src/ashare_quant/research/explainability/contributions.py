"""TreeSHAP contribution calculation for persisted LightGBM rankers."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]


class ExplainableModel(Protocol):
    """Prediction contract required from a LightGBM Booster."""

    @overload
    def predict(self, data: pd.DataFrame) -> FloatArray: ...

    @overload
    def predict(self, data: pd.DataFrame, *, pred_contrib: Literal[True]) -> FloatArray: ...

    def predict(self, data: pd.DataFrame, *, pred_contrib: Literal[True] = True) -> FloatArray:
        """Return raw ranking scores or feature contributions."""


class ShapExplanation(Protocol):
    """Array fields returned by a SHAP explainer."""

    values: object
    base_values: object


class ShapExplainer(Protocol):
    """Callable SHAP explanation interface."""

    def __call__(self, matrix: pd.DataFrame) -> ShapExplanation:
        """Explain one feature matrix."""


@dataclass(frozen=True, slots=True)
class ContributionMatrix:
    """Local additive contributions and their common calculation method."""

    values: FloatArray
    base_values: FloatArray
    method: str


def compute_tree_contributions(
    model: ExplainableModel,
    matrix: pd.DataFrame,
) -> ContributionMatrix:
    """Prefer SHAP TreeExplainer and fall back to LightGBM native TreeSHAP."""

    shap_factory = _optional_shap_factory()
    if shap_factory is not None:
        explainer = shap_factory(model)
        explanation = explainer(matrix)
        values = np.asarray(explanation.values, dtype=float)
        base_values = np.asarray(explanation.base_values, dtype=float)
        if base_values.ndim == 0:
            base_values = np.full(len(matrix), float(base_values), dtype=float)
        elif base_values.ndim > 1:
            base_values = base_values.reshape(len(matrix), -1)[:, 0]
        return _validated_matrix(values, base_values, matrix, "shap_tree_explainer")

    native = np.asarray(model.predict(matrix, pred_contrib=True), dtype=float)
    if native.ndim != 2 or native.shape[1] != len(matrix.columns) + 1:
        raise ValueError(
            "LightGBM pred_contrib returned an invalid shape: "
            f"expected=({len(matrix)}, {len(matrix.columns) + 1}) actual={native.shape}"
        )
    return _validated_matrix(native[:, :-1], native[:, -1], matrix, "lightgbm_pred_contrib")


def _optional_shap_factory() -> Callable[[object], ShapExplainer] | None:
    try:
        module = importlib.import_module("shap")
    except ModuleNotFoundError as error:
        if error.name != "shap":
            raise
        return None
    return cast(Callable[[object], ShapExplainer], vars(module)["TreeExplainer"])


def _validated_matrix(
    values: FloatArray,
    base_values: FloatArray,
    matrix: pd.DataFrame,
    method: str,
) -> ContributionMatrix:
    expected_shape = (len(matrix), len(matrix.columns))
    if values.shape != expected_shape:
        raise ValueError(
            f"{method} returned an invalid contribution shape: "
            f"expected={expected_shape} actual={values.shape}"
        )
    if base_values.shape != (len(matrix),):
        raise ValueError(
            f"{method} returned invalid base values: "
            f"expected={(len(matrix),)} actual={base_values.shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(base_values).all():
        raise ValueError(f"{method} returned non-finite contribution values")
    return ContributionMatrix(values=values, base_values=base_values, method=method)
