"""Same-model historical score positioning for candidate explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.config.settings import ExplainabilitySettings
from ashare_quant.research.explainability.schemas import ExplanationConfidence


def current_score_percentiles(predictions: pd.DataFrame) -> dict[str, float]:
    """Return same-session empirical score percentiles; larger scores rank higher."""

    ranked = predictions.assign(
        score_percentile=predictions["prediction_score"].rank(method="average", pct=True)
    )
    return dict(
        zip(
            ranked["ts_code"].astype(str),
            ranked["score_percentile"].astype(float),
            strict=True,
        )
    )


def load_same_model_history(
    reports_root: Path,
    *,
    as_of: str,
    model_id: str,
    maximum_sessions: int,
) -> tuple[np.ndarray, int]:
    """Load only prior published prediction scores from the identical model."""

    eligible_paths = sorted(
        (
            path
            for path in reports_root.glob("*/predictions.parquet")
            if path.parent.name.isdigit() and path.parent.name < as_of
        ),
        key=lambda path: path.parent.name,
    )[-maximum_sessions:]
    scores: list[np.ndarray] = []
    sessions = 0
    for path in eligible_paths:
        frame = pd.read_parquet(path, columns=["trade_date", "prediction_score", "model_id"])
        same_model = frame.loc[
            (frame["trade_date"].astype(str) < as_of) & (frame["model_id"].astype(str) == model_id),
            "prediction_score",
        ]
        numeric = pd.to_numeric(same_model, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if len(finite):
            scores.append(finite)
            sessions += 1
    if not scores:
        return np.array([], dtype=float), 0
    return np.concatenate(scores), sessions


def historical_percentile(score: float, history: np.ndarray) -> float | None:
    """Return an empirical CDF position within prior same-model scores."""

    if history.size == 0:
        return None
    return float(np.searchsorted(np.sort(history), score, side="right") / history.size)


def history_assessment(
    sessions: int,
    settings: ExplainabilitySettings,
) -> tuple[str, ExplanationConfidence]:
    """Describe history availability without claiming forecast confidence."""

    if sessions == 0:
        return "insufficient_same_model_history", "unavailable"
    if sessions < settings.minimum_history_sessions:
        return "limited_same_model_history", "low"
    if sessions < settings.high_confidence_history_sessions:
        return "available_same_model_history", "medium"
    return "established_same_model_history", "high"
