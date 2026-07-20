"""Typed records produced by the read-only model explainability layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

type SignalStrength = Literal["strong", "moderate", "weak"]
type ExplanationConfidence = Literal["high", "medium", "low", "unavailable"]


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One feature's local contribution to a LightGBM raw ranking score."""

    feature: str
    value: float | None
    shap: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StockExplanation:
    """Local explanation for one unchanged research candidate score."""

    ts_code: str
    model_rank: int
    candidate_rank: int
    prediction_score: float
    score_percentile: float
    historical_score_percentile: float | None
    history_status: str
    signal_strength: SignalStrength
    confidence: ExplanationConfidence
    base_value: float
    positive_contributions: tuple[FeatureContribution, ...]
    negative_contributions: tuple[FeatureContribution, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["positive_contributions"] = [
            contribution.to_dict() for contribution in self.positive_contributions
        ]
        payload["negative_contributions"] = [
            contribution.to_dict() for contribution in self.negative_contributions
        ]
        return payload


@dataclass(frozen=True, slots=True)
class ExplainabilityResult:
    """Published explanation artifact identity."""

    as_of: str
    model_id: str
    candidate_count: int
    method: str
    json_path: str
    markdown_path: str
