"""Immutable model-promotion governance foundations."""

from ashare_quant.models.promotion.evidence import PromotionEvidencePaths
from ashare_quant.models.promotion.gates import PromotionGateEngine, PromotionGateEvaluation
from ashare_quant.models.promotion.service import (
    PromotionGovernanceResult,
    PromotionGovernanceService,
)

__all__ = [
    "PromotionEvidencePaths",
    "PromotionGateEngine",
    "PromotionGateEvaluation",
    "PromotionGovernanceResult",
    "PromotionGovernanceService",
]
