"""Prospective, label-free shadow prediction bundles."""

from ashare_quant.models.shadow.schemas import ShadowPredictionResult
from ashare_quant.models.shadow.service import ShadowPredictionService

__all__ = ["ShadowPredictionResult", "ShadowPredictionService"]
