"""Fixed LightGBM Ranker fitting for governed retraining."""

from __future__ import annotations

import numpy as np

from ashare_quant.config.settings import AppSettings
from ashare_quant.models.ranker import feature_importance, fit_ranker
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.retraining.execution.schemas import PreparedTrainingData, TrainedRanker


class GovernedRankerTrainer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def train(self, prepared: PreparedTrainingData) -> TrainedRanker:
        model = fit_ranker(prepared.train, prepared.validation, self.settings.ranker)
        predictions = np.asarray(model.predict(prepared.validation.features), dtype=float)
        metrics = evaluate_ranker(
            prepared.validation,
            predictions,
            self.settings.ranker.ndcg_at,
            self.settings.ranker.portfolio_fractions,
        )
        return TrainedRanker(model, metrics, feature_importance(model, prepared.features))
