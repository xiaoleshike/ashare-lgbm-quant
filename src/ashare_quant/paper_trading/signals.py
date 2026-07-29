"""Label-free daily signal resolution for isolated paper portfolios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_quant.config.settings import PaperPortfolioSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import score_registered_model_range
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.paper_trading.storage import file_sha256, payload_sha256

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class PaperSignal:
    """One portfolio's deterministic same-date ranking."""

    portfolio_id: str
    as_of: str
    model_id: str
    feature_hash: str
    source_signal_manifest_hash: str
    ranking: DataFrame


class PaperSignalProvider:
    """Resolve Champion, Challenger, and rank-ensemble signals without labels."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root

    def load(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        production_summary_path: Path,
    ) -> PaperSignal:
        """Return a frozen top-to-bottom ranking for one completed signal date."""

        production = _load_json(production_summary_path, "production summary")
        if str(production.get("as_of", "")) != as_of:
            raise DataValidationError("paper signal date differs from production summary date")
        if not production.get("run_id"):
            raise DataValidationError("production summary lacks run_id provenance")
        if portfolio.signal_type == "champion":
            return self._champion(portfolio, as_of, production_summary_path, production)
        if portfolio.signal_type == "model":
            assert portfolio.model_id is not None
            return self._model(
                portfolio,
                as_of,
                production_summary_path,
                self._registered(portfolio.model_id),
            )
        return self._ensemble(portfolio, as_of, production_summary_path)

    def _champion(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        production_summary_path: Path,
        production: dict[str, object],
    ) -> PaperSignal:
        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("paper trading requires a registered champion")
        if production.get("model_id") != champion.model_id:
            raise DataValidationError("production summary model_id is not the current champion")
        candidate_path = self.reports_root / as_of / "candidates.csv"
        candidate_manifest = self.reports_root / as_of / "candidates_manifest.json"
        candidates = _read_candidates(candidate_path, as_of, champion.model_id)
        digest = payload_sha256(
            {
                "production_summary": file_sha256(production_summary_path),
                "candidate_manifest": file_sha256(candidate_manifest),
                "model_manifest": file_sha256(Path(champion.artifact_path) / "manifest.json"),
            }
        )
        return PaperSignal(
            portfolio.portfolio_id,
            as_of,
            champion.model_id,
            champion.feature_hash,
            digest,
            _rank(candidates),
        )

    def _model(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        production_summary_path: Path,
        model: RegisteredModel,
    ) -> PaperSignal:
        batch = score_registered_model_range(
            model,
            processed_root=self.processed_root,
            start_date=as_of,
            end_date=as_of,
        )
        digest = payload_sha256(
            {
                "production_summary": file_sha256(production_summary_path),
                "features_manifest": file_sha256(
                    self.processed_root / "features_daily" / "_manifest.json"
                ),
                "universe_manifest": file_sha256(
                    self.processed_root / "universe_daily" / "_manifest.json"
                ),
                "model_manifest": file_sha256(Path(model.artifact_path) / "manifest.json"),
            }
        )
        return PaperSignal(
            portfolio.portfolio_id,
            as_of,
            model.model_id,
            batch.feature_hash,
            digest,
            _rank(batch.predictions),
        )

    def _ensemble(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        production_summary_path: Path,
    ) -> PaperSignal:
        components = [self._registered(model_id) for model_id in portfolio.component_model_ids]
        if not components:
            raise DataValidationError("paper ensemble has no component models")
        if len({model.feature_hash for model in components}) != 1:
            raise DataValidationError("paper ensemble component feature hashes differ")
        scored = [
            score_registered_model_range(
                model,
                processed_root=self.processed_root,
                start_date=as_of,
                end_date=as_of,
            ).predictions
            for model in components
        ]
        base_keys = scored[0].loc[:, ["trade_date", "ts_code"]].reset_index(drop=True)
        percentile_columns: list[pd.Series[float]] = []
        for model, frame in zip(components, scored, strict=True):
            keys = frame.loc[:, ["trade_date", "ts_code"]].reset_index(drop=True)
            if not keys.equals(base_keys):
                raise DataValidationError(
                    f"paper ensemble model universe differs: {model.model_id}"
                )
            percentile_columns.append(
                frame.groupby("trade_date", sort=False)["prediction_score"].rank(
                    method="average", pct=True
                )
            )
        ranking = base_keys.copy()
        ranking["prediction_score"] = pd.concat(percentile_columns, axis=1).mean(axis=1)
        ensemble_id = "ensemble:" + payload_sha256([model.model_id for model in components])[:16]
        digest = payload_sha256(
            {
                "production_summary": file_sha256(production_summary_path),
                "features_manifest": file_sha256(
                    self.processed_root / "features_daily" / "_manifest.json"
                ),
                "universe_manifest": file_sha256(
                    self.processed_root / "universe_daily" / "_manifest.json"
                ),
                "component_manifests": {
                    model.model_id: file_sha256(Path(model.artifact_path) / "manifest.json")
                    for model in components
                },
                "method": "daily_equal_weight_percentile_rank",
            }
        )
        return PaperSignal(
            portfolio.portfolio_id,
            as_of,
            ensemble_id,
            components[0].feature_hash,
            digest,
            _rank(ranking),
        )

    def _registered(self, model_id: str) -> RegisteredModel:
        try:
            model = next(
                record for record in self.registry.list_models() if record.model_id == model_id
            )
        except StopIteration as error:
            raise DataValidationError(
                f"paper portfolio model is not registered: {model_id}"
            ) from error
        if model.status == "retired":
            raise DataValidationError(f"paper portfolio model is retired: {model_id}")
        return model


def _read_candidates(path: Path, as_of: str, model_id: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"production candidates do not exist: {path}")
    frame = pd.read_csv(path, dtype={"ts_code": str, "trade_date": str})
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    if not required <= set(frame.columns):
        raise DataValidationError("production candidates have an invalid schema")
    if set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("production candidates contain a different date")
    if set(frame["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("production candidates contain a different model")
    return frame


def _rank(frame: DataFrame) -> DataFrame:
    result = frame.loc[:, ["trade_date", "ts_code", "prediction_score"]].copy()
    result = result.sort_values(
        ["prediction_score", "ts_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    if result["ts_code"].duplicated().any():
        raise DataValidationError("paper signal contains duplicate stock codes")
    return result


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise DataValidationError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"{label} must be a JSON object: {path}")
    return payload
