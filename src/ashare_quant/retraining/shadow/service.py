"""Governed prospective scoring for validated retrained Challengers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import PredictionModel
from ashare_quant.models.shadow.scoring import add_score_percentile, score_challenger
from ashare_quant.models.shadow.service import PREDICTION_COLUMNS
from ashare_quant.models.shadow.storage import (
    canonical_payload_hash,
    logical_prediction_hash,
    publish_shadow_bundle,
    read_complete_manifest,
)
from ashare_quant.retraining.execution.schemas import QualificationExecutionContext
from ashare_quant.retraining.shadow.schemas import (
    RetrainedShadowContext,
    RetrainedShadowResult,
)
from ashare_quant.retraining.shadow.validation import validate_retrained_shadow_eligibility
from ashare_quant.utils.manifest import config_hash, current_git_info

type DataFrame = pd.DataFrame


class RetrainedChallengerShadowService:
    """Publish one validated retrained candidate as an immutable daily sidecar."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.model_loader = model_loader
        resolved = config_path.resolve()
        project_root = resolved.parent.parent if resolved.parent.name == "config" else Path.cwd()
        self.runs_root = project_root / "runs"

    def predict(
        self,
        model_id: str,
        *,
        as_of: str | None = None,
        qualification: QualificationExecutionContext | None = None,
    ) -> RetrainedShadowResult:
        """Score only current as-of features after governed validation eligibility."""

        resolved_as_of = as_of or self._latest_production_date()
        context, champion_keys = validate_retrained_shadow_eligibility(
            model_id=model_id,
            as_of=resolved_as_of,
            settings=self.settings,
            config_path=self.config_path,
            runs_root=self.runs_root,
            qualification=qualification,
        )
        identity = canonical_payload_hash(
            {
                "as_of": resolved_as_of,
                "production_run_id": context.production_run_id,
                "production_shadow_run_id": context.production_shadow_run_id,
                "model_id": model_id,
                "artifact_hash": context.artifact_hash,
                "feature_hash": context.feature_hash,
                "universe_hash": context.current_universe_hash,
                "validation_manifest_hash": context.validation_manifest_hash,
                "lineage": context.lineage.model_dump(mode="json"),
                "config_hash": config_hash(self.config_path),
                "qualification": (qualification.model_dump(mode="json") if qualification else None),
            }
        )
        shadow_run_id = f"shadow_{resolved_as_of}_{identity[:16]}"
        output_dir = self._output_dir(
            resolved_as_of,
            model_id,
            qualification_run_id=(qualification.qualification_run_id if qualification else None),
        )
        existing = read_complete_manifest(output_dir)
        if existing is not None:
            if existing.get("shadow_run_id") != shadow_run_id:
                raise DataValidationError(
                    "retrained shadow identity cannot overwrite existing output"
                )
            return RetrainedShadowResult(
                model_id,
                resolved_as_of,
                shadow_run_id,
                int(existing.get("prediction_rows", 0)),
                output_dir,
                True,
            )
        if output_dir.exists():
            raise DataValidationError(f"incomplete retrained shadow output exists: {output_dir}")

        scored = add_score_percentile(
            score_challenger(
                context.model,
                processed_root=self.settings.paths.processed_data,
                as_of=resolved_as_of,
                model_loader=self.model_loader,
            )
        )
        _require_same_keys(champion_keys, scored)
        predictions = scored.loc[
            :, ["trade_date", "ts_code", "prediction_score", "rank", "score_percentile"]
        ].copy()
        predictions["model_id"] = model_id
        predictions["model_role"] = f"challenger_h{context.horizon}"
        predictions["model_origin"] = context.lineage.model_origin
        predictions["native_horizon"] = context.horizon
        predictions["production_run_id"] = context.production_run_id
        predictions["shadow_run_id"] = shadow_run_id
        predictions["prediction_hash"] = ""
        predictions["feature_hash"] = context.feature_hash
        predictions["universe_hash"] = context.current_universe_hash
        predictions["access_policy"] = "prospective_production"
        predictions["generated_at"] = context.generated_at
        predictions["parent_model_id"] = context.lineage.parent_model_id
        predictions["training_request_id"] = context.lineage.training_request_id
        predictions["training_run_id"] = context.lineage.training_run_id
        predictions["validation_run_id"] = context.lineage.validation_run_id
        predictions = (
            predictions.loc[:, list(PREDICTION_COLUMNS)]
            .sort_values(["trade_date", "model_id", "ts_code"], kind="mergesort")
            .reset_index(drop=True)
        )
        if qualification is not None:
            predictions["qualification_run_id"] = qualification.qualification_run_id
            predictions["qualification_only"] = True
        prediction_hash = logical_prediction_hash(predictions)
        predictions["prediction_hash"] = prediction_hash
        manifest = self._manifest(
            context=context,
            shadow_run_id=shadow_run_id,
            prediction_hash=prediction_hash,
            prediction_rows=len(predictions),
            qualification=qualification,
        )
        publish_shadow_bundle(
            output_dir=output_dir,
            predictions=predictions,
            manifest_without_file_hash=manifest,
        )
        return RetrainedShadowResult(
            model_id,
            resolved_as_of,
            shadow_run_id,
            len(predictions),
            output_dir,
        )

    def status(self, model_id: str, *, as_of: str | None = None) -> dict[str, Any]:
        resolved = as_of or self._latest_shadow_date(model_id)
        if resolved is None:
            return {"model_id": model_id, "status": "missing", "as_of": None}
        output = self._output_dir(resolved, model_id)
        manifest = read_complete_manifest(output)
        if manifest is None:
            return {
                "model_id": model_id,
                "status": "incomplete" if output.exists() else "missing",
                "as_of": resolved,
                "output": str(output),
            }
        return {
            "model_id": model_id,
            "status": "complete",
            "as_of": resolved,
            "shadow_run_id": manifest.get("shadow_run_id"),
            "prediction_rows": manifest.get("prediction_rows"),
            "output": str(output),
        }

    def _manifest(
        self,
        *,
        context: RetrainedShadowContext,
        shadow_run_id: str,
        prediction_hash: str,
        prediction_rows: int,
        qualification: QualificationExecutionContext | None,
    ) -> dict[str, Any]:
        git = current_git_info()
        lineage = context.lineage.model_dump(mode="json")
        model_record = {
            **lineage,
            "model_role": f"challenger_h{context.horizon}",
            "native_horizon": context.horizon,
            "prediction_hash": prediction_hash,
            "feature_hash": context.feature_hash,
            "universe_hash": context.current_universe_hash,
            "training_universe_hash": context.training_universe_hash,
            "source_models": [],
            "fusion_method": None,
            "access_policy": "prospective_production",
            "qualification_run_id": (qualification.qualification_run_id if qualification else None),
            "qualification_only": qualification is not None,
            "qualification_phase": (qualification.qualification_phase if qualification else None),
            "qualification_source": (qualification.qualification_source if qualification else None),
            "promotion_forbidden": qualification is not None,
            "trading_forbidden": qualification is not None,
        }
        return {
            "schema_version": 1,
            "artifact_name": "shadow_prediction_bundle",
            "bundle_kind": "retrained_challenger_sidecar",
            "production_run_id": context.production_run_id,
            "production_shadow_run_id": context.production_shadow_run_id,
            "shadow_run_id": shadow_run_id,
            "prediction_hash": prediction_hash,
            "feature_hash": context.feature_hash,
            "universe_hash": context.current_universe_hash,
            "prediction_rows": prediction_rows,
            "generated_at": context.generated_at,
            "execution_generated_at": datetime.now(UTC).isoformat(),
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_hash": config_hash(self.config_path),
            **lineage,
            "models": [model_record],
            "access_policy": "prospective_production",
            "qualification_run_id": (qualification.qualification_run_id if qualification else None),
            "qualification_only": qualification is not None,
            "qualification_phase": (qualification.qualification_phase if qualification else None),
            "qualification_source": (qualification.qualification_source if qualification else None),
            "promotion_forbidden": qualification is not None,
            "trading_forbidden": qualification is not None,
            "validation_manifest_hash": context.validation_manifest_hash,
            "contracts": {
                "champion_recomputed": False,
                "labels_loaded": False,
                "future_data_loaded": False,
                "historical_evaluation_loaded": False,
                "registry_modified": False,
                "candidate_selection_called": False,
                "paper_trading_called": False,
                "promotion_called": False,
            },
        }

    def _latest_production_date(self) -> str:
        root = self.settings.paths.reports / "shadow_predictions"
        dates = (
            [
                path.name
                for path in root.iterdir()
                if path.is_dir()
                and len(path.name) == 8
                and path.name.isdigit()
                and read_complete_manifest(path) is not None
            ]
            if root.is_dir()
            else []
        )
        if not dates:
            raise DataValidationError("SHADOW_NOT_ELIGIBLE: no production shadow bundle exists")
        return max(dates)

    def _latest_shadow_date(self, model_id: str) -> str | None:
        root = self.settings.paths.reports / "shadow_predictions"
        dates = (
            [
                path.name
                for path in root.iterdir()
                if path.is_dir() and self._output_dir(path.name, model_id).is_dir()
            ]
            if root.is_dir()
            else []
        )
        return max(dates) if dates else None

    def _output_dir(
        self,
        as_of: str,
        model_id: str,
        *,
        qualification_run_id: str | None = None,
    ) -> Path:
        base = self.settings.paths.reports / "shadow_predictions" / as_of
        if qualification_run_id is not None:
            return base / "qualification" / qualification_run_id / model_id
        return base / "retrained" / model_id


def _require_same_keys(champion: DataFrame, challenger: DataFrame) -> None:
    columns = ["trade_date", "ts_code"]
    left = set(champion[columns].astype(str).itertuples(index=False, name=None))
    right = set(challenger[columns].astype(str).itertuples(index=False, name=None))
    if left != right:
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: retrained keys differ from Champion")
