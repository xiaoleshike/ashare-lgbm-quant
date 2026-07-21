"""Phase 2.6.2A read-only model drift diagnostic orchestration."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.drift_data import DriftDataLoader, DriftModelContext
from ashare_quant.models.drift_metrics import (
    build_feature_drift,
    build_feature_response_drift,
    build_score_drift,
)
from ashare_quant.utils.manifest import (
    atomic_write_json,
    config_hash,
    current_git_info,
    read_manifest,
)

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class ModelDriftDiagnosticResult:
    """Published model drift diagnostic artifact."""

    run_id: str
    output_dir: Path
    model_id: str
    feature_count: int
    months: int


class ModelDriftDiagnosticEngine:
    """Diagnose a registered model from frozen predictions and generated PIT artifacts."""

    def __init__(
        self,
        *,
        processed_root: Path,
        models_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.processed_root = processed_root
        self.models_root = models_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path

    def run(
        self,
        *,
        model_id: str,
        start_date: str,
        end_date: str,
    ) -> ModelDriftDiagnosticResult:
        """Run diagnostics without model fitting, rescoring, or source artifact writes."""

        _validate_dates(start_date, end_date)
        loader = DriftDataLoader(
            processed_root=self.processed_root,
            reports_root=self.reports_root,
            models_root=self.models_root,
        )
        context = loader.resolve_context(model_id, start_date, end_date)
        drift = self.settings.diagnostics.model_drift

        predictions = loader.load_predictions(context, start_date, end_date)
        reference_features, evaluation_features = loader.load_feature_samples(
            context,
            start_date,
            end_date,
            reference_rows=drift.reference_sample_rows,
            evaluation_rows_per_month=drift.evaluation_sample_rows_per_month,
        )
        reference_coverage, evaluation_coverage = loader.load_feature_coverage(
            context, start_date, end_date
        )
        feature_drift = build_feature_drift(
            reference_features,
            evaluation_features,
            reference_coverage,
            evaluation_coverage,
            context.feature_names,
            psi_bins=drift.psi_bins,
        )
        score_drift, score_reference_months = build_score_drift(
            predictions,
            reference_months=drift.score_reference_months,
            psi_bins=drift.psi_bins,
        )

        # Labels enter only after predictions and label-independent drift are frozen in memory.
        reference_response, evaluation_response = loader.load_response_samples(
            context,
            start_date,
            end_date,
            horizon=drift.label_horizon,
            reference_rows=drift.reference_sample_rows,
            evaluation_rows_per_month=drift.evaluation_sample_rows_per_month,
        )
        feature_response = build_feature_response_drift(
            reference_response,
            evaluation_response,
            context.feature_names,
            bucket_counts=drift.response_bucket_counts,
            minimum_cross_section=drift.minimum_daily_cross_section,
        )

        run_id = _run_id(model_id, start_date, end_date, context.feature_hash)
        output_dir = self.reports_root / "model_diagnostics" / run_id
        summary = self._summary(
            context,
            start_date,
            end_date,
            predictions,
            feature_drift,
            score_drift,
            feature_response,
            score_reference_months,
            len(reference_response),
            len(evaluation_response),
        )
        manifest = self._manifest(
            run_id,
            context,
            start_date,
            end_date,
            summary,
            score_reference_months,
        )
        _publish(
            output_dir,
            feature_drift,
            score_drift,
            feature_response,
            summary,
            manifest,
        )
        return ModelDriftDiagnosticResult(
            run_id=run_id,
            output_dir=output_dir,
            model_id=model_id,
            feature_count=len(context.feature_names),
            months=int(score_drift["month"].nunique()),
        )

    def _summary(
        self,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
        predictions: DataFrame,
        feature_drift: DataFrame,
        score_drift: DataFrame,
        feature_response: DataFrame,
        score_reference_months: tuple[str, ...],
        reference_response_rows: int,
        evaluation_response_rows: int,
    ) -> dict[str, Any]:
        settings = self.settings.diagnostics.model_drift
        severe = feature_drift["psi"] >= settings.psi_severe_threshold
        warning = (feature_drift["psi"] >= settings.psi_warning_threshold) & ~severe
        sign_changes = feature_response[["month", "feature", "ic_sign_change"]].drop_duplicates()
        worst_features = (
            feature_drift.groupby("feature", sort=True)["psi"]
            .max()
            .sort_values(ascending=False)
            .head(10)
        )
        return cast(
            dict[str, Any],
            _json_safe(
                {
                    "schema_version": 1,
                    "artifact_name": "model_drift_diagnostics",
                    "model_id": context.model.model_id,
                    "model_status": context.model.status,
                    "feature_hash": context.feature_hash,
                    "feature_count": len(context.feature_names),
                    "requested_start_date": start_date,
                    "requested_end_date": end_date,
                    "training_reference": context.model.training_date_range,
                    "score_reference_months": list(score_reference_months),
                    "prediction_rows": len(predictions),
                    "months": int(score_drift["month"].nunique()),
                    "feature_drift": {
                        "rows": len(feature_drift),
                        "warning_rows": int(warning.sum()),
                        "severe_rows": int(severe.sum()),
                        "worst_features_by_max_psi": [
                            {"feature": str(feature), "max_psi": float(value)}
                            for feature, value in worst_features.items()
                        ],
                    },
                    "score_drift": {
                        "maximum_score_psi": float(score_drift["score_psi"].max()),
                        "minimum_normalized_breadth": float(
                            score_drift["normalized_breadth"].min()
                        ),
                        "maximum_top1_concentration": float(
                            score_drift["top1_concentration"].max()
                        ),
                        "maximum_top10_concentration": float(
                            score_drift["top10_concentration"].max()
                        ),
                    },
                    "feature_response_drift": {
                        "rows": len(feature_response),
                        "reference_rows": reference_response_rows,
                        "evaluation_rows": evaluation_response_rows,
                        "month_feature_sign_changes": int(sign_changes["ic_sign_change"].sum()),
                    },
                    "scientific_scope": {
                        "uses_frozen_historical_predictions": True,
                        "model_rescored": False,
                        "model_fitted": False,
                        "labels_loaded_after_prediction_and_distribution_drift": True,
                        "labels_used_only_for_post_hoc_feature_response": True,
                        "results_used_for_model_selection": False,
                    },
                }
            ),
        )

    def _manifest(
        self,
        run_id: str,
        context: DriftModelContext,
        start_date: str,
        end_date: str,
        summary: dict[str, Any],
        score_reference_months: tuple[str, ...],
    ) -> dict[str, Any]:
        git = current_git_info()
        return {
            "schema_version": 1,
            "artifact_name": "model_drift_diagnostics_manifest",
            "run_id": run_id,
            "model_id": context.model.model_id,
            "model_status": context.model.status,
            "model_artifact": context.model.artifact_path,
            "feature_list": list(context.feature_names),
            "feature_hash": context.feature_hash,
            "training_date_range": context.model.training_date_range,
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "score_reference_method": "first_n_requested_months_from_frozen_predictions",
            "score_reference_months": list(score_reference_months),
            "diagnostic_config": self.settings.diagnostics.model_drift.model_dump(mode="json"),
            "source_backtest_path": str(context.backtest_dir),
            "source_backtest_manifest": context.backtest_manifest,
            "source_processed_manifests": {
                name: read_manifest(self.processed_root / name)
                for name in ("features_daily", "universe_daily", "labels_forward")
            },
            "output_rows": {
                "feature_drift": summary["feature_drift"]["rows"],
                "score_drift": summary["months"],
                "feature_response": summary["feature_response_drift"]["rows"],
            },
            "leakage_contract": summary["scientific_scope"],
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "completed_at": datetime.now(UTC).isoformat(),
        }


def _validate_dates(start_date: str, end_date: str) -> None:
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if len(value) != 8 or not value.isdigit():
            raise DataValidationError(f"{name} must use YYYYMMDD: {value}")
    if start_date > end_date:
        raise DataValidationError("model drift start_date is after end_date")


def _run_id(model_id: str, start_date: str, end_date: str, feature_hash: str) -> str:
    safe_model = "".join(character if character.isalnum() else "_" for character in model_id)
    return f"{safe_model}_{start_date}_{end_date}_{feature_hash[:8]}"


def _publish(
    output_dir: Path,
    feature_drift: DataFrame,
    score_drift: DataFrame,
    feature_response: DataFrame,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        feature_drift.to_parquet(staging / "feature_drift.parquet", index=False)
        score_drift.to_parquet(staging / "score_drift.parquet", index=False)
        feature_response.to_parquet(staging / "feature_response.parquet", index=False)
        atomic_write_json(staging / "summary.json", summary)
        (staging / "diagnostics_report.md").write_text(_render_report(summary), encoding="utf-8")
        atomic_write_json(staging / "manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "feature_drift.parquet",
            "score_drift.parquet",
            "feature_response.parquet",
            "summary.json",
            "diagnostics_report.md",
            "manifest.json",
        ):
            os.replace(staging / filename, output_dir / filename)


def _render_report(summary: dict[str, Any]) -> str:
    feature = summary["feature_drift"]
    score = summary["score_drift"]
    response = summary["feature_response_drift"]
    lines = [
        "# Model Drift Diagnostics",
        "",
        f"- Model: `{summary['model_id']}` ({summary['model_status']})",
        f"- Evaluation: {summary['requested_start_date']} to {summary['requested_end_date']}",
        f"- Training reference: {summary['training_reference']['start']} to "
        f"{summary['training_reference']['end']}",
        f"- Features: {summary['feature_count']}",
        f"- Prediction rows: {summary['prediction_rows']}",
        "",
        "## Feature Drift",
        "",
        f"- Material PSI rows: {feature['warning_rows']}",
        f"- Severe PSI rows: {feature['severe_rows']}",
        "",
        "| Feature | Maximum PSI |",
        "| --- | ---: |",
    ]
    for row in feature["worst_features_by_max_psi"]:
        lines.append(f"| {row['feature']} | {row['max_psi']:.4f} |")
    lines.extend(
        [
            "",
            "## Score Drift",
            "",
            f"- Maximum monthly score PSI: {score['maximum_score_psi']:.4f}",
            f"- Minimum normalized breadth: {score['minimum_normalized_breadth']:.4f}",
            f"- Maximum Top 1% concentration: {score['maximum_top1_concentration']:.2%}",
            f"- Maximum Top 10% concentration: {score['maximum_top10_concentration']:.2%}",
            "",
            "## Feature-Response Drift",
            "",
            f"- Monthly feature sign changes: {response['month_feature_sign_changes']}",
            f"- Matured evaluation rows: {response['evaluation_rows']}",
            "",
            "> Labels are used only for post-hoc feature-response analysis. This report does not "
            "fit, rescore, select, promote, or modify any model.",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value
