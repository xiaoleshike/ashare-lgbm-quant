"""Fair, immutable Champion-versus-Challenger post-hoc evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import AppSettings, ChallengerEvaluationSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger_metrics import (
    build_promotion_gate,
    evaluate_comparison,
)
from ashare_quant.models.inference import (
    PredictionModel,
    load_registered_feature_list,
    score_registered_model_range,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

EVALUATION_MANIFEST_SCHEMA_VERSION = 1

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class ChallengerEvaluationResult:
    """One immutable Champion-versus-Challenger evaluation report."""

    run_id: str
    champion_model_id: str
    challenger_model_id: str
    horizon: int
    labelled_rows: int
    evaluation_dates: int
    eligible_for_manual_review: bool
    output_dir: Path


class ChallengerEvaluationEngine:
    """Evaluate two frozen Rankers on identical rows and realized labels."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self._model_loader = model_loader

    def evaluate(self, challenger_model_id: str) -> ChallengerEvaluationResult:
        """Compare a candidate with the current champion without changing registry state."""

        challenger = _find_model(self.registry, challenger_model_id)
        if challenger.status != "candidate":
            raise DataValidationError("challenger evaluation requires a candidate model")
        champion = self.registry.get_champion("lightgbm_ranker")
        if champion is None:
            raise DataValidationError("no lightgbm_ranker champion is registered")
        if champion.model_id == challenger.model_id:
            raise DataValidationError("champion and challenger must be different models")
        champion_features, champion_hash = load_registered_feature_list(
            Path(champion.artifact_path), champion
        )
        challenger_features, challenger_hash = load_registered_feature_list(
            Path(challenger.artifact_path), challenger
        )
        if challenger_hash != champion_hash or challenger_features != champion_features:
            raise DataValidationError(
                "champion and challenger feature hashes differ; fair comparison is rejected"
            )
        prediction_dir = self.reports_root / "challenger_predictions" / challenger_model_id
        prediction_manifest = _load_json(
            prediction_dir / "manifest.json", "challenger prediction manifest"
        )
        challenger_predictions = _load_predictions(
            prediction_dir / "predictions.parquet",
            challenger_model_id,
            prediction_manifest,
        )
        horizon, ranges = _validate_prediction_contract(
            prediction_manifest,
            challenger,
            processed_root=self.processed_root,
        )
        start_date = min(start for start, _ in ranges)
        end_date = max(end for _, end in ranges)
        champion_batch = score_registered_model_range(
            champion,
            processed_root=self.processed_root,
            start_date=start_date,
            end_date=end_date,
            allowed_ranges=ranges,
            model_loader=self._model_loader,
        )
        _require_identical_prediction_keys(challenger_predictions, champion_batch.predictions)
        labels = _load_mature_labels(
            self.processed_root,
            horizon,
            dates=tuple(sorted(challenger_predictions["trade_date"].astype(str).unique())),
            maximum_mature_date=str(prediction_manifest["maximum_mature_evaluation_date"]),
        )
        comparison = _comparison_frame(
            challenger_predictions,
            champion_batch.predictions,
            labels,
        )
        metric_rows, overall = evaluate_comparison(
            comparison,
            champion,
            challenger,
            self.settings.models.challenger_evaluation,
        )
        gate = build_promotion_gate(
            overall[champion.model_id],
            overall[challenger.model_id],
            self.settings.models.challenger_evaluation,
        )
        identity = _evaluation_identity(
            champion,
            challenger,
            prediction_manifest,
            processed_root=self.processed_root,
            config_path=self.config_path,
        )
        run_id = f"challenger_eval_h{horizon}_{identity[:16]}"
        output_dir = self.reports_root / "challenger_evaluation" / run_id
        existing = _existing_result(output_dir, identity)
        if existing is not None:
            return existing
        summary = {
            "schema_version": 1,
            "artifact_name": "challenger_evaluation",
            "run_id": run_id,
            "champion_model_id": champion.model_id,
            "challenger_model_id": challenger.model_id,
            "horizon": horizon,
            "holding_period": prediction_manifest["holding_period"],
            "execution_rule": prediction_manifest["execution_rule"],
            "feature_hash": champion_hash,
            "universe_hash": prediction_manifest["universe_hash"],
            "evaluation_dates": int(comparison["trade_date"].nunique()),
            "prediction_rows": len(challenger_predictions),
            "labelled_rows": int(comparison["future_excess_ret"].notna().sum()),
            "overall_metrics": overall,
            "promotion_gate": gate,
            "scientific_scope": {
                "same_dates": True,
                "same_universe_rows": True,
                "same_feature_hash": True,
                "same_labels": True,
                "same_execution_rule": True,
                "labels_used_only_post_hoc": True,
                "unmatured_test_dates_excluded": True,
                "production_observation_loaded": False,
                "candidate_selection_loaded": False,
                "registry_modified": False,
                "automatic_promotion": False,
            },
        }
        manifest = _evaluation_manifest(
            identity=identity,
            run_id=run_id,
            champion=champion,
            challenger=challenger,
            prediction_manifest=prediction_manifest,
            summary=summary,
            processed_root=self.processed_root,
            config_path=self.config_path,
            evaluation_settings=self.settings.models.challenger_evaluation,
        )
        _publish(output_dir, summary, manifest, pd.DataFrame(metric_rows))
        return ChallengerEvaluationResult(
            run_id=run_id,
            champion_model_id=champion.model_id,
            challenger_model_id=challenger.model_id,
            horizon=horizon,
            labelled_rows=int(summary["labelled_rows"]),
            evaluation_dates=int(summary["evaluation_dates"]),
            eligible_for_manual_review=bool(gate["eligible_for_manual_review"]),
            output_dir=output_dir,
        )


def _load_mature_labels(
    processed_root: Path,
    horizon: int,
    *,
    dates: tuple[str, ...],
    maximum_mature_date: str,
) -> DataFrame:
    if not dates or max(dates) > maximum_mature_date:
        raise DataValidationError("evaluation requested immature final-test labels")
    label_dir = processed_root / "labels_forward"
    if not list(label_dir.glob("**/*.parquet")):
        raise DataValidationError("labels_forward artifact does not exist")
    label_glob = label_dir / "**" / "*.parquet"
    query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date,
               CAST(ts_code AS VARCHAR) AS ts_code,
               CAST(future_excess_ret AS DOUBLE) AS future_excess_ret,
               CAST(benchmark_forward_ret AS DOUBLE) AS benchmark_forward_ret
        FROM read_parquet('{label_glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(horizon AS INTEGER) = ?
          AND CAST(is_label_available AS BOOLEAN)
          AND CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
          AND future_excess_ret IS NOT NULL
          AND benchmark_forward_ret IS NOT NULL
        ORDER BY trade_date, ts_code
    """  # noqa: S608 -- fixed local dataset and parameterized values
    try:
        with duckdb.connect() as connection:
            labels = connection.execute(query, [horizon, min(dates), max(dates)]).fetch_df()
    except duckdb.Error as error:
        raise DataValidationError(f"cannot load challenger evaluation labels: {error}") from error
    labels = labels.loc[labels["trade_date"].astype(str).isin(dates)].reset_index(drop=True)
    if labels.empty:
        raise DataValidationError("challenger evaluation has no mature available labels")
    if labels.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("challenger evaluation labels contain duplicate keys")
    return labels


def _comparison_frame(challenger: DataFrame, champion: DataFrame, labels: DataFrame) -> DataFrame:
    keys = ["trade_date", "ts_code"]
    challenger_scores = challenger[keys + ["prediction_score"]].rename(
        columns={"prediction_score": "challenger_score"}
    )
    champion_scores = champion[keys + ["prediction_score"]].rename(
        columns={"prediction_score": "champion_score"}
    )
    scores = challenger_scores.merge(champion_scores, on=keys, validate="one_to_one")
    comparison = scores.merge(labels, on=keys, how="left", validate="one_to_one")
    if not comparison["future_excess_ret"].notna().any():
        raise DataValidationError("no prediction rows have mature labels")
    return comparison.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _require_identical_prediction_keys(challenger: DataFrame, champion: DataFrame) -> None:
    keys = ["trade_date", "ts_code"]
    if challenger.duplicated(keys).any() or champion.duplicated(keys).any():
        raise DataValidationError("model predictions contain duplicate keys")
    challenger_keys = pd.MultiIndex.from_frame(challenger[keys])
    champion_keys = pd.MultiIndex.from_frame(champion[keys])
    if len(challenger) != len(champion) or set(challenger_keys) != set(champion_keys):
        raise DataValidationError(
            "champion and challenger predictions do not use the same universe rows"
        )


def _validate_prediction_contract(
    manifest: dict[str, Any],
    challenger: RegisteredModel,
    *,
    processed_root: Path,
) -> tuple[int, tuple[tuple[str, str], ...]]:
    if manifest.get("artifact_name") != "challenger_predictions":
        raise DataValidationError("prediction artifact is not a challenger prediction")
    if manifest.get("model_id") != challenger.model_id:
        raise DataValidationError("challenger prediction model_id mismatch")
    if manifest.get("feature_hash") != challenger.feature_hash:
        raise DataValidationError("challenger prediction feature_hash mismatch")
    current_universe_hash = _file_hash(processed_root / "universe_daily" / "_manifest.json")
    if manifest.get("universe_hash") != current_universe_hash:
        raise DataValidationError("challenger prediction universe hash differs from current data")
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, dict):
        raise DataValidationError("challenger prediction input manifests are missing")
    if inputs.get("features_daily") != _file_hash(
        processed_root / "features_daily" / "_manifest.json"
    ):
        raise DataValidationError("challenger prediction features manifest has changed")
    if inputs.get("universe_daily") != current_universe_hash:
        raise DataValidationError("challenger prediction universe manifest has changed")
    horizon = _required_int(manifest, "horizon")
    if manifest.get("holding_period") != horizon:
        raise DataValidationError("challenger prediction holding period differs from horizon")
    if manifest.get("execution_rule") != "next_open":
        raise DataValidationError("challenger evaluation requires next_open execution")
    raw_ranges = manifest.get("evaluation_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise DataValidationError("challenger prediction evaluation ranges are missing")
    ranges = tuple(
        (str(item.get("start_date", "")), str(item.get("end_date", "")))
        for item in raw_ranges
        if isinstance(item, dict)
    )
    if len(ranges) != len(raw_ranges):
        raise DataValidationError("challenger prediction evaluation ranges are invalid")
    maximum_mature = str(manifest.get("maximum_mature_evaluation_date", ""))
    if any(end > maximum_mature for _, end in ranges):
        raise DataValidationError("challenger prediction includes immature test ranges")
    return horizon, ranges


def _load_predictions(path: Path, model_id: str, manifest: dict[str, Any]) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"challenger predictions do not exist: {path}")
    predictions = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "prediction_score", "model_id", "rank"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise DataValidationError(f"challenger predictions lack columns: {missing}")
    if predictions.empty:
        raise DataValidationError("challenger predictions are empty")
    if set(predictions["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("challenger predictions contain another model_id")
    if len(predictions) != int(manifest.get("prediction_rows", -1)):
        raise DataValidationError("challenger prediction row count differs from manifest")
    if predictions.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("challenger predictions contain duplicate keys")
    if not np.isfinite(pd.to_numeric(predictions["prediction_score"], errors="coerce")).all():
        raise DataValidationError("challenger predictions contain non-finite scores")
    return predictions.sort_values(["trade_date", "rank", "ts_code"], kind="mergesort").reset_index(
        drop=True
    )


def _evaluation_identity(
    champion: RegisteredModel,
    challenger: RegisteredModel,
    prediction_manifest: dict[str, Any],
    *,
    processed_root: Path,
    config_path: Path,
) -> str:
    git = current_git_info()
    payload = {
        "champion_model_id": champion.model_id,
        "champion_manifest_hash": _file_hash(Path(champion.artifact_path) / "manifest.json"),
        "challenger_model_id": challenger.model_id,
        "challenger_prediction_identity": prediction_manifest.get("prediction_identity"),
        "labels_manifest_hash": _file_hash(processed_root / "labels_forward" / "_manifest.json"),
        "feature_hash": champion.feature_hash,
        "universe_hash": prediction_manifest.get("universe_hash"),
        "config_hash": config_hash(config_path),
        "git_commit": git["commit"],
    }
    return _payload_hash(payload)


def _evaluation_manifest(
    *,
    identity: str,
    run_id: str,
    champion: RegisteredModel,
    challenger: RegisteredModel,
    prediction_manifest: dict[str, Any],
    summary: dict[str, Any],
    processed_root: Path,
    config_path: Path,
    evaluation_settings: ChallengerEvaluationSettings,
) -> dict[str, Any]:
    git = current_git_info()
    return {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "artifact_name": "challenger_evaluation_manifest",
        "evaluation_identity": identity,
        "run_id": run_id,
        "champion_model_id": champion.model_id,
        "challenger_model_id": challenger.model_id,
        "feature_hash": champion.feature_hash,
        "universe_hash": prediction_manifest["universe_hash"],
        "horizon": summary["horizon"],
        "holding_period": summary["holding_period"],
        "execution_rule": summary["execution_rule"],
        "evaluation_config": evaluation_settings.model_dump(mode="json"),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "config_path": str(config_path),
        "config_hash": config_hash(config_path),
        "input_manifests": {
            "champion_model": _file_hash(Path(champion.artifact_path) / "manifest.json"),
            "challenger_model": _file_hash(Path(challenger.artifact_path) / "manifest.json"),
            "challenger_predictions": prediction_manifest,
            "features_daily": _file_hash(processed_root / "features_daily" / "_manifest.json"),
            "universe_daily": _file_hash(processed_root / "universe_daily" / "_manifest.json"),
            "labels_forward": _file_hash(processed_root / "labels_forward" / "_manifest.json"),
        },
        "promotion_gate": summary["promotion_gate"],
        "registry_modified": False,
        "automatic_promotion": False,
    }


def _existing_result(output_dir: Path, identity: str) -> ChallengerEvaluationResult | None:
    if not output_dir.exists():
        return None
    manifest = _load_json(output_dir / "manifest.json", "challenger evaluation manifest")
    if manifest.get("evaluation_identity") != identity:
        raise DataValidationError(
            f"immutable challenger evaluation has a different identity: {output_dir}"
        )
    summary = _load_json(output_dir / "summary.json", "challenger evaluation summary")
    return ChallengerEvaluationResult(
        run_id=str(summary["run_id"]),
        champion_model_id=str(summary["champion_model_id"]),
        challenger_model_id=str(summary["challenger_model_id"]),
        horizon=int(summary["horizon"]),
        labelled_rows=int(summary["labelled_rows"]),
        evaluation_dates=int(summary["evaluation_dates"]),
        eligible_for_manual_review=bool(summary["promotion_gate"]["eligible_for_manual_review"]),
        output_dir=output_dir,
    )


def _publish(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    metrics: DataFrame,
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".evaluation-") as temporary:
        staging = Path(temporary)
        atomic_write_json(staging / "summary.json", summary)
        metrics.sort_values(["period_type", "period", "model_role"], kind="mergesort").to_csv(
            staging / "metrics.csv", index=False
        )
        (staging / "evaluation_report.md").write_text(_render_report(summary), encoding="utf-8")
        atomic_write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise DataValidationError(
                f"immutable challenger evaluation already exists: {output_dir}"
            )
        staging.rename(output_dir)


def _render_report(summary: dict[str, Any]) -> str:
    champion = summary["overall_metrics"][summary["champion_model_id"]]
    challenger = summary["overall_metrics"][summary["challenger_model_id"]]
    gate = summary["promotion_gate"]
    lines = [
        "# Challenger Evaluation",
        "",
        f"- Champion: `{summary['champion_model_id']}`",
        f"- Challenger: `{summary['challenger_model_id']}`",
        f"- Horizon / holding period: {summary['horizon']} trading days",
        f"- Execution: `{summary['execution_rule']}`",
        f"- Evaluation dates: {summary['evaluation_dates']}",
        f"- Labelled rows: {summary['labelled_rows']}",
        "",
        "## Overall Ranking",
        "",
        "| Model | Rank IC | ICIR | Positive IC ratio |",
        "| --- | ---: | ---: | ---: |",
        _ranking_line(summary["champion_model_id"], champion),
        _ranking_line(summary["challenger_model_id"], challenger),
        "",
        "## Manual Promotion Gate",
        "",
        f"Eligible for manual review: **{gate['eligible_for_manual_review']}**",
        "",
    ]
    lines.extend(
        f"- {criterion['name']}: value={criterion['value']} "
        f"threshold={criterion['threshold']} passed={criterion['passed']}"
        for criterion in gate["criteria"]
    )
    lines.extend(
        [
            "",
            "This report does not promote a model, modify the registry, "
            "or generate trading signals.",
            "",
        ]
    )
    return "\n".join(lines)


def _ranking_line(model_id: str, metrics: dict[str, Any]) -> str:
    return (
        f"| `{model_id}` | {_number(metrics['rank_ic'])} | "
        f"{_number(metrics['icir'])} | {_number(metrics['positive_ic_ratio'])} |"
    )


def _number(value: object) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value):.6f}"


def _find_model(registry: ModelRegistry, model_id: str) -> RegisteredModel:
    try:
        return next(model for model in registry.list_models() if model.model_id == model_id)
    except StopIteration as error:
        raise DataValidationError(f"model_id is not registered: {model_id}") from error


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"challenger evaluation {field} must be an integer")
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must be a JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"manifest source does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()
