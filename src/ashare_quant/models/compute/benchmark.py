"""Isolated CPU/CUDA Ranker benchmark and behavioral consistency comparison."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.compute.backend import resolve_training_backend
from ashare_quant.models.compute.probe import lightgbm_build_identity
from ashare_quant.models.compute.schemas import TrainingBackend
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.feature_provenance import (
    feature_provenance_hash,
    validate_governed_feature_set,
)
from ashare_quant.models.ranker import (
    feature_importance,
    fit_ranker,
    ranker_semantic_parameters,
)
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.models.walk_forward_evaluation import (
    EVALUATION_CONTRACT_VERSION,
    validate_completed_walk_forward_artifact,
)
from ashare_quant.utils.manifest import atomic_write_json, current_git_info

BENCHMARK_SCHEMA_VERSION = 2
BENCHMARK_FILES = frozenset(
    {"benchmark.json", "metrics.json", "environment.json", "predictions.parquet", "report.md"}
)
COMPARISON_FILES = frozenset({"comparison.json", "report.md"})


@dataclass(frozen=True, slots=True)
class TrainingBackendBenchmarkResult:
    benchmark_id: str
    status: str
    output_dir: Path
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class TrainingBackendComparisonResult:
    comparison_id: str
    status: str
    output_dir: Path
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    """Portable training/validation contract for one benchmark input."""

    source_kind: str
    source_id: str
    features: tuple[str, ...]
    feature_hash: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    horizon: int
    semantic_parameters: dict[str, Any]
    fold_identity: str
    source_identity: str
    lineage: dict[str, Any]


class TrainingBackendBenchmarkService:
    """Benchmark one immutable model experiment without publishing a model artifact."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.models_root = settings.paths.models
        self.reports_root = settings.paths.reports
        self.output_root = self.reports_root / "training_backend_benchmarks"

    def run(
        self,
        *,
        backend: TrainingBackend,
        experiment_id: str | None = None,
        walk_forward_run_id: str | None = None,
        fold_id: str | None = None,
        feature_provenance_path: Path | None = None,
    ) -> TrainingBackendBenchmarkResult:
        if (experiment_id is None) == (walk_forward_run_id is None):
            raise DataValidationError(
                "benchmark requires exactly one of experiment_id or walk_forward_run_id"
            )
        source = (
            self._legacy_source(experiment_id)
            if experiment_id is not None
            else self._walk_forward_source(
                str(walk_forward_run_id), fold_id, feature_provenance_path
            )
        )
        if source.train_end >= source.validation_start:
            raise DataValidationError("benchmark source train/validation periods overlap")
        if source.horizon not in {5, 10, 20, 60}:
            raise DataValidationError("benchmark source horizon is unsupported")
        semantic = ranker_semantic_parameters(self.settings.ranker)
        if source.semantic_parameters != semantic:
            mismatches = [
                name
                for name, value in semantic.items()
                if source.semantic_parameters.get(name) != value
            ]
            raise DataValidationError(
                f"benchmark source semantic parameters differ from current settings: {mismatches}"
            )
        load_started = perf_counter()
        loader = RankerDataLoader(
            self.settings.paths.processed_data,
            source.horizon,
            self.settings.ranker.minimum_group_size,
        )
        train = loader.load(
            source.train_start,
            source.train_end,
            source.features,
            self.settings.ranker.relevance_grades,
        )
        validation = loader.load(
            source.validation_start,
            source.validation_end,
            source.features,
            self.settings.ranker.relevance_grades,
        )
        data_load_seconds = perf_counter() - load_started
        actual_data_identity = _actual_data_identity(train, validation, source.features)
        backend_settings = self.settings.ranker.training_backend.model_copy(
            update={
                "device_type": backend,
                "allow_cpu_fallback": False,
                "require_cuda_probe": True,
            }
        )
        runtime = resolve_training_backend(backend_settings)
        build_identity = lightgbm_build_identity(runtime.lightgbm_version)
        logical = {
            "source_kind": source.source_kind,
            "source_id": source.source_id,
            "source_identity": source.source_identity,
            "actual_data_identity": actual_data_identity,
            "actual_data_identity_hash": canonical_payload_hash(actual_data_identity),
            "feature_hash": source.feature_hash,
            "fold_identity": source.fold_identity,
            "horizon": source.horizon,
            "semantic_parameter_hash": canonical_payload_hash(semantic),
            "random_seed": self.settings.ranker.random_seed,
            "lightgbm_version": runtime.lightgbm_version,
            "lightgbm_build_identity": build_identity,
            "training_compute": runtime.identity_payload(),
        }
        benchmark_id = f"backend_benchmark_{canonical_payload_hash(logical)[:24]}"
        output = self.output_root / benchmark_id
        if output.exists():
            self._validate_existing(output, benchmark_id)
            return TrainingBackendBenchmarkResult(benchmark_id, "COMPLETED", output, True)
        total_started = perf_counter()
        started = perf_counter()
        model = fit_ranker(train, validation, self.settings.ranker, runtime=runtime)
        training_seconds = perf_counter() - started
        started = perf_counter()
        predictions = np.asarray(model.predict(validation.features), dtype=float)
        prediction_seconds = perf_counter() - started
        metrics = evaluate_ranker(
            validation,
            predictions,
            self.settings.ranker.ndcg_at,
            self.settings.ranker.portfolio_fractions,
        )
        metrics["feature_importance"] = feature_importance(model, source.features)
        prediction_frame = validation.frame.loc[:, ["trade_date", "ts_code"]].copy()
        prediction_frame["prediction"] = predictions
        benchmark = {
            "schema_version": 2,
            "artifact_name": "training_backend_benchmark",
            "benchmark_id": benchmark_id,
            **logical,
            "requested_device_type": runtime.requested_device_type,
            "effective_device_type": runtime.effective_device_type,
            "source_lineage": source.lineage,
            "train_start": source.train_start,
            "train_end": source.train_end,
            "validation_start": source.validation_start,
            "validation_end": source.validation_end,
            "train_rows": len(train.frame),
            "validation_rows": len(validation.frame),
            "feature_count": len(source.features),
            "n_estimators": self.settings.ranker.n_estimators,
            "data_load_wall_seconds": data_load_seconds,
            "training_wall_seconds": training_seconds,
            "prediction_wall_seconds": prediction_seconds,
            "total_benchmark_wall_seconds": perf_counter() - total_started + data_load_seconds,
            "probe_and_warmup_in_timing": False,
            "status": "COMPLETED",
        }
        environment = {
            "lightgbm_version": runtime.lightgbm_version,
            "lightgbm_build_identity": build_identity,
            "requested_device_type": runtime.requested_device_type,
            "effective_device_type": runtime.effective_device_type,
            "gpu_device_id": runtime.gpu_device_id,
            "device_name": runtime.device_name,
            "runtime_information": runtime.runtime_information,
            "git": current_git_info(),
        }
        self._publish(output, benchmark, metrics, environment, prediction_frame)
        return TrainingBackendBenchmarkResult(benchmark_id, "COMPLETED", output)

    def compare(
        self, *, cpu_benchmark_id: str, cuda_benchmark_id: str
    ) -> TrainingBackendComparisonResult:
        cpu_dir = self.output_root / cpu_benchmark_id
        cuda_dir = self.output_root / cuda_benchmark_id
        self._validate_existing(cpu_dir, cpu_benchmark_id)
        self._validate_existing(cuda_dir, cuda_benchmark_id)
        cpu = _json(cpu_dir / "benchmark.json")
        cuda = _json(cuda_dir / "benchmark.json")
        if cpu.get("effective_device_type") != "cpu" or cuda.get("effective_device_type") != "cuda":
            raise DataValidationError("comparison requires effective CPU and CUDA benchmarks")
        identity_fields = (
            "source_identity",
            "actual_data_identity_hash",
            "feature_hash",
            "fold_identity",
            "horizon",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "semantic_parameter_hash",
            "random_seed",
            "lightgbm_version",
            "lightgbm_build_identity",
        )
        mismatches = [name for name in identity_fields if cpu.get(name) != cuda.get(name)]
        if mismatches:
            raise DataValidationError(f"benchmark source/parameter identity mismatch: {mismatches}")
        cpu_predictions = pd.read_parquet(cpu_dir / "predictions.parquet")
        cuda_predictions = pd.read_parquet(cuda_dir / "predictions.parquet")
        _validate_predictions(cpu_predictions, "CPU")
        _validate_predictions(cuda_predictions, "CUDA")
        keys = ["trade_date", "ts_code"]
        if not cpu_predictions[keys].equals(cuda_predictions[keys]):
            raise DataValidationError("benchmark prediction row identity mismatch")
        pearson = float(cpu_predictions["prediction"].corr(cuda_predictions["prediction"]))
        spearman = float(
            cpu_predictions["prediction"].corr(cuda_predictions["prediction"], method="spearman")
        )
        if not np.isfinite([pearson, spearman]).all():
            raise DataValidationError("benchmark prediction correlations are non-finite")
        daily_consistency = _daily_prediction_consistency(cpu_predictions, cuda_predictions)
        top_n_overlap = _daily_top_n_overlap(cpu_predictions, cuda_predictions, (10, 20, 50))
        cpu_metrics = _json(cpu_dir / "metrics.json")
        cuda_metrics = _json(cuda_dir / "metrics.json")
        rank_delta = abs(float(cpu_metrics["rank_ic"]) - float(cuda_metrics["rank_ic"]))
        ndcg_deltas = {
            cutoff: abs(
                float(cpu_metrics[f"ndcg_at_{cutoff}"]) - float(cuda_metrics[f"ndcg_at_{cutoff}"])
            )
            for cutoff in (10, 50)
        }
        importance = pd.DataFrame(cpu_metrics["feature_importance"]).merge(
            pd.DataFrame(cuda_metrics["feature_importance"]),
            on="feature",
            suffixes=("_cpu", "_cuda"),
            validate="one_to_one",
        )
        importance_spearman = float(
            importance["gain_cpu"].corr(importance["gain_cuda"], method="spearman")
        )
        portfolio_deltas = {
            name: abs(float(cpu_metrics[name]) - float(cuda_metrics[name]))
            for name in cpu_metrics
            if name.startswith("top_") and name.endswith("_mean_future_excess_ret")
        }
        consistency = self.settings.ranker.training_backend.consistency
        checks = {
            "prediction_pearson": pearson >= consistency.minimum_prediction_pearson,
            "prediction_spearman": spearman >= consistency.minimum_prediction_spearman,
            "rank_ic_delta": rank_delta <= consistency.maximum_rank_ic_absolute_delta,
            "ndcg_at_10_delta": ndcg_deltas[10] <= consistency.maximum_ndcg_absolute_delta,
            "ndcg_at_50_delta": ndcg_deltas[50] <= consistency.maximum_ndcg_absolute_delta,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        speedup = float(cpu["training_wall_seconds"]) / float(cuda["training_wall_seconds"])
        comparison = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "artifact_name": "training_backend_comparison",
            "cpu_benchmark_id": cpu_benchmark_id,
            "cuda_benchmark_id": cuda_benchmark_id,
            "status": status,
            "prediction_pearson": pearson,
            "prediction_spearman": spearman,
            "daily_prediction_consistency": daily_consistency,
            "daily_top_n_overlap": top_n_overlap,
            "rank_ic_absolute_delta": rank_delta,
            "ndcg_at_10_absolute_delta": ndcg_deltas[10],
            "ndcg_at_50_absolute_delta": ndcg_deltas[50],
            "feature_importance_spearman": importance_spearman,
            "portfolio_proxy_absolute_deltas": portfolio_deltas,
            "training_speedup": speedup,
            "checks": checks,
            "consistency_policy": consistency.model_dump(mode="json"),
        }
        comparison_id = f"backend_comparison_{canonical_payload_hash(comparison)[:24]}"
        comparison["comparison_id"] = comparison_id
        output = self.output_root / "comparisons" / comparison_id
        if output.exists():
            _validate_manifest(output, comparison_id)
            return TrainingBackendComparisonResult(comparison_id, status, output, True)
        self._publish_json_bundle(output, comparison_id, comparison)
        return TrainingBackendComparisonResult(comparison_id, status, output)

    def _legacy_source(self, experiment_id: str) -> BenchmarkSource:
        if Path(experiment_id).name != experiment_id:
            raise DataValidationError("benchmark experiment_id must be one path component")
        candidates = (
            self.models_root / experiment_id,
            self.models_root / "challengers" / experiment_id,
        )
        matches = [path for path in candidates if (path / "manifest.json").is_file()]
        if len(matches) != 1:
            raise DataValidationError(
                "benchmark experiment must resolve to one immutable model artifact: "
                f"{experiment_id}"
            )
        source = matches[0]
        manifest = _json(source / "manifest.json")
        features_payload = _json(source / "feature_list.json")
        features_raw = features_payload.get("features")
        if not isinstance(features_raw, list) or not features_raw:
            raise DataValidationError("benchmark source feature list is invalid")
        features = tuple(map(str, features_raw))
        feature_hash = feature_list_hash(features)
        declared = [
            str(manifest[name])
            for name in ("feature_hash", "feature_list_hash")
            if manifest.get(name) is not None
        ]
        if not declared or any(value != feature_hash for value in declared):
            raise DataValidationError("benchmark source feature hash mismatch")
        semantic = ranker_semantic_parameters(self.settings.ranker)
        _validate_semantic_parameters(manifest, semantic)
        dates = {
            name: _date(manifest, name)
            for name in ("train_start", "train_end", "validation_start", "validation_end")
        }
        horizon = int(manifest.get("horizon", manifest.get("label_horizon", -1)))
        source_identity = canonical_payload_hash(
            {
                "manifest_sha256": file_sha256(source / "manifest.json"),
                "feature_list_sha256": file_sha256(source / "feature_list.json"),
                "source_manifests": manifest.get("source_manifests"),
            }
        )
        fold_identity = canonical_payload_hash(
            {
                "fold_manifest_hash": manifest.get("fold_manifest_hash"),
                "fold_id": manifest.get("fold_id"),
                **dates,
            }
        )
        return BenchmarkSource(
            source_kind="legacy_model_artifact",
            source_id=experiment_id,
            features=features,
            feature_hash=feature_hash,
            horizon=horizon,
            semantic_parameters=semantic,
            fold_identity=fold_identity,
            source_identity=source_identity,
            lineage={"model_manifest_sha256": file_sha256(source / "manifest.json")},
            **dates,
        )

    def _walk_forward_source(
        self,
        run_id: str,
        fold_id: str | None,
        feature_provenance_path: Path | None,
    ) -> BenchmarkSource:
        if Path(run_id).name != run_id:
            raise DataValidationError("walk-forward run ID must be one path component")
        if feature_provenance_path is None:
            raise DataValidationError("walk-forward benchmark requires --feature-provenance")
        root = self.reports_root / "research" / "walk_forward" / run_id
        root_manifest = validate_completed_walk_forward_artifact(root)
        if (
            root_manifest.get("schema_version") != 3
            or root_manifest.get("evaluation_contract_version") != EVALUATION_CONTRACT_VERSION
        ):
            raise DataValidationError(
                "walk-forward benchmark requires current evaluation-contract evidence"
            )
        provenance = validate_governed_feature_set(
            feature_provenance_path,
            reports_root=self.reports_root,
        )
        provenance_hash = feature_provenance_hash(feature_provenance_path)
        expected_feature = (
            root_manifest.get("feature_set_id"),
            root_manifest.get("feature_set_hash"),
            root_manifest.get("feature_provenance_hash"),
        )
        actual_feature = (
            provenance.feature_set_id,
            provenance.feature_list_hash,
            provenance_hash,
        )
        if actual_feature != expected_feature:
            raise DataValidationError("benchmark walk-forward feature provenance mismatch")
        experiment = root_manifest.get("experiment")
        if not isinstance(experiment, dict):
            raise DataValidationError("walk-forward root lacks experiment definition")
        period = experiment.get("selection_period")
        references = period.get("folds") if isinstance(period, dict) else None
        if not isinstance(references, list) or not references:
            raise DataValidationError("walk-forward experiment has no selection folds")
        valid_references = [item for item in references if isinstance(item, dict)]
        if len(valid_references) != len(references):
            raise DataValidationError("walk-forward selection fold references are invalid")
        if fold_id is None:
            selected_reference = max(
                valid_references,
                key=lambda item: (
                    str(item.get("evaluation_start", "")),
                    str(item.get("fold_id", "")),
                ),
            )
            fold_id = str(selected_reference.get("fold_id", ""))
        matches = [item for item in valid_references if item.get("fold_id") == fold_id]
        if len(matches) != 1:
            raise DataValidationError("benchmark fold must belong to the selection period")
        fold_dir = root / "folds" / fold_id
        fold_manifest = _json(fold_dir / "manifest.json")
        fold = fold_manifest.get("fold")
        semantic = fold_manifest.get("semantic_parameters")
        if not isinstance(fold, dict) or not isinstance(semantic, dict):
            raise DataValidationError("walk-forward fold manifest lacks training contract")
        if (
            fold_manifest.get("feature_set_id"),
            fold_manifest.get("feature_set_hash"),
            fold_manifest.get("feature_provenance_hash"),
        ) != actual_feature:
            raise DataValidationError("benchmark fold feature provenance mismatch")
        horizon = fold_manifest.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise DataValidationError("walk-forward fold horizon is invalid")
        dates = {
            name: _date(fold, name)
            for name in ("train_start", "train_end", "validation_start", "validation_end")
        }
        fold_manifest_hash = file_sha256(fold_dir / "manifest.json")
        root_manifest_hash = file_sha256(root / "manifest.json")
        source_identity = canonical_payload_hash(
            {
                "root_manifest_hash": root_manifest_hash,
                "fold_manifest_hash": fold_manifest_hash,
                "feature_provenance_hash": provenance_hash,
                "processed_source_identity": fold_manifest.get("source_identity"),
            }
        )
        return BenchmarkSource(
            source_kind="walk_forward_fold",
            source_id=f"{run_id}:{fold_id}",
            features=provenance.features,
            feature_hash=provenance.feature_list_hash,
            horizon=horizon,
            semantic_parameters=cast(dict[str, Any], semantic),
            fold_identity=canonical_payload_hash(
                {
                    "run_id": run_id,
                    "fold_id": fold_id,
                    "fold_manifest_hash": fold_manifest_hash,
                    **dates,
                }
            ),
            source_identity=source_identity,
            lineage={
                "walk_forward_run_id": run_id,
                "fold_id": fold_id,
                "root_manifest_sha256": root_manifest_hash,
                "fold_manifest_sha256": fold_manifest_hash,
                "feature_provenance_sha256": provenance_hash,
                "feature_set_id": provenance.feature_set_id,
            },
            **dates,
        )

    def _publish(
        self,
        output: Path,
        benchmark: dict[str, Any],
        metrics: dict[str, object],
        environment: dict[str, Any],
        predictions: pd.DataFrame,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.tmp-"))
        try:
            atomic_write_json(staging / "benchmark.json", benchmark)
            atomic_write_json(staging / "metrics.json", metrics)
            atomic_write_json(staging / "environment.json", environment)
            predictions.to_parquet(staging / "predictions.parquet", index=False)
            (staging / "report.md").write_text(
                f"# Training Backend Benchmark\n\nStatus: COMPLETED\n\nBackend: "
                f"{benchmark['effective_device_type']}\n",
                encoding="utf-8",
            )
            hashes = {
                name: file_sha256(staging / name)
                for name in (
                    "benchmark.json",
                    "metrics.json",
                    "environment.json",
                    "predictions.parquet",
                    "report.md",
                )
            }
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "artifact_name": "training_backend_benchmark_manifest",
                    "identity": benchmark["benchmark_id"],
                    "file_hashes": hashes,
                    "manifest_written_last": True,
                },
            )
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _publish_json_bundle(self, output: Path, identity: str, payload: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{identity}.tmp-"))
        try:
            atomic_write_json(staging / "comparison.json", payload)
            (staging / "report.md").write_text(
                f"# CPU/CUDA Training Consistency\n\nStatus: {payload['status']}\n",
                encoding="utf-8",
            )
            hashes = {
                name: file_sha256(staging / name) for name in ("comparison.json", "report.md")
            }
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "artifact_name": "training_backend_comparison_manifest",
                    "identity": identity,
                    "file_hashes": hashes,
                    "manifest_written_last": True,
                },
            )
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _validate_existing(self, output: Path, identity: str) -> None:
        _validate_manifest(output, identity)


def _validate_manifest(output: Path, identity: str) -> None:
    manifest = _json(output / "manifest.json")
    if (
        manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or manifest.get("identity") != identity
        or manifest.get("manifest_written_last") is not True
    ):
        raise DataValidationError(f"benchmark manifest identity mismatch: {output}")
    hashes = manifest.get("file_hashes")
    identity_file = "benchmark.json" if (output / "benchmark.json").is_file() else "comparison.json"
    required = BENCHMARK_FILES if identity_file == "benchmark.json" else COMPARISON_FILES
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise DataValidationError(f"benchmark manifest lacks file hashes: {output}")
    for name, digest in hashes.items():
        relative = Path(str(name))
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or relative.is_absolute()
            or len(relative.parts) != 1
            or ".." in relative.parts
        ):
            raise DataValidationError("benchmark manifest hash entry is invalid")
        if file_sha256(output / name) != digest:
            raise DataValidationError(f"benchmark artifact hash mismatch: {name}")
    if {item.name for item in output.iterdir()} != required | {"manifest.json"}:
        raise DataValidationError(f"benchmark artifact set mismatch: {output}")
    identity_field = "benchmark_id" if identity_file == "benchmark.json" else "comparison_id"
    payload = _json(output / identity_file)
    if payload.get(identity_field) != identity:
        raise DataValidationError(f"benchmark payload identity mismatch: {output}")
    if identity_file == "benchmark.json":
        if (
            payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION
            or payload.get("status") != "COMPLETED"
        ):
            raise DataValidationError(f"benchmark payload contract mismatch: {output}")
        _validate_predictions(pd.read_parquet(output / "predictions.parquet"), "stored")


def _validate_semantic_parameters(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    fixed = manifest.get("fixed_parameters")
    if not isinstance(fixed, dict):
        raise DataValidationError("benchmark source lacks fixed Ranker parameters")
    mismatches = [name for name, value in expected.items() if fixed.get(name) != value]
    if mismatches:
        raise DataValidationError(
            f"benchmark source semantic parameters differ from current settings: {mismatches}"
        )


def _actual_data_identity(
    train: RankerDataset,
    validation: RankerDataset,
    features: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "features": list(features),
        "train": _dataset_identity(train.frame, train.groups, features),
        "validation": _dataset_identity(validation.frame, validation.groups, features),
    }


def _dataset_identity(
    frame: pd.DataFrame,
    groups: list[int],
    features: tuple[str, ...],
) -> dict[str, Any]:
    columns = [
        "trade_date",
        "ts_code",
        *features,
        "future_excess_ret_5d",
        "relevance",
    ]
    if frame.duplicated(subset=["trade_date", "ts_code"]).any():
        raise DataValidationError("benchmark dataset contains duplicate security keys")
    selected = frame.loc[:, columns]
    hashed = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    content_hash = hashlib.sha256(hashed.to_numpy(dtype="uint64").tobytes()).hexdigest()
    return {
        "rows": len(selected),
        "columns": columns,
        "dtypes": [str(selected[column].dtype) for column in columns],
        "groups": groups,
        "content_hash": content_hash,
    }


def _validate_predictions(frame: pd.DataFrame, description: str) -> None:
    required = {"trade_date", "ts_code", "prediction"}
    if set(frame.columns) != required:
        raise DataValidationError(f"{description} benchmark prediction schema is invalid")
    if frame.duplicated(subset=["trade_date", "ts_code"]).any():
        raise DataValidationError(f"{description} benchmark prediction keys are duplicated")
    values = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise DataValidationError(f"{description} benchmark predictions are non-finite or empty")


def _daily_prediction_consistency(cpu: pd.DataFrame, cuda: pd.DataFrame) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for trade_date, cpu_daily in cpu.groupby("trade_date", sort=True):
        cuda_daily = cuda.loc[cuda["trade_date"].astype(str) == str(trade_date)]
        pearson = float(cpu_daily["prediction"].corr(cuda_daily["prediction"]))
        spearman = float(cpu_daily["prediction"].corr(cuda_daily["prediction"], method="spearman"))
        records.append({"trade_date": str(trade_date), "pearson": pearson, "spearman": spearman})
    valid = [item for item in records if np.isfinite([item["pearson"], item["spearman"]]).all()]
    if not valid:
        raise DataValidationError("benchmark has no finite daily prediction correlations")
    return {
        "dates": len(records),
        "finite_dates": len(valid),
        "pearson_mean": float(np.mean([item["pearson"] for item in valid])),
        "pearson_minimum": float(np.min([item["pearson"] for item in valid])),
        "spearman_mean": float(np.mean([item["spearman"] for item in valid])),
        "spearman_minimum": float(np.min([item["spearman"] for item in valid])),
        "by_date": records,
    }


def _daily_top_n_overlap(
    cpu: pd.DataFrame,
    cuda: pd.DataFrame,
    cutoffs: tuple[int, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for cutoff in cutoffs:
        values: list[float] = []
        for trade_date, cpu_daily in cpu.groupby("trade_date", sort=True):
            cuda_daily = cuda.loc[cuda["trade_date"].astype(str) == str(trade_date)]
            count = min(cutoff, len(cpu_daily))
            cpu_top = set(cpu_daily.nlargest(count, "prediction", keep="first")["ts_code"])
            cuda_top = set(cuda_daily.nlargest(count, "prediction", keep="first")["ts_code"])
            values.append(float(len(cpu_top & cuda_top) / count))
        output[str(cutoff)] = {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "by_date": values,
        }
    return output


def _date(manifest: dict[str, Any], name: str) -> str:
    value = str(manifest.get(name, ""))
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"benchmark source lacks {name}")
    return value


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required benchmark artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid benchmark JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"benchmark JSON must contain an object: {path}")
    return cast(dict[str, Any], payload)
