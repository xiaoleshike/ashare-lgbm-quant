"""Isolated CPU/CUDA Ranker benchmark and behavioral consistency comparison."""

from __future__ import annotations

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
from ashare_quant.models.compute.schemas import TrainingBackend
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.ranker import (
    feature_importance,
    fit_ranker,
    ranker_semantic_parameters,
)
from ashare_quant.models.ranker_data import RankerDataLoader
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json, current_git_info


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


class TrainingBackendBenchmarkService:
    """Benchmark one immutable model experiment without publishing a model artifact."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.models_root = settings.paths.models
        self.reports_root = settings.paths.reports
        self.output_root = self.reports_root / "training_backend_benchmarks"

    def run(
        self, *, backend: TrainingBackend, experiment_id: str
    ) -> TrainingBackendBenchmarkResult:
        source = self._source(experiment_id)
        manifest = _json(source / "manifest.json")
        features_payload = _json(source / "feature_list.json")
        features_raw = features_payload.get("features")
        if not isinstance(features_raw, list) or not features_raw:
            raise DataValidationError("benchmark source feature list is invalid")
        features = tuple(map(str, features_raw))
        feature_hash = feature_list_hash(features)
        declared_feature_hashes = [
            str(manifest[name])
            for name in ("feature_hash", "feature_list_hash")
            if manifest.get(name) is not None
        ]
        if not declared_feature_hashes or any(
            value != feature_hash for value in declared_feature_hashes
        ):
            raise DataValidationError("benchmark source feature hash mismatch")
        train_start = _date(manifest, "train_start")
        train_end = _date(manifest, "train_end")
        validation_start = _date(manifest, "validation_start")
        validation_end = _date(manifest, "validation_end")
        if train_end >= validation_start:
            raise DataValidationError("benchmark source train/validation periods overlap")
        horizon = int(manifest.get("horizon", manifest.get("label_horizon", -1)))
        if horizon not in {5, 10, 20, 60}:
            raise DataValidationError("benchmark source horizon is unsupported")
        semantic = ranker_semantic_parameters(self.settings.ranker)
        _validate_semantic_parameters(manifest, semantic)
        backend_settings = self.settings.ranker.training_backend.model_copy(
            update={
                "device_type": backend,
                "allow_cpu_fallback": False,
                "require_cuda_probe": True,
            }
        )
        runtime = resolve_training_backend(backend_settings)
        fold_identity = canonical_payload_hash(
            {
                "fold_manifest_hash": manifest.get("fold_manifest_hash"),
                "fold_id": manifest.get("fold_id"),
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
            }
        )
        source_identity = canonical_payload_hash(
            {
                "manifest_sha256": file_sha256(source / "manifest.json"),
                "feature_list_sha256": file_sha256(source / "feature_list.json"),
                "source_manifests": manifest.get("source_manifests"),
            }
        )
        logical = {
            "experiment_id": experiment_id,
            "source_identity": source_identity,
            "feature_hash": feature_hash,
            "fold_identity": fold_identity,
            "horizon": horizon,
            "semantic_parameter_hash": canonical_payload_hash(semantic),
            "random_seed": self.settings.ranker.random_seed,
            "training_compute": runtime.identity_payload(),
        }
        benchmark_id = f"backend_benchmark_{canonical_payload_hash(logical)[:24]}"
        output = self.output_root / benchmark_id
        if output.exists():
            self._validate_existing(output, benchmark_id)
            return TrainingBackendBenchmarkResult(benchmark_id, "COMPLETED", output, True)
        loader = RankerDataLoader(
            self.settings.paths.processed_data,
            horizon,
            self.settings.ranker.minimum_group_size,
        )
        train = loader.load(train_start, train_end, features, self.settings.ranker.relevance_grades)
        validation = loader.load(
            validation_start,
            validation_end,
            features,
            self.settings.ranker.relevance_grades,
        )
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
        metrics["feature_importance"] = feature_importance(model, features)
        prediction_frame = validation.frame.loc[:, ["trade_date", "ts_code"]].copy()
        prediction_frame["prediction"] = predictions
        benchmark = {
            "schema_version": 1,
            "artifact_name": "training_backend_benchmark",
            "benchmark_id": benchmark_id,
            **logical,
            "requested_device_type": runtime.requested_device_type,
            "effective_device_type": runtime.effective_device_type,
            "train_start": train_start,
            "train_end": train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "train_rows": len(train.frame),
            "validation_rows": len(validation.frame),
            "feature_count": len(features),
            "n_estimators": self.settings.ranker.n_estimators,
            "training_wall_seconds": training_seconds,
            "prediction_wall_seconds": prediction_seconds,
            "status": "COMPLETED",
        }
        environment = {
            "lightgbm_version": runtime.lightgbm_version,
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
            "feature_hash",
            "fold_identity",
            "horizon",
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "semantic_parameter_hash",
            "random_seed",
        )
        mismatches = [name for name in identity_fields if cpu.get(name) != cuda.get(name)]
        if mismatches:
            raise DataValidationError(f"benchmark source/parameter identity mismatch: {mismatches}")
        cpu_predictions = pd.read_parquet(cpu_dir / "predictions.parquet")
        cuda_predictions = pd.read_parquet(cuda_dir / "predictions.parquet")
        keys = ["trade_date", "ts_code"]
        if not cpu_predictions[keys].equals(cuda_predictions[keys]):
            raise DataValidationError("benchmark prediction row identity mismatch")
        pearson = float(cpu_predictions["prediction"].corr(cuda_predictions["prediction"]))
        spearman = float(
            cpu_predictions["prediction"].corr(cuda_predictions["prediction"], method="spearman")
        )
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
            "schema_version": 1,
            "artifact_name": "training_backend_comparison",
            "cpu_benchmark_id": cpu_benchmark_id,
            "cuda_benchmark_id": cuda_benchmark_id,
            "status": status,
            "prediction_pearson": pearson,
            "prediction_spearman": spearman,
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

    def _source(self, experiment_id: str) -> Path:
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
        return matches[0]

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
                    "schema_version": 1,
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
                    "schema_version": 1,
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
    if manifest.get("identity") != identity or manifest.get("manifest_written_last") is not True:
        raise DataValidationError(f"benchmark manifest identity mismatch: {output}")
    hashes = manifest.get("file_hashes")
    if not isinstance(hashes, dict):
        raise DataValidationError(f"benchmark manifest lacks file hashes: {output}")
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise DataValidationError("benchmark manifest hash entry is invalid")
        if file_sha256(output / name) != digest:
            raise DataValidationError(f"benchmark artifact hash mismatch: {name}")
    identity_file = "benchmark.json" if (output / "benchmark.json").is_file() else "comparison.json"
    identity_field = "benchmark_id" if identity_file == "benchmark.json" else "comparison_id"
    if _json(output / identity_file).get(identity_field) != identity:
        raise DataValidationError(f"benchmark payload identity mismatch: {output}")


def _validate_semantic_parameters(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    fixed = manifest.get("fixed_parameters")
    if not isinstance(fixed, dict):
        raise DataValidationError("benchmark source lacks fixed Ranker parameters")
    mismatches = [name for name, value in expected.items() if fixed.get(name) != value]
    if mismatches:
        raise DataValidationError(
            f"benchmark source semantic parameters differ from current settings: {mismatches}"
        )


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
