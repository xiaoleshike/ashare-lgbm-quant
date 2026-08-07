"""CPU-only tests for governed LightGBM training compute backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ashare_quant.cli import main
from ashare_quant.config import load_settings
from ashare_quant.config.settings import (
    AppSettings,
    PathSettings,
    RankerSettings,
    TrainingBackendSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.compute.backend import (
    resolve_training_backend,
    training_backend_parameters,
)
from ashare_quant.models.compute.benchmark import TrainingBackendBenchmarkService
from ashare_quant.models.compute.probe import probe_training_backend
from ashare_quant.models.compute.schemas import (
    ProbeStatus,
    TrainingBackend,
    TrainingBackendProbeResult,
    TrainingRuntimeMetadata,
)
from ashare_quant.models.ranker import (
    fit_ranker,
    ranker_parameters,
    ranker_semantic_parameters,
    training_runtime_metadata,
)
from ashare_quant.models.ranker_data import RankerDataset
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.utils.manifest import atomic_write_json


def _probe(
    requested: TrainingBackend,
    status: ProbeStatus = "AVAILABLE",
    effective: TrainingBackend | None = None,
    message: str = "fixture",
) -> TrainingBackendProbeResult:
    return TrainingBackendProbeResult(
        requested_device_type=requested,
        effective_device_type=effective or (requested if status == "AVAILABLE" else None),
        gpu_device_id=0,
        lightgbm_version="4.fixture",
        status=status,
        message=message,
        checked_at="2026-08-07T00:00:00+00:00",
    )


def _runtime(device: TrainingBackend) -> TrainingRuntimeMetadata:
    return TrainingRuntimeMetadata(
        requested_device_type=device,
        effective_device_type=device,
        gpu_device_id=0,
        allow_cpu_fallback=False,
        fallback_used=False,
        lightgbm_version="4.fixture",
        probe_status="AVAILABLE",
        probe_message="fixture",
    )


def test_training_backend_settings_are_strict_and_default_to_cpu() -> None:
    configured = load_settings("config/default.yaml").ranker.training_backend

    assert configured.device_type == "cpu"
    assert configured.allow_cpu_fallback is False
    assert TrainingBackendSettings(device_type="cuda", gpu_device_id=2).gpu_device_id == 2
    with pytest.raises(ValidationError):
        TrainingBackendSettings(device_type="gpu")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TrainingBackendSettings(gpu_device_id=-1)
    with pytest.raises(ValidationError):
        TrainingBackendSettings(unknown=True)  # type: ignore[call-arg]


def test_backend_parameters_change_only_execution_fields() -> None:
    settings = RankerSettings()
    semantic = ranker_semantic_parameters(settings)
    cpu = ranker_parameters(settings, _runtime("cpu"))
    cuda = ranker_parameters(settings, _runtime("cuda"))

    assert training_backend_parameters(_runtime("cpu")) == {
        "device_type": "cpu",
        "n_jobs": -1,
    }
    assert training_backend_parameters(_runtime("cuda")) == {
        "device_type": "cuda",
        "gpu_device_id": 0,
    }
    assert {name: cpu[name] for name in semantic} == semantic
    assert {name: cuda[name] for name in semantic} == semantic
    assert "max_bin" not in cpu
    assert "max_bin" not in cuda


def test_probe_reports_cpu_and_cuda_without_requiring_hardware() -> None:
    cpu = probe_training_backend(TrainingBackendSettings())
    called: list[int] = []
    cuda = probe_training_backend(
        TrainingBackendSettings(device_type="cuda", gpu_device_id=3),
        smoke_trainer=lambda _module, device: called.append(device),
    )
    unavailable = probe_training_backend(
        TrainingBackendSettings(device_type="cuda"),
        smoke_trainer=lambda _module, _device: (_ for _ in ()).throw(RuntimeError("no CUDA")),
    )

    assert cpu.status == "AVAILABLE" and cpu.effective_device_type == "cpu"
    assert cuda.status == "AVAILABLE" and cuda.effective_device_type == "cuda"
    assert called == [3]
    assert unavailable.status == "UNAVAILABLE"
    assert unavailable.effective_device_type is None
    assert "no CUDA" in unavailable.message


def test_cuda_resolution_fails_closed_or_records_explicit_fallback() -> None:
    unavailable = lambda settings: _probe(  # noqa: E731
        settings.device_type,
        "UNAVAILABLE" if settings.device_type == "cuda" else "AVAILABLE",
        message="CUDA learner unavailable",
    )
    with pytest.raises(DataValidationError, match="fallback is disabled"):
        resolve_training_backend(TrainingBackendSettings(device_type="cuda"), probe=unavailable)

    runtime = resolve_training_backend(
        TrainingBackendSettings(device_type="cuda", allow_cpu_fallback=True),
        probe=unavailable,
    )

    assert runtime.requested_device_type == "cuda"
    assert runtime.effective_device_type == "cpu"
    assert runtime.fallback_used is True
    assert runtime.fallback_reason == "CUDA learner unavailable"


def test_common_ranker_fit_receives_backend_and_attaches_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeRanker:
        def __init__(self, **parameters: Any) -> None:
            captured.update(parameters)

        def fit(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr("ashare_quant.models.ranker.lgb.LGBMRanker", FakeRanker)
    frame = pd.DataFrame(
        {
            "trade_date": ["20260102", "20260102"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "signal": np.asarray([0.1, 0.2], dtype=np.float32),
            "future_excess_ret_5d": [0.01, 0.02],
            "relevance": np.asarray([0, 1], dtype=np.int32),
        }
    )
    dataset = RankerDataset(frame, ("signal",))

    model = fit_ranker(dataset, dataset, RankerSettings(), _runtime("cuda"))

    assert captured["device_type"] == "cuda"
    assert captured["gpu_device_id"] == 0
    assert "n_jobs" not in captured
    assert training_runtime_metadata(model).effective_device_type == "cuda"


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        paths=PathSettings(
            raw_data=tmp_path / "raw",
            processed_data=tmp_path / "processed",
            models=tmp_path / "models",
            reports=tmp_path / "reports",
        )
    )


def _benchmark(
    root: Path,
    benchmark_id: str,
    backend: str,
    predictions: list[float],
    *,
    rank_ic: float = 0.10,
    ndcg: float = 0.80,
    feature_hash: str = "f" * 64,
    identity_updates: dict[str, object] | None = None,
) -> None:
    output = root / "reports/training_backend_benchmarks" / benchmark_id
    common = {
        "source_identity": "s" * 64,
        "feature_hash": feature_hash,
        "fold_identity": "d" * 64,
        "horizon": 10,
        "train_start": "20200101",
        "train_end": "20211231",
        "validation_start": "20220101",
        "validation_end": "20221231",
        "semantic_parameter_hash": "p" * 64,
        "random_seed": 42,
    }
    common.update(identity_updates or {})
    atomic_write_json(
        output / "benchmark.json",
        {
            "benchmark_id": benchmark_id,
            "effective_device_type": backend,
            "training_wall_seconds": 2.0 if backend == "cpu" else 1.0,
            **common,
        },
    )
    atomic_write_json(
        output / "metrics.json",
        {
            "rank_ic": rank_ic,
            "ndcg_at_10": ndcg,
            "ndcg_at_50": ndcg,
            "top_10pct_mean_future_excess_ret": 0.01,
            "feature_importance": [
                {"feature": "f1", "gain": 1.0, "split": 2},
                {"feature": "f2", "gain": 2.0, "split": 1},
            ],
        },
    )
    pd.DataFrame(
        {
            "trade_date": ["20220103"] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "prediction": predictions,
        }
    ).to_parquet(output / "predictions.parquet", index=False)
    files = ("benchmark.json", "metrics.json", "predictions.parquet")
    atomic_write_json(
        output / "manifest.json",
        {
            "identity": benchmark_id,
            "manifest_written_last": True,
            "file_hashes": {name: file_sha256(output / name) for name in files},
        },
    )


def test_benchmark_comparison_checks_correctness_independently_of_speed(tmp_path: Path) -> None:
    service = TrainingBackendBenchmarkService(_settings(tmp_path))
    _benchmark(tmp_path, "cpu", "cpu", [1.0, 2.0, 3.0])
    _benchmark(tmp_path, "cuda", "cuda", [1.0, 2.0, 3.0])

    passed = service.compare(cpu_benchmark_id="cpu", cuda_benchmark_id="cuda")
    repeated = service.compare(cpu_benchmark_id="cpu", cuda_benchmark_id="cuda")

    assert passed.status == "PASS"
    assert repeated.comparison_id == passed.comparison_id
    assert repeated.idempotent is True
    comparison = json.loads((passed.output_dir / "comparison.json").read_text())
    assert comparison["prediction_pearson"] == pytest.approx(1.0)
    assert comparison["training_speedup"] == pytest.approx(2.0)


def test_benchmark_metric_failure_is_not_overridden_by_speedup(tmp_path: Path) -> None:
    service = TrainingBackendBenchmarkService(_settings(tmp_path))
    _benchmark(tmp_path, "cpu", "cpu", [1.0, 2.0, 3.0], rank_ic=0.10)
    _benchmark(tmp_path, "cuda", "cuda", [3.0, 2.0, 1.0], rank_ic=0.20)

    result = service.compare(cpu_benchmark_id="cpu", cuda_benchmark_id="cuda")

    assert result.status == "FAIL"


@pytest.mark.parametrize(
    "identity_updates",
    [
        {"feature_hash": "x" * 64},
        {"fold_identity": "x" * 64},
        {"train_end": "20201231"},
        {"random_seed": 7},
        {"semantic_parameter_hash": "x" * 64},
    ],
)
def test_benchmark_rejects_different_source_or_parameter_identity(
    tmp_path: Path,
    identity_updates: dict[str, object],
) -> None:
    service = TrainingBackendBenchmarkService(_settings(tmp_path))
    _benchmark(tmp_path, "cpu", "cpu", [1.0, 2.0, 3.0])
    _benchmark(
        tmp_path,
        "cuda",
        "cuda",
        [1.0, 2.0, 3.0],
        identity_updates=identity_updates,
    )

    with pytest.raises(DataValidationError, match="identity mismatch"):
        service.compare(cpu_benchmark_id="cpu", cuda_benchmark_id="cuda")


def test_benchmark_ndcg_delta_can_fail_consistency(tmp_path: Path) -> None:
    service = TrainingBackendBenchmarkService(_settings(tmp_path))
    _benchmark(tmp_path, "cpu", "cpu", [1.0, 2.0, 3.0], ndcg=0.80)
    _benchmark(tmp_path, "cuda", "cuda", [1.0, 2.0, 3.0], ndcg=0.70)

    result = service.compare(cpu_benchmark_id="cpu", cuda_benchmark_id="cuda")

    assert result.status == "FAIL"


def test_training_backend_status_cli_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    available = _probe("cpu")
    monkeypatch.setattr("ashare_quant.cli.probe_training_backend", lambda _settings: available)
    monkeypatch.setattr(
        "ashare_quant.cli.resolve_training_backend", lambda _settings: _runtime("cpu")
    )
    assert main(["--config", "config/default.yaml", "models", "training-backend-status"]) == 0

    monkeypatch.setattr(
        "ashare_quant.cli.resolve_training_backend",
        lambda _settings: (_ for _ in ()).throw(DataValidationError("unavailable")),
    )
    unavailable = available.model_copy(update={"status": "UNAVAILABLE"})
    monkeypatch.setattr("ashare_quant.cli.probe_training_backend", lambda _settings: unavailable)
    assert main(["--config", "config/default.yaml", "models", "training-backend-status"]) == 1
