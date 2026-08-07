"""Read-only LightGBM CPU/CUDA application-level capability probes."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from ashare_quant.config.settings import TrainingBackendSettings
from ashare_quant.models.compute.schemas import TrainingBackendProbeResult


class Predictor(Protocol):
    def predict(self, data: object) -> object: ...


class LightGBMModule(Protocol):
    __version__: str
    Dataset: Callable[..., object]
    train: Callable[..., Predictor]


SmokeTrainer = Callable[[LightGBMModule, int], None]


def probe_training_backend(
    settings: TrainingBackendSettings,
    *,
    smoke_trainer: SmokeTrainer | None = None,
    now: Callable[[], datetime] | None = None,
) -> TrainingBackendProbeResult:
    """Probe LightGBM itself; CUDA presence outside LightGBM is insufficient."""

    checked = (now or (lambda: datetime.now(UTC)))()
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=UTC)
    try:
        lightgbm = cast(LightGBMModule, importlib.import_module("lightgbm"))
        version = str(lightgbm.__version__)
    except Exception as error:
        return _result(settings, "ERROR", None, None, f"LightGBM import failed: {error}", checked)
    if settings.device_type == "cpu":
        return _result(settings, "AVAILABLE", "cpu", version, "CPU backend is available", checked)
    if not settings.require_cuda_probe:
        return _result(
            settings,
            "AVAILABLE",
            "cuda",
            version,
            "CUDA smoke probe is disabled by configuration",
            checked,
        )
    try:
        (smoke_trainer or _cuda_smoke_train)(lightgbm, settings.gpu_device_id)
    except Exception as error:
        return _result(
            settings,
            "UNAVAILABLE",
            None,
            version,
            f"LightGBM CUDA smoke training failed: {type(error).__name__}: {error}",
            checked,
        )
    return _result(
        settings,
        "AVAILABLE",
        "cuda",
        version,
        "LightGBM CUDA smoke training succeeded",
        checked,
    )


def _cuda_smoke_train(lightgbm: LightGBMModule, gpu_device_id: int) -> None:
    features = np.asarray(
        [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 2, 1, 2, 0, 1], dtype=np.int32)
    dataset = lightgbm.Dataset(features, label=labels, group=[3, 3], free_raw_data=True)
    booster = lightgbm.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "device_type": "cuda",
            "gpu_device_id": gpu_device_id,
            "verbosity": -1,
            "min_data_in_leaf": 1,
            "num_leaves": 3,
        },
        dataset,
        num_boost_round=1,
    )
    predictions = np.asarray(cast(npt.ArrayLike, booster.predict(features)), dtype=float)
    if len(predictions) != len(features) or not np.isfinite(predictions).all():
        raise RuntimeError("CUDA smoke predictions are invalid")


def _result(
    settings: TrainingBackendSettings,
    status: str,
    effective: str | None,
    version: str | None,
    message: str,
    checked: datetime,
) -> TrainingBackendProbeResult:
    return TrainingBackendProbeResult(
        requested_device_type=settings.device_type,
        effective_device_type=effective,  # type: ignore[arg-type]
        gpu_device_id=settings.gpu_device_id,
        lightgbm_version=version,
        status=status,  # type: ignore[arg-type]
        message=message,
        checked_at=checked.astimezone(UTC).isoformat(),
    )
