"""Backend parameter composition and fail-closed runtime resolution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ashare_quant.config.settings import TrainingBackendSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.compute.probe import probe_training_backend
from ashare_quant.models.compute.schemas import (
    TrainingBackendProbeResult,
    TrainingRuntimeMetadata,
)

Probe = Callable[[TrainingBackendSettings], TrainingBackendProbeResult]
LOGGER = logging.getLogger(__name__)


def training_backend_parameters(runtime: TrainingRuntimeMetadata) -> dict[str, Any]:
    """Return execution-only LightGBM parameters for the effective backend."""

    if runtime.effective_device_type == "cpu":
        return {"device_type": "cpu", "n_jobs": -1}
    return {"device_type": "cuda", "gpu_device_id": runtime.gpu_device_id}


def resolve_training_backend(
    settings: TrainingBackendSettings,
    *,
    probe: Probe = probe_training_backend,
) -> TrainingRuntimeMetadata:
    """Resolve the requested backend, making any fallback explicit and auditable."""

    result = probe(settings)
    version = result.lightgbm_version
    if settings.device_type == "cpu":
        if result.status != "AVAILABLE" or version is None:
            raise DataValidationError(f"CPU training backend probe failed: {result.message}")
        return _runtime(settings, result, effective="cpu")
    if result.status == "AVAILABLE" and result.effective_device_type == "cuda" and version:
        return _runtime(settings, result, effective="cuda")
    if not settings.allow_cpu_fallback:
        raise DataValidationError(
            f"CUDA training backend is unavailable and CPU fallback is disabled: {result.message}"
        )
    cpu = probe(settings.model_copy(update={"device_type": "cpu"}))
    if cpu.status != "AVAILABLE" or cpu.lightgbm_version is None:
        raise DataValidationError(
            "CUDA training backend is unavailable and CPU fallback probe failed: "
            f"cuda={result.message}; cpu={cpu.message}"
        )
    LOGGER.warning("CUDA unavailable; using explicitly configured CPU fallback: %s", result.message)
    return TrainingRuntimeMetadata(
        requested_device_type="cuda",
        effective_device_type="cpu",
        gpu_device_id=settings.gpu_device_id,
        allow_cpu_fallback=True,
        fallback_used=True,
        fallback_reason=result.message,
        lightgbm_version=cpu.lightgbm_version,
        device_name=result.device_name,
        runtime_information=result.runtime_information,
        probe_status=result.status,
        probe_message=result.message,
    )


def _runtime(
    settings: TrainingBackendSettings,
    result: TrainingBackendProbeResult,
    *,
    effective: str,
) -> TrainingRuntimeMetadata:
    assert result.lightgbm_version is not None
    return TrainingRuntimeMetadata(
        requested_device_type=settings.device_type,
        effective_device_type=effective,  # type: ignore[arg-type]
        gpu_device_id=settings.gpu_device_id,
        allow_cpu_fallback=settings.allow_cpu_fallback,
        fallback_used=False,
        lightgbm_version=result.lightgbm_version,
        device_name=result.device_name,
        runtime_information=result.runtime_information,
        probe_status=result.status,
        probe_message=result.message,
    )
