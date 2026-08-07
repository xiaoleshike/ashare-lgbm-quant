"""Typed contracts for LightGBM training compute provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TrainingBackend = Literal["cpu", "cuda"]
ProbeStatus = Literal["AVAILABLE", "UNAVAILABLE", "ERROR"]


class TrainingBackendProbeResult(BaseModel):
    """Read-only application-level backend capability result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    requested_device_type: TrainingBackend
    effective_device_type: TrainingBackend | None
    gpu_device_id: int = Field(ge=0)
    lightgbm_version: str | None
    status: ProbeStatus
    device_name: str | None = None
    runtime_information: str | None = None
    message: str
    checked_at: str


class TrainingRuntimeMetadata(BaseModel):
    """Immutable execution provenance attached to one fitted Ranker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    requested_device_type: TrainingBackend
    effective_device_type: TrainingBackend
    gpu_device_id: int = Field(ge=0)
    allow_cpu_fallback: bool
    fallback_used: bool
    fallback_reason: str | None = None
    lightgbm_version: str
    device_name: str | None = None
    runtime_information: str | None = None
    probe_status: ProbeStatus
    probe_message: str

    def identity_payload(self) -> dict[str, object]:
        """Return stable execution identity fields, excluding descriptive probe metadata."""

        return {
            "requested_device_type": self.requested_device_type,
            "effective_device_type": self.effective_device_type,
            "gpu_device_id": self.gpu_device_id,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "fallback_used": self.fallback_used,
            "lightgbm_version": self.lightgbm_version,
        }
