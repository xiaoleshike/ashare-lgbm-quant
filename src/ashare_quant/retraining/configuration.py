"""Versioned configuration for governed retraining trigger evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.models.shadow.storage import canonical_payload_hash


class ObservationRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    h5: int = Field(default=60, gt=0)
    h10: int = Field(default=60, gt=0)
    h20: int = Field(default=90, gt=0)
    h60: int = Field(default=120, gt=0)


class AlphaDecayTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    threshold: float = 0.7


class IcDeclineTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    rolling_window: Literal[20, 60, 120] = 60
    threshold: float = 0.0


class FeatureDriftTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    psi_threshold: float = Field(default=0.2, ge=0)


class CriticalAlertTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True


class RetrainingTriggers(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha_decay: AlphaDecayTrigger = Field(default_factory=AlphaDecayTrigger)
    ic_decline: IcDeclineTrigger = Field(default_factory=IcDeclineTrigger)
    feature_drift: FeatureDriftTrigger = Field(default_factory=FeatureDriftTrigger)
    critical_alert: CriticalAlertTrigger = Field(default_factory=CriticalAlertTrigger)


class RetrainingLifecyclePolicy(BaseModel):
    """Operational limits and prospective evidence requirements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    max_parallel_runs: Literal[1] = 1
    max_daily_training_runs: int = Field(default=1, ge=1)
    cooldown_days: int = Field(default=30, ge=0)
    minimum_prospective_sessions: ObservationRequirements = Field(
        default_factory=ObservationRequirements
    )

    def required_sessions(self, horizon: int) -> int:
        value = getattr(self.minimum_prospective_sessions, f"h{horizon}", None)
        if not isinstance(value, int):
            raise ValueError(f"unsupported lifecycle horizon: {horizon}")
        return value


class RetrainingQualificationPolicy(BaseModel):
    """Operator-controlled safeguards for real lifecycle qualification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    phase: Literal["2.8.2G"] = "2.8.2G"
    require_clean_worktree: bool = False
    require_dry_run_ready: bool = True
    require_readiness_ready: bool = True
    allow_real_training: bool = False
    require_manual_stage_advance: Literal[True] = True
    allow_real_shadow: bool = False
    maximum_qualification_runs_per_day: int = Field(default=1, ge=1)
    allowed_stop_points: tuple[
        Literal[
            "preflight", "dry-run", "readiness", "training", "validation", "shadow", "observation"
        ],
        ...,
    ] = (
        "preflight",
        "dry-run",
        "readiness",
        "training",
        "validation",
        "shadow",
        "observation",
    )
    minimum_free_disk_bytes: int | None = Field(default=None, ge=0)
    minimum_available_memory_bytes: int | None = Field(default=None, ge=0)
    promotion_forbidden: Literal[True] = True
    trading_forbidden: Literal[True] = True

    @property
    def policy_hash(self) -> str:
        return canonical_payload_hash(self.model_dump(mode="json"))


class RetrainingPolicy(BaseModel):
    """Deterministic policy used only to create training requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    policy_version: str = "v1"
    enabled: bool = True
    cooldown_days: int = Field(default=30, ge=0)
    minimum_observation_sessions: ObservationRequirements = Field(
        default_factory=ObservationRequirements
    )
    triggers: RetrainingTriggers = Field(default_factory=RetrainingTriggers)
    lifecycle: RetrainingLifecyclePolicy = Field(default_factory=RetrainingLifecyclePolicy)
    qualification: RetrainingQualificationPolicy = Field(
        default_factory=RetrainingQualificationPolicy
    )

    @property
    def policy_hash(self) -> str:
        # Lifecycle scheduling does not alter the trigger decision frozen in old requests.
        return canonical_payload_hash(
            self.model_dump(mode="json", exclude={"lifecycle", "qualification"})
        )

    @property
    def lifecycle_policy_hash(self) -> str:
        return canonical_payload_hash(self.lifecycle.model_dump(mode="json"))

    def required_sessions(self, horizon: int) -> int:
        value = getattr(self.minimum_observation_sessions, f"h{horizon}", None)
        if not isinstance(value, int):
            raise ValueError(f"unsupported retraining horizon: {horizon}")
        return value


def load_retraining_policy(path: Path) -> RetrainingPolicy:
    """Load a strict standalone policy without affecting the production config hash."""

    if not path.is_file():
        raise ValueError(f"retraining policy does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("retraining"), dict):
        raise ValueError("retraining policy must contain a `retraining` mapping")
    return RetrainingPolicy.model_validate(payload["retraining"])
