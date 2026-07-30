"""Typed alert contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal


class AlertSeverity(StrEnum):
    """Supported operational severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertState(StrEnum):
    """Append-only alert lifecycle states."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True, slots=True)
class AlertRule:
    """One configuration-driven threshold rule."""

    alert_type: str
    source: Literal["health", "performance", "portfolio"]
    metric_name: str
    direction: Literal["lower", "upper"]
    warning_threshold: float
    critical_threshold: float
    optional: bool = False
    absolute_value: bool = False


@dataclass(frozen=True, slots=True)
class AlertCandidate:
    """One currently triggered logical condition before lifecycle handling."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    model_id: str | None
    portfolio_id: str | None
    metric_name: str
    metric_value: float
    threshold: float
    source_artifact_hash: str


@dataclass(frozen=True, slots=True)
class Alert:
    """One immutable daily alert lifecycle event."""

    alert_id: str
    alert_type: str
    severity: AlertSeverity
    status: AlertState
    first_seen: str
    last_seen: str
    model_id: str | None
    portfolio_id: str | None
    metric_name: str
    metric_value: float
    threshold: float
    source_artifact_hash: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/Parquet-compatible record."""

        payload = asdict(self)
        payload["severity"] = self.severity.value
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True, slots=True)
class AlertEvaluationResult:
    """Current threshold evaluation before lifecycle state transitions."""

    candidates: tuple[AlertCandidate, ...]
    evaluated_alert_ids: frozenset[str]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlertBuild:
    """Deterministic alert output and complete append-only history."""

    as_of: str
    alerts: tuple[Alert, ...]
    alerts_payload: dict[str, Any]
    report: str
    manifest: dict[str, Any]
    history: Any


@dataclass(frozen=True, slots=True)
class AlertMonitorResult:
    """Published alert artifact identity."""

    as_of: str
    output_dir: Path
    alert_count: int
    critical_count: int
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class AlertValidationResult:
    """Read-only alert validation or status result."""

    as_of: str
    valid: bool
    exists: bool
    alert_count: int
    warnings: tuple[str, ...] = ()
    error: str | None = None
