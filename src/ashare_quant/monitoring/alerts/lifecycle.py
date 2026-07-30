"""Append-only NEW, ACTIVE, and RECOVERED alert transitions."""

from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.alerts.schemas import (
    Alert,
    AlertEvaluationResult,
    AlertSeverity,
    AlertState,
)

type DataFrame = pd.DataFrame


def apply_lifecycle(
    evaluation: AlertEvaluationResult,
    history: DataFrame,
    as_of: str,
) -> tuple[Alert, ...]:
    """Produce one event per evaluated logical alert without duplicate daily events."""

    previous = _latest_states(history, as_of)
    current = {candidate.alert_id: candidate for candidate in evaluation.candidates}
    events: list[Alert] = []
    for alert_id in sorted(evaluation.evaluated_alert_ids):
        candidate = current.get(alert_id)
        prior = previous.get(alert_id)
        if candidate is not None:
            active_before = prior is not None and prior["status"] in {
                AlertState.NEW.value,
                AlertState.ACTIVE.value,
            }
            first_seen = str(prior["first_seen"]) if active_before and prior is not None else as_of
            events.append(
                Alert(
                    alert_id=alert_id,
                    alert_type=candidate.alert_type,
                    severity=candidate.severity,
                    status=AlertState.ACTIVE if active_before else AlertState.NEW,
                    first_seen=first_seen,
                    last_seen=as_of,
                    model_id=candidate.model_id,
                    portfolio_id=candidate.portfolio_id,
                    metric_name=candidate.metric_name,
                    metric_value=candidate.metric_value,
                    threshold=candidate.threshold,
                    source_artifact_hash=candidate.source_artifact_hash,
                    created_at=_deterministic_created_at(as_of),
                )
            )
        elif prior is not None and prior["status"] in {
            AlertState.NEW.value,
            AlertState.ACTIVE.value,
        }:
            events.append(_recovered(prior, as_of))
    return tuple(sorted(events, key=lambda item: item.alert_id))


def append_history(history: DataFrame, alerts: tuple[Alert, ...]) -> DataFrame:
    """Append new events and reject conflicting duplicate identities."""

    additions = pd.DataFrame.from_records([alert.to_dict() for alert in alerts])
    if additions.empty:
        return history.copy()
    combined = pd.concat([history, additions], ignore_index=True)
    duplicate_key = ["alert_id", "last_seen"]
    duplicates = combined.loc[combined.duplicated(duplicate_key, keep=False)]
    if not duplicates.empty:
        for _, group in duplicates.groupby(duplicate_key, sort=True):
            normalized = group.astype(object).where(group.notna(), None).to_dict("records")
            if any(record != normalized[0] for record in normalized[1:]):
                raise DataValidationError("duplicate alert identity has conflicting content")
        combined = combined.drop_duplicates(duplicate_key, keep="first")
    return combined.sort_values(["last_seen", "alert_id"], kind="mergesort").reset_index(drop=True)


def _latest_states(history: DataFrame, as_of: str) -> dict[str, dict[str, Any]]:
    if history.empty:
        return {}
    if (history["last_seen"].astype(str) >= as_of).any():
        raise DataValidationError("alert history contains current or future lifecycle events")
    ordered = history.sort_values(["last_seen", "alert_id"], kind="mergesort")
    records = cast(
        list[dict[str, Any]],
        ordered.groupby("alert_id", sort=False).tail(1).to_dict("records"),
    )
    return {str(row["alert_id"]): row for row in records}


def _recovered(prior: dict[str, Any], as_of: str) -> Alert:
    return Alert(
        alert_id=str(prior["alert_id"]),
        alert_type=str(prior["alert_type"]),
        severity=AlertSeverity.INFO,
        status=AlertState.RECOVERED,
        first_seen=str(prior["first_seen"]),
        last_seen=as_of,
        model_id=_optional_string(prior.get("model_id")),
        portfolio_id=_optional_string(prior.get("portfolio_id")),
        metric_name=str(prior["metric_name"]),
        metric_value=float(prior["metric_value"]),
        threshold=float(prior["threshold"]),
        source_artifact_hash=str(prior["source_artifact_hash"]),
        created_at=_deterministic_created_at(as_of),
    )


def _optional_string(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _deterministic_created_at(as_of: str) -> str:
    return f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}T00:00:00+08:00"
