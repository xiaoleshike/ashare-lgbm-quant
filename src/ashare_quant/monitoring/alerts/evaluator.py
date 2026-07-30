"""Pure threshold evaluation over existing monitoring metrics."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.monitoring.alerts.schemas import (
    AlertCandidate,
    AlertEvaluationResult,
    AlertRule,
    AlertSeverity,
)

type DataFrame = pd.DataFrame


def evaluate_alerts(
    *,
    health: dict[str, Any],
    performance: DataFrame,
    portfolios: DataFrame,
    rules: tuple[AlertRule, ...],
    source_artifact_hash: str,
) -> AlertEvaluationResult:
    """Evaluate enabled rules without side effects or source-data access."""

    _validate_inputs(health, performance, portfolios)
    candidates: list[AlertCandidate] = []
    evaluated: set[str] = set()
    warnings: list[str] = []
    for rule in rules:
        measurements = _measurements(rule, health, performance, portfolios)
        if not measurements:
            if rule.source == "portfolio" and portfolios.empty:
                warnings.append("insufficient observations: no monitored portfolios")
                continue
            if rule.optional:
                warnings.append(f"optional metric unavailable: {rule.metric_name}")
                continue
            raise DataValidationError(f"required alert metric is unavailable: {rule.metric_name}")
        for model_id, portfolio_id, metric_name, raw_value in measurements:
            if raw_value is None or not math.isfinite(raw_value):
                if rule.optional:
                    warnings.append(
                        f"optional metric unavailable: {rule.metric_name}: "
                        f"model={model_id} portfolio={portfolio_id}"
                    )
                    continue
                raise DataValidationError(
                    f"required alert metric is non-finite: {rule.metric_name}"
                )
            value = abs(raw_value) if rule.absolute_value else raw_value
            alert_id = deterministic_alert_id(
                rule.alert_type,
                model_id,
                portfolio_id,
                metric_name,
            )
            evaluated.add(alert_id)
            severity, threshold = _severity(rule, value)
            if severity is None or threshold is None:
                continue
            candidates.append(
                AlertCandidate(
                    alert_id=alert_id,
                    alert_type=rule.alert_type,
                    severity=severity,
                    model_id=model_id,
                    portfolio_id=portfolio_id,
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=threshold,
                    source_artifact_hash=source_artifact_hash,
                )
            )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.alert_id,
                -_severity_rank(item.severity),
            ),
        )
    )
    return AlertEvaluationResult(ordered, frozenset(evaluated), tuple(sorted(set(warnings))))


def deterministic_alert_id(
    alert_type: str,
    model_id: str | None,
    portfolio_id: str | None,
    metric_name: str,
) -> str:
    """Hash only the logical alert identity specified by the contract."""

    return canonical_payload_hash(
        {
            "alert_type": alert_type,
            "model_id": model_id,
            "portfolio_id": portfolio_id,
            "metric_name": metric_name,
        }
    )


def _measurements(
    rule: AlertRule,
    health: dict[str, Any],
    performance: DataFrame,
    portfolios: DataFrame,
) -> list[tuple[str | None, str | None, str, float | None]]:
    if rule.source == "performance":
        rows: list[tuple[str | None, str | None, str, float | None]] = []
        for row in performance.sort_values(["model_id", "horizon"], kind="mergesort").to_dict(
            "records"
        ):
            horizon = int(row["horizon"])
            metric_name = f"{rule.metric_name}_h{horizon}"
            if rule.metric_name == "rank_ic_delta":
                value = _difference(row.get("rolling_20_ic_mean"), row.get("rank_ic"))
            else:
                value = _float_or_none(row.get(rule.metric_name))
            rows.append((str(row["model_id"]), None, metric_name, value))
        return rows
    if rule.source == "portfolio":
        metric_column = "drawdown" if rule.metric_name == "current_drawdown" else rule.metric_name
        return [
            (
                None,
                str(row["portfolio_id"]),
                rule.metric_name,
                _float_or_none(row.get(metric_column)),
            )
            for row in portfolios.sort_values("portfolio_id", kind="mergesort").to_dict("records")
        ]
    value = _health_metric(health, rule.metric_name)
    return [(str(health["model_id"]), None, rule.metric_name, value)]


def _health_metric(health: dict[str, Any], metric_name: str) -> float | None:
    if metric_name == "prediction_coverage":
        denominator = _float_or_none(health.get("model_universe_size"))
        numerator = _float_or_none(health.get("prediction_count"))
        if denominator is None or denominator == 0.0 or numerator is None:
            return None
        return numerator / denominator
    if metric_name in {
        "maximum_feature_psi",
        "maximum_feature_ks",
        "maximum_missing_ratio_drift",
    }:
        reference = health.get("drift_reference")
        metrics = reference.get("metrics") if isinstance(reference, dict) else None
        return _float_or_none(metrics.get(metric_name)) if isinstance(metrics, dict) else None
    return _float_or_none(health.get(metric_name))


def _severity(
    rule: AlertRule,
    value: float,
) -> tuple[AlertSeverity | None, float | None]:
    if rule.direction == "lower":
        if value < rule.critical_threshold:
            return AlertSeverity.CRITICAL, rule.critical_threshold
        if value < rule.warning_threshold:
            return AlertSeverity.WARNING, rule.warning_threshold
        return None, None
    if value > rule.critical_threshold:
        return AlertSeverity.CRITICAL, rule.critical_threshold
    if value > rule.warning_threshold:
        return AlertSeverity.WARNING, rule.warning_threshold
    return None, None


def _validate_inputs(
    health: dict[str, Any],
    performance: DataFrame,
    portfolios: DataFrame,
) -> None:
    health_required = {
        "as_of",
        "model_id",
        "model_universe_size",
        "prediction_count",
        "score_std",
        "unique_score_ratio",
    }
    missing_health = sorted(health_required - set(health))
    if missing_health:
        raise DataValidationError(f"health metrics lack alert columns: {missing_health}")
    performance_required = {
        "model_id",
        "horizon",
        "rank_ic",
        "rolling_20_ic_mean",
        "alpha_decay_ratio",
    }
    missing_performance = sorted(performance_required - set(performance.columns))
    if missing_performance:
        raise DataValidationError(f"performance metrics lack alert columns: {missing_performance}")
    portfolio_required = {
        "portfolio_id",
        "drawdown",
        "max_drawdown",
        "max_position_weight",
        "top5_concentration",
        "rejected_order_ratio",
        "failed_execution_ratio",
    }
    missing_portfolio = sorted(portfolio_required - set(portfolios.columns))
    if missing_portfolio and not portfolios.empty:
        raise DataValidationError(f"portfolio metrics lack alert columns: {missing_portfolio}")


def _difference(left: object, right: object) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _severity_rank(value: AlertSeverity) -> int:
    return {
        AlertSeverity.INFO: 1,
        AlertSeverity.WARNING: 2,
        AlertSeverity.CRITICAL: 3,
    }[value]
