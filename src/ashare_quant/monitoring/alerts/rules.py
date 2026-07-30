"""Configuration-driven alert rule definitions."""

from __future__ import annotations

from ashare_quant.config.settings import MonitoringAlertSettings
from ashare_quant.monitoring.alerts.schemas import AlertRule


def configured_rules(settings: MonitoringAlertSettings) -> tuple[AlertRule, ...]:
    """Translate validated settings into deterministic rules."""

    rules: list[AlertRule] = []
    if settings.alpha_decay.enabled:
        rules.append(
            AlertRule(
                "model_alpha_decay",
                "performance",
                "alpha_decay_ratio",
                "lower",
                settings.alpha_decay.warning,
                settings.alpha_decay.critical,
                optional=True,
            )
        )
    if settings.rank_ic_decline.enabled:
        rules.append(
            AlertRule(
                "rank_ic_decline",
                "performance",
                "rank_ic_delta",
                "lower",
                settings.rank_ic_decline.warning,
                settings.rank_ic_decline.critical,
                optional=True,
            )
        )
    if settings.score_collapse.enabled:
        rules.extend(
            (
                AlertRule(
                    "score_collapse",
                    "health",
                    "score_std",
                    "lower",
                    settings.score_collapse.score_std_warning,
                    settings.score_collapse.score_std_critical,
                ),
                AlertRule(
                    "score_collapse",
                    "health",
                    "unique_score_ratio",
                    "lower",
                    settings.score_collapse.unique_ratio_warning,
                    settings.score_collapse.unique_ratio_critical,
                ),
            )
        )
    if settings.feature_drift.enabled:
        rules.extend(
            (
                AlertRule(
                    "feature_drift",
                    "health",
                    "maximum_feature_psi",
                    "upper",
                    settings.feature_drift.psi_warning,
                    settings.feature_drift.psi_critical,
                    optional=True,
                ),
                AlertRule(
                    "feature_drift",
                    "health",
                    "maximum_feature_ks",
                    "upper",
                    settings.feature_drift.ks_warning,
                    settings.feature_drift.ks_critical,
                    optional=True,
                ),
                AlertRule(
                    "feature_drift",
                    "health",
                    "maximum_missing_ratio_drift",
                    "upper",
                    settings.feature_drift.missing_ratio_warning,
                    settings.feature_drift.missing_ratio_critical,
                    optional=True,
                    absolute_value=True,
                ),
            )
        )
    if settings.universe_coverage.enabled:
        rules.extend(
            (
                AlertRule(
                    "universe_coverage",
                    "health",
                    "prediction_coverage",
                    "lower",
                    settings.universe_coverage.prediction_warning,
                    settings.universe_coverage.prediction_critical,
                ),
                AlertRule(
                    "universe_coverage",
                    "health",
                    "universe_size_deviation_ratio",
                    "upper",
                    settings.universe_coverage.universe_deviation_warning,
                    settings.universe_coverage.universe_deviation_critical,
                    optional=True,
                    absolute_value=True,
                ),
            )
        )
    if settings.drawdown.enabled:
        for metric in ("current_drawdown", "max_drawdown"):
            rules.append(
                AlertRule(
                    "portfolio_drawdown",
                    "portfolio",
                    metric,
                    "upper",
                    settings.drawdown.warning,
                    settings.drawdown.critical,
                    absolute_value=True,
                )
            )
    if settings.concentration.enabled:
        rules.extend(
            (
                AlertRule(
                    "concentration_risk",
                    "portfolio",
                    "max_position_weight",
                    "upper",
                    settings.concentration.max_weight_warning,
                    settings.concentration.max_weight_critical,
                ),
                AlertRule(
                    "concentration_risk",
                    "portfolio",
                    "top5_concentration",
                    "upper",
                    settings.concentration.top5_warning,
                    settings.concentration.top5_critical,
                ),
                AlertRule(
                    "concentration_risk",
                    "portfolio",
                    "industry_concentration",
                    "upper",
                    settings.concentration.industry_warning,
                    settings.concentration.industry_critical,
                    optional=True,
                ),
            )
        )
    if settings.execution_quality.enabled:
        rules.extend(
            (
                AlertRule(
                    "execution_quality",
                    "portfolio",
                    "rejected_order_ratio",
                    "upper",
                    settings.execution_quality.rejected_warning,
                    settings.execution_quality.rejected_critical,
                ),
                AlertRule(
                    "execution_quality",
                    "portfolio",
                    "failed_execution_ratio",
                    "upper",
                    settings.execution_quality.failed_warning,
                    settings.execution_quality.failed_critical,
                ),
            )
        )
    return tuple(rules)
