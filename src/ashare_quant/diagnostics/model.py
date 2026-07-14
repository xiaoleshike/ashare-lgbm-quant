"""Fixed-parameter LightGBM diagnostics and validation-period evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from ashare_quant.config.settings import DiagnosticSettings

type DataFrame = pd.DataFrame


def train_diagnostic_model(
    train: DataFrame, feature_names: Sequence[str], settings: DiagnosticSettings
) -> lgb.Booster:
    """Fit a deterministic regression model for importance diagnostics."""

    features = clean_matrix(train, feature_names)
    target = pd.to_numeric(train["target"], errors="coerce")
    valid = target.notna() & np.isfinite(target)
    dataset = lgb.Dataset(
        features.loc[valid],
        label=target.loc[valid],
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    params: dict[str, object] = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "learning_rate": settings.lgbm_learning_rate,
        "num_leaves": settings.lgbm_num_leaves,
        "min_data_in_leaf": settings.lgbm_min_data_in_leaf,
        "feature_fraction": settings.lgbm_feature_fraction,
        "bagging_fraction": settings.lgbm_bagging_fraction,
        "bagging_freq": settings.lgbm_bagging_freq,
        "seed": settings.random_seed,
        "feature_fraction_seed": settings.random_seed,
        "bagging_seed": settings.random_seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    return lgb.train(params, dataset, num_boost_round=settings.lgbm_num_boost_round)


def model_importance(model: lgb.Booster) -> DataFrame:
    """Return split and gain importance from one fitted model."""

    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    total_gain = float(gain.sum())
    return pd.DataFrame(
        {
            "feature": model.feature_name(),
            "split_importance": split.astype(int),
            "gain_importance": gain,
            "gain_fraction": gain / total_gain if total_gain > 0 else np.zeros_like(gain),
        }
    )


def evaluate_model(
    model: lgb.Booster,
    frame: DataFrame,
    feature_names: Sequence[str],
    settings: DiagnosticSettings,
) -> dict[str, float]:
    """Evaluate cross-sectional predictions and a frictionless top-bucket diagnostic."""

    if frame.empty:
        return empty_evaluation()
    working = frame[["trade_date", "ts_code", "target", *feature_names]].copy()
    working["prediction"] = model.predict(clean_matrix(working, feature_names))
    daily_records: list[dict[str, float | str]] = []
    previous_names: set[str] | None = None
    turnovers: list[float] = []
    for trade_date, daily in working.groupby("trade_date", sort=True):
        valid = daily["target"].notna() & daily["prediction"].notna()
        sample = daily.loc[valid]
        if len(sample) < settings.minimum_daily_cross_section:
            continue
        rank_ic = sample["prediction"].corr(sample["target"], method="spearman")
        count = max(1, int(math.ceil(len(sample) * settings.top_fraction)))
        top = sample.nlargest(count, "prediction", keep="first")
        selected = set(top["ts_code"].astype(str))
        if previous_names is not None:
            denominator = max(len(previous_names), len(selected), 1)
            turnovers.append(1.0 - len(previous_names & selected) / denominator)
        previous_names = selected
        daily_records.append(
            {
                "trade_date": str(trade_date),
                "rank_ic": float(rank_ic),
                "top_excess_ret": float(top["target"].mean()),
            }
        )
    if not daily_records:
        return empty_evaluation()
    daily_metrics = pd.DataFrame.from_records(daily_records)
    top_returns = daily_metrics["top_excess_ret"]
    equity = (1.0 + top_returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    rank_std = float(daily_metrics["rank_ic"].std(ddof=1))
    return_std = float(top_returns.std(ddof=1))
    yearly = daily_metrics.assign(year=daily_metrics["trade_date"].str[:4]).groupby("year")
    yearly_positive = yearly["rank_ic"].mean() > 0
    return {
        "days": float(len(daily_metrics)),
        "rank_ic": float(daily_metrics["rank_ic"].mean()),
        "rank_icir": safe_ratio(float(daily_metrics["rank_ic"].mean()), rank_std),
        "top_decile_excess_return": float(top_returns.mean()),
        "sharpe": safe_ratio(float(top_returns.mean()), return_std)
        * math.sqrt(settings.annualization_days),
        "maximum_drawdown": float(drawdown.min()),
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "year_positive_ratio": float(yearly_positive.mean()),
    }


def permutation_importance(
    model: lgb.Booster,
    validation: DataFrame,
    feature_names: Sequence[str],
    settings: DiagnosticSettings,
) -> DataFrame:
    """Measure validation Rank-IC decline after within-date feature permutation."""

    baseline = evaluate_model(model, validation, feature_names, settings)["rank_ic"]
    random = np.random.default_rng(settings.random_seed)
    records = []
    for feature in feature_names:
        declines = []
        for _ in range(settings.permutation_repeats):
            permuted = validation.copy()
            permuted[feature] = permuted.groupby("trade_date", sort=False)[feature].transform(
                lambda values: random.permutation(values.to_numpy())
            )
            score = evaluate_model(model, permuted, feature_names, settings)["rank_ic"]
            declines.append(baseline - score)
        records.append(
            {
                "feature": feature,
                "baseline_rank_ic": baseline,
                "permutation_importance": float(np.mean(declines)),
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_feature_sets(
    train: DataFrame,
    validation: DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    settings: DiagnosticSettings,
) -> DataFrame:
    """Train and evaluate named feature sets strictly on train/validation periods."""

    records = []
    for name, features in feature_sets.items():
        if not features:
            continue
        model = train_diagnostic_model(train, features, settings)
        records.append(
            {
                "set_name": name,
                "feature_count": len(features),
                **evaluate_model(model, validation, features, settings),
            }
        )
    return pd.DataFrame.from_records(records)


def choose_feature_count(validation_results: DataFrame) -> str:
    """Choose by balanced metric ranks, never by one validation return."""

    if validation_results.empty:
        raise ValueError("no validation feature-set results are available")
    working = validation_results.copy()
    higher_is_better = [
        "rank_ic",
        "rank_icir",
        "top_decile_excess_return",
        "sharpe",
        "maximum_drawdown",
        "year_positive_ratio",
    ]
    ranks = [working[column].rank(ascending=False, method="average") for column in higher_is_better]
    ranks.append(working["turnover"].rank(ascending=True, method="average"))
    working["selection_score"] = pd.concat(ranks, axis=1).mean(axis=1)
    return str(
        working.sort_values(
            ["selection_score", "feature_count", "set_name"], ascending=[True, True, True]
        ).iloc[0]["set_name"]
    )


def clean_matrix(frame: DataFrame, feature_names: Sequence[str]) -> DataFrame:
    """Convert feature columns to numeric and preserve missing values for LightGBM."""

    matrix = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    cleaned: DataFrame = matrix.replace([np.inf, -np.inf], np.nan)
    return cleaned


def empty_evaluation() -> dict[str, float]:
    """Return an explicit empty-period metric payload."""

    return {
        "days": 0.0,
        "rank_ic": 0.0,
        "rank_icir": 0.0,
        "top_decile_excess_return": 0.0,
        "sharpe": 0.0,
        "maximum_drawdown": 0.0,
        "turnover": 0.0,
        "year_positive_ratio": 0.0,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or zero."""

    if not math.isfinite(denominator) or denominator == 0:
        return 0.0
    return numerator / denominator
