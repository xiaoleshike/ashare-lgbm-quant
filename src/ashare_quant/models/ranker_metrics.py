"""Cross-sectional ranking metrics and non-backtest portfolio proxies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ashare_quant.models.ranker_data import RankerDataset

type DataFrame = pd.DataFrame


def evaluate_ranker(
    dataset: RankerDataset,
    predictions: np.ndarray,
    ndcg_at: Sequence[int],
    portfolio_fractions: Sequence[float],
) -> dict[str, object]:
    """Evaluate daily ranks against continuous returns and graded relevance."""

    if len(predictions) != len(dataset.frame):
        raise ValueError("prediction count does not match Ranker dataset rows")
    working = dataset.frame[
        [
            "trade_date",
            "ts_code",
            "future_excess_ret_5d",
            "relevance",
        ]
    ].copy()
    working["prediction"] = predictions
    daily_records: list[dict[str, object]] = []
    for trade_date, daily in working.groupby("trade_date", sort=True):
        rank_ic = daily["prediction"].corr(daily["future_excess_ret_5d"], method="spearman")
        record: dict[str, object] = {
            "trade_date": str(trade_date),
            "rank_ic": float(rank_ic),
        }
        relevance = daily["relevance"].to_numpy(dtype=np.float64)
        scores = daily["prediction"].to_numpy(dtype=np.float64)
        for cutoff in ndcg_at:
            record[f"ndcg_at_{cutoff}"] = ndcg(relevance, scores, cutoff)
        for fraction in portfolio_fractions:
            count = max(1, int(math.ceil(len(daily) * fraction)))
            top = daily.nlargest(count, "prediction", keep="first")
            record[portfolio_metric_name(fraction)] = float(top["future_excess_ret_5d"].mean())
        daily_records.append(record)
    daily = pd.DataFrame.from_records(daily_records)
    rank_mean = float(daily["rank_ic"].mean())
    rank_std = float(daily["rank_ic"].std(ddof=1))
    metrics: dict[str, object] = {
        "rows": len(dataset.frame),
        "groups": len(daily),
        "rank_ic": rank_mean,
        "rank_icir": safe_ratio(rank_mean, rank_std),
    }
    for cutoff in ndcg_at:
        metrics[f"ndcg_at_{cutoff}"] = float(daily[f"ndcg_at_{cutoff}"].mean())
    for fraction in portfolio_fractions:
        name = portfolio_metric_name(fraction)
        metrics[name] = float(daily[name].mean())
    metrics["yearly"] = yearly_stability(daily, portfolio_fractions)
    return metrics


def ndcg(relevance: np.ndarray, scores: np.ndarray, cutoff: int) -> float:
    """Compute normalized discounted cumulative gain for one date."""

    count = min(cutoff, len(relevance))
    if count == 0:
        return 0.0
    predicted_order = np.argsort(-scores, kind="stable")[:count]
    ideal_order = np.argsort(-relevance, kind="stable")[:count]
    discounts = np.log2(np.arange(2, count + 2, dtype=np.float64))
    predicted_gain = np.power(2.0, relevance[predicted_order]) - 1.0
    ideal_gain = np.power(2.0, relevance[ideal_order]) - 1.0
    dcg = float(np.sum(predicted_gain / discounts))
    ideal_dcg = float(np.sum(ideal_gain / discounts))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def yearly_stability(
    daily: DataFrame, portfolio_fractions: Sequence[float]
) -> list[dict[str, object]]:
    """Return year-by-year Rank IC and top-bucket excess-return means."""

    working = daily.copy()
    working["year"] = working["trade_date"].astype(str).str[:4]
    records: list[dict[str, object]] = []
    for year, group in working.groupby("year", sort=True):
        record: dict[str, object] = {
            "year": str(year),
            "days": len(group),
            "rank_ic": float(group["rank_ic"].mean()),
        }
        for fraction in portfolio_fractions:
            name = portfolio_metric_name(fraction)
            record[name] = float(group[name].mean())
        records.append(record)
    return records


def portfolio_metric_name(fraction: float) -> str:
    """Return a stable JSON key for one ranking portfolio proxy."""

    percentage = int(round(fraction * 100))
    return f"top_{percentage}pct_mean_future_excess_ret"


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite mean-to-dispersion ratio or zero."""

    if denominator == 0 or not math.isfinite(denominator):
        return 0.0
    return numerator / denominator
