"""Human-readable descriptions for production Ranker features."""

from __future__ import annotations

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "market_excess_ret_120d": "长期市场超额收益",
    "turnover_rate": "成交活跃度",
    "cs_rank_logret_sum_1d": "当日收益的市场横截面位置",
    "cs_rank_market_excess_ret_20d": "20日市场超额收益的横截面位置",
    "amount_cv_60d": "60日成交额稳定性",
    "residual_vol_20d": "20日市场调整后波动",
    "cs_rank_market_excess_ret_5d": "5日市场超额收益的横截面位置",
    "residual_vol_60d": "60日市场调整后波动",
    "dist_low_120d": "相对120日最低价的位置",
    "amount_cv_20d": "20日成交额稳定性",
    "market_excess_ret_5d": "短期市场超额收益",
    "market_excess_ret_60d": "中期市场超额收益",
    "cs_rank_positive_ret_ratio_5d": "近期上涨天数比例的横截面位置",
    "market_excess_ret_1d": "当日市场超额收益",
    "logret_sum_120d": "长期累计对数收益",
    "amihud_20d": "流动性压力",
    "cs_rank_ma_ratio_5d": "相对5日均线位置的横截面排名",
    "market_excess_ret_10d": "10日市场超额收益",
    "range_pos_120d": "120日价格区间位置",
    "close_location_value": "收盘价在当日振幅中的位置",
}


def describe_feature(feature: str) -> str:
    """Return a stable description without inferring causal meaning."""

    return FEATURE_DESCRIPTIONS.get(feature, f"模型特征 {feature}")
