"""Append-only, broker-free paper-trading services."""

from ashare_quant.paper_trading.service import (
    PaperTradingDailyResult,
    PaperTradingExecutionResult,
    PaperTradingInitResult,
    PaperTradingRebalanceResult,
    PaperTradingReportResult,
    PaperTradingService,
)

__all__ = [
    "PaperTradingDailyResult",
    "PaperTradingExecutionResult",
    "PaperTradingInitResult",
    "PaperTradingRebalanceResult",
    "PaperTradingReportResult",
    "PaperTradingService",
]
