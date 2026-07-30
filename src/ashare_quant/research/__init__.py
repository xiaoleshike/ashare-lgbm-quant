"""Human-readable quantitative research reports."""

from ashare_quant.research.agent import ResearchAgentService
from ashare_quant.research.daily_report import DailyReportResult, DailyResearchReportGenerator
from ashare_quant.research.decision_support import (
    DecisionSupportResult,
    InvestmentDecisionSupport,
)
from ashare_quant.research.explainability import ExplainabilityEngine, ExplainabilityResult

__all__ = [
    "DailyReportResult",
    "DailyResearchReportGenerator",
    "DecisionSupportResult",
    "ExplainabilityEngine",
    "ExplainabilityResult",
    "InvestmentDecisionSupport",
    "ResearchAgentService",
]
