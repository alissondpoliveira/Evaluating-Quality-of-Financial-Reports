"""
Advisor-Brain-FSA
Financial Statement Analysis module based on CFA Level 2 curriculum.
"""

from .beneish_mscore import BeneishMScore
from .data_fetcher import CVMDataFetcher, fetch_data
from .accruals import CashFlowQuality, AlertLevel
from .rank_market import rank_market, detect_red_flags, DEFAULT_WATCHLIST
from .report_generator import generate_report, build_markdown

__all__ = [
    "BeneishMScore",
    "CVMDataFetcher",
    "fetch_data",
    "CashFlowQuality",
    "AlertLevel",
    "rank_market",
    "detect_red_flags",
    "DEFAULT_WATCHLIST",
    "generate_report",
    "build_markdown",
]
__version__ = "0.3.0"
