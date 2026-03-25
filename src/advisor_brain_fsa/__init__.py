"""
Advisor-Brain-FSA
Financial Statement Analysis module based on CFA Level 2 curriculum.
"""

from .beneish_mscore import BeneishMScore
from .data_fetcher import CVMDataFetcher, fetch_data

__all__ = ["BeneishMScore", "CVMDataFetcher", "fetch_data"]
__version__ = "0.2.0"
