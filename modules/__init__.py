"""
Core modules for the trading web API.
"""

from modules.data_fetcher import DataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.signal_generator import SignalGenerator
from modules.backtester import Backtester
from modules.portfolio_manager import PortfolioManager

__all__ = [
    "DataFetcher",
    "TechnicalAnalyzer",
    "SignalGenerator",
    "Backtester",
    "PortfolioManager",
]
