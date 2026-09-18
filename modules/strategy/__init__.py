"""
Strategy Package
"""
from modules.strategy.base import BaseStrategy
from modules.strategy.ma_cross import MovingAverageCrossStrategy

__all__ = [
    "BaseStrategy",
    "MovingAverageCrossStrategy",
]
