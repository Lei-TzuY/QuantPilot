"""
Base Strategy Interface
Defines the event-driven strategy contract.
Strategies emit SignalEvents and never interact directly with BrokerAdapters or order APIs.
"""
from abc import ABC, abstractmethod
from typing import Optional

from modules.execution.events import BarEvent, SignalEvent


class BaseStrategy(ABC):
    """
    Abstract Strategy Interface.
    Decoupled from execution modes (Backtest, Paper, Live).
    """

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id

    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Receives a completed bar event and optionally generates a SignalEvent.
        
        CRITICAL TIMING CONTRACT:
        Signal generated from bar t cannot execute at bar t close.
        It is an intent passed to RiskEngine and OMS for execution on subsequent ticks/bars.
        """
        pass
