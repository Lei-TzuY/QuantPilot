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

    def __init__(self, strategy_id: str, required_warmup_bars: int = 0):
        self.strategy_id = strategy_id
        self.required_warmup_bars = max(0, required_warmup_bars)
        self._warmup_bars_received = 0
        self.strategy_ready = (self.required_warmup_bars == 0)

    @property
    def current_warmup_bars(self) -> int:
        return self._warmup_bars_received

    def warmup(self, historical_bars: list) -> None:
        """
        Feeds historical bars into strategy indicators without triggering signals.
        Marks strategy_ready = True when sufficient lookback is populated.
        """
        for bar in historical_bars:
            self.on_warmup_bar(bar)
            self._warmup_bars_received += 1

        if self._warmup_bars_received >= self.required_warmup_bars:
            self.strategy_ready = True

    def record_bar_received(self) -> None:
        """Increments warmup count for streaming live bars."""
        if not self.strategy_ready:
            self._warmup_bars_received += 1
            if self._warmup_bars_received >= self.required_warmup_bars:
                self.strategy_ready = True

    def on_warmup_bar(self, bar: BarEvent) -> None:
        """Optional hook for updating indicator states during historical warmup."""
        pass

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        """
        Receives a completed bar event and optionally generates a SignalEvent.
        
        CRITICAL TIMING CONTRACT:
        Signal generated from bar t cannot execute at bar t close.
        It is an intent passed to RiskEngine and OMS for execution on subsequent ticks/bars.
        """
        self.record_bar_received()
        return None
