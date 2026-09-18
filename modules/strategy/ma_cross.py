"""
Moving Average Crossover Strategy
Clean reference implementation of BaseStrategy.
"""
from collections import deque
from datetime import datetime
from typing import Dict, Optional

from modules.strategy.base import BaseStrategy
from modules.execution.events import BarEvent, SignalEvent


class MovingAverageCrossStrategy(BaseStrategy):
    """
    Classic Moving Average Crossover strategy implementing event-driven BaseStrategy.
    """

    def __init__(self, strategy_id: str = "ma_cross", fast_period: int = 5, slow_period: int = 20):
        super().__init__(strategy_id)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._history: Dict[str, deque] = {}
        self._position_state: Dict[str, str] = {}  # "LONG", "FLAT"

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        symbol = bar.symbol
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self.slow_period + 1)
            self._position_state[symbol] = "FLAT"

        history = self._history[symbol]
        history.append(bar.close)

        if len(history) < self.slow_period:
            return None

        # Calculate fast and slow MA
        closes = list(history)
        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes[-self.slow_period:]) / self.slow_period

        current_state = self._position_state[symbol]

        if fast_ma > slow_ma and current_state != "LONG":
            self._position_state[symbol] = "LONG"
            return SignalEvent(
                signal_id=f"SIG-BUY-{symbol}-{bar.timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
                metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "bar_close": bar.close},
            )
        elif fast_ma <= slow_ma and current_state == "LONG":
            self._position_state[symbol] = "FLAT"
            return SignalEvent(
                signal_id=f"SIG-SELL-{symbol}-{bar.timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=symbol,
                side="SELL",
                strength=1.0,
                strategy_id=self.strategy_id,
                metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "bar_close": bar.close},
            )

        return None
