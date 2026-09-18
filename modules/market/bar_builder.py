"""
Bar Builder
Aggregates ticks into standardized OHLCV BarEvents.
"""
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from modules.execution.events import BarEvent


class BarBuilder:
    """
    Constructs OHLCV BarEvents from streaming tick data.
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval = timedelta(seconds=interval_seconds)
        self._current_bars: Dict[str, Dict] = {}
        self._bar_callbacks: List[Callable[[BarEvent], None]] = []

    def register_bar_callback(self, callback: Callable[[BarEvent], None]) -> None:
        self._bar_callbacks.append(callback)

    def _emit_bar(self, bar: BarEvent) -> None:
        for cb in self._bar_callbacks:
            try:
                cb(bar)
            except Exception as e:
                print(f"Error in bar callback: {e}")

    def on_tick(self, symbol: str, price: float, volume: float, timestamp: datetime) -> Optional[BarEvent]:
        """Processes incoming tick. Emits completed BarEvent if bar interval rolled over."""
        emitted_bar = None

        if symbol not in self._current_bars:
            self._current_bars[symbol] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "start_time": timestamp,
            }
            return None

        current = self._current_bars[symbol]

        # Check if tick belongs to a new bar
        if (timestamp - current["start_time"]) >= self.interval:
            emitted_bar = BarEvent(
                timestamp=current["start_time"],
                symbol=symbol,
                open=current["open"],
                high=current["high"],
                low=current["low"],
                close=current["close"],
                volume=current["volume"],
            )
            self._emit_bar(emitted_bar)

            # Start new bar
            self._current_bars[symbol] = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "start_time": timestamp,
            }
        else:
            # Update current bar
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["volume"] += volume

        return emitted_bar
