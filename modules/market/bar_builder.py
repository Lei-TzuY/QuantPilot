"""
Bar Builder
Aggregates normalized high-frequency TickEvents into standardized, immutable OHLCV BarEvents.
Enforces calendar-minute boundaries (e.g. 09:00:00 - 09:00:59.999999) and strict bar immutability.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import Callable, Dict, List, Optional

from modules.execution.events import BarEvent, TickEvent

logger = logging.getLogger("QuantPilot.BarBuilder")


@dataclass
class _MutableBar:
    symbol: str
    open_time: datetime
    close_time: datetime
    finalize_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int = 1

    def to_immutable_bar(self) -> BarEvent:
        return BarEvent(
            timestamp=self.finalize_time,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            interval="1m",
            open_time=self.open_time,
            close_time=self.close_time,
        )


class BarBuilder:
    """
    Constructs immutable 1-minute OHLCV BarEvents from streaming tick data.
    
    Guarantees:
    1. Calendar-minute snapping:
       - open_time = T.replace(second=0, microsecond=0)
       - close_time = open_time + 59.999999s
       - finalize_time = open_time + 60s
    2. Strict Immutability:
       - Once rolled over and emitted, a finalized bar is completely frozen.
       - Late/out-of-order ticks arriving for an already finalized minute are rejected
         and cannot retroactively alter historical OHLCV prices.
    3. Multi-symbol thread-safe aggregation.
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self.interval = timedelta(seconds=interval_seconds)

        self._lock = threading.RLock()
        self._current_bars: Dict[str, _MutableBar] = {}
        self._finalized_bar_keys: Dict[str, set] = {}  # {symbol: {open_time}}
        self._bar_callbacks: List[Callable[[BarEvent], None]] = []
        self._rejected_late_ticks_count = 0

    def register_bar_callback(self, callback: Callable[[BarEvent], None]) -> None:
        with self._lock:
            self._bar_callbacks.append(callback)

    def _emit_bar(self, bar: BarEvent) -> None:
        for cb in list(self._bar_callbacks):
            try:
                cb(bar)
            except Exception as e:
                logger.error(f"Error in BarBuilder callback: {e}", exc_info=True)

    def _snap_to_open_time(self, ts: datetime) -> datetime:
        """Snaps timestamp down to the nearest calendar minute open."""
        return ts.replace(second=0, microsecond=0)

    def on_tick_event(self, tick: TickEvent) -> Optional[BarEvent]:
        """Processes a normalized TickEvent."""
        return self.on_tick(
            symbol=tick.symbol,
            price=tick.price,
            volume=tick.volume,
            timestamp=tick.timestamp,
        )

    def on_tick(self, symbol: str, price: float, volume: float, timestamp: datetime) -> Optional[BarEvent]:
        """
        Processes incoming tick. Emits completed BarEvent if a bar interval rolls over.
        Rejects ticks belonging to already-finalized minutes to guarantee bar immutability.
        """
        with self._lock:
            if symbol not in self._finalized_bar_keys:
                self._finalized_bar_keys[symbol] = set()

            tick_minute_open = self._snap_to_open_time(timestamp)

            # Immutability check: Reject tick if it belongs to an already-finalized minute
            if tick_minute_open in self._finalized_bar_keys[symbol]:
                self._rejected_late_ticks_count += 1
                logger.warning(
                    f"[{symbol}] BarBuilder rejected late tick @ {timestamp} (price={price}): "
                    f"minute {tick_minute_open} has already been finalized and is immutable."
                )
                return None

            emitted_bar: Optional[BarEvent] = None

            # If no active bar for symbol, initialize current bar
            if symbol not in self._current_bars:
                open_t = tick_minute_open
                close_t = open_t + self.interval - timedelta(microseconds=1)
                final_t = open_t + self.interval
                self._current_bars[symbol] = _MutableBar(
                    symbol=symbol,
                    open_time=open_t,
                    close_time=close_t,
                    finalize_time=final_t,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                    tick_count=1,
                )
                return None

            current = self._current_bars[symbol]

            # Check if tick belongs to a new minute (timestamp >= current.finalize_time)
            if timestamp >= current.finalize_time:
                # 1. Finalize current bar
                emitted_bar = current.to_immutable_bar()
                self._finalized_bar_keys[symbol].add(current.open_time)
                # Keep rolling set of finalized minutes to prevent unbounded memory growth
                if len(self._finalized_bar_keys[symbol]) > 1000:
                    self._finalized_bar_keys[symbol].pop()

                self._emit_bar(emitted_bar)

                # 2. Start new bar snapped to tick's minute open
                open_t = tick_minute_open
                close_t = open_t + self.interval - timedelta(microseconds=1)
                final_t = open_t + self.interval
                self._current_bars[symbol] = _MutableBar(
                    symbol=symbol,
                    open_time=open_t,
                    close_time=close_t,
                    finalize_time=final_t,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                    tick_count=1,
                )
            else:
                # Update current open bar
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
                current.volume += volume
                current.tick_count += 1

            return emitted_bar

    def flush_symbol(self, symbol: str) -> Optional[BarEvent]:
        """Manually finalizes and emits the active bar for a symbol (e.g. session end)."""
        with self._lock:
            if symbol in self._current_bars:
                current = self._current_bars.pop(symbol)
                emitted_bar = current.to_immutable_bar()
                self._finalized_bar_keys[symbol].add(current.open_time)
                self._emit_bar(emitted_bar)
                return emitted_bar
            return None

    def flush_all(self) -> List[BarEvent]:
        """Flushes and finalizes active bars across all symbols."""
        with self._lock:
            flushed = []
            for sym in list(self._current_bars.keys()):
                bar = self.flush_symbol(sym)
                if bar:
                    flushed.append(bar)
            return flushed
