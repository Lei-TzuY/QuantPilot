"""
Quote Book & Top-of-Book State Aggregation
Maintains per-symbol consolidated market quote state by joining independent
Trade Tick and Bid/Ask depth streams, exposing quote age and spread freshness.
"""
from dataclasses import dataclass, field
from datetime import datetime
import logging
import threading
from typing import Dict, List, Optional

from modules.execution.events import BidAskEvent, TickEvent

logger = logging.getLogger("QuantPilot.QuoteBook")


@dataclass
class SymbolQuoteState:
    symbol: str
    last_trade_price: Optional[float] = None
    last_trade_volume: Optional[float] = None
    last_trade_time: Optional[datetime] = None
    best_bid_price: Optional[float] = None
    best_ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    bidask_time: Optional[datetime] = None
    last_update_receive_time: Optional[datetime] = None
    bid_depth: Optional[List[Dict[str, float]]] = None
    ask_depth: Optional[List[Dict[str, float]]] = None

    @property
    def has_trade(self) -> bool:
        return self.last_trade_price is not None and self.last_trade_price > 0

    @property
    def has_book(self) -> bool:
        return (
            self.best_bid_price is not None
            and self.best_ask_price is not None
            and self.best_bid_price > 0
            and self.best_ask_price > 0
        )

    @property
    def spread(self) -> Optional[float]:
        if self.has_book:
            return self.best_ask_price - self.best_bid_price
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.has_book:
            return (self.best_bid_price + self.best_ask_price) / 2.0
        return self.last_trade_price

    def bidask_age_seconds(self, current_time: Optional[datetime] = None) -> Optional[float]:
        if self.bidask_time is None:
            return None
        if current_time is None:
            now = datetime.now(self.bidask_time.tzinfo) if self.bidask_time.tzinfo else datetime.now()
        else:
            now = current_time
            if self.bidask_time.tzinfo is not None and now.tzinfo is None:
                now = now.replace(tzinfo=self.bidask_time.tzinfo)
            elif self.bidask_time.tzinfo is None and now.tzinfo is not None:
                now = now.replace(tzinfo=None)
        return max(0.0, (now - self.bidask_time).total_seconds())

    def trade_age_seconds(self, current_time: Optional[datetime] = None) -> Optional[float]:
        if self.last_trade_time is None:
            return None
        if current_time is None:
            now = datetime.now(self.last_trade_time.tzinfo) if self.last_trade_time.tzinfo else datetime.now()
        else:
            now = current_time
            if self.last_trade_time.tzinfo is not None and now.tzinfo is None:
                now = now.replace(tzinfo=self.last_trade_time.tzinfo)
            elif self.last_trade_time.tzinfo is None and now.tzinfo is not None:
                now = now.replace(tzinfo=None)
        return max(0.0, (now - self.last_trade_time).total_seconds())

    def is_bidask_fresh(
        self,
        max_age_seconds: float = 10.0,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Determines whether the top-of-book bid/ask quote is fresh.
        If bidask is missing or age exceeds max_age_seconds, returns False.
        """
        if not self.has_book:
            return False
        age = self.bidask_age_seconds(current_time)
        if age is None or age > max_age_seconds:
            return False
        return True


class QuoteBook:
    """
    Thread-safe repository of current top-of-book and trade quotes.
    Integrates independent Tick and BidAsk streams to prevent fabricating
    artificial spreads from trade prices alone.
    """

    def __init__(self, default_freshness_seconds: float = 10.0):
        self.default_freshness_seconds = default_freshness_seconds
        self._lock = threading.RLock()
        self._quotes: Dict[str, SymbolQuoteState] = {}

    def _get_or_create(self, symbol: str) -> SymbolQuoteState:
        if symbol not in self._quotes:
            self._quotes[symbol] = SymbolQuoteState(symbol=symbol)
        return self._quotes[symbol]

    def update_tick(self, tick: TickEvent) -> SymbolQuoteState:
        """Updates trade price and volume from a normalized trade TickEvent."""
        with self._lock:
            state = self._get_or_create(tick.symbol)
            state.last_trade_price = float(tick.price)
            state.last_trade_volume = float(tick.volume)
            state.last_trade_time = tick.timestamp
            state.last_update_receive_time = tick.receive_timestamp or datetime.now()
            # If tick has embedded bid/ask (synthetic/replay feeds), update top-of-book state
            if tick.bid_price is not None and tick.ask_price is not None and tick.bid_price > 0 and tick.ask_price > 0:
                state.best_bid_price = float(tick.bid_price)
                state.best_ask_price = float(tick.ask_price)
                state.bid_volume = float(tick.bid_volume) if tick.bid_volume is not None else 1.0
                state.ask_volume = float(tick.ask_volume) if tick.ask_volume is not None else 1.0
                state.bidask_time = tick.timestamp
            return state

    def update_bidask(self, bidask: BidAskEvent) -> SymbolQuoteState:
        """Updates top-of-book and depth from a normalized BidAskEvent."""
        with self._lock:
            state = self._get_or_create(bidask.symbol)
            state.best_bid_price = float(bidask.bid_price)
            state.best_ask_price = float(bidask.ask_price)
            state.bid_volume = float(bidask.bid_volume)
            state.ask_volume = float(bidask.ask_volume)
            state.bidask_time = bidask.timestamp
            state.last_update_receive_time = bidask.receive_timestamp or datetime.now()
            state.bid_depth = bidask.bid_depth
            state.ask_depth = bidask.ask_depth
            return state

    def get_quote(self, symbol: str) -> Optional[SymbolQuoteState]:
        with self._lock:
            return self._quotes.get(symbol)

    def is_bidask_fresh(
        self,
        symbol: str,
        max_age_seconds: Optional[float] = None,
        current_time: Optional[datetime] = None,
    ) -> bool:
        threshold = max_age_seconds if max_age_seconds is not None else self.default_freshness_seconds
        with self._lock:
            q = self._quotes.get(symbol)
            if not q:
                return False
            return q.is_bidask_fresh(max_age_seconds=threshold, current_time=current_time)

    def get_all_quotes(self) -> Dict[str, SymbolQuoteState]:
        with self._lock:
            return dict(self._quotes)

    def clear(self) -> None:
        with self._lock:
            self._quotes.clear()
