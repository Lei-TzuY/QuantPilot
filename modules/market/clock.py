"""
Taiwan Equity Market Clock
Tracks trading sessions, market open/closed states, and intraday cutoff safety deadlines.
"""
from datetime import datetime, time
from enum import Enum
from typing import Optional


class MarketSession(str, Enum):
    PRE_MARKET = "PRE_MARKET"   # 08:30 - 09:00 (Call auction collection)
    OPEN = "OPEN"               # 09:00 - 13:25 (Continuous trading)
    CLOSING = "CLOSING"         # 13:25 - 13:30 (Closing auction)
    CLOSED = "CLOSED"           # After 13:30, before 08:30, or weekends


class MarketClock:
    """
    Market Clock for Taiwan Stock Exchange (TWSE).
    Enforces trading hours and intraday entry cutoffs.
    """

    def __init__(
        self,
        pre_market_start: time = time(8, 30),
        market_open: time = time(9, 0),
        closing_start: time = time(13, 25),
        market_close: time = time(13, 30),
        intraday_cutoff: time = time(13, 15),
    ):
        self.pre_market_start = pre_market_start
        self.market_open = market_open
        self.closing_start = closing_start
        self.market_close = market_close
        self.intraday_cutoff = intraday_cutoff

    def get_session(self, current_dt: Optional[datetime] = None) -> MarketSession:
        dt = current_dt or datetime.now()

        # Weekend check (0=Mon, 6=Sun)
        if dt.weekday() >= 5:
            return MarketSession.CLOSED

        t = dt.time()
        if self.pre_market_start <= t < self.market_open:
            return MarketSession.PRE_MARKET
        elif self.market_open <= t < self.closing_start:
            return MarketSession.OPEN
        elif self.closing_start <= t < self.market_close:
            return MarketSession.CLOSING
        else:
            return MarketSession.CLOSED

    def is_market_open(self, current_dt: Optional[datetime] = None) -> bool:
        return self.get_session(current_dt) == MarketSession.OPEN

    def is_order_placement_allowed(
        self,
        current_dt: Optional[datetime] = None,
        allow_pre_market: bool = False,
    ) -> bool:
        session = self.get_session(current_dt)
        if session == MarketSession.OPEN:
            return True
        if allow_pre_market and session == MarketSession.PRE_MARKET:
            return True
        return False

    def is_past_intraday_cutoff(self, current_dt: Optional[datetime] = None) -> bool:
        """Checks if current time has passed the safety deadline to open new intraday positions."""
        dt = current_dt or datetime.now()
        t = dt.time()
        return t >= self.intraday_cutoff
