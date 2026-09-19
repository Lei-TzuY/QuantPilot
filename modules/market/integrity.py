"""
Market Data Integrity Layer
Validates real-time TickEvents against data-feed anomalies, stale timestamps,
out-of-order sequences, impossible price shocks, and maintains symbol health state.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from typing import Dict, List, Optional, Tuple

from modules.execution.events import BidAskEvent, TickEvent

logger = logging.getLogger("QuantPilot.MarketIntegrity")


class MarketHealthStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"


@dataclass
class SymbolDataHealth:
    symbol: str
    status: MarketHealthStatus = MarketHealthStatus.HEALTHY
    tick_stream_health: MarketHealthStatus = MarketHealthStatus.INITIALIZING
    bidask_stream_health: MarketHealthStatus = MarketHealthStatus.INITIALIZING
    last_exchange_timestamp: Optional[datetime] = None
    last_receive_timestamp: Optional[datetime] = None
    last_tick_timestamp: Optional[datetime] = None
    last_bidask_timestamp: Optional[datetime] = None
    last_sequence: int = 0
    last_price: float = 0.0
    total_ticks_received: int = 0
    total_ticks_valid: int = 0
    total_ticks_rejected: int = 0
    total_bidask_received: int = 0
    total_bidask_valid: int = 0
    total_bidask_rejected: int = 0
    duplicate_count: int = 0
    out_of_order_count: int = 0
    stale_count: int = 0
    bidask_stale_count: int = 0
    price_anomaly_count: int = 0
    bidask_anomaly_count: int = 0
    reconnect_gaps_count: int = 0
    last_anomaly_reason: Optional[str] = None
    last_anomaly_timestamp: Optional[datetime] = None
    has_incident: bool = False


class MarketDataIntegrityChecker:
    """
    Evaluates streaming market ticks against strict domain and exchange invariants.
    Maintains per-symbol health state and feeds health metrics to RiskEngine.
    """

    def __init__(
        self,
        max_stale_seconds: float = 60.0,
        max_price_jump_pct: float = 0.15,  # 15% single-tick jump without circuit notification
        allow_simtrade: bool = False,
    ):
        self.max_stale_seconds = max_stale_seconds
        self.max_price_jump_pct = max_price_jump_pct
        self.allow_simtrade = allow_simtrade

        self._lock = threading.RLock()
        self._symbol_health: Dict[str, SymbolDataHealth] = {}
        self._recent_tick_signatures: Dict[str, set] = {}

    def _get_or_create(self, symbol: str) -> SymbolDataHealth:
        if symbol not in self._symbol_health:
            self._symbol_health[symbol] = SymbolDataHealth(symbol=symbol)
            self._recent_tick_signatures[symbol] = set()
        return self._symbol_health[symbol]

    def validate_tick(self, tick: TickEvent, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates incoming TickEvent.
        Returns:
            (is_valid: bool, rejection_reason: Optional[str])
        """
        with self._lock:
            if current_time is not None:
                now = current_time
                if tick.timestamp.tzinfo is not None and now.tzinfo is None:
                    now = now.replace(tzinfo=tick.timestamp.tzinfo)
                elif tick.timestamp.tzinfo is None and now.tzinfo is not None:
                    now = now.replace(tzinfo=None)
            else:
                now = datetime.now(tick.timestamp.tzinfo) if tick.timestamp.tzinfo else datetime.now()

            health = self._get_or_create(tick.symbol)
            health.total_ticks_received += 1

            # 1. Price non-zero & positive
            if tick.price <= 0:
                health.total_ticks_rejected += 1
                health.price_anomaly_count += 1
                health.status = MarketHealthStatus.DEGRADED
                reason = f"INVALID_PRICE: price {tick.price} <= 0"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                logger.warning(f"[{tick.symbol}] Integrity failure: {reason}")
                return False, reason

            # 2. Volume non-negative
            if tick.volume < 0:
                health.total_ticks_rejected += 1
                health.status = MarketHealthStatus.DEGRADED
                reason = f"INVALID_VOLUME: volume {tick.volume} < 0"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                logger.warning(f"[{tick.symbol}] Integrity failure: {reason}")
                return False, reason

            # 3. Duplicate tick detection (timestamp + price + volume)
            sig = (tick.timestamp, round(tick.price, 4), round(tick.volume, 4))
            sig_set = self._recent_tick_signatures[tick.symbol]
            if sig in sig_set:
                health.total_ticks_rejected += 1
                health.duplicate_count += 1
                reason = f"DUPLICATE_TICK: signature {sig} already processed"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                return False, reason

            # Add to rolling signature set (keep max 1000)
            sig_set.add(sig)
            if len(sig_set) > 1000:
                sig_set.pop()

            # 4. Out-of-order exchange timestamp or regression
            if health.last_exchange_timestamp:
                if tick.timestamp < health.last_exchange_timestamp:
                    health.total_ticks_rejected += 1
                    health.out_of_order_count += 1
                    health.status = MarketHealthStatus.DEGRADED
                    reason = (
                        f"TIMESTAMP_REGRESSION: incoming {tick.timestamp} < last {health.last_exchange_timestamp}"
                    )
                    health.last_anomaly_reason = reason
                    health.last_anomaly_timestamp = now
                    logger.warning(f"[{tick.symbol}] Integrity failure: {reason}")
                    return False, reason

            # 5. Stale tick check (relative to now)
            if self.max_stale_seconds > 0:
                tick_age = (now - tick.timestamp).total_seconds()
                # Only check staleness if current_time was explicitly provided (e.g. controlled tests)
                # or if tick is reasonably close to wall-clock time (< 2 hours).
                # Historical replays with dates far in the past are not rejected as live feed staleness.
                if current_time is not None or (0 < tick_age <= 7200):
                    if tick_age > self.max_stale_seconds:
                        health.total_ticks_rejected += 1
                        health.stale_count += 1
                        health.status = MarketHealthStatus.STALE
                        reason = f"STALE_TICK: age {tick_age:.1f}s > max {self.max_stale_seconds}s"
                        health.last_anomaly_reason = reason
                        health.last_anomaly_timestamp = now
                        logger.warning(f"[{tick.symbol}] Integrity failure: {reason}")
                        return False, reason

            # 6. Impossible single-tick price jump check
            if health.last_price > 0 and self.max_price_jump_pct > 0:
                jump_pct = abs(tick.price - health.last_price) / health.last_price
                if jump_pct > self.max_price_jump_pct:
                    health.total_ticks_rejected += 1
                    health.price_anomaly_count += 1
                    health.status = MarketHealthStatus.DEGRADED
                    reason = (
                        f"ABNORMAL_PRICE_JUMP: {jump_pct*100:.2f}% jump from {health.last_price} to {tick.price}"
                    )
                    health.last_anomaly_reason = reason
                    health.last_anomaly_timestamp = now
                    logger.critical(f"[{tick.symbol}] Integrity shock: {reason}")
                    return False, reason

            # 7. Simulated exchange ticks check
            if tick.simtrade and not self.allow_simtrade:
                health.total_ticks_rejected += 1
                reason = "SIMULATED_EXCHANGE_TICK_REJECTED"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                return False, reason

            # All checks passed! Update healthy state
            health.total_ticks_valid += 1
            health.last_exchange_timestamp = tick.timestamp
            health.last_tick_timestamp = tick.timestamp
            health.last_receive_timestamp = tick.receive_timestamp or now
            health.last_sequence = tick.sequence
            health.last_price = tick.price
            health.tick_stream_health = MarketHealthStatus.HEALTHY
            if tick.bid_price is not None and tick.ask_price is not None and tick.bid_price > 0 and tick.ask_price > 0:
                health.last_bidask_timestamp = tick.timestamp
                health.bidask_stream_health = MarketHealthStatus.HEALTHY
            if not health.has_incident and health.status != MarketHealthStatus.DISCONNECTED:
                health.status = MarketHealthStatus.HEALTHY
            return True, None

    def validate_bidask(self, bidask: BidAskEvent, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates incoming BidAskEvent depth/top-of-book data.
        Returns (is_valid, rejection_reason).
        """
        with self._lock:
            if current_time is not None:
                now = current_time
                if bidask.timestamp.tzinfo is not None and now.tzinfo is None:
                    now = now.replace(tzinfo=bidask.timestamp.tzinfo)
                elif bidask.timestamp.tzinfo is None and now.tzinfo is not None:
                    now = now.replace(tzinfo=None)
            else:
                now = datetime.now(bidask.timestamp.tzinfo) if bidask.timestamp.tzinfo else datetime.now()

            health = self._get_or_create(bidask.symbol)
            health.total_bidask_received += 1

            # 1. Price validity
            if bidask.bid_price <= 0 or bidask.ask_price <= 0:
                health.total_bidask_rejected += 1
                health.bidask_anomaly_count += 1
                health.bidask_stream_health = MarketHealthStatus.DEGRADED
                reason = f"INVALID_BIDASK_PRICE: bid={bidask.bid_price}, ask={bidask.ask_price}"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                logger.warning(f"[{bidask.symbol}] Integrity failure: {reason}")
                return False, reason

            # 2. Volume validity
            if bidask.bid_volume < 0 or bidask.ask_volume < 0:
                health.total_bidask_rejected += 1
                health.bidask_anomaly_count += 1
                health.bidask_stream_health = MarketHealthStatus.DEGRADED
                reason = f"INVALID_BIDASK_VOLUME: bid_vol={bidask.bid_volume}, ask_vol={bidask.ask_volume}"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                return False, reason

            # 3. Crossed market check (bid > ask is only valid during pre-market auction / simtrade)
            if bidask.bid_price > bidask.ask_price and not bidask.simtrade:
                health.total_bidask_rejected += 1
                health.bidask_anomaly_count += 1
                health.bidask_stream_health = MarketHealthStatus.DEGRADED
                reason = f"CROSSED_BIDASK_SPREAD: bid {bidask.bid_price} > ask {bidask.ask_price}"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                logger.warning(f"[{bidask.symbol}] Crossed spread anomaly: {reason}")
                return False, reason

            # 4. Out-of-order timestamp check
            if health.last_bidask_timestamp and bidask.timestamp < health.last_bidask_timestamp:
                health.total_bidask_rejected += 1
                health.out_of_order_count += 1
                health.bidask_stream_health = MarketHealthStatus.DEGRADED
                reason = f"BIDASK_TIMESTAMP_REGRESSION: incoming {bidask.timestamp} < last {health.last_bidask_timestamp}"
                health.last_anomaly_reason = reason
                health.last_anomaly_timestamp = now
                return False, reason

            # 5. Staleness check
            if self.max_stale_seconds > 0:
                age = (now - bidask.timestamp).total_seconds()
                if current_time is not None or (0 < age <= 7200):
                    if age > self.max_stale_seconds:
                        health.total_bidask_rejected += 1
                        health.bidask_stale_count += 1
                        health.bidask_stream_health = MarketHealthStatus.STALE
                        reason = f"STALE_BIDASK: age {age:.1f}s > max {self.max_stale_seconds}s"
                        health.last_anomaly_reason = reason
                        health.last_anomaly_timestamp = now
                        return False, reason

            # All BidAsk checks passed
            health.total_bidask_valid += 1
            health.last_bidask_timestamp = bidask.timestamp
            health.bidask_stream_health = MarketHealthStatus.HEALTHY
            return True, None

    def record_incident(self, symbol: str, reason: str) -> None:
        """Records an external/queue incident and flags symbol health as DEGRADED."""
        with self._lock:
            h = self._get_or_create(symbol)
            h.status = MarketHealthStatus.DEGRADED
            h.has_incident = True
            h.last_anomaly_reason = reason
            h.last_anomaly_timestamp = datetime.now()

    def record_disconnect(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                h = self._get_or_create(symbol)
                h.status = MarketHealthStatus.DISCONNECTED
                h.tick_stream_health = MarketHealthStatus.DISCONNECTED
                h.bidask_stream_health = MarketHealthStatus.DISCONNECTED
            else:
                for h in self._symbol_health.values():
                    h.status = MarketHealthStatus.DISCONNECTED
                    h.tick_stream_health = MarketHealthStatus.DISCONNECTED
                    h.bidask_stream_health = MarketHealthStatus.DISCONNECTED

    def record_reconnect(self, symbol: Optional[str] = None) -> None:
        with self._lock:
            if symbol:
                h = self._get_or_create(symbol)
                h.reconnect_gaps_count += 1
                h.has_incident = False
                h.status = MarketHealthStatus.HEALTHY
                h.tick_stream_health = MarketHealthStatus.HEALTHY
                h.bidask_stream_health = MarketHealthStatus.HEALTHY
            else:
                for h in self._symbol_health.values():
                    h.reconnect_gaps_count += 1
                    h.has_incident = False
                    h.status = MarketHealthStatus.HEALTHY
                    h.tick_stream_health = MarketHealthStatus.HEALTHY
                    h.bidask_stream_health = MarketHealthStatus.HEALTHY

    def get_health(self, symbol: str) -> MarketHealthStatus:
        with self._lock:
            if symbol not in self._symbol_health:
                return MarketHealthStatus.HEALTHY
            return self._symbol_health[symbol].status

    def is_symbol_healthy(self, symbol: str) -> bool:
        with self._lock:
            if symbol not in self._symbol_health:
                return True
            h = self._symbol_health[symbol]
            # Must not be DEGRADED, STALE, or DISCONNECTED
            if h.status != MarketHealthStatus.HEALTHY:
                return False
            if h.bidask_stream_health in (MarketHealthStatus.STALE, MarketHealthStatus.DEGRADED, MarketHealthStatus.DISCONNECTED):
                return False
            return True

    def is_bidask_healthy(self, symbol: str) -> bool:
        with self._lock:
            if symbol not in self._symbol_health:
                return True
            h = self._symbol_health[symbol]
            return h.bidask_stream_health not in (
                MarketHealthStatus.STALE,
                MarketHealthStatus.DEGRADED,
                MarketHealthStatus.DISCONNECTED,
            )

    def get_stream_health(self, symbol: str) -> Dict[str, MarketHealthStatus]:
        with self._lock:
            h = self._get_or_create(symbol)
            return {
                "tick": h.tick_stream_health,
                "bidask": h.bidask_stream_health,
                "overall": h.status,
            }

    def get_symbol_report(self, symbol: str) -> Optional[SymbolDataHealth]:
        with self._lock:
            return self._symbol_health.get(symbol)

    def get_all_reports(self) -> Dict[str, SymbolDataHealth]:
        with self._lock:
            return dict(self._symbol_health)
