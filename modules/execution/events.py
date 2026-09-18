"""
Execution Event Models
Event-driven quantitative trading domain events.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    BAR = "BAR"
    SIGNAL = "SIGNAL"
    RISK_DECISION = "RISK_DECISION"
    RISK_REJECTED = "RISK_REJECTED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_ACCEPTED = "ORDER_ACCEPTED"
    ORDER_REJECTED = "ORDER_REJECTED"
    FILL = "FILL"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    POSITION_UPDATE = "POSITION_UPDATE"
    KILL_SWITCH = "KILL_SWITCH"
    SYSTEM = "SYSTEM"


@dataclass(frozen=True)
class MarketEvent:
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    volume: float = 0.0


@dataclass(frozen=True)
class TickEvent:
    timestamp: datetime  # Exchange timestamp
    symbol: str
    price: float
    volume: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    receive_timestamp: Optional[datetime] = None
    enqueue_timestamp: Optional[datetime] = None
    dequeue_timestamp: Optional[datetime] = None
    enqueue_ns: Optional[int] = None
    dequeue_ns: Optional[int] = None
    sequence: int = 0
    tick_type: str = "trade"
    source: str = "shioaji"
    simtrade: bool = False
    is_replay: bool = False


@dataclass(frozen=True)
class BarEvent:
    timestamp: datetime  # Bar finalization timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str = "1m"
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None

    def __post_init__(self):
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High price {self.high} cannot be less than open/close/low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low price {self.low} cannot be greater than open/close/high")

    @property
    def finalize_time(self) -> datetime:
        return self.timestamp


@dataclass(frozen=True)
class SignalEvent:
    signal_id: str
    timestamp: datetime
    symbol: str
    side: str  # "BUY", "SELL", "HOLD"
    strength: float = 1.0
    strategy_id: str = "default_strategy"
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: Optional[str] = None
    rule_violated: Optional[str] = None
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    event_type: EventType
    timestamp: datetime
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    message: str = ""
