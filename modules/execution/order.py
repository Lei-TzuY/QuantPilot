"""
Order Domain Models and State Machine
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Set


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    ROD = "ROD"  # Rest of Day (Standard for TW stocks)
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class OrderStatus(str, Enum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# Valid state transitions
VALID_TRANSITIONS: dict[OrderStatus, Set[OrderStatus]] = {
    OrderStatus.NEW: {OrderStatus.SUBMITTED, OrderStatus.REJECTED},
    OrderStatus.SUBMITTED: {OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED},
    OrderStatus.ACCEPTED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING, OrderStatus.REJECTED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING},
    OrderStatus.CANCEL_PENDING: {OrderStatus.CANCELLED, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED},
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.ROD
    strategy_id: str = "default"
    signal_id: Optional[str] = None
    client_order_id: Optional[str] = None

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {self.quantity}")
        if self.order_type == OrderType.LIMIT and (self.price is None or self.price <= 0):
            raise ValueError("Limit orders must specify a positive price")


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.ROD
    status: OrderStatus = OrderStatus.NEW
    broker_order_id: Optional[str] = None
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float = 0.0
    strategy_id: str = "default"
    signal_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    fills: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.remaining_quantity == 0 and self.status == OrderStatus.NEW:
            self.remaining_quantity = self.quantity

    @property
    def is_active(self) -> bool:
        return self.status in {
            OrderStatus.NEW,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.CANCEL_PENDING,
        }

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }

    def transition_to(self, new_status: OrderStatus, reason: str = "") -> None:
        """Transitions order status enforcing valid state machine progression."""
        if new_status == self.status:
            return

        valid_next_states = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in valid_next_states:
            raise ValueError(
                f"Invalid order transition from {self.status.value} to {new_status.value} "
                f"(order_id={self.order_id})"
            )

        self.status = new_status
        self.updated_at = datetime.now()
        if reason:
            self.rejection_reason = reason
