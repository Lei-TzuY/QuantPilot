"""
Fill and Execution Domain Models
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from modules.execution.order import OrderSide


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    tax: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    broker_order_id: Optional[str] = None

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Fill quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Fill price must be positive, got {self.price}")
        if self.commission < 0 or self.tax < 0:
            raise ValueError("Fees and taxes cannot be negative")

    @property
    def gross_value(self) -> float:
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Net cash impact: outflow on BUY, inflow on SELL."""
        if self.side == OrderSide.BUY:
            return -(self.gross_value + self.commission + self.tax)
        else:
            return self.gross_value - self.commission - self.tax
