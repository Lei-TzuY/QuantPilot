"""
Abstract Broker Adapter Interface
Unified interface for Paper and Live broker implementations.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from modules.execution.order import Order
from modules.execution.fills import Fill
from modules.execution.position import Position


class BrokerAdapter(ABC):
    """
    Abstract Broker Adapter.
    Strategies, Risk Engine, and Order Management interact solely via this contract.
    """

    def __init__(self):
        self._order_callbacks: List[Callable[[Order], None]] = []
        self._fill_callbacks: List[Callable[[Fill], None]] = []

    def register_order_callback(self, callback: Callable[[Order], None]) -> None:
        self._order_callbacks.append(callback)

    def register_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        self._fill_callbacks.append(callback)

    def _notify_order(self, order: Order) -> None:
        for cb in self._order_callbacks:
            try:
                cb(order)
            except Exception as e:
                print(f"Error in order callback: {e}")

    def _notify_fill(self, fill: Fill) -> None:
        for cb in self._fill_callbacks:
            try:
                cb(fill)
            except Exception as e:
                print(f"Error in fill callback: {e}")

    @abstractmethod
    def connect(self) -> bool:
        """Connects to the broker service."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the broker service."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the connection is active and healthy."""
        pass

    @abstractmethod
    def get_account(self) -> Dict[str, Any]:
        """Returns account summary, e.g. cash, buying_power, total_equity."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Returns current broker authoritative positions."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Returns active/open orders on the broker."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submits an order to the broker. Returns updated order object."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels an existing order by order_id or broker_order_id."""
        pass

    @abstractmethod
    def subscribe_market_data(self, symbols: List[str]) -> None:
        """Subscribes to market data for symbols."""
        pass

    @abstractmethod
    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        """Unsubscribes from market data for symbols."""
        pass

    @abstractmethod
    def heartbeat(self) -> Dict[str, Any]:
        """Returns broker health and connectivity metrics."""
        pass
