"""
Order Management System (OMS)
Maintains order lifecycle, state machine transitions, and idempotency.
"""
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Optional
import uuid

from modules.execution.order import Order, OrderRequest, OrderStatus, OrderSide
from modules.execution.fills import Fill


class OrderManager:
    """
    Order Management System.
    Responsible for order creation, tracking, status transitions, and deduplication.
    """

    def __init__(self, duplicate_window_seconds: float = 5.0):
        self._orders: Dict[str, Order] = {}
        self._broker_order_map: Dict[str, str] = {}  # broker_order_id -> internal order_id
        self._lock = threading.RLock()
        self._duplicate_window = timedelta(seconds=duplicate_window_seconds)
        self._recent_requests: Dict[str, datetime] = {}  # request_hash -> timestamp
        self._counter = 0

    def _generate_order_id(self) -> str:
        with self._lock:
            self._counter += 1
            now_str = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"ORD-{now_str}-{self._counter:04d}-{uuid.uuid4().hex[:4].upper()}"

    def _get_request_fingerprint(self, request: OrderRequest) -> str:
        price_str = f"{request.price:.4f}" if request.price is not None else "MKT"
        return f"{request.symbol}:{request.side.value}:{request.quantity}:{price_str}:{request.signal_id}"

    def is_duplicate(self, request: OrderRequest) -> bool:
        """Checks if an identical order request was received within the duplicate window."""
        with self._lock:
            fingerprint = self._get_request_fingerprint(request)
            now = datetime.now()
            if fingerprint in self._recent_requests:
                last_time = self._recent_requests[fingerprint]
                if now - last_time < self._duplicate_window:
                    return True
            return False

    def create_order(self, request: OrderRequest) -> Order:
        """Creates and registers a new order from an OrderRequest."""
        with self._lock:
            if self.is_duplicate(request):
                raise ValueError(
                    f"Duplicate order detected for {request.symbol} {request.side.value} "
                    f"within {self._duplicate_window.total_seconds()}s"
                )

            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                price=request.price,
                time_in_force=request.time_in_force,
                status=OrderStatus.NEW,
                strategy_id=request.strategy_id,
                signal_id=request.signal_id,
            )
            self._orders[order_id] = order
            self._recent_requests[self._get_request_fingerprint(request)] = datetime.now()
            return order

    def record_submission(self, order_id: str, broker_order_id: Optional[str] = None) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            order.transition_to(OrderStatus.SUBMITTED)
            if broker_order_id:
                order.broker_order_id = broker_order_id
                self._broker_order_map[broker_order_id] = order_id
            return order

    def record_acceptance(self, order_id: str, broker_order_id: Optional[str] = None) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            order.transition_to(OrderStatus.ACCEPTED)
            if broker_order_id:
                order.broker_order_id = broker_order_id
                self._broker_order_map[broker_order_id] = order_id
            return order

    def record_fill(self, fill: Fill) -> Order:
        with self._lock:
            order = self.get_order(fill.order_id)
            if not order:
                raise KeyError(f"Order {fill.order_id} not found")

            # If order was already updated to reflect this fill (e.g. from local broker simulation)
            if order.filled_quantity >= order.quantity and order.status == OrderStatus.FILLED:
                return order

            if fill.quantity > order.remaining_quantity:
                raise ValueError(
                    f"Fill quantity {fill.quantity} exceeds order remaining {order.remaining_quantity}"
                )

            # Update VWAP fill price
            prev_filled = order.filled_quantity
            prev_cost = order.average_fill_price * prev_filled
            new_cost = prev_cost + (fill.price * fill.quantity)
            total_filled = prev_filled + fill.quantity

            order.filled_quantity = total_filled
            order.remaining_quantity = order.quantity - total_filled
            order.average_fill_price = new_cost / total_filled if total_filled > 0 else 0.0

            if order.remaining_quantity == 0:
                order.transition_to(OrderStatus.FILLED)
            else:
                order.transition_to(OrderStatus.PARTIALLY_FILLED)

            return order

    def record_cancellation(self, order_id: str, reason: str = "") -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            order.transition_to(OrderStatus.CANCELLED, reason=reason)
            return order

    def record_rejection(self, order_id: str, reason: str) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            order.transition_to(OrderStatus.REJECTED, reason=reason)
            return order

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            return self._orders.get(order_id)

    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        with self._lock:
            internal_id = self._broker_order_map.get(broker_order_id)
            if internal_id:
                return self._orders.get(internal_id)
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        with self._lock:
            orders = [o for o in self._orders.values() if o.is_active]
            if symbol:
                orders = [o for o in orders if o.symbol == symbol]
            return orders

    def get_all_orders(self) -> List[Order]:
        with self._lock:
            return list(self._orders.values())
