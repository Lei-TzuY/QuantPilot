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
        self._seen_fill_ids: set[str] = set()
        self._counter = 0

    def load_state(self, orders: Dict[str, Order], seen_fill_ids: Optional[set[str]] = None) -> None:
        """Restores orders and seen fill IDs into OMS memory from durable journal."""
        with self._lock:
            self._orders = orders
            self._broker_order_map = {
                o.broker_order_id: o.order_id for o in orders.values() if o.broker_order_id
            }
            if seen_fill_ids:
                self._seen_fill_ids = set(seen_fill_ids)

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
            # If already at or past SUBMITTED, treat callback as idempotent
            if order.status in {
                OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            }:
                if broker_order_id and not order.broker_order_id:
                    order.broker_order_id = broker_order_id
                    self._broker_order_map[broker_order_id] = order_id
                return order

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
            # If already at or past ACCEPTED (including terminal states), ignore late out-of-order callback
            if order.status in {
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            }:
                if broker_order_id and not order.broker_order_id:
                    order.broker_order_id = broker_order_id
                    self._broker_order_map[broker_order_id] = order_id
                return order

            order.transition_to(OrderStatus.ACCEPTED)
            if broker_order_id:
                order.broker_order_id = broker_order_id
                self._broker_order_map[broker_order_id] = order_id
            return order

    def record_fill(self, fill: Fill) -> Order:
        with self._lock:
            # 1. Idempotent deduplication by fill_id
            if fill.fill_id in self._seen_fill_ids:
                return self.get_order(fill.order_id)

            order = self.get_order(fill.order_id)
            if not order:
                raise KeyError(f"Order {fill.order_id} not found")

            # 2. Reject fill on terminal rejected or cancelled orders
            if order.status in {OrderStatus.REJECTED, OrderStatus.CANCELLED}:
                raise ValueError(
                    f"Cannot fill terminal order {order.order_id} with status {order.status.value}"
                )

            # 3. Overfill protection
            if fill.quantity > order.remaining_quantity:
                raise ValueError(
                    f"Overfill detected: fill quantity {fill.quantity} exceeds order remaining {order.remaining_quantity}"
                )

            # 4. State transition check before modifying quantities
            target_status = OrderStatus.FILLED if fill.quantity == order.remaining_quantity else OrderStatus.PARTIALLY_FILLED
            order.transition_to(target_status)

            # 5. Apply fill accounting
            prev_filled = order.filled_quantity
            prev_cost = order.average_fill_price * prev_filled
            new_cost = prev_cost + (fill.price * fill.quantity)
            total_filled = prev_filled + fill.quantity

            order.filled_quantity = total_filled
            order.remaining_quantity = order.quantity - total_filled
            order.average_fill_price = new_cost / total_filled if total_filled > 0 else 0.0
            order.fills.append(fill)
            self._seen_fill_ids.add(fill.fill_id)
            return order

    def record_cancellation(self, order_id: str, reason: str = "") -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            if order.status == OrderStatus.CANCELLED:
                return order
            if order.status == OrderStatus.FILLED:
                raise ValueError(f"Cannot cancel completely filled order {order_id}")
            if order.status == OrderStatus.REJECTED:
                raise ValueError(f"Cannot cancel rejected order {order_id}")
            order.transition_to(OrderStatus.CANCELLED, reason=reason)
            return order

    def record_rejection(self, order_id: str, reason: str) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if not order:
                raise KeyError(f"Order {order_id} not found")
            if order.status == OrderStatus.REJECTED:
                return order
            if order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                raise ValueError(f"Cannot reject already filled order {order_id}")
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
