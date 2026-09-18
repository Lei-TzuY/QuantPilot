"""
Paper Trading Broker Adapter
Realistic simulated execution with partial fills, commissions, tax, slippage, and buying power checks.
"""
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional
import uuid

from modules.brokers.base import BrokerAdapter
from modules.execution.order import Order, OrderSide, OrderStatus, OrderType
from modules.execution.fills import Fill
from modules.execution.position import Position


class PaperBrokerAdapter(BrokerAdapter):
    """
    Simulated broker adapter providing high-fidelity order execution without real capital.
    Uses the identical BrokerAdapter interface as live broker implementations.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.001425,  # 0.1425% Taiwan standard
        tax_rate: float = 0.003,            # 0.3% Taiwan securities tax on SELL
        slippage_pct: float = 0.001,        # 0.1% slippage
        min_commission: float = 20.0,       # Minimum TWD 20 commission
        enable_partial_fills: bool = False,
        partial_fill_fraction: float = 0.5, # When enabled, fills in 50% chunks
    ):
        super().__init__()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
        self.enable_partial_fills = enable_partial_fills
        self.partial_fill_fraction = partial_fill_fraction

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._latest_prices: Dict[str, float] = {}
        self._connected = False
        self._lock = threading.RLock()
        self._counter = 0

    def connect(self) -> bool:
        with self._lock:
            self._connected = True
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_market_price(self, symbol: str, price: float) -> None:
        """Injects current market price for realistic simulation."""
        with self._lock:
            self._latest_prices[symbol] = price
            # Check if any pending limit orders trigger with new price
            self._check_pending_limit_orders(symbol, price)

    def get_account(self) -> Dict[str, Any]:
        with self._lock:
            total_equity = self.cash
            for sym, pos in self._positions.items():
                mkt_price = self._latest_prices.get(sym, pos.avg_price)
                total_equity += pos.market_value(mkt_price)

            return {
                "broker": "PaperBroker",
                "connected": self._connected,
                "cash": self.cash,
                "total_equity": total_equity,
                "positions_count": len(self._positions),
                "open_orders_count": len([o for o in self._orders.values() if o.is_active]),
            }

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            # Return copies to prevent external mutation
            return {
                sym: Position(
                    symbol=p.symbol,
                    quantity=p.quantity,
                    avg_price=p.avg_price,
                    cost_basis=p.cost_basis,
                    realized_pnl=p.realized_pnl,
                    total_commission=p.total_commission,
                    total_tax=p.total_tax,
                    last_price=self._latest_prices.get(sym, p.last_price),
                    updated_at=p.updated_at,
                )
                for sym, p in self._positions.items()
                if p.quantity > 0
            }

    def get_open_orders(self) -> List[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    def _calculate_costs(self, side: OrderSide, quantity: int, price: float) -> tuple[float, float, float]:
        gross_value = quantity * price
        comm = max(self.min_commission, gross_value * self.commission_rate)
        tax = gross_value * self.tax_rate if side == OrderSide.SELL else 0.0
        slippage = gross_value * self.slippage_pct
        return comm, tax, slippage

    def submit_order(self, order: Order) -> Order:
        with self._lock:
            if not self._connected:
                order.transition_to(OrderStatus.REJECTED, reason="BROKER_DISCONNECTED")
                self._notify_order(order)
                return order

            # Assign broker order ID
            self._counter += 1
            broker_id = f"PAPER-{datetime.now().strftime('%Y%m%d')}-{self._counter:05d}"
            order.broker_order_id = broker_id
            self._orders[order.order_id] = order

            order.transition_to(OrderStatus.SUBMITTED)
            self._notify_order(order)

            # Determine execution reference price
            ref_price = self._latest_prices.get(order.symbol, order.price)
            if ref_price is None or ref_price <= 0:
                order.transition_to(OrderStatus.REJECTED, reason=f"NO_MARKET_PRICE_FOR_{order.symbol}")
                self._notify_order(order)
                return order

            # Pre-trade cash / share validation
            if order.side == OrderSide.BUY:
                est_exec_price = ref_price * (1.0 + self.slippage_pct)
                comm, tax, _ = self._calculate_costs(order.side, order.quantity, est_exec_price)
                total_required = (order.quantity * est_exec_price) + comm + tax
                if total_required > self.cash:
                    order.transition_to(
                        OrderStatus.REJECTED,
                        reason=f"INSUFFICIENT_FUNDS: required {total_required:.2f}, available {self.cash:.2f}",
                    )
                    self._notify_order(order)
                    return order
            elif order.side == OrderSide.SELL:
                current_pos = self._positions.get(order.symbol)
                avail_shares = current_pos.quantity if current_pos else 0
                if avail_shares < order.quantity:
                    order.transition_to(
                        OrderStatus.REJECTED,
                        reason=f"INSUFFICIENT_POSITION: requested {order.quantity}, available {avail_shares}",
                    )
                    self._notify_order(order)
                    return order

            order.transition_to(OrderStatus.ACCEPTED)
            self._notify_order(order)

            # Execution logic
            if order.order_type == OrderType.MARKET:
                self._execute_fill(order, ref_price)
            elif order.order_type == OrderType.LIMIT:
                # Immediate check if limit price crosses market
                if order.side == OrderSide.BUY and ref_price <= (order.price or 0.0):
                    self._execute_fill(order, ref_price)
                elif order.side == OrderSide.SELL and ref_price >= (order.price or 0.0):
                    self._execute_fill(order, ref_price)
                # Else remains ACCEPTED waiting for price tick

            return order

    def _execute_fill(self, order: Order, base_price: float) -> None:
        """Executes a fill or partial fill against base price."""
        if not order.is_active:
            return

        # Apply slippage
        if order.side == OrderSide.BUY:
            exec_price = base_price * (1.0 + self.slippage_pct)
        else:
            exec_price = base_price * (1.0 - self.slippage_pct)

        fill_qty = order.remaining_quantity
        if self.enable_partial_fills and order.remaining_quantity > 1:
            chunk = int(order.quantity * self.partial_fill_fraction)
            fill_qty = max(1, min(chunk, order.remaining_quantity))

        comm, tax, slippage_amt = self._calculate_costs(order.side, fill_qty, exec_price)

        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8].upper()}",
            order_id=order.order_id,
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=exec_price,
            commission=comm,
            tax=tax,
            slippage=slippage_amt,
            timestamp=datetime.now(),
        )

        # Update broker position and cash
        if fill.side == OrderSide.BUY:
            self.cash -= (fill_qty * exec_price) + comm + tax
            if order.symbol not in self._positions:
                self._positions[order.symbol] = Position(symbol=order.symbol)
            self._positions[order.symbol].apply_fill(fill)
        else:
            self.cash += (fill_qty * exec_price) - comm - tax
            if order.symbol in self._positions:
                self._positions[order.symbol].apply_fill(fill)
                if self._positions[order.symbol].quantity == 0:
                    del self._positions[order.symbol]

        # Update order progress
        prev_filled = order.filled_quantity
        new_filled = prev_filled + fill_qty
        new_remaining = order.quantity - new_filled
        prev_cost = order.average_fill_price * prev_filled
        new_avg = (prev_cost + (exec_price * fill_qty)) / new_filled

        order.filled_quantity = new_filled
        order.remaining_quantity = new_remaining
        order.average_fill_price = new_avg

        if new_remaining == 0:
            order.transition_to(OrderStatus.FILLED)
        else:
            order.transition_to(OrderStatus.PARTIALLY_FILLED)

        self._notify_fill(fill)
        self._notify_order(order)

    def _check_pending_limit_orders(self, symbol: str, price: float) -> None:
        """Evaluates pending limit orders against new price."""
        for order in list(self._orders.values()):
            if order.symbol == symbol and order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
                if order.order_type == OrderType.LIMIT and order.price is not None:
                    if order.side == OrderSide.BUY and price <= order.price:
                        self._execute_fill(order, price)
                    elif order.side == OrderSide.SELL and price >= order.price:
                        self._execute_fill(order, price)

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                # Try finding by broker order id
                for o in self._orders.values():
                    if o.broker_order_id == order_id:
                        order = o
                        break

            if not order:
                return False

            if order.is_active:
                order.transition_to(OrderStatus.CANCELLED, reason="USER_CANCELLED")
                self._notify_order(order)
                return True
            return False

    def subscribe_market_data(self, symbols: List[str]) -> None:
        pass

    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        pass

    def heartbeat(self) -> Dict[str, Any]:
        return {
            "status": "HEALTHY" if self._connected else "DISCONNECTED",
            "cash": self.cash,
            "timestamp": datetime.now().isoformat(),
        }
