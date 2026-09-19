"""
Paper Trading Broker Adapter
Realistic simulated execution with partial fills, commissions, tax, slippage, and buying power checks.
"""
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional
import uuid

from modules.brokers.base import BrokerAdapter
from modules.execution.fees import TaiwanFeeModel
from modules.execution.order import Order, OrderSide, OrderStatus, OrderType
from modules.execution.fills import Fill
from modules.execution.position import Position
from modules.market.tick_size import TaiwanTickSizeModel, default_tick_model


class PaperBrokerAdapter(BrokerAdapter):
    """
    Simulated broker adapter providing high-fidelity order execution without real capital.
    Uses the identical BrokerAdapter interface as live broker implementations.
    Integrates TaiwanFeeModel for accurate 0.3% ordinary vs 0.15% day-trade tax calculations.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.001425,  # 0.1425% Taiwan standard
        tax_rate: float = 0.003,            # 0.3% Taiwan securities tax on ordinary SELL
        slippage_pct: float = 0.001,        # 0.1% slippage
        min_commission: float = 20.0,       # Minimum TWD 20 commission
        enable_partial_fills: bool = False,
        partial_fill_fraction: float = 0.5, # When enabled, fills in 50% chunks
        fee_model: Optional[TaiwanFeeModel] = None,
        tick_model: Optional[TaiwanTickSizeModel] = None,
    ):
        super().__init__()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.trading_mode = "paper"
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.slippage_pct = slippage_pct
        self.min_commission = min_commission
        self.enable_partial_fills = enable_partial_fills
        self.partial_fill_fraction = partial_fill_fraction
        self.fee_model = fee_model or TaiwanFeeModel(
            commission_rate=commission_rate,
            ordinary_tax_rate=tax_rate,
            slippage_pct=slippage_pct,
            min_commission=min_commission,
        )
        self.tick_model = tick_model or default_tick_model

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._latest_prices: Dict[str, float] = {}
        self._latest_quotes: Dict[str, Dict[str, Optional[float]]] = {}
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

    def set_market_quote(
        self,
        symbol: str,
        price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
    ) -> None:
        """
        Injects top-of-book quote (last trade, best bid, best ask).
        
        LIMITATION NOTE:
        PaperBroker simulates execution using top-of-book best bid/ask quotes.
        Depth-of-book queue priority (order-book matching queue) is not modeled
        due to absence of Level 2/3 exchange order-book market feeds.
        """
        with self._lock:
            self._latest_prices[symbol] = price
            self._latest_quotes[symbol] = {
                "last": price,
                "bid": bid_price,
                "ask": ask_price,
            }
            # Check if any pending limit orders trigger with new quote
            self._check_pending_limit_orders(symbol, price, bid_price, ask_price)

    def set_market_price(self, symbol: str, price: float) -> None:
        """Backwards-compatible market price injection."""
        self.set_market_quote(symbol, price=price)

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

    def _calculate_costs(self, side: OrderSide, quantity: int, price: float, symbol: str) -> tuple[float, float, float, float]:
        breakdown = self.fee_model.calculate_execution_costs(side, quantity, price, symbol)
        return breakdown.commission, breakdown.tax, breakdown.slippage, breakdown.tax_rebate

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

            # Determine execution reference price taking top-of-book quotes into account
            quote = self._latest_quotes.get(order.symbol, {})
            last_p = self._latest_prices.get(order.symbol, order.price)
            bid_p = quote.get("bid")
            ask_p = quote.get("ask")

            if order.side == OrderSide.BUY:
                # Market BUY order executes against Ask when available
                ref_price = ask_p if (ask_p is not None and ask_p > 0) else last_p
            else:
                # Market SELL order executes against Bid when available
                ref_price = bid_p if (bid_p is not None and bid_p > 0) else last_p

            if ref_price is None or ref_price <= 0:
                order.transition_to(OrderStatus.REJECTED, reason=f"NO_MARKET_PRICE_FOR_{order.symbol}")
                self._notify_order(order)
                return order

            # Pre-trade cash / share validation
            if order.side == OrderSide.BUY:
                est_exec_price = ref_price * (1.0 + self.slippage_pct)
                comm, tax, _, _ = self._calculate_costs(order.side, order.quantity, est_exec_price, order.symbol)
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
                if order.price is not None and not self.tick_model.is_valid_tick(order.price):
                    order.price = self.tick_model.normalize_limit_price(order.price, order.side)
                # Immediate check if limit price crosses market
                if order.side == OrderSide.BUY and ref_price <= (order.price or 0.0):
                    self._execute_fill(order, ref_price)
                elif order.side == OrderSide.SELL and ref_price >= (order.price or 0.0):
                    self._execute_fill(order, ref_price)
                # Else remains ACCEPTED waiting for price tick

            return order

    def _execute_fill(self, order: Order, base_price: float) -> None:
        """Executes a fill or partial fill against base price adhering to TWSE tick sizes."""
        if not order.is_active:
            return

        # Apply slippage conforming to Taiwan exchange tick sizes
        exec_price = self.tick_model.apply_execution_slippage(
            base_price=base_price,
            side=order.side,
            slippage_pct=self.slippage_pct,
        )

        fill_qty = order.remaining_quantity
        if self.enable_partial_fills and order.remaining_quantity > 1:
            chunk = int(order.quantity * self.partial_fill_fraction)
            fill_qty = max(1, min(chunk, order.remaining_quantity))

        comm, tax, slippage_amt, tax_rebate = self._calculate_costs(order.side, fill_qty, exec_price, order.symbol)

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
            self.cash += tax_rebate  # Credit tax rebate if covering earlier short day-trade
            if order.symbol not in self._positions:
                self._positions[order.symbol] = Position(symbol=order.symbol)
            self._positions[order.symbol].apply_fill(fill)
        else:
            self.cash += (fill_qty * exec_price) - comm - tax
            if order.symbol in self._positions:
                self._positions[order.symbol].apply_fill(fill)
                if self._positions[order.symbol].quantity == 0:
                    del self._positions[order.symbol]

        # Calculate fill metrics
        prev_filled = order.filled_quantity
        new_filled = prev_filled + fill_qty
        new_remaining = order.quantity - new_filled
        prev_cost = order.average_fill_price * prev_filled
        new_avg = (prev_cost + (exec_price * fill_qty)) / new_filled

        # Notify fill listener first (so OMS can process fill transitions and invariants)
        self._notify_fill(fill)

        # Update order progress if not already updated by OMS fill listener
        if fill not in order.fills:
            order.fills.append(fill)

        if order.filled_quantity < new_filled:
            order.filled_quantity = new_filled
            order.remaining_quantity = new_remaining
            order.average_fill_price = new_avg

            if new_remaining == 0:
                order.transition_to(OrderStatus.FILLED)
            else:
                order.transition_to(OrderStatus.PARTIALLY_FILLED)

        self._notify_order(order)

    def _check_pending_limit_orders(
        self,
        symbol: str,
        price: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
    ) -> None:
        """Evaluates pending limit orders against top-of-book bid/ask quotes and traded price."""
        for order in list(self._orders.values()):
            if order.symbol == symbol and order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
                if order.order_type == OrderType.LIMIT and order.price is not None:
                    if order.side == OrderSide.BUY:
                        # Executable ask or trade price reaches or crosses limit
                        exec_ref = ask_price if (ask_price is not None and ask_price > 0) else price
                        if exec_ref <= order.price:
                            self._execute_fill(order, exec_ref)
                    elif order.side == OrderSide.SELL:
                        # Executable bid or trade price reaches or crosses limit
                        exec_ref = bid_price if (bid_price is not None and bid_price > 0) else price
                        if exec_ref >= order.price:
                            self._execute_fill(order, exec_ref)

    def _process_open_orders(self, symbol: str) -> None:
        """Convenience alias to re-evaluate open orders against current market quote."""
        with self._lock:
            quote = self._latest_quotes.get(symbol, {})
            price = self._latest_prices.get(symbol, 0.0)
            self._check_pending_limit_orders(
                symbol=symbol,
                price=price,
                bid_price=quote.get("bid"),
                ask_price=quote.get("ask"),
            )

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

    def get_order(self, order_id: str) -> Optional[Order]:
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                for o in self._orders.values():
                    if o.broker_order_id == order_id:
                        return o
            return order

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
