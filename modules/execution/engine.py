"""
Execution Engine
Coordinates event-driven pipeline: MarketData -> Strategy -> Risk -> OMS -> Broker -> Fills -> Position.
"""
from datetime import datetime
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from modules.brokers.base import BrokerAdapter
from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.events import (
    AuditLogEntry,
    BarEvent,
    EventType,
    RiskDecision,
    SignalEvent,
)
from modules.execution.fills import Fill
from modules.execution.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from modules.execution.order_manager import OrderManager
from modules.execution.persistence import ExecutionStatePersistence
from modules.execution.position import Position
from modules.execution.reconciliation import Reconciler, ReconciliationReport
from modules.market.clock import MarketClock
from modules.risk.engine import RiskEngine
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy

logger = logging.getLogger("QuantPilot.ExecutionEngine")


class ExecutionEngine:
    """
    Central Execution Engine for QuantPilot Algorithmic Trading.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        risk_engine: RiskEngine,
        order_manager: Optional[OrderManager] = None,
        market_clock: Optional[MarketClock] = None,
        persistence: Optional[ExecutionStatePersistence] = None,
        reconciler: Optional[Reconciler] = None,
        trading_mode: str = "paper",
        default_order_shares: int = 1000,  # Standard Taiwan round lot: 1,000 shares
    ):
        self.trading_mode = trading_mode.lower()
        self.broker = broker
        self.risk_engine = risk_engine
        self.order_manager = order_manager or OrderManager()
        self.market_clock = market_clock or MarketClock()
        self.persistence = persistence or ExecutionStatePersistence()
        self.reconciler = reconciler or Reconciler()
        self.default_order_shares = default_order_shares

        self._positions: Dict[str, Position] = {}
        self._strategies: Dict[str, BaseStrategy] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._latest_bars: Dict[str, BarEvent] = {}
        self._lock = threading.RLock()
        self._is_running = False

        # Hook broker callbacks
        self.broker.register_order_callback(self._on_broker_order_update)
        self.broker.register_fill_callback(self._on_broker_fill)

    def register_strategy(self, strategy: BaseStrategy) -> None:
        with self._lock:
            self._strategies[strategy.strategy_id] = strategy

    def get_audit_log(self, limit: int = 100) -> List[AuditLogEntry]:
        with self._lock:
            return self._audit_log[-limit:]

    def _log_event(self, event_type: EventType, payload: Dict[str, Any], correlation_id: Optional[str] = None, message: str = "") -> None:
        entry = AuditLogEntry(
            event_type=event_type,
            timestamp=datetime.now(),
            payload=payload,
            correlation_id=correlation_id,
            message=message,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > 5000:
            self._audit_log.pop(0)

    def start(self, reconcile_on_startup: bool = True) -> bool:
        """
        Starts the execution engine.
        Ensures broker connection, restores persisted state, and performs reconciliation.
        """
        with self._lock:
            # 1. Connect broker
            if not self.broker.is_connected():
                self.broker.connect()

            # 2. Restore state from persistence
            saved = self.persistence.load_state()
            if saved:
                self._positions = saved["positions"]
                for ord_obj in saved["orders"]:
                    self.order_manager._orders[ord_obj.order_id] = ord_obj
                    if ord_obj.broker_order_id:
                        self.order_manager._broker_order_map[ord_obj.broker_order_id] = ord_obj.order_id

            # 3. Reconcile with authoritative broker state
            if reconcile_on_startup:
                report = self.reconcile_with_broker()
                if not report.is_clean and report.critical_count > 0:
                    logger.critical(f"Critical reconciliation mismatch on startup: {report.discrepancies}")
                    # Safety: halt trading if critical discrepancies detected
                    self.risk_engine.kill_switch.halt(
                        reason=f"CRITICAL_STARTUP_RECONCILIATION_MISMATCH: {report.critical_count} critical issues"
                    )

            self._is_running = True
            self._log_event(EventType.SYSTEM, {"action": "START", "trading_mode": self.trading_mode})
            return True

    def stop(self) -> None:
        with self._lock:
            self._is_running = False
            self._persist_current_state()
            self._log_event(EventType.SYSTEM, {"action": "STOP"})

    def _persist_current_state(self) -> None:
        all_orders = self.order_manager.get_all_orders()
        fills: List[Fill] = []  # Can be expanded
        self.persistence.save_state(
            positions=self._positions,
            orders=all_orders,
            fills=fills,
            realized_pnl=sum(p.realized_pnl for p in self._positions.values()),
            daily_trades=self.risk_engine._daily_trades_count,
            session_status="HALTED" if self.risk_engine.kill_switch.is_halted() else "RUNNING",
        )

    def on_bar(self, bar: BarEvent) -> None:
        """
        Processes a completed bar event.
        Dispatches to strategies, routes signals to RiskEngine, and submits approved orders.
        """
        with self._lock:
            if not self._is_running:
                return

            self._latest_bars[bar.symbol] = bar
            self._log_event(
                EventType.BAR,
                {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                },
                correlation_id=f"{bar.symbol}-{bar.timestamp.isoformat()}",
            )

            # Update paper broker price if applicable
            if isinstance(self.broker, PaperBrokerAdapter):
                self.broker.set_market_price(bar.symbol, bar.close)

            # Evaluate registered strategies
            for strat_id, strategy in self._strategies.items():
                try:
                    signal = strategy.on_bar(bar)
                    if signal:
                        self._process_signal(signal, bar)
                except Exception as e:
                    logger.error(f"Strategy {strat_id} failed on_bar: {e}")

    def _process_signal(self, signal: SignalEvent, bar: BarEvent) -> Optional[Order]:
        """
        Processes a SignalEvent through RiskEngine and OMS.
        """
        self._log_event(
            EventType.SIGNAL,
            {
                "signal_id": signal.signal_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "strategy_id": signal.strategy_id,
                "strength": signal.strength,
            },
            correlation_id=signal.signal_id,
        )

        if signal.side not in ("BUY", "SELL"):
            return None

        # Sizing and Order Request creation
        side = OrderSide.BUY if signal.side == "BUY" else OrderSide.SELL
        order_qty = self.default_order_shares

        # If SELL, cap at available position
        if side == OrderSide.SELL:
            current_pos = self._positions.get(signal.symbol)
            if not current_pos or current_pos.quantity <= 0:
                logger.warning(f"Rejecting SELL signal for {signal.symbol}: No position available")
                return None
            order_qty = min(order_qty, current_pos.quantity)

        # Intraday cutoff check: do not open new positions past safety cutoff
        if side == OrderSide.BUY and self.market_clock.is_past_intraday_cutoff(bar.timestamp):
            self._log_event(
                EventType.RISK_REJECTED,
                {"reason": "PAST_INTRADAY_CUTOFF", "signal_id": signal.signal_id},
                correlation_id=signal.signal_id,
                message="Order rejected: Past intraday entry cutoff",
            )
            return None

        request = OrderRequest(
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=order_qty,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
        )

        # Evaluate against RiskEngine
        decision = self.risk_engine.evaluate_order(
            request=request,
            current_positions=self._positions,
            market_price=bar.close,
            market_price_timestamp=bar.timestamp,
            current_time=bar.timestamp,
        )

        if not decision.allowed:
            self._log_event(
                EventType.RISK_REJECTED,
                {
                    "signal_id": signal.signal_id,
                    "symbol": request.symbol,
                    "reason": decision.reason,
                    "rule": decision.rule_violated,
                },
                correlation_id=signal.signal_id,
                message=f"Risk rejected: {decision.reason}",
            )
            logger.warning(f"Order for {signal.symbol} rejected by RiskEngine: {decision.reason}")
            return None

        # Risk approved -> Create order via OrderManager
        try:
            order = self.order_manager.create_order(request)
        except ValueError as e:
            # E.g. Duplicate order detected
            self._log_event(
                EventType.RISK_REJECTED,
                {"reason": str(e), "signal_id": signal.signal_id},
                correlation_id=signal.signal_id,
                message=str(e),
            )
            return None

        self._log_event(
            EventType.ORDER_SUBMITTED,
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
            },
            correlation_id=order.order_id,
        )

        # Submit to BrokerAdapter
        updated_order = self.broker.submit_order(order)
        self._persist_current_state()
        return updated_order

    def submit_manual_order(self, request: OrderRequest, market_price: Optional[float] = None) -> Order:
        """
        Safely submits a manual order.
        MANDATORY: Gated by RiskEngine before reaching the broker.
        """
        with self._lock:
            # Risk check
            decision = self.risk_engine.evaluate_order(
                request=request,
                current_positions=self._positions,
                market_price=market_price or request.price,
                market_price_timestamp=datetime.now(),
            )

            if not decision.allowed:
                raise PermissionError(f"Manual order rejected by RiskEngine: {decision.reason}")

            order = self.order_manager.create_order(request)
            submitted_order = self.broker.submit_order(order)
            self._persist_current_state()
            return submitted_order

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            return self.broker.cancel_order(order_id)

    def _on_broker_order_update(self, order: Order) -> None:
        """Handles order lifecycle updates from broker."""
        with self._lock:
            self._log_event(
                EventType.ORDER_ACCEPTED if order.status == OrderStatus.ACCEPTED else EventType.SYSTEM,
                {"order_id": order.order_id, "status": order.status.value, "reason": order.rejection_reason},
                correlation_id=order.order_id,
            )
            self._persist_current_state()

    def _on_broker_fill(self, fill: Fill) -> None:
        """Handles fill events from broker."""
        with self._lock:
            # 1. Update OMS order
            self.order_manager.record_fill(fill)

            # 2. Update Position
            if fill.symbol not in self._positions:
                self._positions[fill.symbol] = Position(symbol=fill.symbol)

            realized_pnl = self._positions[fill.symbol].apply_fill(fill)

            # Clean up zero-share positions
            if self._positions[fill.symbol].quantity == 0:
                del self._positions[fill.symbol]

            # 3. Notify RiskEngine
            self.risk_engine.record_trade_execution(realized_pnl)

            # 4. Audit log
            self._log_event(
                EventType.FILL,
                {
                    "fill_id": fill.fill_id,
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "commission": fill.commission,
                    "tax": fill.tax,
                    "realized_pnl": realized_pnl,
                },
                correlation_id=fill.order_id,
            )

            self._persist_current_state()

    def reconcile_with_broker(self) -> ReconciliationReport:
        """
        Executes full state reconciliation between internal state and broker authoritative state.
        """
        with self._lock:
            broker_positions = self.broker.get_positions()
            broker_open_orders = self.broker.get_open_orders()
            internal_open_orders = self.order_manager.get_open_orders()

            report = self.reconciler.run_full_reconciliation(
                internal_positions=self._positions,
                broker_positions=broker_positions,
                internal_open_orders=internal_open_orders,
                broker_open_orders=broker_open_orders,
            )

            self._log_event(
                EventType.SYSTEM,
                {
                    "action": "RECONCILIATION",
                    "is_clean": report.is_clean,
                    "critical_count": report.critical_count,
                    "warning_count": report.warning_count,
                },
            )

            return report

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            kill_status = self.risk_engine.kill_switch.get_status()
            market_session = self.market_clock.get_session()
            broker_acc = self.broker.get_account()

            total_unrealized_pnl = sum(
                p.unrealized_pnl(self._latest_bars[p.symbol].close if p.symbol in self._latest_bars else None)
                for p in self._positions.values()
            )
            total_realized_pnl = sum(p.realized_pnl for p in self._positions.values())

            return {
                "running": self._is_running,
                "trading_mode": self.trading_mode,
                "market_session": market_session.value,
                "kill_switch": {
                    "status": kill_status.status.value,
                    "reason": kill_status.reason,
                    "operator_id": kill_status.operator_id,
                },
                "positions_count": len(self._positions),
                "open_orders_count": len(self.order_manager.get_open_orders()),
                "daily_trades": self.risk_engine._daily_trades_count,
                "daily_realized_loss": self.risk_engine._daily_realized_loss,
                "total_realized_pnl": total_realized_pnl,
                "total_unrealized_pnl": total_unrealized_pnl,
                "broker": broker_acc,
            }
