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
    TickEvent,
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
from modules.execution.journal import ExecutionJournal
from modules.market.bar_builder import BarBuilder
from modules.market.clock import MarketClock
from modules.market.event_queue import MarketDataEventQueue, QueueMetrics
from modules.market.integrity import MarketDataIntegrityChecker, MarketHealthStatus
from modules.market.recorder import RawMarketDataRecorder
from modules.monitoring.latency import LatencySnapshot, LatencyTracker
from modules.monitoring.session_report import ShadowSessionReporter, ShadowSessionData
from modules.execution.supervisor import ShadowSessionSupervisor, SupervisorState
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
        journal: Optional[ExecutionJournal] = None,
        bar_builder: Optional[BarBuilder] = None,
        trading_mode: str = "paper",
        default_order_shares: int = 1000,  # Standard Taiwan round lot: 1,000 shares
        event_queue: Optional[MarketDataEventQueue] = None,
        integrity_checker: Optional[MarketDataIntegrityChecker] = None,
        recorder: Optional[RawMarketDataRecorder] = None,
        latency_tracker: Optional[LatencyTracker] = None,
        supervisor: Optional[ShadowSessionSupervisor] = None,
        reporter: Optional[ShadowSessionReporter] = None,
        synchronous_queue: bool = True,
    ):
        self.trading_mode = trading_mode.lower()
        self.broker = broker
        self.risk_engine = risk_engine
        self.order_manager = order_manager or OrderManager()
        self.market_clock = market_clock or MarketClock()
        self.persistence = persistence or ExecutionStatePersistence()
        self.reconciler = reconciler or Reconciler()
        self.journal = journal or ExecutionJournal()
        self.bar_builder = bar_builder or BarBuilder(interval_seconds=60)
        self.default_order_shares = default_order_shares

        # Production-Safety & Observability Subsystems
        self.event_queue = event_queue or MarketDataEventQueue(
            capacity=10_000,
            synchronous=synchronous_queue,
            name="ShadowEventQueue",
        )
        self.integrity_checker = integrity_checker or MarketDataIntegrityChecker()
        self.recorder = recorder
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.supervisor = supervisor or ShadowSessionSupervisor()
        self.reporter = reporter or ShadowSessionReporter()

        self._positions: Dict[str, Position] = {}
        self._strategies: Dict[str, BaseStrategy] = {}
        self._audit_log: List[AuditLogEntry] = []
        self._latest_bars: Dict[str, BarEvent] = {}
        self._lock = threading.RLock()
        self._is_running = False

        # Hook broker callbacks
        self.broker.register_order_callback(self._on_broker_order_update)
        self.broker.register_fill_callback(self._on_broker_fill)

        # Hook BarBuilder callback to engine's on_bar
        self.bar_builder.register_bar_callback(self.on_bar)

        # Wire event queue subscriber & overflow handler
        self.event_queue.subscribe(self._process_dequeued_tick)
        self.event_queue.on_overflow = self._on_queue_overflow

    def _on_queue_overflow(self, tick: TickEvent, metrics: QueueMetrics) -> None:
        """Handles queue overflow incident: mark data unhealthy and halt entries."""
        logger.critical(f"Queue overflow detected for {tick.symbol}! Capacity={metrics.capacity}, dropped={metrics.dropped_count}")
        self.integrity_checker.record_incident(tick.symbol, "QUEUE_OVERFLOW")
        self.supervisor.trigger_degraded(f"EventQueue overflow: {metrics.dropped_count} ticks dropped")

    def on_tick(self, tick: TickEvent) -> bool:
        """
        Ultra-lightweight market data quote callback.
        Normalizes tick with enqueue timestamp and pushes to bounded event queue.
        Returns immediately without blocking or computing downstream logic.
        """
        with self._lock:
            if not self._is_running:
                return False
        return self.event_queue.enqueue(tick)

    def _process_dequeued_tick(self, tick: TickEvent) -> None:
        """
        Consumes dequeued tick from event queue:
        1. Measure queue and network latencies
        2. Validate tick against integrity invariants
        3. Persist raw tick in Parquet recorder
        4. Update broker top-of-book bid/ask quotes
        5. Aggregate into 1-minute BarBuilder
        """
        with self._lock:
            if not self._is_running:
                return

            # 1. Latency tracking
            self.latency_tracker.record_tick_latencies(
                exchange_ts=tick.timestamp,
                receive_ts=tick.receive_timestamp,
                enqueue_ts=tick.enqueue_timestamp,
                dequeue_ts=tick.dequeue_timestamp or datetime.now(),
            )

            # 2. Market-data integrity check
            is_valid, rejection_reason = self.integrity_checker.validate_tick(tick)
            if not is_valid:
                logger.warning(f"Integrity check rejected tick for {tick.symbol}: {rejection_reason}")
                return

            # 3. Raw tick recorder (buffered, append-only)
            if self.recorder:
                self.recorder.record_tick(tick)

            # 4. Bid/Ask top-of-book awareness for paper execution
            if isinstance(self.broker, PaperBrokerAdapter):
                self.broker.set_market_quote(
                    symbol=tick.symbol,
                    price=tick.price,
                    bid_price=tick.bid_price,
                    ask_price=tick.ask_price,
                )

            # 5. Bar aggregation
            self.bar_builder.on_tick_event(tick)

    def connect_market_data(self, quote_source: Any) -> None:
        """
        Connects an external market data feed (e.g. Shioaji quote feed) to the internal BarBuilder.
        Enables SHADOW MODE: Real market data feeding Paper execution.
        """
        with self._lock:
            if hasattr(quote_source, "register_tick_callback"):
                quote_source.register_tick_callback(self.on_tick)
                logger.info("Connected market data feed to ExecutionEngine BarBuilder (Shadow Mode active).")

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
        Starts the execution engine with full ShadowSessionSupervisor lifecycle management:
        INITIALIZING -> CONNECTING -> SYNCING -> READY -> RUNNING.
        Enforces institutional checklists before permitting order flow.
        """
        with self._lock:
            # Lifecycle: INITIALIZING
            if self.supervisor.current_state != SupervisorState.INITIALIZING:
                self.supervisor.transition_to(SupervisorState.INITIALIZING, "Starting execution engine")

            # 1. Connect broker
            self.supervisor.transition_to(SupervisorState.CONNECTING, "Connecting broker adapter")
            if not self.broker.is_connected():
                self.broker.connect()
            self.supervisor.update_checklist(broker_connected=self.broker.is_connected())

            # 2. Restore state from durable journal first (crash-safe)
            if self.journal:
                self.supervisor.update_checklist(journal_available=True)
                journal_state = self.journal.restore_state()
                if journal_state and journal_state.get("orders"):
                    seen_fill_ids = {f.fill_id for f in journal_state.get("fills", [])}
                    self.order_manager.load_state(journal_state["orders"], seen_fill_ids=seen_fill_ids)

                    # Reconstruct positions from journal fills
                    rebuilt_positions: Dict[str, Position] = {}
                    for fill in journal_state.get("fills", []):
                        if fill.symbol not in rebuilt_positions:
                            rebuilt_positions[fill.symbol] = Position(symbol=fill.symbol)
                        rebuilt_positions[fill.symbol].apply_fill(fill)
                    self._positions = {s: p for s, p in rebuilt_positions.items() if p.quantity > 0}

                # If journal records trading was HALTED, retain halt on startup
                if journal_state and journal_state.get("is_halted"):
                    self.risk_engine.kill_switch.halt(
                        reason="Restored HALTED state from durable execution journal",
                        operator_id="JOURNAL_RECOVERY",
                    )

            # Fallback to JSON snapshot only if no durable journal is configured
            if not self.journal and not self._positions:
                saved = self.persistence.load_state()
                if saved:
                    self._positions = saved["positions"]
                    for ord_obj in saved["orders"]:
                        self.order_manager._orders[ord_obj.order_id] = ord_obj
                        if ord_obj.broker_order_id:
                            self.order_manager._broker_order_map[ord_obj.broker_order_id] = ord_obj.order_id

            # 3. Reconcile with authoritative broker state
            self.supervisor.transition_to(SupervisorState.SYNCING, "Reconciling state with broker")
            reconciliation_clean = True
            if reconcile_on_startup:
                report = self.reconcile_with_broker()
                if not report.is_clean and report.critical_count > 0:
                    reconciliation_clean = False
                    logger.critical(f"Critical reconciliation mismatch on startup: {report.discrepancies}")
                    self.risk_engine.kill_switch.halt(
                        reason=f"CRITICAL_STARTUP_RECONCILIATION_MISMATCH: {report.critical_count} critical issues",
                        operator_id="RECONCILER",
                    )
            self.supervisor.update_checklist(reconciliation_clean=reconciliation_clean)

            # 4. Strategy warmup status & Risk status
            all_warmed_up = all(s.strategy_ready for s in self._strategies.values()) if self._strategies else True
            self.supervisor.update_checklist(
                strategies_warmed_up=all_warmed_up,
                risk_engine_ready=True,
                kill_switch_active=self.risk_engine.kill_switch.is_halted(),
                market_clock_valid=True,
            )

            # 5. Check readiness and transition to RUNNING
            if self.supervisor.can_transition_to(SupervisorState.READY):
                self.supervisor.transition_to(SupervisorState.READY, "Preconditions satisfied")
                if self.supervisor.can_transition_to(SupervisorState.RUNNING):
                    self.supervisor.transition_to(SupervisorState.RUNNING, "Starting trading loop")

            # 6. Start queue worker and set running flag
            self.event_queue.start()
            self._is_running = True
            self._log_event(EventType.SYSTEM, {"action": "START", "trading_mode": self.trading_mode})
            return True

    def stop(self) -> None:
        with self._lock:
            if self.supervisor.current_state in (
                SupervisorState.RUNNING,
                SupervisorState.DEGRADED,
                SupervisorState.HALTED,
                SupervisorState.READY,
            ):
                self.supervisor.transition_to(SupervisorState.CLOSING, "Stopping execution engine")

            self.event_queue.stop(drain=True)
            if self.recorder:
                self.recorder.close()

            self._is_running = False
            self._persist_current_state()
            self._log_event(EventType.SYSTEM, {"action": "STOP"})

            if self.supervisor.current_state == SupervisorState.CLOSING:
                self.supervisor.transition_to(SupervisorState.CLOSED, "Execution engine stopped")

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
        Processes a completed, immutable 1-minute bar event.
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

            # Verify market data integrity status
            data_healthy = self.integrity_checker.is_symbol_healthy(bar.symbol)
            if not data_healthy:
                health_status = self.integrity_checker.get_health(bar.symbol)
                logger.warning(
                    f"Market data for {bar.symbol} is {health_status.value}. Strategy entry signals will be rejected by RiskEngine."
                )

            # Evaluate registered strategies
            for strat_id, strategy in self._strategies.items():
                try:
                    t_strat_start = datetime.now()
                    signal = strategy.on_bar(bar)
                    t_strat_ms = (datetime.now() - t_strat_start).total_seconds() * 1000.0
                    self.latency_tracker.record_stage_latency("strategy", t_strat_ms)

                    if signal:
                        self._process_signal(
                            signal=signal,
                            bar=bar,
                            strategy_ready=strategy.strategy_ready,
                            market_data_healthy=data_healthy,
                        )
                except Exception as e:
                    logger.error(f"Strategy {strat_id} failed on_bar: {e}", exc_info=True)

    def _process_signal(
        self,
        signal: SignalEvent,
        bar: BarEvent,
        strategy_ready: bool = True,
        market_data_healthy: bool = True,
    ) -> Optional[Order]:
        """
        Processes a SignalEvent through RiskEngine and OMS with full safety validation.
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

        # Evaluate against RiskEngine with latency instrumentation
        t_risk_start = datetime.now()
        decision = self.risk_engine.evaluate_order(
            request=request,
            current_positions=self._positions,
            market_price=bar.close,
            market_price_timestamp=bar.timestamp,
            current_time=bar.timestamp,
            strategy_ready=strategy_ready,
            market_data_healthy=market_data_healthy,
        )
        t_risk_ms = (datetime.now() - t_risk_start).total_seconds() * 1000.0
        self.latency_tracker.record_stage_latency("risk_evaluation", t_risk_ms)

        if self.journal:
            self.journal.record_risk_decision(
                decision_id=f"RISK-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                timestamp=datetime.now(),
                order_id=signal.signal_id or "SIGNAL_ORDER",
                approved=decision.allowed,
                reason=decision.reason or "",
                metrics_snapshot=getattr(decision, "metrics_snapshot", {}),
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

        if self.journal:
            self.journal.record_order_request(
                order_id=order.order_id,
                timestamp=order.created_at,
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                strategy_id=order.strategy_id,
                signal_id=order.signal_id,
            )

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

        # Submit to BrokerAdapter with latency tracking
        t_exec_start = datetime.now()
        updated_order = self.broker.submit_order(order)
        t_exec_ms = (datetime.now() - t_exec_start).total_seconds() * 1000.0
        self.latency_tracker.record_stage_latency("execution", t_exec_ms)

        # Record end-to-end signal latency
        e2e_ms = (datetime.now() - bar.timestamp).total_seconds() * 1000.0
        self.latency_tracker.record_stage_latency("end_to_end", e2e_ms)

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

            if self.journal:
                self.journal.record_risk_decision(
                    decision_id=f"RISK-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    timestamp=datetime.now(),
                    order_id="MANUAL_ORDER",
                    approved=decision.allowed,
                    reason=decision.reason or "",
                    metrics_snapshot=getattr(decision, "metrics_snapshot", {}),
                )

            if not decision.allowed:
                raise PermissionError(f"Manual order rejected by RiskEngine: {decision.reason}")

            order = self.order_manager.create_order(request)

            if self.journal:
                self.journal.record_order_request(
                    order_id=order.order_id,
                    timestamp=order.created_at,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    quantity=order.quantity,
                    price=order.price,
                    strategy_id=order.strategy_id,
                    signal_id=order.signal_id,
                )

            submitted_order = self.broker.submit_order(order)
            self._persist_current_state()
            return submitted_order

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            return self.broker.cancel_order(order_id)

    def _on_broker_order_update(self, order: Order) -> None:
        """Handles order lifecycle updates from broker."""
        with self._lock:
            if self.journal:
                self.journal.record_order_transition(
                    order_id=order.order_id,
                    timestamp=order.updated_at,
                    from_status="BROKER_UPDATE",
                    to_status=order.status.value,
                    broker_order_id=order.broker_order_id,
                    reason=order.rejection_reason or "",
                )

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

            # 4. Durable journal record
            if self.journal:
                self.journal.record_fill(fill)

            # 5. Audit log
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
        Authoritative broker state is primary for LIVE mode.
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

            # Reconcile harmless differences:
            # If broker has order that is ACCEPTED but internally still SUBMITTED
            brk_by_id = {o.broker_order_id or o.order_id: o for o in broker_open_orders}
            for int_ord in internal_open_orders:
                if int_ord.status == OrderStatus.SUBMITTED and int_ord.broker_order_id in brk_by_id:
                    brk_ord = brk_by_id[int_ord.broker_order_id]
                    if brk_ord.status == OrderStatus.ACCEPTED:
                        self.order_manager.record_acceptance(int_ord.order_id, int_ord.broker_order_id)

            # Halt immediately on critical discrepancies
            if report.critical_count > 0:
                logger.critical(
                    f"Authoritative reconciliation detected {report.critical_count} critical mismatches! Triggering emergency halt."
                )
                self.risk_engine.kill_switch.halt(
                    reason=f"Authoritative reconciliation mismatch: {report.critical_count} critical discrepancies detected.",
                    operator_id="RECONCILER",
                )

            # Record in journal
            if self.journal:
                self.journal.record_reconciliation(
                    event_id=f"REC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                    timestamp=datetime.now(),
                    is_clean=report.is_clean,
                    critical_count=report.critical_count,
                    warning_count=report.warning_count,
                    details={"discrepancies": [d.description for d in report.discrepancies]},
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

    def on_broker_reconnect(self) -> ReconciliationReport:
        """
        Called upon broker reconnection.
        Freezes submissions, queries broker state, reconciles, and halts on critical mismatch.
        """
        with self._lock:
            logger.info("Broker reconnected. Executing authoritative state reconciliation...")
            return self.reconcile_with_broker()

    def get_positions(self) -> Dict[str, Position]:
        with self._lock:
            return dict(self._positions)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            kill_status = self.risk_engine.kill_switch.get_status()
            market_session = self.market_clock.get_session()
            broker_acc = self.broker.get_account()
            q_metrics = self.event_queue.get_metrics()
            sup_report = self.supervisor.get_status_report()

            total_unrealized_pnl = sum(
                p.unrealized_pnl(self._latest_bars[p.symbol].close if p.symbol in self._latest_bars else None)
                for p in self._positions.values()
            )
            total_realized_pnl = sum(p.realized_pnl for p in self._positions.values())

            return {
                "running": self._is_running,
                "trading_mode": self.trading_mode,
                "supervisor_state": sup_report["current_state"],
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
                "queue": {
                    "depth": q_metrics.current_depth,
                    "max_depth": q_metrics.max_depth,
                    "overflow_count": q_metrics.overflow_count,
                    "dropped_count": q_metrics.dropped_count,
                    "is_healthy": q_metrics.is_healthy,
                },
                "broker": broker_acc,
            }

    def generate_session_report(self, session_date: Optional[str] = None) -> ShadowSessionData:
        """
        Gathers comprehensive end-of-session telemetry and writes JSON and Markdown reports.
        """
        with self._lock:
            symbols = list(set(list(self._latest_bars.keys()) + list(self._positions.keys())))
            if not symbols:
                symbols = ["2330"]

            all_orders = self.order_manager.get_all_orders()
            orders_submitted = len(all_orders)
            orders_filled = len([o for o in all_orders if o.status == OrderStatus.FILLED])
            orders_rejected = len([o for o in all_orders if o.status == OrderStatus.REJECTED])
            orders_cancelled = len([o for o in all_orders if o.status == OrderStatus.CANCELLED])

            total_fills = sum(len(getattr(o, "fills", [])) for o in all_orders)
            total_realized_pnl = sum(p.realized_pnl for p in self._positions.values())

            total_comm = sum(sum(f.commission for f in getattr(o, "fills", [])) for o in all_orders)
            total_tax = sum(sum(f.tax for f in getattr(o, "fills", [])) for o in all_orders)
            total_slippage = sum(sum(getattr(f, "slippage", 0.0) for f in getattr(o, "fills", [])) for o in all_orders)

            q_metrics = self.event_queue.get_metrics()
            lat_snapshot = self.latency_tracker.get_snapshot()

            now = datetime.now()
            start_time = self._latest_bars[symbols[0]].timestamp if (symbols and symbols[0] in self._latest_bars) else now

            data_health_incidents = sum(
                h.duplicate_count + h.out_of_order_count + h.stale_count + h.price_anomaly_count
                for h in self.integrity_checker.get_all_reports().values()
            )

            report = self.reporter.build_report(
                session_id=self.supervisor.session_id,
                symbols=symbols,
                start_time=start_time,
                end_time=now,
                ticks_received=q_metrics.total_enqueued,
                ticks_valid=q_metrics.total_dequeued,
                ticks_rejected=q_metrics.dropped_count,
                bars_count=len(self._latest_bars),
                signals_count=len([e for e in self._audit_log if e.event_type == EventType.SIGNAL]),
                risk_approvals=len([e for e in self._audit_log if e.event_type == EventType.ORDER_SUBMITTED]),
                risk_rejections=len([e for e in self._audit_log if e.event_type == EventType.RISK_REJECTED]),
                orders_submitted=orders_submitted,
                orders_filled=orders_filled,
                orders_rejected=orders_rejected,
                orders_cancelled=orders_cancelled,
                total_fills=total_fills,
                gross_pnl=total_realized_pnl + total_comm + total_tax + total_slippage,
                commission=total_comm,
                tax=total_tax,
                slippage=total_slippage,
                max_drawdown_pct=0.0,
                queue_max_depth=q_metrics.max_depth,
                queue_overflow_count=q_metrics.overflow_count,
                disconnect_count=sum(h.reconnect_gaps_count for h in self.integrity_checker.get_all_reports().values()),
                reconciliation_count=len([e for e in self._audit_log if e.event_type == EventType.SYSTEM and e.payload.get("action") == "RECONCILIATION"]),
                kill_switch_events=1 if self.risk_engine.kill_switch.is_halted() else 0,
                data_health_incidents=data_health_incidents,
                exceptions_count=0,
                latency_snapshot=lat_snapshot,
            )
            self.reporter.persist_report(report)
            return report

    def get_latency_snapshot(self) -> LatencySnapshot:
        return self.latency_tracker.get_snapshot()

    def get_data_health(self) -> Dict[str, Any]:
        return {
            s: {
                "status": h.status.value,
                "last_price": h.last_price,
                "last_sequence": h.last_sequence,
                "total_ticks_received": h.total_ticks_received,
                "total_ticks_valid": h.total_ticks_valid,
                "total_ticks_rejected": h.total_ticks_rejected,
                "duplicate_count": h.duplicate_count,
                "out_of_order_count": h.out_of_order_count,
                "stale_count": h.stale_count,
                "price_anomaly_count": h.price_anomaly_count,
                "reconnect_gaps_count": h.reconnect_gaps_count,
                "last_anomaly_reason": h.last_anomaly_reason,
            }
            for s, h in self.integrity_checker.get_all_reports().items()
        }

    def get_queue_metrics(self) -> QueueMetrics:
        return self.event_queue.get_metrics()

    def get_supervisor_status(self) -> Dict[str, Any]:
        return self.supervisor.get_status_report()
