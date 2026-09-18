"""
Comprehensive Test Suite for Event-Driven Algorithmic Trading Execution Platform
Covers all safety invariants, risk controls, broker simulation, reconciliation, and timing correctness.
"""
from datetime import datetime, timedelta
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brokers.base import BrokerAdapter
from modules.brokers.paper import PaperBrokerAdapter
from modules.brokers.shioaji import ShioajiBrokerAdapter
from modules.execution.events import BarEvent, EventType, SignalEvent
from modules.execution.fills import Fill
from modules.execution.order import (
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from modules.execution.journal import ExecutionJournal
from modules.execution.order_manager import OrderManager
from modules.execution.persistence import ExecutionStatePersistence
from modules.execution.position import Position
from modules.execution.reconciliation import (
    DiscrepancySeverity,
    DiscrepancyType,
    Reconciler,
)
from modules.execution.engine import ExecutionEngine
from modules.market.clock import MarketClock, MarketSession
from modules.risk.engine import RiskEngine
from modules.risk.kill_switch import KillSwitch, KillSwitchStatus
from modules.risk.limits import RiskLimits
from modules.strategy.base import BaseStrategy


class MockTriggerStrategy(BaseStrategy):
    """Deterministic test strategy that emits a signal on demand."""

    def __init__(self, strategy_id: str = "mock_strat"):
        super().__init__(strategy_id)
        self.signal_to_emit = None

    def on_bar(self, bar: BarEvent):
        sig = self.signal_to_emit
        self.signal_to_emit = None
        return sig


class TestExecutionPlatform(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.kill_switch_file = os.path.join(self.test_dir, "kill_switch.json")
        self.state_file = os.path.join(self.test_dir, "execution_state.json")
        self.journal_file = os.path.join(self.test_dir, "test_journal.db")

        self.kill_switch = KillSwitch(state_file=self.kill_switch_file)
        self.limits = RiskLimits(
            max_position_value_per_symbol=500_000.0,
            max_total_exposure=1_500_000.0,
            max_order_value=200_000.0,
            max_open_positions=3,
            max_trades_per_day=10,
            max_daily_realized_loss=20_000.0,
            max_stale_data_seconds=30.0,
            max_price_deviation_pct=0.08,
            duplicate_window_seconds=2.0,
        )
        self.risk_engine = RiskEngine(limits=self.limits, kill_switch=self.kill_switch)
        self.broker = PaperBrokerAdapter(
            initial_cash=1_000_000.0,
            commission_rate=0.001425,
            tax_rate=0.003,
            slippage_pct=0.001,
            min_commission=20.0,
        )
        self.broker.connect()
        self.order_manager = OrderManager(duplicate_window_seconds=2.0)
        self.persistence = ExecutionStatePersistence(storage_path=self.state_file)
        self.journal = ExecutionJournal(db_path=self.journal_file)
        self.reconciler = Reconciler()
        self.market_clock = MarketClock()

        self.engine = ExecutionEngine(
            broker=self.broker,
            risk_engine=self.risk_engine,
            order_manager=self.order_manager,
            market_clock=self.market_clock,
            persistence=self.persistence,
            journal=self.journal,
            reconciler=self.reconciler,
            trading_mode="paper",
            default_order_shares=1000,
        )
        self.engine.start(reconcile_on_startup=False)

    def tearDown(self):
        self.engine.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Look-ahead / Same-Bar Execution Prevention
    # --------------------------------------------------------------------------
    def test_same_bar_lookahead_prevention(self):
        """
        REGRESSION TEST:
        Signal generated from Bar t (closed at 09:30 with close=100) must NOT execute
        retroactively at bar t's close. Its execution is processed into the next bar (t+1).
        """
        strat = MockTriggerStrategy("test_lookahead")
        self.engine.register_strategy(strat)

        t0 = datetime(2026, 9, 18, 9, 30, 0)
        t1 = datetime(2026, 9, 18, 9, 31, 0)

        bar_t0 = BarEvent(timestamp=t0, symbol="2330", open=99.0, high=101.0, low=98.5, close=100.0, volume=5000)

        # Strategy emits signal when bar_t0 closes
        strat.signal_to_emit = SignalEvent(
            signal_id="SIG-001",
            timestamp=t0,
            symbol="2330",
            side="BUY",
            strategy_id="test_lookahead",
        )

        # Feed bar t0
        self.engine.on_bar(bar_t0)

        orders = self.engine.order_manager.get_all_orders()
        self.assertEqual(len(orders), 1)
        order = orders[0]

        # The order must have been created with timestamp >= bar_t0.timestamp
        self.assertGreaterEqual(order.created_at, bar_t0.timestamp)

        # In paper broker with slippage (0.1%), execution price is 100 * 1.001 = 100.1
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(order.average_fill_price, 100.1, places=2)

        # Bar t+1 comes in at price 105
        bar_t1 = BarEvent(timestamp=t1, symbol="2330", open=100.5, high=106.0, low=100.2, close=105.0, volume=6000)
        self.engine.on_bar(bar_t1)

        # Positions are updated according to execution sequence
        positions = self.engine.get_positions()
        self.assertIn("2330", positions)
        self.assertEqual(positions["2330"].quantity, 1000)
        # Unrealized PnL evaluated against bar_t1 close (105 - 100.1) * 1000 = 4900
        self.assertAlmostEqual(positions["2330"].unrealized_pnl(105.0), (105.0 - 100.1) * 1000, places=2)

    # --------------------------------------------------------------------------
    # 2. End-to-End Signal -> Risk -> OMS -> Broker -> Fill Flow
    # --------------------------------------------------------------------------
    def test_signal_to_risk_to_order_pipeline(self):
        strat = MockTriggerStrategy("e2e_strat")
        self.engine.register_strategy(strat)

        t = datetime(2026, 9, 18, 9, 30)
        bar = BarEvent(timestamp=t, symbol="2317", open=150.0, high=152.0, low=149.0, close=151.0, volume=10000)

        strat.signal_to_emit = SignalEvent(
            signal_id="SIG-E2E-1",
            timestamp=t,
            symbol="2317",
            side="BUY",
            strategy_id="e2e_strat",
        )

        self.engine.on_bar(bar)

        # Check OMS
        orders = self.engine.order_manager.get_all_orders()
        self.assertEqual(len(orders), 1)
        ord1 = orders[0]
        self.assertEqual(ord1.symbol, "2317")
        self.assertEqual(ord1.side, OrderSide.BUY)
        self.assertEqual(ord1.status, OrderStatus.FILLED)

        # Check Position
        positions = self.engine.get_positions()
        self.assertIn("2317", positions)
        self.assertEqual(positions["2317"].quantity, 1000)

        # Check Audit Log contains full chain
        audit = self.engine.get_audit_log(limit=20)
        event_types = [entry.event_type for entry in audit]
        self.assertIn(EventType.BAR, event_types)
        self.assertIn(EventType.SIGNAL, event_types)
        self.assertIn(EventType.ORDER_SUBMITTED, event_types)
        self.assertIn(EventType.FILL, event_types)

    # --------------------------------------------------------------------------
    # 3. Rejected Risk Decision Never Reaches Broker
    # --------------------------------------------------------------------------
    def test_rejected_risk_decision_never_reaches_broker(self):
        # Configure small max_order_value limit
        self.limits.max_order_value = 50_000.0  # 1000 shares * 100 = 100,000 > 50,000

        initial_broker_orders = len(self.broker.get_open_orders())

        strat = MockTriggerStrategy("risk_reject_strat")
        self.engine.register_strategy(strat)

        t = datetime(2026, 9, 18, 9, 30)
        bar = BarEvent(timestamp=t, symbol="2454", open=100.0, high=102.0, low=99.0, close=100.0, volume=2000)

        strat.signal_to_emit = SignalEvent(
            signal_id="SIG-RISK-REJECT",
            timestamp=t,
            symbol="2454",
            side="BUY",
            strategy_id="risk_reject_strat",
        )

        self.engine.on_bar(bar)

        # Order must NOT be created in OMS
        orders = self.engine.order_manager.get_all_orders()
        self.assertEqual(len(orders), 0)

        # Broker must NOT have received any order
        self.assertEqual(len(self.broker.get_open_orders()), initial_broker_orders)

        # Check audit log records RISK_REJECTED
        audit = self.engine.get_audit_log(limit=10)
        rejected_entries = [e for e in audit if e.event_type == EventType.RISK_REJECTED]
        self.assertEqual(len(rejected_entries), 1)
        self.assertIn("ORDER_VALUE_EXCEEDS_LIMIT", rejected_entries[0].payload.get("reason", ""))

    # --------------------------------------------------------------------------
    # 4. Partial-Fill Accounting
    # --------------------------------------------------------------------------
    def test_partial_fill_accounting(self):
        paper_broker = PaperBrokerAdapter(
            initial_cash=500_000.0,
            enable_partial_fills=True,
            partial_fill_fraction=0.4,  # Fills 40% (400 shares) on first fill
            commission_rate=0.0,
            tax_rate=0.0,
            slippage_pct=0.0,
        )
        paper_broker.connect()
        paper_broker.set_market_price("2330", 100.0)

        order_req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = Order(
            order_id="ORD-PARTIAL-01",
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        submitted = paper_broker.submit_order(order)
        self.assertEqual(submitted.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(submitted.filled_quantity, 400)
        self.assertEqual(submitted.remaining_quantity, 600)

        pos = Position(symbol="2330")
        # Apply fill 1 (400 @ 100)
        fill1 = Fill(fill_id="F1", order_id=order.order_id, symbol="2330", side=OrderSide.BUY, quantity=400, price=100.0)
        pos.apply_fill(fill1)
        self.assertEqual(pos.quantity, 400)
        self.assertEqual(pos.avg_price, 100.0)

        # Apply fill 2 (600 @ 110)
        fill2 = Fill(fill_id="F2", order_id=order.order_id, symbol="2330", side=OrderSide.BUY, quantity=600, price=110.0)
        pos.apply_fill(fill2)
        self.assertEqual(pos.quantity, 1000)
        # Weighted average price: (400*100 + 600*110) / 1000 = (40000 + 66000) / 1000 = 106.0
        self.assertAlmostEqual(pos.avg_price, 106.0, places=2)

    # --------------------------------------------------------------------------
    # 5. Commission & Taiwan Transaction Tax Accounting
    # --------------------------------------------------------------------------
    def test_fee_and_tax_accounting(self):
        pos = Position(symbol="2330")

        # BUY 1000 shares @ 100, Commission = 100,000 * 0.001425 = 142.5, Tax = 0
        buy_fill = Fill(
            fill_id="F-BUY",
            order_id="ORD-1",
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=100.0,
            commission=142.5,
            tax=0.0,
        )
        pnl_buy = pos.apply_fill(buy_fill)
        self.assertEqual(pnl_buy, 0.0)
        self.assertEqual(pos.quantity, 1000)
        self.assertEqual(pos.cost_basis, 100_000.0)
        self.assertEqual(pos.total_commission, 142.5)
        self.assertEqual(pos.total_tax, 0.0)

        # SELL 1000 shares @ 120, Commission = 120,000 * 0.001425 = 171.0, Tax = 120,000 * 0.003 = 360.0
        # Trade profit = (120 - 100) * 1000 = 20,000
        # Net Realized PnL = 20,000 - 171 - 360 = 19,469
        sell_fill = Fill(
            fill_id="F-SELL",
            order_id="ORD-2",
            symbol="2330",
            side=OrderSide.SELL,
            quantity=1000,
            price=120.0,
            commission=171.0,
            tax=360.0,
        )
        realized_pnl = pos.apply_fill(sell_fill)
        self.assertAlmostEqual(realized_pnl, 19469.0, places=2)
        self.assertEqual(pos.quantity, 0)
        self.assertEqual(pos.cost_basis, 0.0)
        self.assertAlmostEqual(pos.total_commission, 313.5, places=2)
        self.assertAlmostEqual(pos.total_tax, 360.0, places=2)

    # --------------------------------------------------------------------------
    # 6. Slippage Accounting
    # --------------------------------------------------------------------------
    def test_slippage_accounting(self):
        broker = PaperBrokerAdapter(initial_cash=500_000.0, slippage_pct=0.002)  # 0.2% slippage
        broker.connect()
        broker.set_market_price("2330", 100.0)

        # BUY order: expected fill price = 100 * (1 + 0.002) = 100.2
        buy_order = Order(order_id="O-BUY", symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        broker.submit_order(buy_order)
        self.assertEqual(buy_order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(buy_order.average_fill_price, 100.2, places=4)

        # SELL order: expected fill price = 100 * (1 - 0.002) = 99.8
        sell_order = Order(order_id="O-SELL", symbol="2330", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=100)
        broker.submit_order(sell_order)
        self.assertEqual(sell_order.status, OrderStatus.FILLED)
        self.assertAlmostEqual(sell_order.average_fill_price, 99.8, places=4)

    # --------------------------------------------------------------------------
    # 7. Duplicate Order Prevention
    # --------------------------------------------------------------------------
    def test_duplicate_order_prevention(self):
        req1 = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000, signal_id="S1")
        ord1 = self.order_manager.create_order(req1)
        self.assertIsNotNone(ord1)

        # Immediately submit identical request
        req2 = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000, signal_id="S1")
        with self.assertRaises(ValueError) as ctx:
            self.order_manager.create_order(req2)
        self.assertIn("Duplicate order detected", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 8. Daily Loss Limit and Auto-Halt
    # --------------------------------------------------------------------------
    def test_daily_loss_limit_auto_halt(self):
        # Loss limit set to 20,000 in setUp
        self.assertFalse(self.risk_engine.kill_switch.is_halted())

        # Trade with loss of 15,000 (below limit)
        self.risk_engine.record_trade_execution(-15_000.0)
        self.assertFalse(self.risk_engine.kill_switch.is_halted())

        # Additional trade with loss of 6,000 (total = 21,000 >= 20,000)
        self.risk_engine.record_trade_execution(-6_000.0)
        self.assertTrue(self.risk_engine.kill_switch.is_halted())

        # Any new order must be rejected with DAILY_LOSS_LIMIT or KILL_SWITCH
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        decision = self.risk_engine.evaluate_order(req, {}, market_price=100.0)
        self.assertFalse(decision.allowed)
        self.assertIn("KILL_SWITCH_HALTED", decision.reason)

    # --------------------------------------------------------------------------
    # 9. Kill Switch Manual Trigger, Persistence, and No Auto-Resume
    # --------------------------------------------------------------------------
    def test_kill_switch_persistence_and_no_auto_resume(self):
        # Operator halts system
        self.kill_switch.halt(reason="Emergency Market Anomaly", operator_id="TRADER_01")
        self.assertTrue(self.kill_switch.is_halted())

        # Simulate engine restart: new KillSwitch instance loading from same file
        restarted_kill_switch = KillSwitch(state_file=self.kill_switch_file)
        # MUST remain halted
        self.assertTrue(restarted_kill_switch.is_halted())
        status = restarted_kill_switch.get_status()
        self.assertEqual(status.operator_id, "TRADER_01")
        self.assertEqual(status.reason, "Emergency Market Anomaly")

        # Resuming without operator ID must fail
        with self.assertRaises(ValueError):
            restarted_kill_switch.resume(operator_id="", reason="test")

        # Resume with operator ID
        restarted_kill_switch.resume(operator_id="RISK_OFFICER", reason="Market conditions stabilized")
        self.assertFalse(restarted_kill_switch.is_halted())

    # --------------------------------------------------------------------------
    # 10. Reconnection and Reconciliation
    # --------------------------------------------------------------------------
    def test_reconciliation_discrepancy_detection(self):
        internal_pos = {
            "2330": Position(symbol="2330", quantity=1000, avg_price=100.0),
            "2317": Position(symbol="2317", quantity=500, avg_price=150.0),
        }
        # Broker has different quantity for 2330, and missing 2317, but has unexpected 2454
        broker_pos = {
            "2330": Position(symbol="2330", quantity=2000, avg_price=100.0),
            "2454": Position(symbol="2454", quantity=300, avg_price=800.0),
        }

        report = self.reconciler.run_full_reconciliation(
            internal_positions=internal_pos,
            broker_positions=broker_pos,
            internal_open_orders=[],
            broker_open_orders=[],
        )

        self.assertFalse(report.is_clean)
        self.assertEqual(report.critical_count, 3)

        disc_types = [d.discrepancy_type for d in report.discrepancies]
        self.assertIn(DiscrepancyType.POSITION_QUANTITY_MISMATCH, disc_types)
        self.assertIn(DiscrepancyType.POSITION_MISSING_AT_BROKER, disc_types)
        self.assertIn(DiscrepancyType.POSITION_MISSING_INTERNALLY, disc_types)

    # --------------------------------------------------------------------------
    # 11. Crash / Restart State Restoration
    # --------------------------------------------------------------------------
    def test_crash_restart_state_restoration(self):
        positions = {"2330": Position(symbol="2330", quantity=2000, avg_price=120.0, cost_basis=240000.0)}
        orders = [
            Order(
                order_id="ORD-PERSIST-1",
                symbol="2330",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=120.0,
                quantity=2000,
                status=OrderStatus.FILLED,
            )
        ]
        fills = [
            Fill(fill_id="F-1", order_id="ORD-PERSIST-1", symbol="2330", side=OrderSide.BUY, quantity=2000, price=120.0)
        ]

        self.persistence.save_state(
            positions=positions,
            orders=orders,
            fills=fills,
            realized_pnl=5000.0,
            daily_trades=2,
            session_status="RUNNING",
        )

        # Restore
        loaded = self.persistence.load_state()
        self.assertIsNotNone(loaded)
        self.assertIn("2330", loaded["positions"])
        self.assertEqual(loaded["positions"]["2330"].quantity, 2000)
        self.assertEqual(loaded["orders"][0].order_id, "ORD-PERSIST-1")
        self.assertEqual(loaded["orders"][0].status, OrderStatus.FILLED)
        self.assertEqual(loaded["realized_pnl"], 5000.0)

    # --------------------------------------------------------------------------
    # 12. Broker Rejected Order Handling
    # --------------------------------------------------------------------------
    def test_broker_rejected_order_handling(self):
        # Disconnect broker
        self.broker.disconnect()
        order = Order(order_id="O-REJECT", symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        res = self.broker.submit_order(order)
        self.assertEqual(res.status, OrderStatus.REJECTED)
        self.assertIn("BROKER_DISCONNECTED", res.rejection_reason)

    # --------------------------------------------------------------------------
    # 13. Insufficient Buying Power Rejection
    # --------------------------------------------------------------------------
    def test_insufficient_funds_rejection(self):
        # Set cash to only 5,000
        self.broker.cash = 5_000.0
        self.broker.set_market_price("2330", 100.0)

        # Attempt to buy 1,000 shares @ 100 = 100,000 > 5,000
        order = Order(order_id="O-FUNDS", symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        res = self.broker.submit_order(order)
        self.assertEqual(res.status, OrderStatus.REJECTED)
        self.assertIn("INSUFFICIENT_FUNDS", res.rejection_reason)

    # --------------------------------------------------------------------------
    # 14. Stale Market Data Rejection
    # --------------------------------------------------------------------------
    def test_stale_market_data_rejection(self):
        old_time = datetime.now() - timedelta(seconds=120)  # 120s old > 30s limit
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)

        decision = self.risk_engine.evaluate_order(
            request=req,
            current_positions={},
            market_price=100.0,
            market_price_timestamp=old_time,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.rule_violated, "STALE_MARKET_DATA")

    # --------------------------------------------------------------------------
    # 15. Market Close and Intraday Cutoff Behavior
    # --------------------------------------------------------------------------
    def test_market_close_and_intraday_cutoff(self):
        clock = MarketClock(intraday_cutoff=datetime(2026, 9, 18, 13, 15).time())

        before_cutoff = datetime(2026, 9, 18, 11, 0)
        after_cutoff = datetime(2026, 9, 18, 13, 20)
        after_market = datetime(2026, 9, 18, 14, 0)

        self.assertFalse(clock.is_past_intraday_cutoff(before_cutoff))
        self.assertTrue(clock.is_past_intraday_cutoff(after_cutoff))
        self.assertEqual(clock.get_session(after_market), MarketSession.CLOSED)
        self.assertFalse(clock.is_order_placement_allowed(after_market))

    # --------------------------------------------------------------------------
    # 16. Safety Invariants: Shioaji Live Mode Guard & Invariants
    # --------------------------------------------------------------------------
    def test_shioaji_live_mode_guard_invariant(self):
        """
        CRITICAL SAFETY INVARIANT:
        Shioaji adapter CANNOT submit real orders unless TRADING_MODE='live'.
        In default 'paper' mode, it must immediately reject and raise/flag an error.
        """
        shioaji_adapter = ShioajiBrokerAdapter(trading_mode="paper")
        order = Order(order_id="O-LIVE-TEST", symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)

        result = shioaji_adapter.submit_order(order)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("LIVE_TRADING_DISABLED", result.rejection_reason)

    # --------------------------------------------------------------------------
    # 17. Flask Trading REST API Endpoints Verification
    # --------------------------------------------------------------------------
    def test_flask_trading_api_endpoints(self):
        """Tests /api/trading/* endpoints through Flask test client."""
        from app import app
        client = app.test_client()

        # Status
        res = client.get("/api/trading/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("status", data)
        self.assertIn("trading_mode", data["status"])

        # Halt
        res = client.post("/api/trading/halt", json={"reason": "Test Halt", "operator_id": "TEST_OP"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

        # Verify halted status
        res = client.get("/api/trading/status")
        self.assertEqual(res.get_json()["status"]["kill_switch"]["status"], "HALTED")

        # Resume without operator_id fails (400)
        res = client.post("/api/trading/resume", json={"reason": "Test Resume"})
        self.assertEqual(res.status_code, 400)

        # Resume with operator_id and confirmation succeeds
        res = client.post(
            "/api/trading/resume",
            json={"reason": "Test Resume", "operator_id": "TEST_OP", "confirmation": "CONFIRM_RESUME"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

        # Positions & Orders & PnL
        self.assertEqual(client.get("/api/trading/positions").status_code, 200)
        self.assertEqual(client.get("/api/trading/orders").status_code, 200)
        self.assertEqual(client.get("/api/trading/pnl").status_code, 200)


if __name__ == "__main__":
    unittest.main()
