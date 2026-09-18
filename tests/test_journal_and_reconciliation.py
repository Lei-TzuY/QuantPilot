import os
import shutil
import tempfile
import unittest
from datetime import datetime

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.fills import Fill
from modules.execution.journal import ExecutionJournal
from modules.execution.order import Order, OrderRequest, OrderSide, OrderStatus, OrderType
from modules.execution.order_manager import OrderManager
from modules.execution.position import Position
from modules.execution.reconciliation import Reconciler, DiscrepancySeverity
from modules.risk.engine import RiskEngine
from modules.risk.kill_switch import KillSwitch, KillSwitchStatus
from modules.risk.limits import RiskLimits


class TestJournalAndOrderStateMachine(unittest.TestCase):
    """
    Tests SQLite WAL execution journal, order state machine invariants,
    idempotency, and authoritative broker reconciliation on reconnect.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_journal.db")
        self.kill_switch_path = os.path.join(self.temp_dir, "test_kill_switch.json")
        self.journal = ExecutionJournal(db_path=self.db_path)
        self.oms = OrderManager()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ---------------- 1. Journal & Restart Tests ----------------
    def test_journal_persists_and_restores_state(self):
        """Journal records requests, transitions, and fills, and deterministically restores state."""
        now = datetime.now()
        order_id = "ORD-TEST-001"

        # Record order request
        self.journal.record_order_request(
            order_id=order_id,
            timestamp=now,
            symbol="2330",
            side="BUY",
            order_type="LIMIT",
            quantity=2000,
            price=1000.0,
            strategy_id="ma_strat",
        )

        # Record transitions
        self.journal.record_order_transition(order_id, now, "NEW", "SUBMITTED", broker_order_id="BRK-001")
        self.journal.record_order_transition(order_id, now, "SUBMITTED", "ACCEPTED", broker_order_id="BRK-001")

        # Record partial fill 1000 @ 1000.0
        fill1 = Fill(
            fill_id="FILL-001",
            order_id=order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0,
            commission=1425.0,
            tax=0.0,
            timestamp=now,
            broker_order_id="BRK-001",
        )
        self.journal.record_fill(fill1)

        # Record second fill 1000 @ 1010.0 (Completing the order)
        fill2 = Fill(
            fill_id="FILL-002",
            order_id=order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1010.0,
            commission=1439.0,
            tax=0.0,
            timestamp=now,
            broker_order_id="BRK-001",
        )
        self.journal.record_fill(fill2)
        self.journal.record_order_transition(order_id, now, "PARTIALLY_FILLED", "FILLED", broker_order_id="BRK-001")

        # Restore from clean journal instance
        new_journal = ExecutionJournal(db_path=self.db_path)
        state = new_journal.restore_state()

        self.assertIn(order_id, state["orders"])
        restored_order = state["orders"][order_id]
        self.assertEqual(restored_order.status, OrderStatus.FILLED)
        self.assertEqual(restored_order.filled_quantity, 2000)
        self.assertEqual(restored_order.remaining_quantity, 0)
        # VWAP = (1000 * 1000 + 1000 * 1010) / 2000 = 1005.0
        self.assertEqual(restored_order.average_fill_price, 1005.0)
        self.assertEqual(len(state["fills"]), 2)

    def test_journal_retains_kill_switch_halt_across_restart(self):
        """If KillSwitch was halted before shutdown, restart preserves halted status."""
        self.journal.record_kill_switch(
            event_id="KS-001",
            timestamp=datetime.now(),
            action="HALT",
            operator_id="RISK_MANAGER",
            reason="Emergency risk limit exceeded",
        )

        state = self.journal.restore_state()
        self.assertTrue(state["is_halted"])

    # ---------------- 2. Order State Machine Invariants ----------------
    def test_reject_fill_after_order_rejection(self):
        """An order that has been rejected cannot subsequently accept fills."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_rejection(order.order_id, reason="Insufficient funds")
        self.assertEqual(order.status, OrderStatus.REJECTED)

        fill = Fill(
            fill_id="FILL-LATE",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0,
        )

        with self.assertRaises(ValueError) as ctx:
            self.oms.record_fill(fill)
        self.assertIn("Cannot fill terminal order", str(ctx.exception))
        # Ensure quantities and status were not corrupted
        self.assertEqual(order.filled_quantity, 0)
        self.assertEqual(order.status, OrderStatus.REJECTED)

    def test_overfill_rejected(self):
        """A fill exceeding remaining quantity must be rejected."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id)
        self.oms.record_acceptance(order.order_id)

        overfill = Fill(
            fill_id="FILL-OVER",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1500,  # exceeds 1000!
            price=1000.0,
        )

        with self.assertRaises(ValueError) as ctx:
            self.oms.record_fill(overfill)
        self.assertIn("Overfill detected", str(ctx.exception))
        self.assertEqual(order.filled_quantity, 0)
        self.assertEqual(order.remaining_quantity, 1000)

    def test_duplicate_fill_callback_is_idempotent(self):
        """Duplicate broker fill callback with identical fill_id is ignored safely."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id)
        self.oms.record_acceptance(order.order_id)

        fill = Fill(
            fill_id="FILL-DUP",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=500,
            price=1000.0,
        )

        # First fill
        res1 = self.oms.record_fill(fill)
        self.assertEqual(res1.filled_quantity, 500)
        self.assertEqual(res1.status, OrderStatus.PARTIALLY_FILLED)

        # Second fill callback (network replay)
        res2 = self.oms.record_fill(fill)
        self.assertEqual(res2.filled_quantity, 500)
        self.assertEqual(res2.status, OrderStatus.PARTIALLY_FILLED)

    def test_cancel_fill_race_condition(self):
        """Cannot cancel an order that is already completely filled."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id)
        self.oms.record_acceptance(order.order_id)

        fill = Fill(
            fill_id="FILL-FULL",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0,
        )
        self.oms.record_fill(fill)
        self.assertEqual(order.status, OrderStatus.FILLED)

        # Cancellation attempt after complete fill
        with self.assertRaises(ValueError) as ctx:
            self.oms.record_cancellation(order.order_id, reason="Late user cancel")
        self.assertIn("Cannot cancel completely filled order", str(ctx.exception))
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_late_broker_acceptance_callback_ignored_when_filled(self):
        """Late ACCEPTED callback arriving after order has already FILLED is safely ignored."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id, broker_order_id="BRK-LATE")

        fill = Fill(
            fill_id="FILL-FAST",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0,
            broker_order_id="BRK-LATE",
        )
        self.oms.record_fill(fill)
        self.assertEqual(order.status, OrderStatus.FILLED)

        # Out of order acceptance from broker
        res = self.oms.record_acceptance(order.order_id, broker_order_id="BRK-LATE")
        self.assertEqual(res.status, OrderStatus.FILLED)  # Remains FILLED, not reverted!

    # ---------------- 3. Reconnect & Authoritative Reconciliation ----------------
    def test_reconnect_reconciles_harmless_order_difference(self):
        """When reconnecting, SUBMITTED order that became ACCEPTED at broker is updated cleanly."""
        broker = PaperBrokerAdapter(initial_cash=5_000_000)
        broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=kill_switch)
        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            trading_mode="paper",
        )

        # Create internal order marked SUBMITTED
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=1000, price=900.0)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id, broker_order_id="BRK-999")

        # Mock broker holding this order as ACCEPTED
        mock_broker_order = Order(
            order_id="BRK-999",
            broker_order_id="BRK-999",
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=900.0,
            status=OrderStatus.ACCEPTED,
        )
        broker.get_open_orders = lambda: [mock_broker_order]

        # Trigger reconnect
        report = engine.on_broker_reconnect()
        self.assertEqual(order.status, OrderStatus.ACCEPTED)
        self.assertEqual(report.critical_count, 0)
        self.assertFalse(kill_switch.is_halted())

    def test_reconnect_halts_on_critical_position_mismatch(self):
        """When broker position differs dangerously from internal state, reconnect halts trading."""
        broker = PaperBrokerAdapter(initial_cash=5_000_000)
        broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=kill_switch)
        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            trading_mode="paper",
        )

        # Internal state records 0 shares of 2330
        engine._positions = {}

        # Broker holds 2000 shares of 2330 unexpectedly (e.g. manual fill outside system)
        broker.get_positions = lambda: {"2330": Position(symbol="2330", quantity=2000, avg_price=1000.0)}

        # Reconnect reconciliation
        report = engine.on_broker_reconnect()
        self.assertFalse(report.is_clean)
        self.assertTrue(report.critical_count > 0)

        # Kill switch MUST be engaged automatically
        self.assertTrue(kill_switch.is_halted())
        self.assertEqual(kill_switch.get_status().status, KillSwitchStatus.HALTED)
        self.assertIn("Authoritative reconciliation mismatch", kill_switch.get_status().reason)


if __name__ == "__main__":
    unittest.main()
