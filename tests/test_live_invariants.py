import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from modules.brokers.paper import PaperBrokerAdapter
from modules.brokers.shioaji import ShioajiBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent
from modules.execution.fills import Fill
from modules.execution.journal import ExecutionJournal
from modules.execution.order import OrderRequest, OrderSide, OrderType, OrderStatus
from modules.execution.order_manager import OrderManager
from modules.execution.persistence import ExecutionStatePersistence
from modules.execution.position import Position
from modules.risk.engine import RiskEngine
from modules.risk.kill_switch import KillSwitch, KillSwitchStatus
from modules.risk.limits import RiskLimits
from modules.strategy.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Strategy that can only emit signals, having no broker access."""
    def __init__(self):
        super().__init__("dummy_strat")
        self.signal_to_emit = None

    def on_bar(self, bar: BarEvent):
        if self.signal_to_emit:
            return self.signal_to_emit
        return None


class TestLiveModeSafetyInvariants(unittest.TestCase):
    """
    Dedicated test suite enforcing all 10 production-safety invariants:
    1. CI can never place a live order.
    2. Missing credentials cannot downgrade into unsafe behavior.
    3. TRADING_MODE=paper cannot reach Shioaji live submission.
    4. Kill switch blocks every order path (automated & manual).
    5. Restart while halted remains halted.
    6. Dangerous reconciliation mismatch halts trading.
    7. Stale market data blocks new positions.
    8. Duplicate signal/order callbacks remain idempotent.
    9. No strategy has direct broker access.
    10. No public execution path bypasses RiskEngine.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_journal.db")
        self.kill_switch_path = os.path.join(self.temp_dir, "test_kill_switch.json")
        self.state_file_path = os.path.join(self.temp_dir, "test_state.json")
        self.journal = ExecutionJournal(db_path=self.db_path)
        self.persistence = ExecutionStatePersistence(storage_path=self.state_file_path)
        self.oms = OrderManager()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invariant_1_ci_can_never_place_live_order(self):
        """Invariant 1: In default CI/test environment, ShioajiBrokerAdapter refuses live order placement."""
        # Without ENABLE_LIVE_TRADING=true and TRADING_MODE=live, Shioaji refuses live submission
        with patch.dict(os.environ, {"CI": "true", "TRADING_MODE": "paper"}):
            adapter = ShioajiBrokerAdapter()
            order = self.oms.create_order(
                OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
            )
            result = adapter.submit_order(order)
            self.assertEqual(result.status, OrderStatus.REJECTED)
            self.assertIn("LIVE_TRADING_DISABLED", result.rejection_reason)

    def test_invariant_2_missing_credentials_fails_closed(self):
        """Invariant 2: Missing live broker credentials throws error, never downgrading into unsafe behavior."""
        with patch.dict(os.environ, {"TRADING_MODE": "live"}, clear=True):
            adapter = ShioajiBrokerAdapter(trading_mode="live")
            with self.assertRaises(ValueError) as ctx:
                adapter.connect()
            self.assertIn("Shioaji credentials missing", str(ctx.exception))

    def test_invariant_3_paper_mode_cannot_reach_shioaji(self):
        """Invariant 3: TRADING_MODE=paper engine interacts exclusively with PaperBrokerAdapter."""
        paper_broker = PaperBrokerAdapter(initial_cash=10_000_000)
        paper_broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        limits = RiskLimits(
            max_order_value=2_000_000.0,
            max_position_value_per_symbol=5_000_000.0,
            max_total_exposure=10_000_000.0,
        )
        risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)
        engine = ExecutionEngine(
            broker=paper_broker,
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            persistence=self.persistence,
            trading_mode="paper",
        )
        engine.start(reconcile_on_startup=False)

        # Submit manual order (400 TWD * 1000 = 400k <= 500k max_symbol_exposure)
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000, price=400.0)
        submitted = engine.submit_manual_order(req, market_price=400.0)

        self.assertIsInstance(engine.broker, PaperBrokerAdapter)
        self.assertEqual(submitted.status, OrderStatus.FILLED)
        # Verify Shioaji was never touched
        self.assertNotIn("Shioaji", type(engine.broker).__name__)

    def test_invariant_4_kill_switch_blocks_all_order_paths(self):
        """Invariant 4: Kill switch halts both automated strategy orders and manual API orders."""
        broker = PaperBrokerAdapter(initial_cash=10_000_000)
        broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=kill_switch)
        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            persistence=self.persistence,
            trading_mode="paper",
        )

        strategy = DummyStrategy()
        engine.register_strategy(strategy)

        # Halt system
        kill_switch.halt("Operator emergency halt", operator_id="ADMIN")
        self.assertTrue(kill_switch.is_halted())

        # Path A: Automated bar signal
        now = datetime.now()
        bar = BarEvent("2330", now, 1000.0, 1010.0, 995.0, 1005.0, 5000)
        strategy.signal_to_emit = SignalEvent("SIG-1", "dummy_strat", "2330", now, "BUY", 1.0)
        order_auto = engine.on_bar(bar)
        self.assertIsNone(order_auto)  # Blocked by kill switch!

        # Path B: Manual order submission
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        with self.assertRaises(PermissionError) as ctx:
            engine.submit_manual_order(req, market_price=1000.0)
        self.assertIn("KILL_SWITCH", str(ctx.exception))

    def test_invariant_5_restart_while_halted_remains_halted(self):
        """Invariant 5: Restarting the platform while halted preserves the halt state."""
        # Halt in initial instance
        ks1 = KillSwitch(state_file=self.kill_switch_path)
        ks1.halt("Pre-shutdown emergency halt", operator_id="SRE_LEAD")
        self.assertTrue(ks1.is_halted())

        # Restart with new instance
        ks2 = KillSwitch(state_file=self.kill_switch_path)
        self.assertTrue(ks2.is_halted())
        self.assertEqual(ks2.get_status().status, KillSwitchStatus.HALTED)
        self.assertIn("Pre-shutdown emergency halt", ks2.get_status().reason)

    def test_invariant_6_dangerous_reconciliation_mismatch_halts_trading(self):
        """Invariant 6: Reconciliation mismatch between broker and internal records triggers emergency halt."""
        broker = PaperBrokerAdapter(initial_cash=5_000_000)
        broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=kill_switch)
        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            persistence=self.persistence,
            trading_mode="paper",
        )

        # Internal state records 1,000 shares of 2330
        engine._positions["2330"] = Position(symbol="2330", quantity=1000, avg_price=900.0)
        # Broker authoritative record has 0 shares (position missing at broker!)
        broker.get_positions = lambda: {}

        report = engine.reconcile_with_broker()
        self.assertTrue(report.critical_count > 0)
        self.assertTrue(kill_switch.is_halted())
        self.assertIn("Authoritative reconciliation mismatch", kill_switch.get_status().reason)

    def test_invariant_7_stale_market_data_blocks_new_positions(self):
        """Invariant 7: Stale market data timestamp blocks opening new positions."""
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        limits = RiskLimits(max_stale_data_seconds=10.0)
        risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)

        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        current_time = datetime.now()
        # Market price was last received 60 seconds ago (stale!)
        stale_time = current_time - timedelta(seconds=60)

        decision = risk_engine.evaluate_order(
            request=req,
            current_positions={},
            market_price=1000.0,
            market_price_timestamp=stale_time,
            current_time=current_time,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.rule_violated, "STALE_MARKET_DATA")
        self.assertIn("STALE_MARKET_DATA", decision.reason)

    def test_invariant_8_duplicate_callbacks_remain_idempotent(self):
        """Invariant 8: Duplicate broker fill/order update callbacks do not duplicate quantities or state."""
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        order = self.oms.create_order(req)
        self.oms.record_submission(order.order_id)
        self.oms.record_acceptance(order.order_id)

        fill = Fill(
            fill_id="FILL-UNIQUE-101",
            order_id=order.order_id,
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=1000.0,
        )

        # First fill callback
        res1 = self.oms.record_fill(fill)
        self.assertEqual(res1.status, OrderStatus.FILLED)
        self.assertEqual(res1.filled_quantity, 1000)

        # Duplicate fill callback
        res2 = self.oms.record_fill(fill)
        self.assertEqual(res2.status, OrderStatus.FILLED)
        self.assertEqual(res2.filled_quantity, 1000)  # NOT 2000!

    def test_invariant_9_no_strategy_has_direct_broker_access(self):
        """Invariant 9: Strategy interfaces are pure signal emitters and do not have BrokerAdapter references."""
        strategy = DummyStrategy()
        self.assertFalse(hasattr(strategy, "broker"))
        self.assertFalse(hasattr(strategy, "submit_order"))
        self.assertFalse(hasattr(strategy, "cancel_order"))

    def test_invariant_10_no_execution_path_bypasses_risk_engine(self):
        """Invariant 10: Every execution path routes through RiskEngine pre-trade limits."""
        broker = PaperBrokerAdapter(initial_cash=5_000_000)
        broker.connect()
        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        mock_risk_engine = MagicMock(spec=RiskEngine)
        # Mock risk engine rejecting any order
        mock_risk_engine.evaluate_order.return_value = MagicMock(
            allowed=False, reason="RISK_BLOCKED", rule_violated="TEST_RULE", metrics_snapshot={}
        )
        mock_risk_engine.kill_switch = kill_switch

        engine = ExecutionEngine(
            broker=broker,
            risk_engine=mock_risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            trading_mode="paper",
        )

        # Try manual order
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        with self.assertRaises(PermissionError):
            engine.submit_manual_order(req, market_price=1000.0)

        mock_risk_engine.evaluate_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
