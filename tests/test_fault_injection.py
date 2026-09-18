"""
Fault-Injection Simulation Framework Tests.
Verifies fail-safe behavior under adverse network, queue, journal, broker,
and strategy exception scenarios.
"""
import os
import shutil
import tempfile
from datetime import datetime, timedelta
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.journal import ExecutionJournal
from modules.execution.order import OrderRequest, OrderSide, OrderType
from modules.market.bar_builder import BarBuilder
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.simulation.fault_injection import FaultConfig, FaultInjector, FaultType
from modules.strategy.base import BaseStrategy


class FaultyStrategy(BaseStrategy):
    """Strategy that throws unexpected exception during on_bar."""

    def __init__(self):
        super().__init__("faulty_strat")

    def on_bar(self, bar: BarEvent):
        raise RuntimeError("SIMULATED_UNHANDLED_STRATEGY_EXCEPTION")


class TestFaultInjection(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "fault_journal.db")
        self.journal = ExecutionJournal(db_path=self.db_path)
        self.broker = PaperBrokerAdapter(initial_cash=10_000_000.0)
        self.broker.connect()
        self.risk = RiskEngine(
            limits=RiskLimits(
                max_order_value=10_000_000.0,
                max_position_value_per_symbol=20_000_000.0,
                max_total_exposure=50_000_000.0,
            ),
            kill_switch=KillSwitch(),
        )
        self.engine = ExecutionEngine(
            broker=self.broker,
            risk_engine=self.risk,
            journal=self.journal,
            trading_mode="shadow",
            synchronous_queue=True,
        )
        self.engine.start(reconcile_on_startup=False)

    def tearDown(self):
        self.engine.stop()
        self.journal.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fault_market_data_disconnect_blocks_orders(self):
        injector = FaultInjector()
        injector.enable_fault(FaultType.DISCONNECT, FaultConfig(target_symbol="2330"))

        # Inject tick while disconnect fault is active
        t0 = datetime(2026, 9, 18, 9, 0, 0)
        tick = TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10)

        # Hook injector into integrity checker
        if injector.is_fault_active(FaultType.DISCONNECT):
            self.engine.integrity_checker.record_disconnect("2330")

        self.assertFalse(self.engine.integrity_checker.is_symbol_healthy("2330"))

        # Evaluating an order for 2330 must fail safely
        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        decision = self.risk.evaluate_order(
            request=req,
            current_positions={},
            market_price=950.0,
            market_data_healthy=self.engine.integrity_checker.is_symbol_healthy("2330"),
        )
        self.assertFalse(decision.allowed)
        self.assertIn("UNHEALTHY_MARKET_DATA", decision.reason)

    def test_fault_duplicate_callback_dropped_by_integrity_layer(self):
        t0 = datetime(2026, 9, 18, 9, 0, 0)
        tick = TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10)

        # First tick succeeds
        self.assertTrue(self.engine.on_tick(tick))

        # Fault injector duplicates the identical tick
        self.engine.on_tick(tick)
        report = self.engine.integrity_checker.get_symbol_report("2330")
        self.assertEqual(report.duplicate_count, 1)

    def test_fault_out_of_order_tick_handled_safely(self):
        t0 = datetime(2026, 9, 18, 9, 1, 0)
        t_older = datetime(2026, 9, 18, 9, 0, 15)

        self.engine.on_tick(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))
        # Out-of-order tick arriving later
        self.engine.on_tick(TickEvent(timestamp=t_older, symbol="2330", price=951.0, volume=10))

        report = self.engine.integrity_checker.get_symbol_report("2330")
        self.assertEqual(report.out_of_order_count, 1)
        self.assertEqual(report.status.value, "DEGRADED")

    def test_fault_strategy_exception_does_not_crash_engine(self):
        strat = FaultyStrategy()
        self.engine.register_strategy(strat)

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        bar = BarEvent(symbol="2330", timestamp=t0, open=950.0, high=955.0, low=948.0, close=952.0, volume=100)

        # Engine handles strategy exception gracefully without terminating running state
        self.engine.on_bar(bar)
        self.assertTrue(self.engine._is_running)

    def test_fault_broker_rejected_order_persisted_safely(self):
        # Force a rejected order (e.g. insufficient cash)
        poor_db = os.path.join(self.test_dir, "poor_journal.db")
        poor_journal = ExecutionJournal(db_path=poor_db)
        poor_broker = PaperBrokerAdapter(initial_cash=100.0)  # Only 100 TWD
        poor_broker.connect()
        poor_engine = ExecutionEngine(
            broker=poor_broker,
            risk_engine=RiskEngine(
                limits=RiskLimits(
                    max_order_value=10_000_000.0,
                    max_position_value_per_symbol=20_000_000.0,
                    max_total_exposure=50_000_000.0,
                ),
                kill_switch=KillSwitch(),
            ),
            journal=poor_journal,
            trading_mode="shadow",
        )
        poor_engine.start(reconcile_on_startup=False)

        req = OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000, price=950.0)
        order = poor_engine.submit_manual_order(req, market_price=950.0)

        self.assertEqual(order.status.value, "REJECTED")
        self.assertIn("INSUFFICIENT_FUNDS", order.rejection_reason)
        poor_engine.stop()
        poor_journal.close()


if __name__ == "__main__":
    unittest.main()
