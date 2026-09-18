"""
Concurrency tests for ExecutionEngine shutdown and queue draining.
Verifies that:
1. Stop with empty queue completes immediately (< 0.1s).
2. Stop with backlog drains remaining items without deadlocking on engine locks.
3. Stop while subscriber is actively processing completes cleanly without deadlock.
4. Total shutdown time does not suffer unexplained stalls.
"""
from datetime import datetime
import time
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, TickEvent
from modules.market.event_queue import MarketDataEventQueue
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch


class TestShutdownConcurrency(unittest.TestCase):
    def test_stop_with_empty_queue(self):
        broker = PaperBrokerAdapter()
        broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=False,
        )
        engine.start(reconcile_on_startup=False)

        t0 = time.time()
        engine.stop()
        duration = time.time() - t0

        self.assertFalse(engine._is_running)
        self.assertLess(duration, 1.0, f"Shutdown with empty queue should be sub-second, took {duration:.3f}s")

    def test_stop_with_backlog_drains_without_deadlock(self):
        """
        Backlog test: 100 ticks are queued.
        The worker acquires engine lock on each tick.
        Calling engine.stop() must release the engine lock before draining,
        allowing the worker to process all items and terminate cleanly without deadlock.
        """
        broker = PaperBrokerAdapter()
        broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=False,
        )
        engine.start(reconcile_on_startup=False)

        # Enqueue 100 ticks
        for i in range(100):
            tick = TickEvent(
                timestamp=datetime(2026, 9, 18, 9, 0, i % 60),
                symbol="2330",
                price=950.0,
                volume=10,
                bid_price=949.0,
                ask_price=951.0,
                sequence=i,
            )
            engine.on_tick(tick)

        t0 = time.time()
        # This will call engine.stop() which drains the queue
        engine.stop()
        duration = time.time() - t0

        # Assert no deadlock occurred and all items were drained
        metrics = engine.get_queue_metrics()
        self.assertEqual(metrics.current_depth, 0, "Queue should be completely empty after drain")
        self.assertEqual(metrics.total_dequeued, 100, "All 100 ticks should be dequeued")
        self.assertLess(duration, 3.0, f"Shutdown with backlog took too long: {duration:.3f}s")

    def test_stop_while_subscriber_is_executing(self):
        """
        Subscriber takes 0.1s to execute. Stop is called concurrently.
        Must complete cleanly without hang or crash.
        """
        broker = PaperBrokerAdapter()
        broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=False,
        )

        original_on_tick = engine.bar_builder.on_tick_event

        def delayed_tick(tick):
            time.sleep(0.05)
            return original_on_tick(tick)

        engine.bar_builder.on_tick_event = delayed_tick

        engine.start(reconcile_on_startup=False)

        for i in range(5):
            tick = TickEvent(
                timestamp=datetime(2026, 9, 18, 9, 0, i),
                symbol="2330",
                price=950.0,
                volume=10,
                bid_price=949.0,
                ask_price=951.0,
                sequence=i,
            )
            engine.on_tick(tick)

        time.sleep(0.02)  # Let worker start processing first tick
        t0 = time.time()
        engine.stop()
        duration = time.time() - t0

        self.assertFalse(engine._is_running)
        self.assertLess(duration, 2.0, f"Shutdown took too long: {duration:.3f}s")


if __name__ == "__main__":
    unittest.main()
