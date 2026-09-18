"""
Unit tests for MarketDataEventQueue and MarketDataIntegrityChecker.
Verifies thread-safe bounded queuing, monotonic sequencing, latency preservation,
overflow detection, and data anomaly validation (stale, duplicate, regression, price shock).
"""
from datetime import datetime, timedelta
import queue
import time
import unittest

from modules.execution.events import TickEvent
from modules.market.event_queue import MarketDataEventQueue, QueueMetrics
from modules.market.integrity import MarketDataIntegrityChecker, MarketHealthStatus
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.execution.order import OrderRequest, OrderSide, OrderType


class TestMarketQueueAndIntegrity(unittest.TestCase):

    def test_queue_monotonic_sequence_and_timestamps(self):
        eq = MarketDataEventQueue(capacity=100, synchronous=True)
        received_ticks = []
        eq.subscribe(lambda t: received_ticks.append(t))

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        for i in range(10):
            tick = TickEvent(
                timestamp=t0 + timedelta(seconds=i),
                symbol="2330",
                price=950.0 + i,
                volume=10,
                receive_timestamp=datetime.now(),
            )
            success = eq.enqueue(tick)
            self.assertTrue(success)

        self.assertEqual(len(received_ticks), 10)
        # Check monotonic sequence numbers
        for idx, t in enumerate(received_ticks):
            self.assertEqual(t.sequence, idx + 1)
            self.assertIsNotNone(t.enqueue_timestamp)
            self.assertIsNotNone(t.dequeue_timestamp)
            self.assertGreaterEqual(t.dequeue_timestamp, t.enqueue_timestamp)

    def test_queue_overflow_detection_and_dropped_counter(self):
        # Bounded capacity of 3 ticks, asynchronous worker thread NOT started
        overflow_events = []

        def on_overflow(tick, metrics):
            overflow_events.append((tick, metrics))

        eq = MarketDataEventQueue(capacity=3, synchronous=False, on_overflow=on_overflow)

        # Enqueue 3 items to fill capacity
        t0 = datetime(2026, 9, 18, 9, 0, 0)
        self.assertTrue(eq.enqueue(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=1)))
        self.assertTrue(eq.enqueue(TickEvent(timestamp=t0, symbol="2330", price=951.0, volume=1)))
        self.assertTrue(eq.enqueue(TickEvent(timestamp=t0, symbol="2330", price=952.0, volume=1)))

        # 4th and 5th items must trigger overflow without blocking
        self.assertFalse(eq.enqueue(TickEvent(timestamp=t0, symbol="2330", price=953.0, volume=1)))
        self.assertFalse(eq.enqueue(TickEvent(timestamp=t0, symbol="2330", price=954.0, volume=1)))

        metrics = eq.get_metrics()
        self.assertEqual(metrics.capacity, 3)
        self.assertEqual(metrics.overflow_count, 2)
        self.assertEqual(metrics.dropped_count, 2)
        self.assertFalse(metrics.is_healthy)
        self.assertEqual(len(overflow_events), 2)

    def test_integrity_price_and_volume_validation(self):
        checker = MarketDataIntegrityChecker()
        t0 = datetime(2026, 9, 18, 9, 0, 0)

        # Negative price rejected
        bad_price_tick = TickEvent(timestamp=t0, symbol="2330", price=-10.0, volume=10)
        valid, reason = checker.validate_tick(bad_price_tick)
        self.assertFalse(valid)
        self.assertIn("INVALID_PRICE", reason)

        # Zero price rejected
        zero_price_tick = TickEvent(timestamp=t0, symbol="2330", price=0.0, volume=10)
        valid, reason = checker.validate_tick(zero_price_tick)
        self.assertFalse(valid)
        self.assertIn("INVALID_PRICE", reason)

        # Negative volume rejected
        bad_vol_tick = TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=-5)
        valid, reason = checker.validate_tick(bad_vol_tick)
        self.assertFalse(valid)
        self.assertIn("INVALID_VOLUME", reason)

        # Valid tick accepted
        good_tick = TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10)
        valid, reason = checker.validate_tick(good_tick)
        self.assertTrue(valid)
        self.assertIsNone(reason)
        self.assertEqual(checker.get_health("2330"), MarketHealthStatus.HEALTHY)

    def test_integrity_duplicate_tick_detection(self):
        checker = MarketDataIntegrityChecker()
        t0 = datetime(2026, 9, 18, 9, 0, 0)
        tick = TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10)

        # First tick passes
        valid, _ = checker.validate_tick(tick)
        self.assertTrue(valid)

        # Exact duplicate rejected
        valid2, reason2 = checker.validate_tick(tick)
        self.assertFalse(valid2)
        self.assertIn("DUPLICATE_TICK", reason2)
        report = checker.get_symbol_report("2330")
        self.assertEqual(report.duplicate_count, 1)

    def test_integrity_timestamp_regression_detection(self):
        checker = MarketDataIntegrityChecker()
        t0 = datetime(2026, 9, 18, 9, 1, 0)
        t_older = datetime(2026, 9, 18, 9, 0, 30)

        valid1, _ = checker.validate_tick(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))
        self.assertTrue(valid1)

        # Older timestamp arriving after later timestamp
        valid2, reason2 = checker.validate_tick(TickEvent(timestamp=t_older, symbol="2330", price=951.0, volume=15))
        self.assertFalse(valid2)
        self.assertIn("TIMESTAMP_REGRESSION", reason2)
        self.assertEqual(checker.get_health("2330"), MarketHealthStatus.DEGRADED)

    def test_integrity_price_shock_jump_detection(self):
        # 15% max jump threshold
        checker = MarketDataIntegrityChecker(max_price_jump_pct=0.15)
        t0 = datetime(2026, 9, 18, 9, 0, 0)
        t1 = datetime(2026, 9, 18, 9, 0, 1)

        # Baseline price: 100.0
        checker.validate_tick(TickEvent(timestamp=t0, symbol="2603", price=100.0, volume=10))

        # 20% jump to 120.0 without exchange halt flag
        valid, reason = checker.validate_tick(TickEvent(timestamp=t1, symbol="2603", price=120.0, volume=10))
        self.assertFalse(valid)
        self.assertIn("ABNORMAL_PRICE_JUMP", reason)
        self.assertEqual(checker.get_health("2603"), MarketHealthStatus.DEGRADED)

    def test_integrity_stale_tick_detection(self):
        checker = MarketDataIntegrityChecker(max_stale_seconds=30.0)
        now = datetime(2026, 9, 18, 9, 5, 0)
        stale_time = now - timedelta(seconds=90)  # 90s old > 30s threshold

        valid, reason = checker.validate_tick(
            TickEvent(timestamp=stale_time, symbol="2330", price=950.0, volume=10),
            current_time=now,
        )
        self.assertFalse(valid)
        self.assertIn("STALE_TICK", reason)
        self.assertEqual(checker.get_health("2330"), MarketHealthStatus.STALE)

    def test_risk_engine_blocks_entries_when_market_data_unhealthy(self):
        limits = RiskLimits(max_order_value=2_000_000.0, max_position_value_per_symbol=2_000_000.0)
        kill_switch = KillSwitch()
        risk = RiskEngine(limits=limits, kill_switch=kill_switch)

        req = OrderRequest(
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            strategy_id="strat_test",
        )

        # 1. Evaluated with market_data_healthy = False -> Must REJECT
        decision = risk.evaluate_order(
            request=req,
            current_positions={},
            market_price=950.0,
            market_data_healthy=False,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("UNHEALTHY_MARKET_DATA", decision.reason)

        # 2. Evaluated with market_data_healthy = True -> Must APPROVE
        decision_ok = risk.evaluate_order(
            request=req,
            current_positions={},
            market_price=950.0,
            market_data_healthy=True,
        )
        self.assertTrue(decision_ok.allowed)


if __name__ == "__main__":
    unittest.main()
