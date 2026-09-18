"""
Stress test for queue backpressure, max depth accounting, and overflow gating.
Verifies that:
1. Under high producer arrival rate and slow consumer, queue depth > 0.
2. Max queue depth is accurately recorded after successful enqueue.
3. Overflow is explicitly detected and dropped count matches.
4. Market health transitions to DEGRADED / UNHEALTHY.
5. RiskEngine rejects new entry orders when market data is degraded.
"""
from datetime import datetime
import time
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.order import OrderRequest, OrderSide, OrderType
from modules.market.event_queue import MarketDataEventQueue
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch


class TestQueueBackpressureStress(unittest.TestCase):
    def test_queue_overflow_and_risk_gating(self):
        # Bounded capacity = 20
        capacity = 20
        event_queue = MarketDataEventQueue(
            capacity=capacity,
            synchronous=False,
            name="StressTestQueue",
        )

        broker = PaperBrokerAdapter()
        broker.connect()
        risk_limits = RiskLimits(max_order_value=5_000_000.0)
        risk_engine = RiskEngine(limits=risk_limits, kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            event_queue=event_queue,
            trading_mode="shadow",
            test_only_synchronous=False,
        )

        # Monkey-patch BarBuilder.on_tick_event to simulate a slow consumer
        original_on_tick = engine.bar_builder.on_tick_event

        def slow_on_tick_event(tick):
            time.sleep(0.02)  # 20ms slow consumer
            return original_on_tick(tick)

        engine.bar_builder.on_tick_event = slow_on_tick_event

        engine.start(reconcile_on_startup=False)

        # Producer pushes 60 ticks rapidly into capacity=20 queue
        total_pushed = 60
        enqueued_count = 0
        rejected_count = 0

        for i in range(total_pushed):
            tick = TickEvent(
                timestamp=datetime(2026, 9, 18, 9, 0, i % 60),
                symbol="2330",
                price=950.0 + (i % 5),
                volume=10,
                bid_price=949.0,
                ask_price=951.0,
                sequence=i,
            )
            success = engine.on_tick(tick)
            if success:
                enqueued_count += 1
            else:
                rejected_count += 1

        # Check queue metrics immediately after fast burst
        metrics = engine.get_queue_metrics()

        # 1. Depth > 0 during or after burst
        self.assertGreater(metrics.max_depth, 0, "Max queue depth should be > 0 under stress")
        # 2. Max depth should not exceed capacity
        self.assertLessEqual(metrics.max_depth, capacity, "Max depth cannot exceed queue capacity")
        # 3. Overflow detected and dropped count explicit
        self.assertGreater(metrics.overflow_count, 0, "Overflow should be detected")
        self.assertGreater(metrics.dropped_count, 0, "Dropped count should be explicit")
        self.assertEqual(metrics.dropped_count, rejected_count, "Dropped count must equal rejected on_tick calls")
        self.assertFalse(metrics.is_healthy, "Queue health must become UNHEALTHY / unsafe")

        # 4. Market health becomes degraded
        health_report = engine.get_data_health().get("2330")
        self.assertIsNotNone(health_report)
        self.assertEqual(health_report["status"], "DEGRADED", "Symbol status must be DEGRADED on queue overflow")

        # 5. RiskEngine blocks new entries after unrecoverable overflow
        request = OrderRequest(
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            strategy_id="test_strat",
        )
        decision = risk_engine.evaluate_order(
            request=request,
            current_positions={},
            market_price=950.0,
            market_price_timestamp=datetime.now(),
            market_data_healthy=False,  # Flagged unhealthy due to overflow incident
        )
        self.assertFalse(decision.allowed, "RiskEngine MUST reject entry orders when market data is degraded")
        self.assertTrue("UNHEALTHY_MARKET_DATA" in decision.reason or "MARKET_DATA_UNHEALTHY" in decision.reason)

        engine.stop()


if __name__ == "__main__":
    unittest.main()
