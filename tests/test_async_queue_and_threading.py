"""
Verification of genuine asynchronous ingestion and thread isolation.
Proves that:
1. Shioaji quote callback runs on the producer thread.
2. Callback thread returns immediately without executing downstream logic.
3. Downstream strategy / BarBuilder / RiskEngine runs on a dedicated consumer worker thread.
4. quote callback thread ID != downstream strategy thread ID.
"""
from datetime import datetime
import threading
import time
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, TickEvent
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy


class ThreadRecordingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("thread_recorder")
        self.recorded_thread_ids = []
        self.bar_processed_event = threading.Event()

    def on_bar(self, bar: BarEvent):
        super().on_bar(bar)
        self.recorded_thread_ids.append(threading.get_ident())
        self.bar_processed_event.set()
        return None


class TestAsyncQueueAndThreading(unittest.TestCase):
    def test_quote_callback_thread_differs_from_strategy_thread(self):
        broker = PaperBrokerAdapter()
        broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        # Production / Shadow default is asynchronous (test_only_synchronous=False)
        engine = ExecutionEngine(
            broker=broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=False,
        )
        strategy = ThreadRecordingStrategy()
        engine.register_strategy(strategy)
        engine.start(reconcile_on_startup=False)

        producer_thread_id = None
        callback_returned = threading.Event()

        def producer_quote_callback():
            nonlocal producer_thread_id
            producer_thread_id = threading.get_ident()

            # Enqueue enough ticks to trigger bar completion
            now = datetime.now()
            for i in range(2):
                t = datetime(2026, 9, 18, 9, 0, i * 30)
                tick = TickEvent(
                    timestamp=t,
                    symbol="2330",
                    price=950.0,
                    volume=10,
                    bid_price=949.0,
                    ask_price=951.0,
                )
                success = engine.on_tick(tick)
                assert success is True

            # Boundary tick in next minute closes the bar
            boundary_tick = TickEvent(
                timestamp=datetime(2026, 9, 18, 9, 1, 0),
                symbol="2330",
                price=951.0,
                volume=10,
                bid_price=950.0,
                ask_price=952.0,
            )
            engine.on_tick(boundary_tick)
            callback_returned.set()

        producer_thread = threading.Thread(target=producer_quote_callback, name="SimulatedShioajiCallbackThread")
        producer_thread.start()
        producer_thread.join(timeout=2.0)

        self.assertTrue(callback_returned.is_set(), "Producer callback should return immediately")

        # Wait for downstream worker to process bar in strategy
        strategy_processed = strategy.bar_processed_event.wait(timeout=3.0)
        self.assertTrue(strategy_processed, "Downstream strategy should process bar asynchronously")

        consumer_thread_id = strategy.recorded_thread_ids[0]

        # Verify thread IDs
        self.assertIsNotNone(producer_thread_id)
        self.assertIsNotNone(consumer_thread_id)
        self.assertNotEqual(
            producer_thread_id,
            consumer_thread_id,
            f"Callback thread ({producer_thread_id}) MUST NOT equal consumer worker thread ({consumer_thread_id})!",
        )

        engine.stop()

    def test_synchronous_mode_only_when_explicitly_flagged(self):
        # Default must be asynchronous
        broker = PaperBrokerAdapter()
        broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        default_engine = ExecutionEngine(broker=broker, risk_engine=risk_engine)
        self.assertFalse(
            default_engine.event_queue.synchronous,
            "Default ExecutionEngine constructor MUST NOT choose synchronous queue!",
        )

        # Test-only mode must be explicit
        test_engine = ExecutionEngine(broker=broker, risk_engine=risk_engine, test_only_synchronous=True)
        self.assertTrue(
            test_engine.event_queue.synchronous,
            "ExecutionEngine should enable synchronous queue only when test_only_synchronous=True is passed",
        )


if __name__ == "__main__":
    unittest.main()
