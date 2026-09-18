"""
Tests for clock abstractions, timezone handling, and market-data latency semantics.
Verifies:
1. SystemClock and VirtualClock provide timezone-aware Asia/Taipei time.
2. Monotonic duration measurement via now_ns() and perf_counter.
3. Historical replay does NOT compare past exchange timestamps against wall clock (no fake multi-hour latency).
4. Queue latency uses monotonic nanosecond timestamps.
"""
from datetime import datetime, timedelta
import time
import unittest
import zoneinfo

from modules.common.clock import SystemClock, VirtualClock, TAIPEI_TZ
from modules.execution.events import TickEvent
from modules.monitoring.latency import LatencyTracker


class TestClockAndLatencySemantics(unittest.TestCase):
    def test_clock_timezone_and_virtual_stepping(self):
        sys_clock = SystemClock()
        now_taipei = sys_clock.now()
        self.assertIsNotNone(now_taipei.tzinfo)
        self.assertEqual(now_taipei.tzname(), "CST")  # Asia/Taipei is UTC+8 CST

        # Virtual clock
        v_clock = VirtualClock()
        target_t = datetime(2026, 9, 18, 9, 0, 0, tzinfo=TAIPEI_TZ)
        v_clock.set_time(target_t)
        self.assertEqual(v_clock.now(), target_t)

        v_clock.advance(timedelta(minutes=5))
        self.assertEqual(v_clock.now(), target_t + timedelta(minutes=5))

    def test_monotonic_nanosecond_timing(self):
        clock = SystemClock()
        t1 = clock.now_ns()
        time.sleep(0.01)
        t2 = clock.now_ns()

        self.assertGreater(t2, t1)
        elapsed_ms = clock.elapsed_ms(t1, t2)
        self.assertGreaterEqual(elapsed_ms, 9.0)  # ~10ms
        self.assertLess(elapsed_ms, 100.0)

    def test_historical_replay_suppresses_fake_latency(self):
        """
        Historical ticks have exchange timestamps in the past (e.g. 2026-09-18 09:00:00).
        Wall clock is current time (e.g. 2026-09-18 23:30:00).
        Comparing wall clock against past exchange timestamp would produce fake 14+ hour latency!
        The LatencyTracker MUST NOT record this fake latency when is_replay=True.
        """
        tracker = LatencyTracker()

        historical_exchange_ts = datetime(2026, 9, 18, 9, 0, 0)
        wall_clock_receive_ts = datetime.now()

        # Replay tick
        tracker.record_tick_latencies(
            exchange_ts=historical_exchange_ts,
            receive_ts=wall_clock_receive_ts,
            enqueue_ts=wall_clock_receive_ts,
            dequeue_ts=wall_clock_receive_ts + timedelta(milliseconds=1),
            is_replay=True,
        )

        snapshot = tracker.get_snapshot()
        # Market data latency stage should NOT be populated with fake hours
        md_stage = snapshot.stages.get("market_data")
        self.assertEqual(md_stage.count, 0, "Historical replay should not record fake multi-hour market_data latency")

    def test_monotonic_queue_dwell_time(self):
        tracker = LatencyTracker()
        now_dt = datetime.now()

        t_enqueue_ns = time.perf_counter_ns()
        time.sleep(0.005)  # 5ms dwell
        t_dequeue_ns = time.perf_counter_ns()

        tracker.record_tick_latencies(
            exchange_ts=now_dt,
            receive_ts=now_dt,
            enqueue_ts=now_dt,
            dequeue_ts=now_dt,
            enqueue_ns=t_enqueue_ns,
            dequeue_ns=t_dequeue_ns,
            is_replay=False,
        )

        snapshot = tracker.get_snapshot()
        queue_stage = snapshot.stages.get("queue")
        self.assertIsNotNone(queue_stage)
        self.assertGreaterEqual(queue_stage.mean_ms, 4.0)
        self.assertLess(queue_stage.mean_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
