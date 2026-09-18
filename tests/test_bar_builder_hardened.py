"""
Hardened 1-minute BarBuilder Unit Tests.
Verifies boundary snapping, first/last tick, missing minutes, multi-symbol concurrency,
session flush, and strict bar immutability (late ticks cannot alter finalized bars).
"""
from datetime import datetime, timedelta
import unittest

from modules.execution.events import BarEvent, TickEvent
from modules.market.bar_builder import BarBuilder


class TestBarBuilderHardened(unittest.TestCase):

    def test_minute_boundary_snapping_and_ohlcv(self):
        bb = BarBuilder(interval_seconds=60)
        emitted_bars = []
        bb.register_bar_callback(lambda b: emitted_bars.append(b))

        t0 = datetime(2026, 9, 18, 9, 0, 0)

        # Bar 1 ticks: from 09:00:00 to 09:00:59.999999
        bb.on_tick_event(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=15), symbol="2330", price=955.0, volume=15))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=30), symbol="2330", price=948.0, volume=20))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=59, microseconds=999999), symbol="2330", price=952.0, volume=25))

        # Bar has not finalized yet during the minute
        self.assertEqual(len(emitted_bars), 0)

        # First tick of minute 09:01:00 rolls over Bar 1
        t_next = t0 + timedelta(seconds=60)
        bb.on_tick_event(TickEvent(timestamp=t_next, symbol="2330", price=953.0, volume=5))

        self.assertEqual(len(emitted_bars), 1)
        bar = emitted_bars[0]
        self.assertEqual(bar.symbol, "2330")
        self.assertEqual(bar.open, 950.0)
        self.assertEqual(bar.high, 955.0)
        self.assertEqual(bar.low, 948.0)
        self.assertEqual(bar.close, 952.0)
        self.assertEqual(bar.volume, 70)
        self.assertEqual(bar.open_time, t0)
        self.assertEqual(bar.finalize_time, t_next)

    def test_strict_bar_immutability_rejects_late_ticks(self):
        bb = BarBuilder(interval_seconds=60)
        emitted_bars = []
        bb.register_bar_callback(lambda b: emitted_bars.append(b))

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        # Bar 1 ticks
        bb.on_tick_event(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=30), symbol="2330", price=952.0, volume=10))

        # Roll over Bar 1 with tick at 09:01:00
        t1 = t0 + timedelta(seconds=60)
        bb.on_tick_event(TickEvent(timestamp=t1, symbol="2330", price=955.0, volume=10))
        self.assertEqual(len(emitted_bars), 1)
        original_bar = emitted_bars[0]
        self.assertEqual(original_bar.close, 952.0)
        self.assertEqual(original_bar.volume, 20)

        # LATE TICK arriving for 09:00:45 (after minute has been finalized)
        late_tick = TickEvent(
            timestamp=t0 + timedelta(seconds=45),
            symbol="2330",
            price=999.0,  # Extreme price that would alter high/close if accepted
            volume=1000,
        )
        res = bb.on_tick_event(late_tick)
        self.assertIsNone(res)
        self.assertEqual(bb._rejected_late_ticks_count, 1)

        # Finalized bar values must be completely immutable
        self.assertEqual(original_bar.high, 952.0)
        self.assertEqual(original_bar.close, 952.0)
        self.assertEqual(original_bar.volume, 20)

    def test_missing_minutes_gap_handling(self):
        bb = BarBuilder(interval_seconds=60)
        emitted_bars = []
        bb.register_bar_callback(lambda b: emitted_bars.append(b))

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        bb.on_tick_event(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))

        # Missing minutes 09:01, 09:02, 09:03. Next tick arrives at 09:04:15
        t_gap = t0 + timedelta(minutes=4, seconds=15)
        bb.on_tick_event(TickEvent(timestamp=t_gap, symbol="2330", price=955.0, volume=20))

        # Minute 09:00 was cleanly finalized
        self.assertEqual(len(emitted_bars), 1)
        self.assertEqual(emitted_bars[0].open_time, t0)
        self.assertEqual(emitted_bars[0].close, 950.0)

        # Current active bar is now snapped to 09:04:00
        active_bar = bb._current_bars["2330"]
        self.assertEqual(active_bar.open_time, t0 + timedelta(minutes=4))
        self.assertEqual(active_bar.open, 955.0)

    def test_multi_symbol_concurrent_aggregation(self):
        bb = BarBuilder(interval_seconds=60)
        emitted = []
        bb.register_bar_callback(lambda b: emitted.append(b))

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        # Interleaved ticks for 2330 and 2454
        bb.on_tick_event(TickEvent(timestamp=t0, symbol="2330", price=950.0, volume=10))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=5), symbol="2454", price=1200.0, volume=5))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=30), symbol="2330", price=954.0, volume=15))
        bb.on_tick_event(TickEvent(timestamp=t0 + timedelta(seconds=35), symbol="2454", price=1210.0, volume=10))

        # Roll over at 09:01:00
        t1 = t0 + timedelta(seconds=60)
        bb.on_tick_event(TickEvent(timestamp=t1, symbol="2330", price=955.0, volume=5))
        bb.on_tick_event(TickEvent(timestamp=t1 + timedelta(seconds=1), symbol="2454", price=1205.0, volume=5))

        self.assertEqual(len(emitted), 2)
        symbols_emitted = {b.symbol for b in emitted}
        self.assertEqual(symbols_emitted, {"2330", "2454"})

    def test_session_end_flush(self):
        bb = BarBuilder(interval_seconds=60)
        emitted = []
        bb.register_bar_callback(lambda b: emitted.append(b))

        t0 = datetime(2026, 9, 18, 13, 29, 0)
        bb.on_tick_event(TickEvent(timestamp=t0, symbol="2330", price=960.0, volume=100))

        # Market closes at 13:30:00 without a 13:31 tick arriving
        flushed_bar = bb.flush_symbol("2330")
        self.assertIsNotNone(flushed_bar)
        self.assertEqual(flushed_bar.close, 960.0)
        self.assertEqual(flushed_bar.volume, 100)
        self.assertNotIn("2330", bb._current_bars)


if __name__ == "__main__":
    unittest.main()
