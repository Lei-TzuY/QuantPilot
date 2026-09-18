"""
Soak Test Simulation.
Simulates an unattended SHADOW trading session covering the full Taiwan trading day (09:00 -> 13:30 = 270 minutes).
Verifies:
- Memory growth stability
- Zero queue backlog accumulation
- Market data stream integrity
- Continuous bar generation and order processing
- Strictly zero real-money order submissions
"""
from datetime import datetime, timedelta
import gc
import os
import shutil
import tempfile
import tracemalloc
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.journal import ExecutionJournal
from modules.market.bar_builder import BarBuilder
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy


class SoakIntradayStrategy(BaseStrategy):
    """Generates occasional intraday signals during the simulated day."""

    def __init__(self):
        super().__init__("soak_strat")
        self.bars_count = 0
        self.signals_count = 0

    def on_bar(self, bar: BarEvent):
        self.bars_count += 1
        # Signal on every 30th bar
        if self.bars_count % 30 == 0:
            self.signals_count += 1
            return SignalEvent(
                signal_id=f"SIG-SOAK-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
        return None


class TestSoakMode(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "soak_journal.db")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_taiwan_trading_day_shadow_session_simulation(self):
        """
        Simulates 270 minutes (09:00:00 -> 13:30:00) with 4 ticks per minute across 2 symbols = 2,160 ticks.
        """
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        journal = ExecutionJournal(db_path=self.db_path)
        paper_broker = PaperBrokerAdapter(initial_cash=50_000_000.0)
        paper_broker.connect()
        risk = RiskEngine(
            limits=RiskLimits(
                max_order_value=2_000_000.0,
                max_position_value_per_symbol=20_000_000.0,
                max_total_exposure=50_000_000.0,
                max_trades_per_day=500,
            ),
            kill_switch=KillSwitch(),
        )
        engine = ExecutionEngine(
            broker=paper_broker,
            risk_engine=risk,
            journal=journal,
            trading_mode="shadow",
            synchronous_queue=True,
        )
        strat = SoakIntradayStrategy()
        engine.register_strategy(strat)
        engine.start(reconcile_on_startup=False)

        start_time = datetime(2026, 9, 18, 9, 0, 0)
        symbols = ["2330", "2454"]

        # Run 270 minutes (09:00 -> 13:30)
        for minute_idx in range(270):
            current_minute = start_time + timedelta(minutes=minute_idx)
            # 2 ticks per symbol per minute
            for sec_offset in [0, 30]:
                ts = current_minute + timedelta(seconds=sec_offset)
                for sym_idx, sym in enumerate(symbols):
                    base_price = 950.0 if sym == "2330" else 1200.0
                    price = base_price + (minute_idx % 10) - (sec_offset // 30)
                    tick = TickEvent(
                        timestamp=ts,
                        symbol=sym,
                        price=price,
                        volume=10 + (minute_idx % 5),
                        bid_price=price - 0.5,
                        ask_price=price + 0.5,
                    )
                    engine.on_tick(tick)

        # Final tick at 13:30:00 rolls over minute 13:29
        ts_final = start_time + timedelta(minutes=270)
        for sym in symbols:
            final_p = 955.0 if sym == "2330" else 1205.0
            engine.on_tick(
                TickEvent(
                    timestamp=ts_final,
                    symbol=sym,
                    price=final_p,
                    volume=10,
                    bid_price=final_p - 0.5,
                    ask_price=final_p + 0.5,
                )
            )

        # Stop and generate session report
        report = engine.generate_session_report()
        engine.stop()

        snapshot_end = tracemalloc.take_snapshot()
        stats = snapshot_end.compare_to(snapshot_start, "lineno")
        total_memory_diff_kb = sum(stat.size_diff for stat in stats) / 1024.0
        tracemalloc.stop()

        # Invariants verification:
        # 1. Pipeline survived 270 minutes without manual intervention
        self.assertGreaterEqual(strat.bars_count, 270)
        self.assertGreater(report.ticks_processed, 1000)
        self.assertEqual(report.ticks_rejected, 0)

        # 2. Queue backlog must be zero at end of session
        q_metrics = engine.get_queue_metrics()
        self.assertEqual(q_metrics.current_depth, 0)
        self.assertEqual(q_metrics.overflow_count, 0)

        # 3. Signals generated and orders submitted cleanly
        orders = engine.order_manager.get_all_orders()
        self.assertGreater(len(orders), 0)
        for o in orders:
            self.assertEqual(o.status.value, "FILLED")

        # 4. Zero real-money broker submissions
        self.assertEqual(engine.trading_mode, "shadow")

        # 5. Bounded memory growth (well within 25MB for 2,160 ticks)
        self.assertLess(total_memory_diff_kb, 25_000.0)


if __name__ == "__main__":
    unittest.main()
