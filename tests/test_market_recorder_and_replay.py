"""
Deterministic Replay Engine and Raw Market Data Recorder Tests.
Verifies append-only Parquet tick persistence and reproducible replay across
bars, signals, orders, fills, and PnL.
"""
from datetime import datetime, timedelta
import os
import shutil
import tempfile
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.journal import ExecutionJournal
from modules.execution.order_manager import OrderManager
from modules.execution.persistence import ExecutionStatePersistence
from modules.market.bar_builder import BarBuilder
from modules.market.recorder import RawMarketDataRecorder
from modules.market.replay import ReplayMarketDataSource, ReplaySpeed
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy


class DeterministicReplayStrategy(BaseStrategy):
    """Generates BUY on first green bar, then SELL on first red bar."""

    def __init__(self):
        super().__init__("deterministic_replay_strat")
        self.bars = []
        self.signals = []

    def on_bar(self, bar: BarEvent):
        self.bars.append(bar)
        if bar.close > bar.open and len(self.signals) == 0:
            sig = SignalEvent(
                signal_id=f"SIG-BUY-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
            self.signals.append(sig)
            return sig
        elif bar.close < bar.open and len(self.signals) == 1:
            sig = SignalEvent(
                signal_id=f"SIG-SELL-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="SELL",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
            self.signals.append(sig)
            return sig
        return None


class TestMarketRecorderAndReplay(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.market_data_dir = os.path.join(self.test_dir, "market_data")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_raw_market_data_recorder_parquet_persistence(self):
        recorder = RawMarketDataRecorder(base_dir=self.market_data_dir, buffer_size=5)

        t0 = datetime(2026, 9, 18, 9, 0, 0)
        ticks = [
            TickEvent(
                timestamp=t0 + timedelta(seconds=i * 10),
                symbol="2330",
                price=950.0 + i,
                volume=10 + i,
                bid_price=949.5 + i,
                ask_price=950.5 + i,
                bid_volume=50,
                ask_volume=50,
                sequence=i + 1,
            )
            for i in range(10)
        ]

        for tick in ticks:
            recorder.record_tick(tick)

        recorder.close()

        # Verify Parquet file was written to data/market/YYYY-MM-DD/{symbol}.parquet
        expected_file = os.path.join(self.market_data_dir, "2026-09-18", "2330.parquet")
        self.assertTrue(os.path.exists(expected_file))

        # Read back via PyArrow
        import pyarrow.parquet as pq
        table = pq.read_table(expected_file)
        self.assertEqual(len(table), 10)
        self.assertIn("symbol", table.column_names)
        self.assertIn("price", table.column_names)
        self.assertIn("bid_price", table.column_names)
        self.assertIn("ask_price", table.column_names)

    def test_deterministic_replay_produces_identical_execution(self):
        """
        Given identical market data, config, strategy, and initial state:
        Replay run 1 and Replay run 2 MUST produce identical bars, signals, orders, fills, and PnL.
        """
        # 1. Synthesize and record raw ticks for 3 complete bars
        # Bar 1: 09:00:00 - 09:00:50 (Green bar: 950 -> 955)
        # Bar 2: 09:01:00 - 09:01:50 (Red bar: 955 -> 945)
        # Bar 3: 09:02:00 - 09:02:10 (Closes Bar 2)
        recorder = RawMarketDataRecorder(base_dir=self.market_data_dir, buffer_size=100)
        t0 = datetime(2026, 9, 18, 9, 0, 0)

        raw_ticks = [
            # Bar 1
            TickEvent(timestamp=t0 + timedelta(seconds=0), symbol="2330", price=950.0, volume=10, bid_price=949.5, ask_price=950.5, sequence=1),
            TickEvent(timestamp=t0 + timedelta(seconds=30), symbol="2330", price=955.0, volume=20, bid_price=954.5, ask_price=955.5, sequence=2),
            # Bar 2
            TickEvent(timestamp=t0 + timedelta(seconds=60), symbol="2330", price=955.0, volume=15, bid_price=954.5, ask_price=955.5, sequence=3),
            TickEvent(timestamp=t0 + timedelta(seconds=90), symbol="2330", price=945.0, volume=25, bid_price=944.5, ask_price=945.5, sequence=4),
            # Bar 3 (Triggers Bar 2 close)
            TickEvent(timestamp=t0 + timedelta(seconds=120), symbol="2330", price=946.0, volume=10, bid_price=945.5, ask_price=946.5, sequence=5),
        ]

        for t in raw_ticks:
            recorder.record_tick(t)
        recorder.close()

        def run_replay():
            # Setup isolated engine
            fills = []
            broker = PaperBrokerAdapter(initial_cash=10_000_000.0, commission_rate=0.001425)
            broker.connect()
            broker.register_fill_callback(fills.append)
            limits = RiskLimits(max_order_value=5_000_000.0, max_position_value_per_symbol=5_000_000.0)
            risk = RiskEngine(limits=limits, kill_switch=KillSwitch())
            bar_builder = BarBuilder(interval_seconds=60)
            test_db = os.path.join(self.test_dir, f"replay_{datetime.now().strftime('%H%M%S%f')}.db")
            journal = ExecutionJournal(db_path=test_db)
            state_path = os.path.join(self.test_dir, f"state_{datetime.now().strftime('%H%M%S%f')}.json")
            pers = ExecutionStatePersistence(storage_path=state_path)
            engine = ExecutionEngine(
                broker=broker,
                risk_engine=risk,
                journal=journal,
                persistence=pers,
                bar_builder=bar_builder,
                trading_mode="shadow",
                synchronous_queue=True,
            )
            strat = DeterministicReplayStrategy()
            engine.register_strategy(strat)
            engine.start(reconcile_on_startup=False)

            source = ReplayMarketDataSource(data_dir=self.market_data_dir, speed=ReplaySpeed.MAX_SPEED)
            source.register_tick_callback(engine.on_tick)
            source.start(symbols=["2330"], date_str="2026-09-18", blocking=True)
            engine.stop()
            journal.close()

            orders = engine.order_manager.get_all_orders()
            pnl = sum(p.realized_pnl for p in engine._positions.values())

            return {
                "bars_count": len(strat.bars),
                "signals_count": len(strat.signals),
                "orders_count": len(orders),
                "fills_count": len(fills),
                "order_ids": [o.order_id for o in orders],
                "order_sides": [o.side.value for o in orders],
                "fill_prices": [f.price for f in fills],
                "fill_quantities": [f.quantity for f in fills],
                "realized_pnl": pnl,
            }

        run1 = run_replay()
        run2 = run_replay()

        # Replay runs MUST be 100% bitwise identical in all outcomes
        self.assertEqual(run1["bars_count"], 2)  # 2 bars finalized
        self.assertEqual(run1["signals_count"], 2)  # 1 BUY, 1 SELL
        self.assertEqual(run1["orders_count"], 2)
        self.assertEqual(run1["fills_count"], 2)

        self.assertEqual(run1["bars_count"], run2["bars_count"])
        self.assertEqual(run1["signals_count"], run2["signals_count"])
        self.assertEqual(run1["orders_count"], run2["orders_count"])
        self.assertEqual(run1["fills_count"], run2["fills_count"])
        self.assertEqual(run1["order_sides"], run2["order_sides"])
        self.assertEqual(run1["fill_prices"], run2["fill_prices"])
        self.assertEqual(run1["fill_quantities"], run2["fill_quantities"])
        self.assertEqual(run1["realized_pnl"], run2["realized_pnl"])


if __name__ == "__main__":
    unittest.main()
