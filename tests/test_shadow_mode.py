import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from modules.brokers.paper import PaperBrokerAdapter
from modules.brokers.shioaji import ShioajiBrokerAdapter
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.journal import ExecutionJournal
from modules.execution.order import OrderRequest, OrderSide, OrderStatus, OrderType
from modules.execution.order_manager import OrderManager
from modules.execution.persistence import ExecutionStatePersistence
from modules.market.bar_builder import BarBuilder
from modules.risk.engine import RiskEngine
from modules.risk.kill_switch import KillSwitch
from modules.risk.limits import RiskLimits
from modules.strategy.base import BaseStrategy


class SimpleIntradayStrategy(BaseStrategy):
    """Simple strategy that generates BUY signal upon completed bar if close > open."""

    def __init__(self):
        super().__init__("intraday_momentum")
        self.bars_received = []
        self.signals_generated = []

    def on_bar(self, bar: BarEvent):
        self.bars_received.append(bar)
        if bar.close > bar.open:
            sig = SignalEvent(
                signal_id=f"SIG-{bar.timestamp.strftime('%Y%m%d%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
                target_price=bar.close,
            )
            self.signals_generated.append(sig)
            return sig
        return None


class TestShadowModeLiveMarketToPaperExecution(unittest.TestCase):
    """
    Tests the real-time shadow-trading mode:
    REAL MARKET DATA (Shioaji quote feed)
    → Normalized TickEvent
    → BarBuilder (1-minute aggregation)
    → Completed BarEvent
    → Strategy.on_bar
    → SignalEvent
    → RiskEngine (pre-trade checks)
    → OMS (order creation & tracking)
    → PaperBroker (paper execution, strictly zero live-money risk)
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "shadow_journal.db")
        self.kill_switch_path = os.path.join(self.temp_dir, "shadow_ks.json")
        self.state_file_path = os.path.join(self.temp_dir, "shadow_state.json")
        self.journal = ExecutionJournal(db_path=self.db_path)
        self.persistence = ExecutionStatePersistence(storage_path=self.state_file_path)
        self.oms = OrderManager()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_pipeline_from_shioaji_ticks_to_paper_fill(self):
        """Validates the end-to-end event-driven flow in shadow mode."""
        # 1. Initialize Shioaji market feed adapter (read-only quote feed)
        shioaji_feed = ShioajiBrokerAdapter(trading_mode="paper")
        shioaji_feed.connect()

        # 2. Initialize Paper execution target with 10M TWD
        paper_broker = PaperBrokerAdapter(initial_cash=10_000_000)
        paper_broker.connect()

        kill_switch = KillSwitch(state_file=self.kill_switch_path)
        limits = RiskLimits(
            max_order_value=2_000_000.0,
            max_position_value_per_symbol=5_000_000.0,
            max_total_exposure=10_000_000.0,
        )
        risk_engine = RiskEngine(limits=limits, kill_switch=kill_switch)
        bar_builder = BarBuilder(interval_seconds=60)

        engine = ExecutionEngine(
            broker=paper_broker,  # PAPER execution target
            risk_engine=risk_engine,
            order_manager=self.oms,
            journal=self.journal,
            persistence=self.persistence,
            bar_builder=bar_builder,
            trading_mode="shadow",
            default_order_shares=1000,
        )

        # 4. Wire market data stream into engine
        engine.connect_market_data(shioaji_feed)

        # 5. Register strategy
        strategy = SimpleIntradayStrategy()
        engine.register_strategy(strategy)

        # 6. Start execution engine
        engine.start(reconcile_on_startup=False)

        # 7. Simulate incoming Shioaji quote ticks for Bar 1 (09:00:00 to 09:00:59)
        base_time = datetime(2026, 9, 18, 9, 0, 0)

        # Bar 1 ticks: Open 950, High 955, Low 948, Close 954 (bullish bar)
        ticks_bar1 = [
            TickEvent(timestamp=base_time + timedelta(seconds=0), symbol="2330", price=950.0, volume=10),
            TickEvent(timestamp=base_time + timedelta(seconds=15), symbol="2330", price=948.0, volume=15),
            TickEvent(timestamp=base_time + timedelta(seconds=30), symbol="2330", price=955.0, volume=20),
            TickEvent(timestamp=base_time + timedelta(seconds=45), symbol="2330", price=954.0, volume=25),
        ]

        for tick in ticks_bar1:
            shioaji_feed.simulate_tick(tick)

        # During Bar 1, bar has not completed yet
        self.assertEqual(len(strategy.bars_received), 0)
        self.assertEqual(len(engine.order_manager.get_all_orders()), 0)

        # 8. First tick of Bar 2 arrives at 09:01:00 (rolls over Bar 1!)
        tick_bar2_first = TickEvent(
            timestamp=base_time + timedelta(seconds=60),
            symbol="2330",
            price=956.0,
            volume=30,
        )
        shioaji_feed.simulate_tick(tick_bar2_first)

        # Bar 1 completed!
        self.assertEqual(len(strategy.bars_received), 1)
        completed_bar = strategy.bars_received[0]
        self.assertEqual(completed_bar.open, 950.0)
        self.assertEqual(completed_bar.high, 955.0)
        self.assertEqual(completed_bar.low, 948.0)
        self.assertEqual(completed_bar.close, 954.0)
        self.assertEqual(completed_bar.volume, 70.0)

        # Strategy produced BUY signal since close 954 > open 950
        self.assertEqual(len(strategy.signals_generated), 1)

        # Order was vetted by RiskEngine, created in OMS, and filled by PaperBroker!
        orders = engine.order_manager.get_all_orders()
        self.assertEqual(len(orders), 1)
        order = orders[0]
        self.assertEqual(order.symbol, "2330")
        self.assertEqual(order.quantity, 1000)
        self.assertEqual(order.status, OrderStatus.FILLED)

        # Positions updated in Paper environment
        positions = engine.get_positions()
        self.assertIn("2330", positions)
        pos = positions["2330"]
        self.assertEqual(pos.quantity, 1000)

        # Cash deducted in Paper broker (never real money!)
        acc = paper_broker.get_account()
        self.assertLess(acc["cash"], 10_000_000)
        self.assertEqual(paper_broker.trading_mode, "paper")

    def test_shadow_mode_strictly_disallows_live_order_submission(self):
        """Guarantees that shadow mode never permits real-money orders."""
        shioaji_feed = ShioajiBrokerAdapter(trading_mode="paper")
        # Attempting to submit live order to Shioaji feed in paper/shadow mode returns REJECTED
        order = self.oms.create_order(
            OrderRequest(symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)
        )
        result = shioaji_feed.submit_order(order)
        self.assertEqual(result.status, OrderStatus.REJECTED)
        self.assertIn("LIVE_TRADING_DISABLED", result.rejection_reason)


if __name__ == "__main__":
    unittest.main()
