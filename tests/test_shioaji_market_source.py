"""
Integration & Safety Tests for Shioaji Market Data Transport & Shadow Execution
Tests all callback registrations, dual subscription, payload normalizations,
QuoteBook top-of-book joins, stale BidAsk gating, async thread-isolation,
reconnect resilience, and proves zero live-money order access.
"""
from datetime import datetime, timedelta
from decimal import Decimal
import os
import shutil
import tempfile
import threading
import time
import unittest
from zoneinfo import ZoneInfo

from modules.brokers.paper import PaperBrokerAdapter
from modules.common.clock import SystemClock
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, BidAskEvent, SignalEvent, TickEvent
from modules.execution.journal import ExecutionJournal
from modules.execution.persistence import ExecutionStatePersistence
from modules.market.integrity import MarketDataIntegrityChecker, MarketHealthStatus
from modules.market.quote_book import QuoteBook
from modules.market.recorder import RawMarketDataRecorder
from modules.market.shioaji_source import ConnectionState, ShioajiMarketDataSource
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy

from tests.fake_shioaji import FakeContract, FakeQuoteType, FakeShioajiAPI, TAIPEI_TZ


class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("dummy_strat")
        self.bars = []
        self.signals = []
        self.bar_thread_ids = []

    def on_bar(self, bar: BarEvent):
        super().on_bar(bar)
        self.bars.append(bar)
        self.bar_thread_ids.append(threading.get_ident())
        # Generate BUY signal on first bar
        if len(self.signals) == 0:
            sig = SignalEvent(
                signal_id=f"SIG-1-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
            self.signals.append(sig)
            return sig
        return None


class TestShioajiMarketDataSource(unittest.TestCase):
    def setUp(self):
        self.fake_api = FakeShioajiAPI()
        self.source = ShioajiMarketDataSource(
            api_key="TEST_API_KEY",
            secret_key="TEST_SECRET_KEY",
            simulation=True,
            api_instance=self.fake_api,
        )
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if self.source.is_connected():
            self.source.disconnect()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_callback_registration_on_connect(self):
        """Proves native callbacks are bound to api.quote upon connection."""
        self.assertIsNone(self.fake_api.quote.on_tick_callback)
        self.assertIsNone(self.fake_api.quote.on_bidask_callback)
        self.assertIsNone(self.fake_api.quote.on_event_callback)

        success = self.source.connect()
        self.assertTrue(success)
        self.assertEqual(self.source.state, ConnectionState.AUTHENTICATED)

        # Verify callbacks are registered
        self.assertIsNotNone(self.fake_api.quote.on_tick_callback)
        self.assertIsNotNone(self.fake_api.quote.on_bidask_callback)
        self.assertIsNotNone(self.fake_api.quote.on_event_callback)

    def test_02_dual_subscription_and_duplicate_prevention(self):
        """Verifies both Tick and BidAsk streams are subscribed and duplicates prevented."""
        self.source.connect()
        self.source.subscribe(["2330", "2454"])

        # Check subscriptions recorded in FakeQuote
        subs = self.fake_api.quote.subscriptions
        self.assertIn(("2330", "tick"), subs)
        self.assertIn(("2330", "bidask"), subs)
        self.assertIn(("2454", "tick"), subs)
        self.assertIn(("2454", "bidask"), subs)
        self.assertEqual(len(subs), 4)

        # Duplicate subscription should not duplicate in API
        self.source.subscribe(["2330"])
        self.assertEqual(len(self.fake_api.quote.subscriptions), 4)

        # Test Unsubscribe
        self.source.unsubscribe(["2330"])
        self.assertNotIn(("2330", "tick"), self.fake_api.quote.subscriptions)
        self.assertNotIn(("2330", "bidask"), self.fake_api.quote.subscriptions)
        self.assertIn(("2330", "tick"), self.fake_api.quote.unsubscriptions)
        self.assertIn(("2330", "bidask"), self.fake_api.quote.unsubscriptions)

    def test_03_robust_payload_normalization_and_timezone(self):
        """Tests that native Decimal and timestamps normalize to Asia/Taipei domain events."""
        self.source.connect()

        received_ticks = []
        received_bidask = []
        self.source.register_tick_callback(received_ticks.append)
        self.source.register_bidask_callback(received_bidask.append)

        # Emit native tick
        t0 = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)
        self.fake_api.quote.emit_tick(
            code="2330",
            price=955.0,
            volume=15,
            total_volume=1200,
            timestamp=t0,
            tick_type=1,
            simtrade=False,
        )

        self.assertEqual(len(received_ticks), 1)
        tick_event: TickEvent = received_ticks[0]
        self.assertEqual(tick_event.symbol, "2330")
        self.assertEqual(tick_event.price, 955.0)
        self.assertEqual(tick_event.volume, 15.0)
        self.assertEqual(tick_event.total_volume, 1200.0)
        self.assertEqual(tick_event.timestamp.tzinfo, TAIPEI_TZ)

        # Emit native bidask
        self.fake_api.quote.emit_bidask(
            code="2330",
            bid_price=954.0,
            ask_price=955.0,
            bid_volume=80,
            ask_volume=100,
            timestamp=t0,
        )

        self.assertEqual(len(received_bidask), 1)
        bidask_event: BidAskEvent = received_bidask[0]
        self.assertEqual(bidask_event.symbol, "2330")
        self.assertEqual(bidask_event.bid_price, 954.0)
        self.assertEqual(bidask_event.ask_price, 955.0)
        self.assertEqual(bidask_event.bid_volume, 80.0)
        self.assertEqual(bidask_event.ask_volume, 100.0)
        self.assertEqual(bidask_event.timestamp.tzinfo, TAIPEI_TZ)

    def test_04_malformed_payload_rejection_metrics(self):
        """Malformed records should be rejected and increment metrics without entering pipeline."""
        self.source.connect()
        received = []
        self.source.register_tick_callback(received.append)

        # Tick with invalid empty code
        self.fake_api.quote.emit_tick(code="", price=950.0)
        self.assertEqual(len(received), 0)

        # Tick with negative price
        self.fake_api.quote.emit_tick(code="2330", price=-10.0)
        self.assertEqual(len(received), 0)

        hb = self.source.heartbeat()
        self.assertGreaterEqual(hb["malformed_payloads"], 2)

    def test_05_quotebook_joining_and_stale_bidask_detection(self):
        """Verifies top-of-book joining and stale BidAsk detection."""
        quote_book = QuoteBook(default_freshness_seconds=5.0)
        t0 = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)

        tick = TickEvent(
            timestamp=t0,
            symbol="2330",
            price=950.0,
            volume=10,
        )
        quote_book.update_tick(tick)
        q = quote_book.get_quote("2330")
        self.assertEqual(q.last_trade_price, 950.0)
        self.assertFalse(q.has_book)  # No book yet!

        # Add fresh BidAsk
        bidask = BidAskEvent(
            timestamp=t0,
            symbol="2330",
            bid_price=949.0,
            ask_price=951.0,
            bid_volume=50,
            ask_volume=50,
        )
        quote_book.update_bidask(bidask)
        q = quote_book.get_quote("2330")
        self.assertTrue(q.has_book)
        self.assertEqual(q.spread, 2.0)
        self.assertEqual(q.mid_price, 950.0)
        self.assertTrue(quote_book.is_bidask_fresh("2330", max_age_seconds=5.0, current_time=t0 + timedelta(seconds=2)))

        # Age exceeds threshold: should be marked not fresh
        self.assertFalse(quote_book.is_bidask_fresh("2330", max_age_seconds=5.0, current_time=t0 + timedelta(seconds=10)))

    def test_06_async_thread_isolation(self):
        """Proves strategy execution occurs on worker thread separate from native quote callback."""
        paper_broker = PaperBrokerAdapter(initial_cash=1_000_000.0)
        paper_broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=paper_broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=False,  # Genuine asynchronous queue
            journal=ExecutionJournal(os.path.join(self.test_dir, "j_06.db")),
            persistence=ExecutionStatePersistence(os.path.join(self.test_dir, "p_06.json")),
        )
        strategy = DummyStrategy()
        engine.register_strategy(strategy)
        engine.connect_market_data(self.source)

        engine.start(reconcile_on_startup=False)
        self.source.connect()
        self.source.subscribe(["2330"])

        caller_thread_id = threading.get_ident()

        # Emit 60 ticks across 1 minute to finalize a bar
        t_base = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)
        for s in range(61):
            self.fake_api.quote.emit_tick(
                code="2330",
                price=950.0 + (s % 5),
                volume=10,
                timestamp=t_base + timedelta(seconds=s),
            )

        # Wait for queue to drain
        engine.event_queue.drain(timeout=5.0)
        time.sleep(0.1)

        self.assertGreaterEqual(len(strategy.bars), 1)
        # Verify strategy execution thread is distinct from the caller thread!
        self.assertNotEqual(strategy.bar_thread_ids[0], caller_thread_id)

        engine.stop()

    def test_07_solace_disconnect_reconnect_resubscribe(self):
        """Verifies session recovery, callback re-binding, and re-subscription on reconnect."""
        self.source.connect()
        self.source.subscribe(["2330", "2454"])
        self.assertEqual(len(self.fake_api.quote.subscriptions), 4)

        # Trigger Solace Down event
        self.fake_api.quote.emit_event(resp_code=500, event_code=1, info="Network dropped", event="DOWN_ERROR")
        self.assertEqual(self.source.state, ConnectionState.RECONNECTING)

        # Trigger Solace Up event (reconnected)
        self.fake_api.quote.emit_event(resp_code=200, event_code=0, info="Session connected", event="UP_NOTICE")

        # Subscriptions should be restored
        self.assertEqual(self.source.state, ConnectionState.STREAMING)
        subs = self.fake_api.quote.subscriptions
        self.assertIn(("2330", "tick"), subs)
        self.assertIn(("2454", "bidask"), subs)
        self.assertEqual(self.source.heartbeat()["reconnects"], 1)
        self.assertEqual(self.source.heartbeat()["resubscribes"], 1)

    def test_08_zero_live_money_order_invariant(self):
        """
        CRITICAL SAFETY TEST:
        Monkeypatches Shioaji place_order to raise if touched.
        Runs full shadow pipeline:
        Fake Quote -> ShioajiMarketDataSource -> Normalized Event -> Engine -> Strategy -> Signal -> Risk -> OMS -> PaperBroker -> Fill.
        Proves place_order was never touched (call count == 0).
        """
        paper_broker = PaperBrokerAdapter(initial_cash=5_000_000.0)
        paper_broker.connect()
        risk_engine = RiskEngine(
            limits=RiskLimits(max_order_value=2_000_000.0, max_position_value_per_symbol=5_000_000.0),
            kill_switch=KillSwitch(),
        )

        engine = ExecutionEngine(
            broker=paper_broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=True,  # Synchronous for deterministic execution
            journal=ExecutionJournal(os.path.join(self.test_dir, "j_08.db")),
            persistence=ExecutionStatePersistence(os.path.join(self.test_dir, "p_08.json")),
        )
        strategy = DummyStrategy()
        engine.register_strategy(strategy)
        engine.connect_market_data(self.source)

        engine.start(reconcile_on_startup=False)
        self.source.connect()
        self.source.subscribe(["2330"])

        t_base = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)

        # Emit 61 ticks and periodic BidAsk to reflect realistic streaming quote flow
        for s in range(61):
            curr_ts = t_base + timedelta(seconds=s)
            if s % 10 == 0 or s == 60:
                self.fake_api.quote.emit_bidask(
                    code="2330",
                    bid_price=949.0,
                    ask_price=951.0,
                    bid_volume=100,
                    ask_volume=100,
                    timestamp=curr_ts,
                )
            self.fake_api.quote.emit_tick(
                code="2330",
                price=950.0,
                volume=10,
                timestamp=curr_ts,
            )

        # Confirm paper order was placed and filled
        orders = paper_broker.get_open_orders()
        positions = paper_broker.get_positions()

        self.assertIn("2330", positions)
        self.assertEqual(positions["2330"].quantity, engine.default_order_shares)

        # CRITICAL VERIFICATION: Shioaji.place_order was NEVER invoked
        self.assertEqual(self.fake_api.place_order_call_count, 0)

        engine.stop()

    def test_09_stale_bidask_spread_gating(self):
        """Proves that stale BidAsk spread causes RiskEngine to reject new entry orders."""
        paper_broker = PaperBrokerAdapter(initial_cash=1_000_000.0)
        paper_broker.connect()
        risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

        engine = ExecutionEngine(
            broker=paper_broker,
            risk_engine=risk_engine,
            trading_mode="shadow",
            test_only_synchronous=True,
            journal=ExecutionJournal(os.path.join(self.test_dir, "j_09.db")),
            persistence=ExecutionStatePersistence(os.path.join(self.test_dir, "p_09.json")),
        )
        strategy = DummyStrategy()
        engine.register_strategy(strategy)
        engine.connect_market_data(self.source)

        engine.start(reconcile_on_startup=False)
        self.source.connect()
        self.source.subscribe(["2330"])

        # Emit a BidAsk with old timestamp (> 60s ago)
        t_old = datetime(2026, 9, 19, 9, 0, 0, tzinfo=TAIPEI_TZ)
        self.fake_api.quote.emit_bidask(
            code="2330",
            bid_price=949.0,
            ask_price=951.0,
            timestamp=t_old,
        )

        # Emit 61 ticks at 09:30:00 (30 minutes after BidAsk)
        t_base = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)
        for s in range(61):
            self.fake_api.quote.emit_tick(
                code="2330",
                price=950.0,
                volume=10,
                timestamp=t_base + timedelta(seconds=s),
            )

        # Position should NOT be opened because BidAsk was stale!
        positions = paper_broker.get_positions()
        self.assertNotIn("2330", positions)
        self.assertGreaterEqual(engine._total_risk_rejections, 1)

        engine.stop()

    def test_10_dual_parquet_recording(self):
        """Verifies both Tick and BidAsk streams are captured into Parquet files."""
        recorder = RawMarketDataRecorder(base_dir=self.test_dir, buffer_size=5)
        t0 = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)

        for i in range(5):
            recorder.record_tick(
                TickEvent(
                    timestamp=t0 + timedelta(seconds=i),
                    symbol="2330",
                    price=950.0 + i,
                    volume=10,
                )
            )
            recorder.record_bidask(
                BidAskEvent(
                    timestamp=t0 + timedelta(seconds=i),
                    symbol="2330",
                    bid_price=949.0 + i,
                    ask_price=951.0 + i,
                    bid_volume=50,
                    ask_volume=50,
                )
            )

        recorder.close()

        import pyarrow.parquet as pq

        ticks_file = os.path.join(self.test_dir, "2026-09-19", "2330_ticks.parquet")
        bidask_file = os.path.join(self.test_dir, "2026-09-19", "2330_bidask.parquet")
        compat_file = os.path.join(self.test_dir, "2026-09-19", "2330.parquet")

        self.assertTrue(os.path.exists(ticks_file))
        self.assertTrue(os.path.exists(bidask_file))
        self.assertTrue(os.path.exists(compat_file))

        t_table = pq.read_table(ticks_file)
        b_table = pq.read_table(bidask_file)
        self.assertEqual(len(t_table), 5)
        self.assertEqual(len(b_table), 5)
        self.assertIn("bid_price", b_table.column_names)
        self.assertIn("ask_price", b_table.column_names)


if __name__ == "__main__":
    unittest.main()
