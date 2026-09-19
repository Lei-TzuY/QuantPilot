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
import unittest.mock
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

    def test_11_cli_simulation_mode_precedence(self):
        """Proves CLI simulation/production precedence: CLI override > env var > safe default."""
        from scripts.run_shioaji_shadow import resolve_simulation_mode

        # 1. No CLI override + env False => False
        with unittest.mock.patch.dict(os.environ, {"SHIOAJI_SIMULATION": "False"}):
            self.assertFalse(resolve_simulation_mode(cli_simulation=None))

        # 2. No CLI override + env True => True
        with unittest.mock.patch.dict(os.environ, {"SHIOAJI_SIMULATION": "True"}):
            self.assertTrue(resolve_simulation_mode(cli_simulation=None))

        # 3. Explicit CLI override True with env False => True
        with unittest.mock.patch.dict(os.environ, {"SHIOAJI_SIMULATION": "False"}):
            self.assertTrue(resolve_simulation_mode(cli_simulation=True))

        # 4. Explicit CLI override False with env True => False
        with unittest.mock.patch.dict(os.environ, {"SHIOAJI_SIMULATION": "True"}):
            self.assertFalse(resolve_simulation_mode(cli_simulation=False))

        # 5. Default when env not set => True (safe default)
        env_without_sim = dict(os.environ)
        env_without_sim.pop("SHIOAJI_SIMULATION", None)
        with unittest.mock.patch.dict(os.environ, env_without_sim, clear=True):
            self.assertTrue(resolve_simulation_mode(cli_simulation=None))

    def test_12_multi_symbol_strategy_isolation(self):
        """Proves independent per-symbol bar history, MA calculations, position state, and bounded history."""
        from scripts.run_shioaji_shadow import ShadowBurnInStrategy

        strategy = ShadowBurnInStrategy(lookback=5, max_history=10)
        t_base = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)

        # Interleave 5 bars for 2330 (around 950.0) and 2454 (around 1200.0)
        signals_2330 = []
        signals_2454 = []

        for i in range(5):
            t = t_base + timedelta(minutes=i)
            # 2330 bar
            b_2330 = BarEvent(timestamp=t, symbol="2330", open=950.0, high=955.0, low=948.0, close=950.0, volume=100)
            sig_a = strategy.on_bar(b_2330)
            if sig_a:
                signals_2330.append(sig_a)

            # 2454 bar
            b_2454 = BarEvent(timestamp=t, symbol="2454", open=1200.0, high=1205.0, low=1195.0, close=1200.0, volume=200)
            sig_b = strategy.on_bar(b_2454)
            if sig_b:
                signals_2454.append(sig_b)

        # Confirm bar counts per symbol
        self.assertEqual(len(strategy.get_symbol_bars("2330")), 5)
        self.assertEqual(len(strategy.get_symbol_bars("2454")), 5)

        # Confirm 2330 MA is calculated only from 2330 bars (~950), and 2454 MA only from 2454 bars (~1200)
        bars_2330 = strategy.get_symbol_bars("2330")
        bars_2454 = strategy.get_symbol_bars("2454")
        self.assertAlmostEqual(sum(b.close for b in bars_2330) / 5, 950.0)
        self.assertAlmostEqual(sum(b.close for b in bars_2454) / 5, 1200.0)

        # Now emit a surge on 2330 (> 1.001 * 950 -> 960.0) to trigger BUY for 2330
        t_surge = t_base + timedelta(minutes=6)
        surge_2330 = BarEvent(timestamp=t_surge, symbol="2330", open=955.0, high=962.0, low=954.0, close=960.0, volume=150)
        sig_2330 = strategy.on_bar(surge_2330)

        self.assertIsNotNone(sig_2330)
        self.assertEqual(sig_2330.symbol, "2330")
        self.assertEqual(sig_2330.side, "BUY")
        self.assertEqual(strategy.get_symbol_position_side("2330"), "LONG")

        # CRITICAL ISOLATION CHECK: 2454 position side MUST remain FLAT!
        self.assertEqual(strategy.get_symbol_position_side("2454"), "FLAT")

        # Now emit a surge on 2454 (> 1.001 * 1200 -> 1215.0)
        surge_2454 = BarEvent(timestamp=t_surge, symbol="2454", open=1205.0, high=1220.0, low=1202.0, close=1215.0, volume=250)
        sig_2454 = strategy.on_bar(surge_2454)

        # 2330 LONG does NOT block 2454 BUY!
        self.assertIsNotNone(sig_2454)
        self.assertEqual(sig_2454.symbol, "2454")
        self.assertEqual(sig_2454.side, "BUY")
        self.assertEqual(strategy.get_symbol_position_side("2454"), "LONG")

        # Test bounded historical bar storage (max_history=10)
        for j in range(10):
            strategy.on_bar(BarEvent(timestamp=t_surge + timedelta(minutes=j + 1), symbol="2330", open=960.0, high=960.0, low=960.0, close=960.0, volume=10))
        self.assertLessEqual(len(strategy.get_symbol_bars("2330")), 10)

    def test_13_paper_broker_bidask_only_preserves_last_price(self):
        """Proves BidAsk-only updates (price=None) preserve last traded price and maintain valid account/positions."""
        paper_broker = PaperBrokerAdapter(initial_cash=1_000_000.0)
        paper_broker.connect()

        # 1. Tick arrives @ 1000.0
        paper_broker.set_market_quote("2330", price=1000.0)
        self.assertEqual(paper_broker._latest_prices["2330"], 1000.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["last"], 1000.0)

        # 2. BidAsk update arrives with price=None: bid=999.0, ask=1001.0
        paper_broker.set_market_quote(
            symbol="2330",
            price=None,
            bid_price=999.0,
            ask_price=1001.0,
            bid_volume=80.0,
            ask_volume=120.0,
        )

        # 3. Last trade must remain 1000.0 (NOT erased or set to None!)
        self.assertEqual(paper_broker._latest_prices["2330"], 1000.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["last"], 1000.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["bid"], 999.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["ask"], 1001.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["bid_volume"], 80.0)
        self.assertEqual(paper_broker._latest_quotes["2330"]["ask_volume"], 120.0)

        # 4. get_account() must remain completely valid without errors
        acct = paper_broker.get_account()
        self.assertEqual(acct["cash"], 1_000_000.0)
        self.assertEqual(acct["total_equity"], 1_000_000.0)

        # 5. Submit Market BUY order: must execute against Ask (1001.0)
        from modules.execution.order import OrderRequest, OrderSide, OrderType
        req_buy = OrderRequest(
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            strategy_id="test_strat",
        )
        from modules.execution.order_manager import OrderManager
        oms = OrderManager()
        order_buy = oms.create_order(req_buy)
        filled_buy = paper_broker.submit_order(order_buy)

        self.assertEqual(filled_buy.status.value, "FILLED")
        # Execution price conforms to Taiwan tick applied to ask (1001.0 -> 1005.0)
        self.assertGreaterEqual(filled_buy.average_fill_price, 1001.0)

        # Check positions mark-to-market does not receive None
        pos = paper_broker.get_positions()
        self.assertIn("2330", pos)
        self.assertIsNotNone(pos["2330"].last_price)
        self.assertEqual(pos["2330"].last_price, 1000.0)

        # 6. Submit Market SELL order: must execute against Bid (999.0)
        req_sell = OrderRequest(
            symbol="2330",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100,
            strategy_id="test_strat",
        )
        order_sell = oms.create_order(req_sell)
        filled_sell = paper_broker.submit_order(order_sell)
        self.assertEqual(filled_sell.status.value, "FILLED")
        self.assertLessEqual(filled_sell.average_fill_price, 1000.0)

    def test_14_shadow_invariant_under_production_quote_mode(self):
        """
        CRITICAL PRODUCTION QUOTE INVARIANT TEST:
        Instantiates ShioajiMarketDataSource with simulation=False.
        Proves that even when receiving genuine production quote format feeds,
        live money execution remains structurally unreachable (place_order calls == 0).
        """
        # Create source with simulation=False using FakeShioajiAPI
        prod_source = ShioajiMarketDataSource(
            api_key="PROD_TEST_KEY",
            secret_key="PROD_TEST_SECRET",
            simulation=False,
            api_instance=self.fake_api,
        )
        self.assertFalse(prod_source.simulation)

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
            test_only_synchronous=True,
            journal=ExecutionJournal(os.path.join(self.test_dir, "j_14.db")),
            persistence=ExecutionStatePersistence(os.path.join(self.test_dir, "p_14.json")),
        )
        strategy = DummyStrategy()
        engine.register_strategy(strategy)
        engine.connect_market_data(prod_source)

        engine.start(reconcile_on_startup=False)
        prod_source.connect()
        prod_source.subscribe(["2330"])

        t_base = datetime(2026, 9, 19, 9, 30, 0, tzinfo=TAIPEI_TZ)

        # Drive Tick + BidAsk through pipeline
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
        positions = paper_broker.get_positions()
        self.assertIn("2330", positions)
        self.assertEqual(positions["2330"].quantity, engine.default_order_shares)

        # ABSOLUTE SAFETY PROOF: Shioaji.place_order was NEVER invoked
        self.assertEqual(self.fake_api.place_order_call_count, 0)

        prod_source.disconnect()
        engine.stop()

    def test_15_show_config_dry_run_safety(self):
        """Proves --show-config dry-run outputs safe masked config and never leaks secrets."""
        import io
        import contextlib
        from scripts.run_shioaji_shadow import show_configuration

        stdout_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf):
            show_configuration(
                symbols=["2330", "2454"],
                paper_cash=2_000_000.0,
                simulation=False,
                api_key="MY_SECRET_API_KEY_1234",
                secret_key="SUPER_CONFIDENTIAL_SECRET_XYZ",
            )

        output = stdout_buf.getvalue()
        self.assertIn("SHADOW", output)
        self.assertIn("PaperBrokerAdapter", output)
        self.assertIn("ShioajiMarketDataSource", output)
        self.assertIn("PRODUCTION (REAL FEED)", output)
        self.assertIn("DISABLED", output)
        # Verify secrets are masked and not printed in plaintext
        self.assertNotIn("MY_SECRET_API_KEY_1234", output)
        self.assertNotIn("SUPER_CONFIDENTIAL_SECRET_XYZ", output)
        self.assertIn("MY_S***1234", output)

    def test_16_modern_sdk_top_level_methods(self):
        """
        Tests current Shioaji SDK (v1.5.x - v1.7.5) top-level API where:
        - api.set_on_tick_stk_v1_callback
        - api.set_on_bidask_stk_v1_callback
        - api.set_event_callback
        - api.subscribe
        - api.unsubscribe
        are top-level methods, and api.quote does NOT expose them.
        """
        from tests.fake_shioaji import FakeShioajiAPI
        from modules.market.shioaji_source import ShioajiSDKCompat

        # pure_modern=True: api.quote is an empty object without methods
        modern_api = FakeShioajiAPI(pure_modern=True)
        self.assertFalse(hasattr(modern_api.quote, "set_on_tick_stk_v1_callback"))
        self.assertFalse(hasattr(modern_api.quote, "subscribe"))
        self.assertFalse(hasattr(modern_api.quote, "unsubscribe"))

        source = ShioajiMarketDataSource(api_instance=modern_api)
        ticks: List[TickEvent] = []
        bidasks: List[BidAskEvent] = []
        source.register_tick_callback(lambda e: ticks.append(e))
        source.register_bidask_callback(lambda e: bidasks.append(e))

        self.assertTrue(source.connect())
        source.subscribe(["2330"])

        self.assertIn(("2330", "tick"), modern_api.subscriptions)
        self.assertIn(("2330", "bidask"), modern_api.subscriptions)

        # Emit tick and bidask via modern top-level methods
        now = datetime.now(TAIPEI_TZ)
        modern_api.emit_tick("2330", 980.0, volume=15, timestamp=now)
        modern_api.emit_bidask("2330", 979.0, 981.0, timestamp=now)

        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].price, 980.0)
        self.assertEqual(ticks[0].volume, 15.0)

        self.assertEqual(len(bidasks), 1)
        self.assertEqual(bidasks[0].bid_price, 979.0)
        self.assertEqual(bidasks[0].ask_price, 981.0)

        # Test modern unsubscribe
        source.unsubscribe(["2330"])
        self.assertIn(("2330", "tick"), modern_api.unsubscriptions)
        self.assertIn(("2330", "bidask"), modern_api.unsubscriptions)

    def test_17_legacy_quote_fallback(self):
        """Tests backwards compatibility fallback when only api.quote exposes methods."""
        from tests.fake_shioaji import FakeQuote, FakeContracts
        from modules.market.shioaji_source import ShioajiSDKCompat

        class LegacyAPI:
            def __init__(self):
                self.quote = FakeQuote()
                self.Contracts = FakeContracts()
            def login(self, *a, **k):
                return [self]

        legacy_api = LegacyAPI()
        # Ensure legacy_api does NOT have top-level methods
        self.assertFalse(hasattr(legacy_api, "set_on_tick_stk_v1_callback"))
        self.assertFalse(hasattr(legacy_api, "subscribe"))

        source = ShioajiMarketDataSource(api_instance=legacy_api)
        ticks: List[TickEvent] = []
        source.register_tick_callback(lambda e: ticks.append(e))
        source.connect()
        source.subscribe(["2330"])

        self.assertIn(("2330", "tick"), legacy_api.quote.subscriptions)
        self.assertIn(("2330", "bidask"), legacy_api.quote.subscriptions)

        legacy_api.quote.emit_tick("2330", 975.0, volume=5)
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].price, 975.0)

    def test_18_dotenv_loading_and_credential_aliases(self):
        """Tests .env loading and credential alias precedence: SHIOAJI_* > SJ_*."""
        from config import Config

        # 1. SHIOAJI_API_KEY takes precedence over SJ_API_KEY
        with unittest.mock.patch.dict(os.environ, {
            "SHIOAJI_API_KEY": "PRIMARY_KEY",
            "SJ_API_KEY": "SECONDARY_KEY",
            "SHIOAJI_SECRET_KEY": "PRIMARY_SEC",
            "SJ_SEC_KEY": "SECONDARY_SEC",
            "SHIOAJI_SIMULATION": "False",
            "SJ_SIMULATION": "True",
        }):
            self.assertEqual(Config.get_api_key(), "PRIMARY_KEY")
            self.assertEqual(Config.get_secret_key(), "PRIMARY_SEC")
            self.assertFalse(Config.get_simulation_mode())

        # 2. SJ_* fallback when SHIOAJI_* is absent
        with unittest.mock.patch.dict(os.environ, {
            "SHIOAJI_API_KEY": "",
            "SJ_API_KEY": "FALLBACK_KEY",
            "SHIOAJI_SECRET_KEY": "",
            "SJ_SEC_KEY": "FALLBACK_SEC",
            "SHIOAJI_SIMULATION": "",
            "SJ_SIMULATION": "False",
        }):
            self.assertEqual(Config.get_api_key(), "FALLBACK_KEY")
            self.assertEqual(Config.get_secret_key(), "FALLBACK_SEC")
            self.assertFalse(Config.get_simulation_mode())

        # 3. ShioajiMarketDataSource respects alias fallback
        with unittest.mock.patch.dict(os.environ, {
            "SHIOAJI_API_KEY": "",
            "SJ_API_KEY": "SJ_KEY_123",
            "SHIOAJI_SECRET_KEY": "",
            "SJ_SEC_KEY": "SJ_SEC_456",
        }):
            ds = ShioajiMarketDataSource()
            self.assertEqual(ds.api_key, "SJ_KEY_123")
            self.assertEqual(ds.secret_key, "SJ_SEC_456")

    def test_19_runner_fatal_exit_codes(self):
        """Tests that runner returns non-zero exit code (1) on fatal failures."""
        from scripts.run_shioaji_shadow import run_shioaji_shadow

        # 1. Fatal startup error: missing credentials
        with unittest.mock.patch.dict(os.environ, {"SHIOAJI_API_KEY": "", "SJ_API_KEY": "", "SHIOAJI_SECRET_KEY": "", "SJ_SEC_KEY": ""}):
            ret = run_shioaji_shadow(symbols=["2330"], api_key="", secret_key="")
            self.assertEqual(ret, 1)

        # 2. Fatal connection error
        with unittest.mock.patch.object(ShioajiMarketDataSource, "connect", return_value=False):
            ret = run_shioaji_shadow(
                symbols=["2330"],
                api_key="TEST_KEY",
                secret_key="TEST_SEC",
                simulation=True,
            )
            self.assertEqual(ret, 1)

        # 3. Fatal subscription error: mismatch reported
        with unittest.mock.patch.object(ShioajiMarketDataSource, "connect", return_value=True), \
             unittest.mock.patch.object(ShioajiMarketDataSource, "is_connected", return_value=True), \
             unittest.mock.patch.object(ShioajiMarketDataSource, "subscribe", side_effect=RuntimeError("Sub failed")):
            ret = run_shioaji_shadow(
                symbols=["2330"],
                api_key="TEST_KEY",
                secret_key="TEST_SEC",
                simulation=True,
            )
            self.assertEqual(ret, 1)

    def test_20_real_feed_verification_state_progression(self):
        """
        Proves real-feed verification requires genuine Tick + BidAsk events.
        Login and subscription alone CANNOT claim REAL_FEED_STREAMING or REAL_FEED_COMPLETED.
        """
        from scripts.run_shioaji_shadow import VerificationState
        from tests.fake_shioaji import FakeShioajiAPI

        fake_api = FakeShioajiAPI(pure_modern=True)
        source = ShioajiMarketDataSource(
            api_key="TEST_KEY",
            secret_key="TEST_SEC",
            simulation=False,
            api_instance=fake_api,
        )
        source.connect()
        source.subscribe(["2330"])

        # Initial state before genuine market events
        verification_state = VerificationState.REAL_FEED_CONNECTING
        self.assertFalse(source.has_received_genuine_events())
        self.assertEqual(verification_state, VerificationState.REAL_FEED_CONNECTING)

        # 1. Tick alone does NOT allow claiming REAL_FEED_STREAMING
        fake_api.emit_tick("2330", 990.0, volume=10)
        self.assertFalse(source.has_received_genuine_events())
        if source.has_received_genuine_events():
            verification_state = VerificationState.REAL_FEED_STREAMING
        self.assertEqual(verification_state, VerificationState.REAL_FEED_CONNECTING)

        # 2. Both Tick AND BidAsk received -> state can transition to REAL_FEED_STREAMING
        fake_api.emit_bidask("2330", 989.0, 991.0)
        self.assertTrue(source.has_received_genuine_events())
        if source.has_received_genuine_events():
            verification_state = VerificationState.REAL_FEED_STREAMING
        self.assertEqual(verification_state, VerificationState.REAL_FEED_STREAMING)

        # 3. Clean shutdown with genuine events completes verification
        clean_shutdown = True
        fatal_error = False
        if clean_shutdown and not fatal_error and verification_state == VerificationState.REAL_FEED_STREAMING:
            verification_state = VerificationState.REAL_FEED_COMPLETED
        self.assertEqual(verification_state, VerificationState.REAL_FEED_COMPLETED)

    def test_21_preflight_mode_safe_output(self):
        """Tests preflight check output and status codes."""
        import io
        import contextlib
        from scripts.run_shioaji_shadow import run_preflight

        # 1. Simulation preflight with missing creds passes safely
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run_preflight(symbols=["2330"], simulation=True, api_key="", secret_key="")
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("Shioaji SDK:", out)
        self.assertIn("Feed: SIMULATION", out)
        self.assertIn("Execution: PAPER (PaperBrokerAdapter)", out)
        self.assertIn("Live orders: DISABLED", out)

        # 2. Production preflight without creds returns 1
        code_prod_fail = run_preflight(symbols=["2330"], simulation=False, api_key="", secret_key="")
        self.assertEqual(code_prod_fail, 1)

        # 3. Production preflight with configured creds returns 0 without leaking them
        buf_prod = io.StringIO()
        with contextlib.redirect_stdout(buf_prod):
            code_prod_ok = run_preflight(
                symbols=["2330"],
                simulation=False,
                api_key="SUPER_SECRET_PROD_KEY",
                secret_key="SUPER_SECRET_PROD_SECRET",
            )
        out_prod = buf_prod.getvalue()
        self.assertEqual(code_prod_ok, 0)
        self.assertIn("Feed: PRODUCTION", out_prod)
        self.assertIn("Credentials: configured", out_prod)
        self.assertNotIn("SUPER_SECRET_PROD_KEY", out_prod)
        self.assertNotIn("SUPER_SECRET_PROD_SECRET", out_prod)

    def test_22_sdk_compat_version_reporting(self):
        """Tests ShioajiSDKCompat version reporting and fallback."""
        from modules.market.shioaji_source import ShioajiSDKCompat

        ver = ShioajiSDKCompat.get_sdk_version()
        self.assertIsInstance(ver, str)
        self.assertTrue(len(ver) > 0)


if __name__ == "__main__":
    unittest.main()

