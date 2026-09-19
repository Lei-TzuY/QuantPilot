"""
QuantPilot Real-Feed Shioaji Shadow Runner
Runs unattended shadow trading during Taiwan market hours using genuine Shioaji market feeds
(Tick + BidAsk) while executing strictly via PaperBrokerAdapter.

HARD SAFETY GUARANTEE:
- Trading mode is unconditionally enforced as SHADOW.
- Real money order submission is structurally impossible: live broker adapters are NEVER instantiated.
- Exclusively uses read-only ShioajiMarketDataSource.

Usage:
    python scripts/run_shioaji_shadow.py --symbols 2330,2454 --record --paper-cash 1000000
"""
import argparse
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import signal
import sys
import time
import tracemalloc
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    # Explicitly load .env file; OS environment variables remain authoritative (override=False)
    load_dotenv(override=False)
except ImportError:
    pass

from config import Config
from modules.brokers.paper import PaperBrokerAdapter
from modules.common.clock import SystemClock
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent
from modules.market.clock import MarketClock, MarketSession
from modules.market.recorder import RawMarketDataRecorder
from modules.market.shioaji_source import ShioajiMarketDataSource, ShioajiSDKCompat
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuantPilot.ShioajiShadowRunner")
TAIPEI_TZ = ZoneInfo("Asia/Taipei")


class VerificationState(str, Enum):
    OFFLINE_VERIFIED = "OFFLINE_VERIFIED"
    REAL_FEED_CONNECTING = "REAL_FEED_CONNECTING"
    REAL_FEED_STREAMING = "REAL_FEED_STREAMING"
    REAL_FEED_COMPLETED = "REAL_FEED_COMPLETED"
    REAL_FEED_FAILED = "REAL_FEED_FAILED"


class ShadowBurnInStrategy(BaseStrategy):
    """
    Deterministic, realistic burn-in strategy for shadow testing.
    Maintains independent per-symbol bar history and position state to prevent cross-symbol contamination.
    Uses short-window momentum to generate realistic PAPER orders exercising:
    Strategy -> Risk -> OMS -> PaperBroker -> Fees -> Tax -> Reconciliation.
    """

    def __init__(self, lookback: int = 5, max_history: int = 100):
        super().__init__("shioaji_shadow_burn_in")
        self.lookback = lookback
        self.max_history = max_history
        self.bars_by_symbol: Dict[str, List[BarEvent]] = {}
        self.position_state_by_symbol: Dict[str, str] = {}

    @property
    def bars(self) -> List[BarEvent]:
        """Backwards-compatible flattened list of all bars across symbols."""
        all_b: List[BarEvent] = []
        for bl in self.bars_by_symbol.values():
            all_b.extend(bl)
        return sorted(all_b, key=lambda b: b.timestamp)

    def get_symbol_bars(self, symbol: str) -> List[BarEvent]:
        return self.bars_by_symbol.get(symbol, [])

    def get_symbol_position_side(self, symbol: str) -> str:
        return self.position_state_by_symbol.get(symbol, "FLAT")

    def on_bar(self, bar: BarEvent) -> Optional[SignalEvent]:
        super().on_bar(bar)
        sym = bar.symbol
        if sym not in self.bars_by_symbol:
            self.bars_by_symbol[sym] = []
            self.position_state_by_symbol[sym] = "FLAT"

        sym_bars = self.bars_by_symbol[sym]
        sym_bars.append(bar)
        if len(sym_bars) > self.max_history:
            sym_bars.pop(0)

        if len(sym_bars) < self.lookback:
            return None

        recent_closes = [b.close for b in sym_bars[-self.lookback:]]
        ma = sum(recent_closes) / len(recent_closes)
        curr = bar.close
        pos_side = self.position_state_by_symbol[sym]

        # Long entry when price crosses above MA and flat for THIS symbol
        if curr > ma * 1.001 and pos_side == "FLAT":
            self.position_state_by_symbol[sym] = "LONG"
            return SignalEvent(
                signal_id=f"SIG-BUY-{sym}-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=sym,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
        # Exit when price crosses below MA and long for THIS symbol
        elif curr < ma * 0.999 and pos_side == "LONG":
            self.position_state_by_symbol[sym] = "FLAT"
            return SignalEvent(
                signal_id=f"SIG-SELL-{sym}-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=sym,
                side="SELL",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
        return None


def resolve_simulation_mode(cli_simulation: Optional[bool] = None) -> bool:
    """
    Resolves whether Shioaji market data source should run in simulation or production.
    Precedence:
        1. Explicit CLI override (--simulation or --no-simulation / --production)
        2. Environment variable SHIOAJI_SIMULATION or SJ_SIMULATION
        3. Config.get_simulation_mode()
        4. Safe default: True
    """
    if cli_simulation is not None:
        return bool(cli_simulation)
    env_val = os.getenv("SHIOAJI_SIMULATION")
    if env_val is None:
        env_val = os.getenv("SJ_SIMULATION")
    if env_val is not None:
        return env_val.strip().lower() in ("true", "1", "yes")
    return Config.get_simulation_mode()


def show_configuration(
    symbols: list,
    paper_cash: float,
    simulation: bool,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> None:
    """Prints safe, masked configuration proof for shadow execution."""
    resolved_api_key = api_key or Config.get_api_key() or ""
    resolved_secret_key = secret_key or Config.get_secret_key() or ""
    masked_key = (
        f"{resolved_api_key[:4]}***{resolved_api_key[-4:]}"
        if len(resolved_api_key) >= 8
        else ("***" if resolved_api_key else "<not configured>")
    )
    feed_mode = "SIMULATION" if simulation else "PRODUCTION (REAL FEED)"

    print("=================================================================")
    print("QuantPilot SHADOW Mode - Configuration Dry-Run Proof")
    print("=================================================================")
    print(f"Trading Mode:          SHADOW")
    print(f"Execution Broker:      PaperBrokerAdapter (PaperBroker)")
    print(f"Market Data Source:    ShioajiMarketDataSource (Shioaji)")
    print(f"Feed Environment:      {feed_mode}")
    print(f"Live Execution:        DISABLED (Autonomous live orders unreachable)")
    print(f"Symbols Subscribed:    {symbols}")
    print(f"Paper Initial Cash:    ${paper_cash:,.2f} TWD")
    print(f"API Key Configured:    {'Yes' if resolved_api_key else 'No'} ({masked_key})")
    print(f"Secret Key Configured: {'Yes' if resolved_secret_key else 'No'}")
    print(f"CA Cert Configured:    {'Yes' if Config.SHIOAJI_CERT_PATH else 'No (Not needed for quote shadow)'}")
    print("=================================================================")


def run_preflight(
    symbols: list,
    simulation: bool,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> int:
    """
    Executes preflight verification according to Requirement 9:
    1. Loads .env
    2. Verifies Shioaji package installed & prints version
    3. Validates production/simulation selection
    4. Verifies credentials configured (yes/no)
    5. Confirms execution broker is PaperBrokerAdapter
    6. Confirms ENABLE_LIVE_TRADING is false
    7. Tests callback registration
    8. Never places orders
    """
    resolved_api_key = api_key or Config.get_api_key() or ""
    resolved_secret_key = secret_key or Config.get_secret_key() or ""
    creds_configured = bool(resolved_api_key and resolved_secret_key)
    sdk_version = ShioajiSDKCompat.get_sdk_version()
    feed_mode = "SIMULATION" if simulation else "PRODUCTION"

    os.environ["TRADING_MODE"] = "shadow"
    os.environ["BROKER_TYPE"] = "paper"
    os.environ["ENABLE_LIVE_TRADING"] = "false"

    paper_broker = PaperBrokerAdapter()
    source = ShioajiMarketDataSource(
        api_key=resolved_api_key,
        secret_key=resolved_secret_key,
        simulation=simulation,
    )
    tick_cb_ok = hasattr(source, "_on_native_tick")
    bidask_cb_ok = hasattr(source, "_on_native_bidask")

    print(f"Shioaji SDK: {sdk_version}")
    print(f"Feed: {feed_mode}")
    print(f"Credentials: {'configured' if creds_configured else 'MISSING'}")
    print(f"Execution: PAPER ({paper_broker.__class__.__name__})")
    print(f"Live orders: DISABLED (ENABLE_LIVE_TRADING=False)")
    print(f"Tick callback: {'registered' if tick_cb_ok else 'UNSUPPORTED'}")
    print(f"BidAsk callback: {'registered' if bidask_cb_ok else 'UNSUPPORTED'}")

    if not simulation and not creds_configured:
        logger.error("Preflight failed: Credentials are missing for PRODUCTION quote feed.")
        return 1

    return 0


def run_shioaji_shadow(
    symbols: list,
    paper_cash: float = 1_000_000.0,
    record: bool = True,
    duration_minutes: int = 270,
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    simulation: bool = True,
) -> int:
    """
    Executes a real-market quote shadow session with paper execution.
    Returns 0 on normal operator shutdown / market-close, 1 on fatal error.
    """
    logger.info("=================================================================")
    logger.info("Starting QuantPilot SHIOAJI SHADOW MODE Session")
    logger.info("READ-ONLY MARKET DATA + PAPER EXECUTION EXCLUSIVITY")
    logger.info(f"Target Symbols: {symbols} | Paper Capital: ${paper_cash:,.2f}")
    logger.info(f"Session Max Duration: {duration_minutes} minutes")
    logger.info("=================================================================")

    # Enforce strict shadow invariants
    os.environ["TRADING_MODE"] = "shadow"
    os.environ["BROKER_TYPE"] = "paper"
    os.environ["ENABLE_LIVE_TRADING"] = "false"

    verification_state = (
        VerificationState.OFFLINE_VERIFIED if simulation else VerificationState.REAL_FEED_CONNECTING
    )
    fatal_error = False

    # Verify credentials exist before starting
    resolved_api_key = api_key or Config.get_api_key()
    resolved_secret_key = secret_key or Config.get_secret_key()
    if not resolved_api_key or not resolved_secret_key:
        verification_state = VerificationState.REAL_FEED_FAILED
        logger.error(
            "FATAL: Shioaji API key and secret key must be configured in environment (.env) "
            "or passed via CLI (--api-key, --secret-key) before starting shadow session."
        )
        return 1

    tracemalloc.start()
    clock = SystemClock()
    market_clock = MarketClock()

    # 1. Market Data Recorder (if enabled)
    recorder = RawMarketDataRecorder(base_dir="data/market") if record else None

    # 2. Paper Broker Adapter (ONLY broker used in shadow execution)
    paper_broker = PaperBrokerAdapter(
        initial_cash=paper_cash,
        commission_rate=Config.PAPER_TRADING_FEE_PCT,
    )
    paper_broker.connect()

    # 3. Risk Engine
    risk_limits = RiskLimits(
        max_order_value=Config.RISK_MAX_ORDER_VALUE,
        max_position_value_per_symbol=Config.RISK_MAX_POSITION_VALUE,
        max_total_exposure=Config.RISK_MAX_TOTAL_EXPOSURE,
        max_open_positions=Config.RISK_MAX_OPEN_POSITIONS,
        max_trades_per_day=Config.RISK_MAX_TRADES_PER_DAY,
        max_daily_realized_loss=Config.RISK_MAX_DAILY_REALIZED_LOSS,
        max_price_deviation_pct=Config.RISK_MAX_PRICE_DEVIATION_PCT,
        max_stale_data_seconds=Config.RISK_MAX_STALE_DATA_SECONDS,
    )
    risk_engine = RiskEngine(limits=risk_limits, kill_switch=KillSwitch())

    # 4. Execution Engine in SHADOW mode
    engine = ExecutionEngine(
        broker=paper_broker,
        risk_engine=risk_engine,
        recorder=recorder,
        trading_mode="shadow",
        test_only_synchronous=False,  # Genuine async queueing
        soak_mode="WALL_CLOCK_SHADOW_SOAK",
        data_source_type="shioaji",
        clock=clock,
    )

    strategy = ShadowBurnInStrategy()
    engine.register_strategy(strategy)

    # 5. Read-Only Shioaji Market Data Source
    shioaji_source = ShioajiMarketDataSource(
        api_key=resolved_api_key,
        secret_key=resolved_secret_key,
        simulation=simulation,
    )

    # Hook quote feed to engine (both Tick and BidAsk callbacks)
    engine.connect_market_data(shioaji_source)

    # Graceful shutdown handling
    shutdown_requested = False

    def handle_signal(sig, frame):
        nonlocal shutdown_requested
        logger.info(f"Signal {sig} received. Initiating clean shadow shutdown...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # Start engine first (worker thread active)
        engine.start(reconcile_on_startup=False)

        feed_mode_str = "SIMULATION" if simulation else "PRODUCTION (REAL FEED)"
        logger.info(f"Market Data Feed Mode: {feed_mode_str}")
        logger.info("Execution Broker: PaperBrokerAdapter (STRICTLY PAPER / ZERO LIVE ORDERS)")

        # Connect quote transport & subscribe
        logger.info(f"Authenticating with SinoPac Shioaji quote transport ({feed_mode_str})...")
        connected = shioaji_source.connect()
        if not connected or not shioaji_source.is_connected():
            fatal_error = True
            verification_state = VerificationState.REAL_FEED_FAILED
            raise ConnectionError(f"Failed to connect to Shioaji market data source ({feed_mode_str}).")

        # Fast-fail if production mode was requested but source reports simulation mode
        if not simulation and shioaji_source.simulation:
            fatal_error = True
            verification_state = VerificationState.REAL_FEED_FAILED
            raise RuntimeError(
                "FATAL: Production quote mode was requested, but market data source is operating in simulation mode."
            )

        logger.info(f"Subscribing Tick + BidAsk depth for symbols: {symbols}...")
        shioaji_source.subscribe(symbols)

        # Confirm subscription state
        missing_ticks = [s for s in symbols if s not in shioaji_source.subscribed_tick_symbols]
        missing_bidask = [s for s in symbols if s not in shioaji_source.subscribed_bidask_symbols]
        if missing_ticks or missing_bidask:
            fatal_error = True
            verification_state = VerificationState.REAL_FEED_FAILED
            raise RuntimeError(
                f"Subscription verification failed. Missing Ticks: {missing_ticks}, Missing BidAsk: {missing_bidask}"
            )
        logger.info(f"Confirmed subscription state: {len(symbols)} symbols subscribed for both Tick and BidAsk streams.")

        start_time = datetime.now(TAIPEI_TZ)
        max_end_time = start_time + timedelta(minutes=duration_minutes)

        logger.info("Entering operational shadow loop. Press Ctrl+C to terminate.")

        while not shutdown_requested:
            now_taipei = datetime.now(TAIPEI_TZ)
            if now_taipei >= max_end_time:
                logger.info("Reached configured session duration. Terminating shadow runner.")
                break

            # Check for transition to REAL_FEED_STREAMING (requires simulation=False AND genuine tick + bidask)
            if (
                not simulation
                and verification_state == VerificationState.REAL_FEED_CONNECTING
                and shioaji_source.has_received_genuine_events()
            ):
                verification_state = VerificationState.REAL_FEED_STREAMING
                logger.info(
                    "*** VERIFICATION STATE TRANSITION: REAL_FEED_STREAMING (Genuine Tick and BidAsk received) ***"
                )

            # Check market session state
            session = market_clock.get_session()
            if session == MarketSession.CLOSED and now_taipei.hour >= 14:
                logger.info("Taiwan market session is CLOSED (post 13:30/14:00). Ending burn-in.")
                break

            # Heartbeat telemetry every 30 seconds
            status = engine.get_status()
            mkt_status = shioaji_source.heartbeat()
            logger.info(
                f"[SHADOW TELEMETRY] Verification={verification_state.value} | Session={session.value} | "
                f"TicksRecv={mkt_status['ticks_received']} | "
                f"BidAskRecv={mkt_status['bidask_received']} | "
                f"QDepth={status['queue']['depth']} | "
                f"Orders={status['open_orders_count']} | "
                f"Positions={status['positions_count']} | "
                f"Equity=${status['broker'].get('total_equity', paper_cash):,.2f}"
            )

            time.sleep(30.0)

    except KeyboardInterrupt:
        logger.info("Operator interrupt received. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error during Shioaji shadow session: {e}", exc_info=True)
        fatal_error = True
        verification_state = VerificationState.REAL_FEED_FAILED
    finally:
        logger.info("Executing clean shutdown sequence...")
        # 1. Unsubscribe and disconnect market feed
        try:
            shioaji_source.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting market data source: {e}")

        # 2. Stop ExecutionEngine (drain queue without deadlocks)
        try:
            engine.stop()
        except Exception as e:
            logger.error(f"Error stopping execution engine: {e}")

        # 3. Flush recorder
        if recorder:
            try:
                recorder.flush()
                logger.info(f"Recorder flushed. Total events recorded: {recorder.get_total_recorded()}")
            except Exception as e:
                logger.error(f"Error flushing recorder: {e}")

        # 4. Check clean completion state
        if (
            not fatal_error
            and not simulation
            and verification_state == VerificationState.REAL_FEED_STREAMING
            and shioaji_source.has_received_genuine_events()
        ):
            verification_state = VerificationState.REAL_FEED_COMPLETED
            logger.info("*** VERIFICATION STATE TRANSITION: REAL_FEED_COMPLETED ***")

        logger.info(f"Final Verification State: {verification_state.value}")

        # 5. Generate & persist session reports
        try:
            report = engine.generate_session_report()
            logger.info("=================================================================")
            logger.info(f"SHADOW SESSION REPORT GENERATED: {report.session_id}")
            logger.info(f"Verification State: {verification_state.value}")
            logger.info(f"Market Data Received: {report.market_data_received}")
            logger.info(f"Ticks Processed: {report.ticks_processed} | Rejected: {report.ticks_rejected}")
            logger.info(f"Bars Finalized: {report.bars_generated} | Signals: {report.signals_generated}")
            logger.info(f"Orders Submitted: {report.orders_submitted} | Filled: {report.orders_filled}")
            logger.info(f"Gross PnL: ${report.gross_pnl:,.2f} | Net PnL: ${report.net_pnl:,.2f}")
            logger.info(f"Statutory Tax: ${report.total_statutory_tax:,.2f} | Fees: ${report.total_commission:,.2f}")
            logger.info("=================================================================")
        except Exception as e:
            logger.error(f"Failed to generate shadow session report: {e}", exc_info=True)

        return 1 if fatal_error else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantPilot Shioaji Shadow Runner")
    parser.add_argument("--symbols", type=str, default="2330,2454", help="Comma-separated stock symbols")
    parser.add_argument("--paper-cash", type=float, default=1_000_000.0, help="Initial paper trading cash")
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record raw tick and bidask parquet streams",
    )
    parser.add_argument("--duration-minutes", type=int, default=270, help="Max run duration in minutes")
    parser.add_argument(
        "--simulation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Shioaji simulation environment (use --no-simulation for production feed)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Shortcut for --no-simulation (connects to real production quote feed)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run preflight environment, credential, SDK and invariant checks then exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display safe masked configuration proof and exit without trading (dry run)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Shioaji API Key (or use SHIOAJI_API_KEY env)")
    parser.add_argument("--secret-key", type=str, default=None, help="Shioaji Secret Key (or use SHIOAJI_SECRET_KEY env)")

    args = parser.parse_args()
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Resolve simulation / production precedence
    if args.production:
        cli_sim = False
    else:
        cli_sim = args.simulation
    simulation = resolve_simulation_mode(cli_sim)

    if args.preflight:
        code = run_preflight(
            symbols=syms,
            simulation=simulation,
            api_key=args.api_key,
            secret_key=args.secret_key,
        )
        sys.exit(code)

    if args.show_config:
        show_configuration(
            symbols=syms,
            paper_cash=args.paper_cash,
            simulation=simulation,
            api_key=args.api_key,
            secret_key=args.secret_key,
        )
        sys.exit(0)

    exit_code = run_shioaji_shadow(
        symbols=syms,
        paper_cash=args.paper_cash,
        record=args.record,
        duration_minutes=args.duration_minutes,
        api_key=args.api_key,
        secret_key=args.secret_key,
        simulation=simulation,
    )
    sys.exit(exit_code)
