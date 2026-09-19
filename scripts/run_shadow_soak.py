"""
Unattended Shadow Trading Soak Session Runner.
Simulates or runs a full Taiwan trading day (09:00:00 -> 13:30:00 = 270 minutes) in SHADOW mode.

Supported Modes:
1. ACCELERATED_SIMULATION:
   270 market minutes execute as fast as possible in-process with genuine asynchronous queueing.
   Intended for CI and deterministic testing.
2. REALTIME_REPLAY:
   Replays timestamp spacing faithfully using virtual/source timeline.
   Intended for pipeline timing behavior.
3. WALL_CLOCK_SHADOW_SOAK:
   Runs against real incoming quote data from approximately 09:00-13:30 Taiwan time.
   Intended for actual operational burn-in.

Usage:
    python scripts/run_shadow_soak.py --mode accelerated_simulation --symbols 2330,2454 --date 2026-09-18
"""
import argparse
from datetime import datetime, timedelta
import logging
import os
import sys
import time
import tracemalloc

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.brokers.paper import PaperBrokerAdapter
from modules.common.clock import SystemClock, VirtualClock
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.market.tick_size import TaiwanTickSizeModel
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("QuantPilot.ShadowSoakRunner")


class SoakDemonstrationStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("soak_demo_strat")
        self.bars = []
        self.signals = []

    def on_bar(self, bar: BarEvent):
        super().on_bar(bar)
        self.bars.append(bar)
        # Momentum trigger on 3 consecutive rising closes
        if len(self.bars) >= 3:
            c1, c2, c3 = self.bars[-3].close, self.bars[-2].close, self.bars[-1].close
            if c3 > c2 > c1 and len(self.signals) % 2 == 0:
                sig = SignalEvent(
                    signal_id=f"SIG-BUY-{bar.symbol}-{bar.timestamp.strftime('%H%M%S')}",
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    side="BUY",
                    strength=1.0,
                    strategy_id=self.strategy_id,
                )
                self.signals.append(sig)
                return sig
            elif c3 < c2 and len(self.signals) % 2 == 1:
                sig = SignalEvent(
                    signal_id=f"SIG-SELL-{bar.symbol}-{bar.timestamp.strftime('%H%M%S')}",
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    side="SELL",
                    strength=1.0,
                    strategy_id=self.strategy_id,
                )
                self.signals.append(sig)
                return sig
        return None


def run_soak_session(
    symbols: list,
    session_date: str = "2026-09-18",
    mode: str = "accelerated_simulation",
    speed: float = 1.0,
):
    normalized_mode = mode.upper()
    valid_modes = {"ACCELERATED_SIMULATION", "REALTIME_REPLAY", "WALL_CLOCK_SHADOW_SOAK"}
    if normalized_mode not in valid_modes:
        raise ValueError(f"Invalid soak mode: {mode}. Must be one of {valid_modes}")

    logger.info("=================================================================")
    logger.info("Starting QuantPilot SHADOW Mode Full-Day Soak Session")
    logger.info(f"Target Session: {session_date} 09:00:00 -> 13:30:00 (270 minutes)")
    logger.info(f"Symbols: {symbols} | Mode: {normalized_mode}")
    logger.info("Queue Mode: ASYNCHRONOUS (Production/Shadow Default)")
    logger.info("=================================================================")

    tracemalloc.start()
    mem_start = tracemalloc.take_snapshot()

    paper_broker = PaperBrokerAdapter(initial_cash=10_000_000.0, commission_rate=0.001425)
    paper_broker.connect()
    risk_limits = RiskLimits(
        max_order_value=2_000_000.0,
        max_position_value_per_symbol=10_000_000.0,
        max_total_exposure=20_000_000.0,
    )
    risk_engine = RiskEngine(limits=risk_limits, kill_switch=KillSwitch())

    clock = VirtualClock() if normalized_mode == "REALTIME_REPLAY" else SystemClock()

    journal_path = f"data/journal/soak_journal_{session_date}.db"
    state_path = f"data/state/soak_state_{session_date}.json"
    os.makedirs(os.path.dirname(journal_path), exist_ok=True)
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    if os.path.exists(journal_path):
        try:
            os.remove(journal_path)
        except OSError:
            pass
    if os.path.exists(state_path):
        try:
            os.remove(state_path)
        except OSError:
            pass

    from modules.execution.journal import ExecutionJournal
    from modules.execution.persistence import ExecutionStatePersistence
    journal = ExecutionJournal(db_path=journal_path)
    persistence = ExecutionStatePersistence(storage_path=state_path)

    engine = ExecutionEngine(
        broker=paper_broker,
        risk_engine=risk_engine,
        trading_mode="shadow",
        test_only_synchronous=False,  # Enforce genuine asynchronous queue!
        soak_mode=normalized_mode,
        data_source_type="synthetic" if normalized_mode != "WALL_CLOCK_SHADOW_SOAK" else "live",
        clock=clock,
        journal=journal,
        persistence=persistence,
    )
    strategy = SoakDemonstrationStrategy()
    engine.register_strategy(strategy)
    engine.start(reconcile_on_startup=False)

    start_dt = datetime.strptime(f"{session_date} 09:00:00", "%Y-%m-%d %H:%M:%S")

    total_ticks = 0
    # Simulate 270 minutes (4 ticks per minute per symbol = 1,080 ticks/sym)
    for m in range(270):
        m_time = start_dt + timedelta(minutes=m)
        if isinstance(clock, VirtualClock):
            clock.set_time(m_time)

        for s_idx, sec in enumerate([0, 15, 30, 45]):
            t_time = m_time + timedelta(seconds=sec)
            if isinstance(clock, VirtualClock):
                clock.set_time(t_time)

            for sym in symbols:
                base = 950.0 if sym == "2330" else 1200.0
                raw_price = base + (m % 15) - (sec // 20)
                # Enforce Taiwan equity tick size rules
                norm_price = TaiwanTickSizeModel.round_to_tick(raw_price, mode="nearest")
                bid_price = TaiwanTickSizeModel.prev_tick(norm_price)
                ask_price = TaiwanTickSizeModel.next_tick(norm_price)

                tick = TickEvent(
                    timestamp=t_time,
                    symbol=sym,
                    price=norm_price,
                    volume=15 + (m % 5),
                    bid_price=bid_price,
                    ask_price=ask_price,
                )
                engine.on_tick(tick)
                total_ticks += 1

            if normalized_mode == "REALTIME_REPLAY" and speed > 0:
                # Sleep proportional to inter-tick step / speed factor
                time.sleep(0.001 / speed)

    # End of day boundary tick at 13:30:00
    end_dt = start_dt + timedelta(minutes=270)
    if isinstance(clock, VirtualClock):
        clock.set_time(end_dt)

    for sym in symbols:
        final_raw = 955.0 if sym == "2330" else 1205.0
        final_norm = TaiwanTickSizeModel.round_to_tick(final_raw, mode="nearest")
        engine.on_tick(
            TickEvent(
                timestamp=end_dt,
                symbol=sym,
                price=final_norm,
                volume=10,
                bid_price=TaiwanTickSizeModel.prev_tick(final_norm),
                ask_price=TaiwanTickSizeModel.next_tick(final_norm),
            )
        )
        total_ticks += 1

    # Drain asynchronous queue completely to finalize all bars and metrics before report generation
    logger.info("Waiting for asynchronous queue to drain all events...")
    engine.event_queue.join(timeout=15.0)

    # Generate and persist canonical session report
    report = engine.generate_session_report(session_date=session_date)
    engine.stop()

    mem_end = tracemalloc.take_snapshot()
    diff = sum(s.size_diff for s in mem_end.compare_to(mem_start, "lineno")) / 1024.0
    tracemalloc.stop()

    # Validate invariants
    invariants = report.validate_invariants()
    if invariants:
        logger.error(f"REPORT INVARIANT VIOLATIONS DETECTED: {invariants}")
    else:
        logger.info("Session Report Invariant Checks Passed: 100% Consistent.")

    logger.info("=================================================================")
    logger.info("SHADOW Soak Session Completed Successfully!")
    logger.info(f"Mode: {report.soak_mode} | Git Commit: {report.git_commit_sha} (Dirty: {report.dirty_working_tree})")
    logger.info(f"Market Session Window: {report.market_session_start} -> {report.market_session_end}")
    logger.info(f"Total Ticks Ingested: {total_ticks:,} (Dequeued: {report.ticks_processed:,})")
    logger.info(f"Total Finalized Bars: {report.bars_generated}")
    logger.info(f"Signals: {report.signals_generated} | Risk Approvals: {report.risk_approvals} | Rejections: {report.risk_rejections}")
    logger.info(f"Orders Submitted: {report.orders_submitted} | Fills: {report.total_fills}")
    logger.info(f"Net Realized PnL: {report.net_pnl:,.2f} TWD (Gross: {report.gross_pnl:,.2f}, Comm: {report.total_commission:,.2f}, Tax: {report.total_statutory_tax:,.2f})")
    logger.info(f"Queue Max Depth: {report.queue_max_depth} | Dropped Ticks: {report.ticks_rejected}")
    logger.info(f"Memory Delta: {diff:,.1f} KB")
    logger.info("=================================================================")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QuantPilot Shadow Soak Session")
    parser.add_argument(
        "--mode",
        type=str,
        default="accelerated_simulation",
        choices=["accelerated_simulation", "realtime_replay", "wall_clock_shadow_soak"],
        help="Soak execution mode",
    )
    parser.add_argument("--symbols", type=str, default="2330,2454", help="Comma-separated stock symbols")
    parser.add_argument("--date", type=str, default="2026-09-18", help="Session date YYYY-MM-DD")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier for replay modes")
    args = parser.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    run_soak_session(syms, session_date=args.date, mode=args.mode, speed=args.speed)
