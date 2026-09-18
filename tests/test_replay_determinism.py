"""
Tests for deterministic replay semantics.
Verifies that two runs given identical recorded ticks, config, seed, and virtual clock
produce bitwise identical finalized bars, signals, risk decisions, orders, fills, and PnL.
"""
from datetime import datetime, timedelta
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.common.clock import VirtualClock
from modules.execution.engine import ExecutionEngine
from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.market.tick_size import TaiwanTickSizeModel
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.strategy.base import BaseStrategy


class DeterministicStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("det_strat")
        self.bars = []
        self.signal_count = 0

    def on_bar(self, bar: BarEvent):
        super().on_bar(bar)
        self.bars.append(bar)
        if len(self.bars) >= 3:
            c1, c2, c3 = self.bars[-3].close, self.bars[-2].close, self.bars[-1].close
            if c3 > c2 > c1 and self.signal_count % 2 == 0:
                self.signal_count += 1
                return SignalEvent(
                    signal_id=f"SIG-BUY-{bar.symbol}-{bar.timestamp.strftime('%H%M%S')}",
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    side="BUY",
                    strength=1.0,
                    strategy_id=self.strategy_id,
                )
            elif c3 < c2 and self.signal_count % 2 == 1:
                self.signal_count += 1
                return SignalEvent(
                    signal_id=f"SIG-SELL-{bar.symbol}-{bar.timestamp.strftime('%H%M%S')}",
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    side="SELL",
                    strength=1.0,
                    strategy_id=self.strategy_id,
                )
        return None


def run_single_replay():
    vclock = VirtualClock()
    start_dt = datetime(2026, 9, 18, 9, 0, 0)
    vclock.set_time(start_dt)

    broker = PaperBrokerAdapter(initial_cash=5_000_000.0, commission_rate=0.001425)
    broker.connect()
    risk_engine = RiskEngine(limits=RiskLimits(), kill_switch=KillSwitch())

    # For replay determinism, use test_only_synchronous=True or synchronous event loop
    engine = ExecutionEngine(
        broker=broker,
        risk_engine=risk_engine,
        clock=vclock,
        trading_mode="shadow",
        test_only_synchronous=True,
        data_source_type="replay",
        soak_mode="ACCELERATED_SIMULATION",
    )
    strategy = DeterministicStrategy()
    engine.register_strategy(strategy)
    engine.start(reconcile_on_startup=False)

    # Replay 30 minutes of synthetic ticks
    for m in range(30):
        t_bar = start_dt + timedelta(minutes=m)
        vclock.set_time(t_bar)
        for s in [0, 15, 30, 45]:
            t_tick = t_bar + timedelta(seconds=s)
            price = TaiwanTickSizeModel.round_to_tick(950.0 + (m % 5), "nearest")
            engine.on_tick(
                TickEvent(
                    timestamp=t_tick,
                    symbol="2330",
                    price=price,
                    volume=10,
                    bid_price=TaiwanTickSizeModel.prev_tick(price),
                    ask_price=TaiwanTickSizeModel.next_tick(price),
                    is_replay=True,
                )
            )

    # Close last bar
    vclock.set_time(start_dt + timedelta(minutes=30))
    engine.on_tick(
        TickEvent(
            timestamp=start_dt + timedelta(minutes=30),
            symbol="2330",
            price=950.0,
            volume=10,
            bid_price=949.0,
            ask_price=951.0,
            is_replay=True,
        )
    )

    report = engine.generate_session_report(session_date="2026-09-18")
    engine.stop()
    return report


class TestReplayDeterminism(unittest.TestCase):
    def test_two_replays_are_bitwise_identical(self):
        report1 = run_single_replay()
        report2 = run_single_replay()

        # Finalized bars
        self.assertEqual(report1.bars_generated, report2.bars_generated)
        # Signals emitted
        self.assertEqual(report1.signals_generated, report2.signals_generated)
        # Risk approvals and rejections
        self.assertEqual(report1.risk_approvals, report2.risk_approvals)
        self.assertEqual(report1.risk_rejections, report2.risk_rejections)
        # Orders and Fills
        self.assertEqual(report1.orders_submitted, report2.orders_submitted)
        self.assertEqual(report1.orders_filled, report2.orders_filled)
        self.assertEqual(report1.total_fills, report2.total_fills)
        # PnL, commission, tax
        self.assertEqual(report1.gross_pnl, report2.gross_pnl)
        self.assertEqual(report1.net_pnl, report2.net_pnl)
        self.assertEqual(report1.total_commission, report2.total_commission)
        self.assertEqual(report1.total_statutory_tax, report2.total_statutory_tax)
        self.assertEqual(report1.total_slippage, report2.total_slippage)
        # Session start and end timestamps
        self.assertEqual(report1.market_session_start, report2.market_session_start)
        self.assertEqual(report1.market_session_end, report2.market_session_end)


if __name__ == "__main__":
    unittest.main()
