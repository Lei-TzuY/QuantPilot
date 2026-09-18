"""
Strategy Warmup Unit Tests.
Verifies indicator historical lookback requirements, prevention of bogus startup signals,
and RiskEngine rejection gating until warmup completion.
"""
from datetime import datetime, timedelta
import unittest

from modules.execution.events import BarEvent, SignalEvent
from modules.risk.engine import RiskEngine
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch
from modules.execution.order import OrderRequest, OrderSide, OrderType
from modules.strategy.base import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    """Requires 20 bars of historical close prices before generating signals."""

    def __init__(self):
        super().__init__("ma_cross_strat", required_warmup_bars=20)
        self.closes = []

    def on_bar(self, bar: BarEvent):
        # Base class handles warmup bar counting
        super().on_bar(bar)
        self.closes.append(bar.close)

        if not self.strategy_ready:
            # During warmup, MUST NOT generate trading signals
            return None

        # Compute 20-bar SMA
        sma20 = sum(self.closes[-20:]) / 20.0
        if bar.close > sma20:
            return SignalEvent(
                signal_id=f"SIG-MA-{bar.timestamp.strftime('%H%M%S')}",
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                side="BUY",
                strength=1.0,
                strategy_id=self.strategy_id,
            )
        return None


class TestStrategyWarmup(unittest.TestCase):

    def test_strategy_warmup_gating(self):
        strat = MovingAverageCrossStrategy()
        self.assertFalse(strat.strategy_ready)
        self.assertEqual(strat.current_warmup_bars, 0)
        self.assertEqual(strat.required_warmup_bars, 20)

        t0 = datetime(2026, 9, 18, 9, 0, 0)

        # Feed 19 bars -> Must remain not ready and emit NO signals
        for i in range(19):
            bar = BarEvent(
                symbol="2330",
                timestamp=t0 + timedelta(minutes=i),
                open=950.0,
                high=960.0,
                low=945.0,
                close=955.0,
                volume=100,
            )
            sig = strat.on_bar(bar)
            self.assertIsNone(sig)
            self.assertFalse(strat.strategy_ready)
            self.assertEqual(strat.current_warmup_bars, i + 1)

        # 20th bar arrives -> Strategy completes warmup!
        bar20 = BarEvent(
            symbol="2330",
            timestamp=t0 + timedelta(minutes=19),
            open=950.0,
            high=960.0,
            low=945.0,
            close=958.0,
            volume=100,
        )
        sig20 = strat.on_bar(bar20)
        self.assertTrue(strat.strategy_ready)
        # 20th bar close 958 > SMA(955.15) -> Signal permitted now
        self.assertIsNotNone(sig20)
        self.assertEqual(sig20.side, "BUY")

    def test_risk_engine_rejects_unwarmed_strategy_orders(self):
        limits = RiskLimits(max_order_value=2_000_000.0, max_position_value_per_symbol=2_000_000.0)
        risk = RiskEngine(limits=limits, kill_switch=KillSwitch())

        req = OrderRequest(
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
            strategy_id="ma_cross_strat",
        )

        # When strategy_ready = False
        decision = risk.evaluate_order(
            request=req,
            current_positions={},
            market_price=950.0,
            strategy_ready=False,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("STRATEGY_NOT_WARMED_UP", decision.reason)

        # When strategy_ready = True
        decision_ok = risk.evaluate_order(
            request=req,
            current_positions={},
            market_price=950.0,
            strategy_ready=True,
        )
        self.assertTrue(decision_ok.allowed)

    def test_restart_at_different_times_without_bogus_signals(self):
        """Tests restarting the strategy at 09:00, 10:15, and 12:30."""
        for restart_time in [
            datetime(2026, 9, 18, 9, 0, 0),
            datetime(2026, 9, 18, 10, 15, 0),
            datetime(2026, 9, 18, 12, 30, 0),
        ]:
            strat = MovingAverageCrossStrategy()
            # Feed 10 historical warmup bars
            warmup_bars = [
                BarEvent(
                    symbol="2330",
                    timestamp=restart_time - timedelta(minutes=10 - i),
                    open=950.0,
                    high=970.0,
                    low=948.0,
                    close=950.0 + i,
                    volume=100,
                )
                for i in range(10)
            ]
            strat.warmup(warmup_bars)
            self.assertEqual(strat.current_warmup_bars, 10)
            self.assertFalse(strat.strategy_ready)

            # First live bar arrives immediately after restart
            live_bar = BarEvent(
                symbol="2330",
                timestamp=restart_time,
                open=960.0,
                high=965.0,
                low=959.0,
                close=962.0,
                volume=150,
            )
            sig = strat.on_bar(live_bar)
            # Must NOT create premature signal because 11 < 20
            self.assertIsNone(sig)


if __name__ == "__main__":
    unittest.main()
