import warnings
import unittest
import numpy as np
import pandas as pd

from modules.backtester import Backtester
from modules.ml_backtester import MLBacktester


class TestExecutionTiming(unittest.TestCase):
    """
    Verifies that all supported backtest engines strictly enforce zero look-ahead bias:
    A signal produced from bar t's close is executed earliest at bar t+1's Open.
    """

    def setUp(self):
        # Create a deterministic 5-day dataset
        # Day 1: 100 open, 102 close
        # Day 2: 105 open, 110 close (crossover signal triggers here on close)
        # Day 3: 112 open, 115 close (must execute here on Open, not Day 2 close!)
        # Day 4: 114 open, 108 close (exit signal triggers on close)
        # Day 5: 106 open, 105 close (must exit here on Open)
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        self.df = pd.DataFrame({
            "open": [100.0, 105.0, 112.0, 114.0, 106.0],
            "high": [103.0, 111.0, 116.0, 115.0, 107.0],
            "low": [99.0, 104.0, 111.0, 107.0, 104.0],
            "close": [102.0, 110.0, 115.0, 108.0, 105.0],
            "volume": [1000, 1500, 2000, 1800, 1200],
        }, index=dates)

    def test_backtester_default_is_next_bar_open(self):
        """Rule-based backtester defaults to next_bar_open without lookahead."""
        backtester = Backtester()
        # Mock strategy signal directly onto a test df
        test_df = self.df.copy()
        # Bar 0 (2026-01-01): signal 0
        # Bar 1 (2026-01-02): signal 1 (Buy signal on bar 1 close)
        # Bar 2 (2026-01-03): signal 0
        # Bar 3 (2026-01-04): signal -1 (Sell signal on bar 3 close)
        # Bar 4 (2026-01-05): signal 0
        test_df["signal"] = [0, 1, 0, -1, 0]

        # Use 0 slippage for exact price verification
        risk_params = {"slippage_pct": 0.0, "commission_rate": 0.0, "tax_rate": 0.0}
        trades, final_val, equity_curve, fee_summary = backtester._simulate(
            test_df, initial_capital=100_000, risk_params=risk_params
        )

        self.assertEqual(len(trades), 2)
        buy_trade = trades[0]
        sell_trade = trades[1]

        # Buy signal at bar 1 (2026-01-02 close) MUST execute at bar 2 (2026-01-03 open = 112.0)
        self.assertEqual(buy_trade["type"], "buy")
        self.assertEqual(buy_trade["date"], "2026-01-03")
        self.assertEqual(buy_trade["market_price"], 112.0)
        self.assertEqual(buy_trade["price"], 112.0)

        # Sell signal at bar 3 (2026-01-04 close) MUST execute at bar 4 (2026-01-05 open = 106.0)
        self.assertEqual(sell_trade["type"], "sell")
        self.assertEqual(sell_trade["date"], "2026-01-05")
        self.assertEqual(sell_trade["market_price"], 106.0)
        self.assertEqual(sell_trade["price"], 106.0)

    def test_backtester_same_bar_close_emits_warning(self):
        """Deprecated same_bar_close emits a DeprecationWarning."""
        backtester = Backtester()
        test_df = self.df.copy()
        test_df["signal"] = [0, 1, 0, -1, 0]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backtester._simulate(test_df, initial_capital=100_000, risk_params={"execution_timing": "same_bar_close"})
            deprecation_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
            self.assertTrue(len(deprecation_warnings) >= 1)
            self.assertIn("same_bar_close", str(deprecation_warnings[0].message))

    def test_ml_backtester_default_is_next_bar_open(self):
        """ML backtester executes bar t prediction at bar t+1 Open."""
        ml_backtester = MLBacktester()
        test_df = self.df.copy()
        # Bar 0 (2026-01-01): pred 0
        # Bar 1 (2026-01-02): pred 1 (Buy prediction on bar 1)
        # Bar 2 (2026-01-03): pred 1
        # Bar 3 (2026-01-04): pred 0 (Sell prediction on bar 3)
        # Bar 4 (2026-01-05): pred 0
        predictions = np.array([0, 1, 1, 0, 0])
        probabilities = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.8, 0.2],
        ])

        result = ml_backtester.backtest_ml_strategy(
            test_df,
            predictions,
            probabilities,
            initial_capital=100_000,
            slippage_pct=0.0,
            fee_rate=0.0,
            tax_rate=0.0,
            execution_timing="next_bar_open",
        )

        trades = result["trades"]
        self.assertTrue(len(trades) >= 1)
        trade = trades[0]

        # Entry MUST be on 2026-01-03 at Open (112.0), NOT 2026-01-02 Close (110.0)
        self.assertEqual(trade["entry_date"], test_df.index[2])
        self.assertEqual(trade["entry_price"], 112.0)

        # Exit MUST be on 2026-01-05 at Open (106.0), NOT 2026-01-04 Close (108.0)
        self.assertEqual(trade["exit_date"], test_df.index[4])
        self.assertEqual(trade["exit_price"], 106.0)

    def test_ml_backtester_same_bar_close_emits_warning(self):
        """MLBacktester emits DeprecationWarning when same_bar_close is requested."""
        ml_backtester = MLBacktester()
        predictions = np.array([0, 1, 1, 0, 0])
        probabilities = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.8, 0.2],
        ])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ml_backtester.backtest_ml_strategy(
                self.df,
                predictions,
                probabilities,
                execution_timing="same_bar_close",
            )
            deprecation_warnings = [item for item in w if issubclass(item.category, DeprecationWarning)]
            self.assertTrue(len(deprecation_warnings) >= 1)
            self.assertIn("same_bar_close", str(deprecation_warnings[0].message))


if __name__ == "__main__":
    unittest.main()
