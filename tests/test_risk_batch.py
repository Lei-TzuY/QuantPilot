import unittest
import sys
import os
import json
from flask import Flask

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from modules.backtester import Backtester
from modules.data_fetcher import DataFetcher
import pandas as pd

class TestRiskAndBatch(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.backtester = Backtester()
        self.data_fetcher = DataFetcher()

    def test_backtest_with_stop_loss(self):
        """Test that Stop Loss triggers."""
        # Create dummy data: Price drops significantly
        dates = pd.date_range(start="2023-01-01", periods=10)
        data = {
            "close": [100, 102, 105, 100, 95, 90, 85, 90, 95, 100],
            "high":  [101, 103, 106, 101, 96, 91, 86, 91, 96, 101],
            "low":   [99,  101, 104, 99,  94, 89, 84, 89, 94, 99],
            "open":  [100, 102, 105, 100, 95, 90, 85, 90, 95, 100],
            "signal": [1, 0, 0, 0, 0, 0, 0, 0, 0, -1] # Buy at day 0
        }
        df = pd.DataFrame(data, index=dates)
        
        # Run with SL = 5%
        # Entry @ 100. SL @ 95.
        # Day 4: Low 94 <= 95. Should trigger SL.
        
        trades, final_val, _, _ = self.backtester._simulate(df, 10000, risk_params={"stop_loss_pct": 0.05})
        
        self.assertEqual(len(trades), 2) # Buy, then SL Sell
        self.assertEqual(trades[1]['type'], 'sell')
        self.assertEqual(trades[1]['reason'], 'Stop Loss')
        # SL price should be 95 (unless open was lower, but open was 95)
        self.assertTrue(trades[1]['price'] <= 95)

    def test_batch_api_endpoint(self):
        """Test the batch API endpoint (mocked)."""
        # We assume backtester works, we just check if endpoint responds
        # Using symbols that likely exist or handling error gracefully
        response = self.app.post('/api/batch/backtest', json={
            "symbols": ["2330", "2317"],
            "period": "1mo", # Short period for speed
            "risk_params": {"stop_loss_pct": 0.1}
        })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['results'], list)
        
if __name__ == '__main__':
    unittest.main()
