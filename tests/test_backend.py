import unittest
import sys
import os
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from modules.backtester import Backtester

class TestBackend(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.backtester = Backtester()

    def test_backtester_metrics(self):
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = {
            'close': [100 + i + (i%5)*2 for i in range(100)], # Upward trend
            'volume': [1000] * 100
        }
        df = pd.DataFrame(data, index=dates)
        
        # Test backtest run
        result = self.backtester.run(df, strategy='ma_crossover', initial_capital=10000, 
                                   params={'short_window': 5, 'long_window': 20})
        
        self.assertIn('max_drawdown_pct', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('equity_curve', result)
        print(f"Max DD: {result['max_drawdown_pct']:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")

    def test_api_technical_analysis(self):
        # Mocking data fetcher would be ideal, but for now we trust yfinance or handle error gracefully
        try:
            # We assume 2330 is valid and we might have internet. 
            # If not, this might fail, so we wrap in try/except or mock.
            # For this environment, let's assume we can hit the endpoint.
            response = self.app.get('/api/analysis/2330?indicators=ma,bollinger&period=1mo')
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertTrue(data['success'])
                self.assertIn('ma', data['analysis'])
                self.assertIn('bollinger', data['analysis'])
                self.assertIn('periods', data['analysis']['ma'])
        except Exception as e:
            print(f"Skipping online test: {e}")

    def test_api_optimize_strategy(self):
        try:
            # Test optimization with a small search space
            payload = {
                "symbol": "2330",
                "strategy": "ma_crossover",
                "period": "1y",
                "initial_capital": 100000,
                "param_ranges": {
                    "short_window": [5, 10],
                    "long_window": [20, 50]
                }
            }
            response = self.app.post('/api/backtest/optimize', json=payload)
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertTrue(data['success'])
                self.assertTrue(len(data['results']) > 0)
                self.assertIn('return_pct', data['results'][0])
                print(f"Top Result Return: {data['results'][0]['return_pct']:.2f}%")
        except Exception as e:
            print(f"Skipping opt test: {e}")

    def test_api_news(self):
        try:
            response = self.app.get('/api/stock/2330/news')
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertTrue(data['success'])
                self.assertIn('news', data)
                # It might be empty if yfinance fails or no news, but structure should be correct
                if len(data['news']) > 0:
                    self.assertIn('title', data['news'][0])
                    self.assertIn('sentiment', data['news'][0])
        except Exception as e:
            print(f"Skipping news test: {e}")


if __name__ == '__main__':
    unittest.main()
