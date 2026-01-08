import concurrent.futures
from typing import List, Dict, Any
import pandas as pd
from modules.data_fetcher import DataFetcher
from modules.backtester import Backtester

class BatchProcessor:
    def __init__(self, data_fetcher: DataFetcher, backtester: Backtester):
        self.data_fetcher = data_fetcher
        self.backtester = backtester

    def run_batch_backtest(
        self,
        symbols: List[str],
        strategy: str = "ma_crossover",
        period: str = "1y",
        initial_capital: float = 1_000_000,
        start_date: str = None, # Optional override (not used yet)
        end_date: str = None,   # Optional override
        params: Dict = None,
        risk_params: Dict = None
    ) -> List[Dict]:
        """
        Run backtests for multiple symbols in parallel.
        Returns a list of results sorted by Return %.
        """
        results = []
        
        # Use ThreadPoolExecutor for I/O bound tasks (fetching data)
        # Note: Backtesting itself is CPU bound, but fetching data is the bottleneck usually.
        # If we already have data, ProcessPool might be better, but ThreadPool is safer in Flask context.
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(
                    self._process_single_stock, 
                    symbol, 
                    strategy, 
                    period, 
                    initial_capital, 
                    params, 
                    risk_params
                ): symbol for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    # Capture error result
                    results.append({
                        "symbol": symbol,
                        "error": str(e),
                        "return_pct": -999 # Push to bottom
                    })

        # Sort by return_pct descending
        results.sort(key=lambda x: x.get("return_pct", -999), reverse=True)
        return results

    def _process_single_stock(
        self, symbol, strategy, period, capital, params, risk_params
    ) -> Dict:
        try:
            # Fetch Data
            df = self.data_fetcher.get_stock_data_df(symbol, period=period)
            
            # Run Backtest
            result = self.backtester.run(
                df, 
                strategy=strategy, 
                initial_capital=capital, 
                params=params, 
                risk_params=risk_params
            )
            
            # Inject symbol into result for identification
            result["symbol"] = symbol
            
            # Simplify output for batch view (don't need full equity curve or trades list usually)
            # But kept for now if user wants details. 
            # We can strip heavy data if needed.
            del result["equity_curve"] 
            
            return result
        except Exception as e:
            # print(f"Batch failed for {symbol}: {e}")
            raise e
