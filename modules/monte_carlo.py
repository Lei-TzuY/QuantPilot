"""
Monte Carlo Simulation Module for Backtesting
Generates probabilistic outcomes and confidence intervals.
"""

import numpy as np
from typing import Dict, List, Optional
import pandas as pd


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio risk analysis.
    Uses historical returns to simulate future scenarios.
    """
    
    def __init__(self, n_simulations: int = 1000, trading_days: int = 252):
        self.n_simulations = n_simulations
        self.trading_days = trading_days
    
    def run_simulation(
        self, 
        equity_curve: List[float], 
        initial_capital: float = None,
        forecast_days: int = 252
    ) -> Dict:
        """
        Run Monte Carlo simulation based on historical equity curve.
        
        Args:
            equity_curve: Historical equity values from backtest
            initial_capital: Starting capital (uses last equity value if None)
            forecast_days: Number of days to simulate forward
            
        Returns:
            Dictionary with simulation results
        """
        if len(equity_curve) < 20:
            return {
                'success': False,
                'error': 'Insufficient data for Monte Carlo simulation (need at least 20 data points)'
            }
        
        # Convert to numpy array and calculate returns
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Remove any NaN or inf values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 10:
            return {
                'success': False,
                'error': 'Insufficient valid returns for simulation'
            }
        
        # Calculate return statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Starting value for simulation
        start_value = initial_capital if initial_capital else equity[-1]
        
        # Run simulations
        simulations = np.zeros((self.n_simulations, forecast_days))
        
        for i in range(self.n_simulations):
            # Generate random returns based on historical distribution
            random_returns = np.random.normal(mean_return, std_return, forecast_days)
            
            # Calculate cumulative portfolio values
            cumulative = start_value * np.cumprod(1 + random_returns)
            simulations[i] = cumulative
        
        # Calculate statistics across simulations
        final_values = simulations[:, -1]
        
        # Percentiles for confidence intervals
        percentiles = {
            'p5': float(np.percentile(final_values, 5)),
            'p25': float(np.percentile(final_values, 25)),
            'p50': float(np.percentile(final_values, 50)),
            'p75': float(np.percentile(final_values, 75)),
            'p95': float(np.percentile(final_values, 95))
        }
        
        # VaR calculations
        var_95 = start_value - percentiles['p5']
        var_99 = start_value - float(np.percentile(final_values, 1))
        
        # Expected Shortfall (CVaR) - average loss beyond VaR
        losses = start_value - final_values
        cvar_95 = float(np.mean(losses[losses >= var_95])) if np.any(losses >= var_95) else 0
        
        # Probability of profit
        prob_profit = float(np.sum(final_values > start_value) / self.n_simulations * 100)
        
        # Calculate simulation paths for visualization (sample 10 paths)
        sample_indices = np.linspace(0, self.n_simulations - 1, 10, dtype=int)
        sample_paths = simulations[sample_indices].tolist()
        
        # Confidence bands for chart
        p5_path = np.percentile(simulations, 5, axis=0).tolist()
        p50_path = np.percentile(simulations, 50, axis=0).tolist()
        p95_path = np.percentile(simulations, 95, axis=0).tolist()
        
        return {
            'success': True,
            'start_value': start_value,
            'forecast_days': forecast_days,
            'n_simulations': self.n_simulations,
            'mean_return': float(mean_return * 252 * 100),  # Annualized %
            'volatility': float(std_return * np.sqrt(252) * 100),  # Annualized %
            'percentiles': percentiles,
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'probability_of_profit': prob_profit,
            'expected_value': float(np.mean(final_values)),
            'confidence_bands': {
                'p5': p5_path,
                'p50': p50_path,
                'p95': p95_path
            },
            'sample_paths': sample_paths
        }
    
    def calculate_max_drawdown_distribution(self, equity_curve: List[float]) -> Dict:
        """
        Simulate max drawdown distribution using bootstrap.
        """
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]
        
        if len(returns) < 20:
            return {'success': False, 'error': 'Insufficient data'}
        
        drawdowns = []
        
        for _ in range(self.n_simulations):
            # Bootstrap resample returns
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve from sampled returns
            simulated_equity = 100 * np.cumprod(1 + sampled_returns)
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(simulated_equity)
            dd = (simulated_equity - peak) / peak
            max_dd = float(np.min(dd) * 100)
            drawdowns.append(max_dd)
        
        drawdowns = np.array(drawdowns)
        
        return {
            'success': True,
            'mean_max_dd': float(np.mean(drawdowns)),
            'median_max_dd': float(np.median(drawdowns)),
            'p5_max_dd': float(np.percentile(drawdowns, 5)),
            'p95_max_dd': float(np.percentile(drawdowns, 95)),
            'worst_case': float(np.min(drawdowns))
        }
