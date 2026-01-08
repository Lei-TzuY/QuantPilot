"""
Time Series Decomposition Module
Prophet-like decomposition for trend, seasonality, and forecast.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.optimize import minimize


class TimeSeriesDecomposer:
    """
    Time series decomposition and forecasting.
    Decomposes into trend, seasonality, and residual components.
    """
    
    def __init__(self):
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.period = None
        self.fitted_params = {}
    
    def decompose(self, df: pd.DataFrame, period: int = 5, 
                  column: str = 'close') -> Dict:
        """
        Decompose time series into components.
        
        Args:
            df: DataFrame with time series data
            period: Seasonal period (5 for weekly, 20 for monthly)
            column: Column to decompose
            
        Returns:
            Decomposition results
        """
        data = df[column].values
        n = len(data)
        self.period = period
        
        # Calculate trend using moving average
        if period > 1:
            trend = pd.Series(data).rolling(period, center=True).mean().values
        else:
            trend = data.copy()
        
        # Handle NaN at edges
        for i in range(period // 2):
            trend[i] = np.mean(data[:period])
            trend[-(i+1)] = np.mean(data[-period:])
        
        self.trend = trend
        
        # Detrend to get seasonal + residual
        detrended = data - trend
        
        # Extract seasonal component using period averaging
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            seasonal_value = np.mean(detrended[indices])
            seasonal[indices] = seasonal_value
        
        self.seasonal = seasonal
        
        # Residual
        self.residual = data - trend - seasonal
        
        # Strength of components
        var_total = np.var(data)
        var_residual = np.var(self.residual)
        
        trend_strength = max(0, 1 - var_residual / np.var(data - self.seasonal))
        seasonal_strength = max(0, 1 - var_residual / np.var(data - self.trend))
        
        return {
            "success": True,
            "period": period,
            "n_points": n,
            "trend_strength": round(trend_strength, 4),
            "seasonal_strength": round(seasonal_strength, 4),
            "residual_std": round(float(np.std(self.residual)), 4),
            "components": {
                "trend": self.trend.tolist(),
                "seasonal": self.seasonal.tolist(),
                "residual": self.residual.tolist()
            }
        }
    
    def forecast(self, df: pd.DataFrame, horizon: int = 10, 
                 column: str = 'close') -> Dict:
        """
        Forecast future values.
        
        Args:
            df: Historical DataFrame
            horizon: Number of periods to forecast
            column: Column to forecast
            
        Returns:
            Forecast results
        """
        data = df[column].values
        n = len(data)
        
        # Decompose if not done
        if self.trend is None:
            self.decompose(df, column=column)
        
        # Fit linear trend
        x = np.arange(n)
        slope, intercept = np.polyfit(x, self.trend, 1)
        
        # Forecast trend
        future_x = np.arange(n, n + horizon)
        trend_forecast = slope * future_x + intercept
        
        # Forecast seasonal (repeat pattern)
        period = self.period or 5
        seasonal_forecast = np.array([
            self.seasonal[(n + i) % period] if period > 0 else 0 
            for i in range(horizon)
        ])
        
        # Combine
        forecast = trend_forecast + seasonal_forecast
        
        # Confidence intervals based on residual std
        std = np.std(self.residual) if self.residual is not None else np.std(data) * 0.02
        ci_lower = forecast - 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        ci_upper = forecast + 1.96 * std * np.sqrt(np.arange(1, horizon + 1))
        
        # Last known value
        last_value = data[-1]
        
        return {
            "success": True,
            "last_value": float(last_value),
            "forecast": forecast.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "trend_direction": "up" if slope > 0 else "down",
            "trend_slope": round(float(slope), 4),
            "forecast_change_pct": round((forecast[-1] / last_value - 1) * 100, 2)
        }
    
    def detect_changepoints(self, df: pd.DataFrame, column: str = 'close',
                            threshold: float = 2.0) -> Dict:
        """
        Detect trend changepoints in the data.
        
        Args:
            df: DataFrame
            column: Column to analyze
            threshold: Z-score threshold for changepoint detection
            
        Returns:
            Changepoint detection results
        """
        data = df[column].values
        n = len(data)
        
        # Calculate returns
        returns = np.diff(data) / data[:-1]
        
        # Rolling mean and std of returns
        window = 20
        rolling_mean = pd.Series(returns).rolling(window).mean().values
        rolling_std = pd.Series(returns).rolling(window).std().values
        
        # Z-scores
        z_scores = np.zeros(n - 1)
        for i in range(window, n - 1):
            if rolling_std[i] > 0:
                z_scores[i] = abs(returns[i] - rolling_mean[i]) / rolling_std[i]
        
        # Find changepoints
        changepoints = np.where(z_scores > threshold)[0]
        
        # Get dates if available
        if hasattr(df.index, 'strftime'):
            cp_dates = [str(df.index[i + 1])[:10] for i in changepoints[-10:]]
        else:
            cp_dates = changepoints[-10:].tolist()
        
        # Trend analysis between changepoints
        segments = []
        prev_cp = 0
        for cp in changepoints:
            if cp - prev_cp > 5:  # Minimum segment length
                segment_data = data[prev_cp:cp+1]
                segment_return = (segment_data[-1] / segment_data[0] - 1) * 100
                segments.append({
                    "start_idx": int(prev_cp),
                    "end_idx": int(cp),
                    "return_pct": round(segment_return, 2),
                    "direction": "up" if segment_return > 0 else "down"
                })
            prev_cp = cp
        
        return {
            "success": True,
            "total_changepoints": len(changepoints),
            "recent_changepoints": cp_dates,
            "threshold": threshold,
            "trend_segments": segments[-5:],
            "current_trend": segments[-1]["direction"] if segments else "unknown"
        }
    
    def seasonal_analysis(self, df: pd.DataFrame, column: str = 'close') -> Dict:
        """
        Analyze seasonal patterns in the data.
        
        Returns:
            Seasonal pattern analysis
        """
        data = df[column].values
        returns = np.diff(data) / data[:-1]
        
        # Daily patterns (if enough data)
        if hasattr(df.index, 'dayofweek'):
            daily_returns = {}
            for day in range(5):  # Monday to Friday
                mask = df.index[1:].dayofweek == day
                if mask.sum() > 0:
                    day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][day]
                    daily_returns[day_name] = {
                        "mean_return": round(float(np.mean(returns[mask])) * 100, 4),
                        "positive_rate": round(float((returns[mask] > 0).mean()) * 100, 2),
                        "count": int(mask.sum())
                    }
        else:
            daily_returns = {}
        
        # Monthly patterns (if available)
        monthly_returns = {}
        if hasattr(df.index, 'month') and len(df) > 252:
            for month in range(1, 13):
                mask = df.index[1:].month == month
                if mask.sum() > 0:
                    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
                    monthly_returns[month_name] = {
                        "mean_return": round(float(np.mean(returns[mask])) * 100, 4),
                        "positive_rate": round(float((returns[mask] > 0).mean()) * 100, 2)
                    }
        
        # Best/worst periods
        best_day = max(daily_returns.items(), key=lambda x: x[1]['mean_return'])[0] if daily_returns else "N/A"
        worst_day = min(daily_returns.items(), key=lambda x: x[1]['mean_return'])[0] if daily_returns else "N/A"
        
        return {
            "success": True,
            "daily_patterns": daily_returns,
            "monthly_patterns": monthly_returns,
            "best_day": best_day,
            "worst_day": worst_day,
            "data_points": len(data)
        }
