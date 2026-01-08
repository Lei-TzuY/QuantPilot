"""
Multi-Factor Model Module
Implements factor-based quantitative analysis for stock selection and signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats


class MultiFactor:
    """
    Multi-factor model for quantitative stock analysis.
    Computes and combines various alpha factors.
    """
    
    def __init__(self):
        self.factor_weights = {}
        self.factor_stats = {}
    
    def compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all factors for a given stock.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with factor values
        """
        data = df.copy()
        
        # Momentum Factors
        data = self._momentum_factors(data)
        
        # Mean Reversion Factors
        data = self._mean_reversion_factors(data)
        
        # Volatility Factors
        data = self._volatility_factors(data)
        
        # Volume Factors
        data = self._volume_factors(data)
        
        # Technical Factors
        data = self._technical_factors(data)
        
        return data.dropna()
    
    def _momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based factors."""
        # Price momentum (various windows)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        df['momentum_60'] = df['close'].pct_change(60)
        
        # Rate of Change
        df['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Acceleration (momentum of momentum)
        df['acceleration'] = df['momentum_10'] - df['momentum_10'].shift(5)
        
        # Price vs moving averages
        df['price_ma5_ratio'] = df['close'] / df['close'].rolling(5).mean()
        df['price_ma20_ratio'] = df['close'] / df['close'].rolling(20).mean()
        df['price_ma60_ratio'] = df['close'] / df['close'].rolling(60).mean()
        
        # Golden/Death cross signal
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        df['ma_cross'] = (ma5 - ma20) / df['close']
        
        return df
    
    def _mean_reversion_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion factors."""
        # Z-score (price deviation from mean)
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['zscore_20'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
        
        # Bollinger Band position
        bb_upper = rolling_mean + 2 * rolling_std
        bb_lower = rolling_mean - 2 * rolling_std
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # Distance from 52-week high/low
        rolling_252_high = df['high'].rolling(252, min_periods=60).max()
        rolling_252_low = df['low'].rolling(252, min_periods=60).min()
        df['dist_from_high'] = (df['close'] - rolling_252_high) / rolling_252_high
        df['dist_from_low'] = (df['close'] - rolling_252_low) / rolling_252_low
        
        # RSI as mean reversion indicator
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Centered at 0
        
        return df
    
    def _volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based factors."""
        returns = df['close'].pct_change()
        
        # Historical volatility
        df['volatility_5'] = returns.rolling(5).std() * np.sqrt(252)
        df['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['volatility_5'] / (df['volatility_60'] + 1e-10)
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # Parkinson volatility (high-low based)
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(20).mean())
        
        return df
    
    def _volume_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based factors."""
        if 'volume' not in df.columns:
            df['volume_ratio'] = 1.0
            df['obv_slope'] = 0.0
            df['vwap_ratio'] = 1.0
            return df
        
        # Volume ratio
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1)
        
        # On-Balance Volume trend
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_slope'] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Volume-weighted average price ratio
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_ratio'] = df['close'] / (vwap + 1e-10)
        
        # Money Flow Index
        mf = typical_price * df['volume']
        mf_pos = mf.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        mf_neg = mf.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + mf_pos / (mf_neg + 1e-10)))
        
        return df
    
    def _technical_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicator factors."""
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_normalized'] = df['macd'] / df['close']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ADX (trend strength)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-10))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        return df
    
    def calculate_factor_ic(self, df: pd.DataFrame, forward_returns_days: int = 5) -> Dict:
        """
        Calculate Information Coefficient for each factor.
        IC measures correlation between factor values and future returns.
        
        Args:
            df: DataFrame with computed factors
            forward_returns_days: Days ahead for return calculation
            
        Returns:
            Dictionary with IC values for each factor
        """
        data = self.compute_factors(df)
        
        # Calculate forward returns
        data['forward_returns'] = data['close'].pct_change(forward_returns_days).shift(-forward_returns_days)
        data = data.dropna()
        
        if len(data) < 30:
            return {"error": "Insufficient data for IC calculation"}
        
        # Factor columns (exclude price/volume/forward_returns)
        factor_cols = [col for col in data.columns if col not in 
                       ['open', 'high', 'low', 'close', 'volume', 'forward_returns', 
                        'volume_ma20', 'atr_14']]
        
        ic_results = {}
        for col in factor_cols:
            try:
                ic, p_value = stats.spearmanr(data[col], data['forward_returns'])
                ic_results[col] = {
                    "ic": round(ic, 4),
                    "p_value": round(p_value, 4),
                    "significant": p_value < 0.05
                }
            except Exception:
                continue
        
        # Sort by absolute IC
        ic_results = dict(sorted(ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True))
        
        self.factor_stats = ic_results
        return ic_results
    
    def get_factor_signal(self, df: pd.DataFrame, top_factors: int = 5) -> Dict:
        """
        Generate trading signal based on top factors.
        
        Args:
            df: DataFrame with OHLCV data
            top_factors: Number of top factors to use
            
        Returns:
            Aggregate signal and factor breakdown
        """
        data = self.compute_factors(df)
        
        if len(data) < 2:
            return {"error": "Insufficient data"}
        
        # Get latest factor values
        latest = data.iloc[-1]
        
        # Calculate IC if not done
        if not self.factor_stats:
            self.calculate_factor_ic(df)
        
        # Select top factors by IC
        top_factor_names = list(self.factor_stats.keys())[:top_factors]
        
        signals = {}
        weighted_signal = 0
        total_weight = 0
        
        for factor in top_factor_names:
            if factor not in latest.index:
                continue
            
            value = latest[factor]
            ic = self.factor_stats[factor]['ic']
            
            # Normalize factor value to -1 to 1 range (using z-score)
            factor_mean = data[factor].mean()
            factor_std = data[factor].std()
            z_score = (value - factor_mean) / (factor_std + 1e-10)
            normalized = np.clip(z_score / 3, -1, 1)  # Clip at 3 std
            
            # Weight by IC
            weight = abs(ic)
            direction = np.sign(ic)  # Positive IC = higher factor value means higher returns
            
            signal_contribution = normalized * direction * weight
            weighted_signal += signal_contribution
            total_weight += weight
            
            signals[factor] = {
                "value": round(float(value), 4),
                "z_score": round(float(z_score), 4),
                "contribution": round(float(signal_contribution), 4)
            }
        
        # Final signal
        final_signal = weighted_signal / (total_weight + 1e-10)
        
        if final_signal > 0.3:
            action = "STRONG_BUY"
        elif final_signal > 0.1:
            action = "BUY"
        elif final_signal < -0.3:
            action = "STRONG_SELL"
        elif final_signal < -0.1:
            action = "SELL"
        else:
            action = "HOLD"
        
        return {
            "signal": action,
            "score": round(float(final_signal), 4),
            "factors_used": len(signals),
            "factor_breakdown": signals
        }
