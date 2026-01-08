"""
機器學習特徵工程模組
Feature Engineering Module for ML
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import talib
from ta import momentum, trend, volatility, volume


class FeatureEngineering:
    """特徵工程類 - 為機器學習生成各種特徵"""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有特徵"""
        data = df.copy()
        
        # 確保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 1. 價格特徵
        data = self._add_price_features(data)
        
        # 2. 技術指標特徵
        data = self._add_technical_indicators(data)
        
        # 3. 統計特徵
        data = self._add_statistical_features(data)
        
        # 4. 成交量特徵
        data = self._add_volume_features(data)
        
        # 5. 時間特徵
        data = self._add_time_features(data)
        
        # 6. 高級特徵
        data = self._add_advanced_features(data)
        
        return data
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加價格相關特徵"""
        data = df.copy()
        
        # 收益率
        data['returns_1d'] = data['close'].pct_change(1)
        data['returns_3d'] = data['close'].pct_change(3)
        data['returns_5d'] = data['close'].pct_change(5)
        data['returns_10d'] = data['close'].pct_change(10)
        data['returns_20d'] = data['close'].pct_change(20)
        
        # 對數收益率
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # 價格範圍
        data['price_range'] = data['high'] - data['low']
        data['price_range_pct'] = (data['high'] - data['low']) / data['close']
        
        # 開盤與收盤差異
        data['open_close_diff'] = data['close'] - data['open']
        data['open_close_pct'] = (data['close'] - data['open']) / data['open']
        
        # 上影線和下影線
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        
        # K線實體大小
        data['body_size'] = abs(data['close'] - data['open'])
        data['body_size_pct'] = data['body_size'] / data['close']
        
        return data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術指標特徵"""
        data = df.copy()
        
        # 移動平均線
        for period in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = data['close'].rolling(period).mean()
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        # 移動平均線比率
        data['sma_5_20_ratio'] = data['sma_5'] / data['sma_20']
        data['sma_20_50_ratio'] = data['sma_20'] / data['sma_50']
        data['sma_50_200_ratio'] = data['sma_50'] / data['sma_200']
        
        # RSI (多時間週期)
        for period in [7, 14, 21, 28]:
            delta = data['close'].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # 布林帶
        for period in [10, 20, 30]:
            sma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            data[f'bb_upper_{period}'] = sma + (2 * std)
            data[f'bb_lower_{period}'] = sma - (2 * std)
            data[f'bb_width_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / sma
            data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / (
                data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']
            )
        
        # ATR (平均真實範圍)
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr_14'] = true_range.rolling(14).mean()
        data['atr_20'] = true_range.rolling(20).mean()
        
        # ADX (平均趨向指數)
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        data['adx_14'] = dx.rolling(14).mean()
        
        # CCI (商品通道指數)
        for period in [14, 20]:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
            data[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            data[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 28]:
            high_max = data['high'].rolling(period).max()
            low_min = data['low'].rolling(period).min()
            data[f'williams_r_{period}'] = -100 * (high_max - data['close']) / (high_max - low_min)
        
        return data
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加統計特徵"""
        data = df.copy()
        
        # 波動率 (標準差)
        for period in [5, 10, 20, 30]:
            data[f'volatility_{period}'] = data['returns_1d'].rolling(period).std()
        
        # 偏度 (Skewness)
        for period in [10, 20, 30]:
            data[f'skewness_{period}'] = data['returns_1d'].rolling(period).skew()
        
        # 峰度 (Kurtosis)
        for period in [10, 20, 30]:
            data[f'kurtosis_{period}'] = data['returns_1d'].rolling(period).kurt()
        
        # 價格距離移動平均線
        for period in [5, 10, 20, 50]:
            data[f'distance_sma_{period}'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']
        
        # 動量
        for period in [5, 10, 20]:
            data[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            data[f'momentum_pct_{period}'] = data[f'momentum_{period}'] / data['close'].shift(period)
        
        # ROC (變化率)
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / 
                                     data['close'].shift(period)) * 100
        
        return data
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特徵"""
        data = df.copy()
        
        # 成交量移動平均
        for period in [5, 10, 20, 50]:
            data[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
            data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_sma_{period}']
        
        # 成交量變化率
        data['volume_change'] = data['volume'].pct_change()
        
        # OBV (能量潮)
        data['obv'] = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        data['obv_ema_10'] = data['obv'].ewm(span=10, adjust=False).mean()
        
        # 成交量加權平均價 (VWAP)
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()
        data['vwap_distance'] = (data['close'] - data['vwap']) / data['vwap']
        
        # 價量相關性
        for period in [10, 20]:
            data[f'price_volume_corr_{period}'] = data['close'].rolling(period).corr(data['volume'])
        
        # MFI (資金流量指標)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        
        mfi_ratio = positive_mf / negative_mf
        data['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
        
        return data
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加時間特徵"""
        data = df.copy()
        
        # 確保索引是 DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            return data
        
        # 日期特徵
        data['day_of_week'] = data.index.dayofweek
        data['day_of_month'] = data.index.day
        data['week_of_year'] = data.index.isocalendar().week
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        
        # 是否月初/月末
        data['is_month_start'] = data.index.is_month_start.astype(int)
        data['is_month_end'] = data.index.is_month_end.astype(int)
        data['is_quarter_start'] = data.index.is_quarter_start.astype(int)
        data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        return data
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高級特徵"""
        data = df.copy()
        
        # 趨勢強度
        for period in [10, 20, 30]:
            # 線性回歸斜率
            def linear_slope(series):
                x = np.arange(len(series))
                y = series.values
                if len(x) < 2:
                    return 0
                slope = np.polyfit(x, y, 1)[0]
                return slope
            
            data[f'trend_strength_{period}'] = data['close'].rolling(period).apply(linear_slope, raw=False)
        
        # 支撐和阻力級別
        for period in [20, 50]:
            data[f'resistance_{period}'] = data['high'].rolling(period).max()
            data[f'support_{period}'] = data['low'].rolling(period).min()
            data[f'distance_resistance_{period}'] = (data[f'resistance_{period}'] - data['close']) / data['close']
            data[f'distance_support_{period}'] = (data['close'] - data[f'support_{period}']) / data['close']
        
        # 價格突破
        data['breakout_high_20'] = (data['close'] > data['high'].rolling(20).max().shift(1)).astype(int)
        data['breakout_low_20'] = (data['close'] < data['low'].rolling(20).min().shift(1)).astype(int)
        
        # 缺口 (Gap)
        data['gap_up'] = ((data['low'] > data['high'].shift(1)).astype(int))
        data['gap_down'] = ((data['high'] < data['low'].shift(1)).astype(int))
        data['gap_size'] = data['open'] - data['close'].shift(1)
        
        # 波動性突破
        for period in [10, 20]:
            vol_mean = data['returns_1d'].rolling(period).std()
            vol_std = data['returns_1d'].rolling(period * 2).std()
            data[f'volatility_breakout_{period}'] = (vol_mean > vol_std * 1.5).astype(int)
        
        return data
    
    def get_feature_names(self, exclude_target: bool = True) -> List[str]:
        """獲取所有特徵名稱"""
        if not self.feature_names:
            return []
        
        if exclude_target:
            return [f for f in self.feature_names if f != 'target']
        return self.feature_names
    
    def select_top_features(self, df: pd.DataFrame, target: str, n_features: int = 50) -> List[str]:
        """
        使用特徵重要性選擇最重要的特徵
        
        Args:
            df: 包含所有特徵的DataFrame
            target: 目標變量名稱
            n_features: 要選擇的特徵數量
        
        Returns:
            選擇的特徵名稱列表
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        # 移除 NaN 和 inf
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 準備特徵和目標
        feature_cols = [col for col in df_clean.columns if col != target]
        X = df_clean[feature_cols]
        y = df_clean[target]
        
        # 使用隨機森林獲取特徵重要性
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 獲取特徵重要性
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 選擇前 n 個特徵
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        return top_features
