"""
XGBoost/LightGBM Gradient Boosting Module
High-performance gradient boosting for stock prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import pickle

# Try to import gradient boosting libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class GradientBoostPredictor:
    """
    Gradient Boosting predictor using XGBoost or LightGBM.
    Provides classification and regression for stock prediction.
    """
    
    def __init__(self, model_type: str = "auto", task: str = "classification"):
        """
        Initialize gradient boosting predictor.
        
        Args:
            model_type: "xgboost", "lightgbm", or "auto"
            task: "classification" (up/down) or "regression" (returns)
        """
        self.task = task
        
        if model_type == "auto":
            if XGB_AVAILABLE:
                self.model_type = "xgboost"
            elif LGB_AVAILABLE:
                self.model_type = "lightgbm"
            else:
                self.model_type = None
        else:
            self.model_type = model_type
        
        self.model = None
        self.feature_columns = []
        self.feature_importance = {}
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training."""
        data = df.copy()
        
        # Price-based features
        data['returns_1d'] = data['close'].pct_change()
        data['returns_5d'] = data['close'].pct_change(5)
        data['returns_10d'] = data['close'].pct_change(10)
        data['returns_20d'] = data['close'].pct_change(20)
        
        # Log returns
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving average ratios
        for window in [5, 10, 20, 50]:
            ma = data['close'].rolling(window).mean()
            data[f'ma_{window}_ratio'] = data['close'] / ma
            data[f'ma_{window}_slope'] = ma.pct_change(5)
        
        # Volatility features
        data['volatility_5d'] = data['returns_1d'].rolling(5).std()
        data['volatility_20d'] = data['returns_1d'].rolling(20).std()
        data['volatility_ratio'] = data['volatility_5d'] / (data['volatility_20d'] + 1e-10)
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = (ema12 - ema26) / data['close']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        bb_mid = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = (bb_mid + 2 * bb_std - data['close']) / data['close']
        data['bb_lower'] = (data['close'] - (bb_mid - 2 * bb_std)) / data['close']
        data['bb_width'] = (4 * bb_std) / bb_mid
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_ma20'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / (data['volume_ma20'] + 1)
            data['price_volume_corr'] = data['close'].rolling(20).corr(data['volume'])
        
        # ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['atr'] = tr.rolling(14).mean() / data['close']
        
        # Momentum indicators
        data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        
        # Day of week (cyclical encoding)
        if hasattr(data.index, 'dayofweek'):
            data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 5)
            data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 5)
        
        return data.dropna()
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns."""
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 
                   'volume_ma20', 'adj close', 'Adj Close']
        return [c for c in df.columns if c.lower() not in [x.lower() for x in exclude] 
                and not c.startswith('target')]
    
    def train(self, df: pd.DataFrame, target_days: int = 5, 
              test_size: float = 0.2, **kwargs) -> Dict:
        """
        Train gradient boosting model.
        
        Args:
            df: OHLCV DataFrame
            target_days: Days ahead for prediction
            test_size: Fraction of data for testing
            **kwargs: Additional model parameters
            
        Returns:
            Training results
        """
        if self.model_type is None:
            return {"success": False, "error": "No gradient boosting library available"}
        
        # Prepare data
        data = self._prepare_features(df)
        
        # Create target
        if self.task == "classification":
            data['target'] = (data['close'].shift(-target_days) > data['close']).astype(int)
        else:
            data['target'] = data['close'].pct_change(target_days).shift(-target_days)
        
        data = data.dropna()
        
        if len(data) < 100:
            return {"success": False, "error": "Insufficient data for training"}
        
        # Get features
        self.feature_columns = self._get_feature_columns(data)
        X = data[self.feature_columns].values
        y = data['target'].values
        
        # Train/test split (temporal)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Default parameters
        params = {
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 6),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'random_state': 42
        }
        
        # Train model
        if self.model_type == "xgboost" and XGB_AVAILABLE:
            if self.task == "classification":
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
        elif self.model_type == "lightgbm" and LGB_AVAILABLE:
            params['verbose'] = -1
            if self.task == "classification":
                self.model = lgb.LGBMClassifier(**params)
            else:
                self.model = lgb.LGBMRegressor(**params)
        else:
            return {"success": False, "error": f"{self.model_type} not available"}
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance = dict(sorted(
            zip(self.feature_columns, importances),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Predictions for metrics
        if self.task == "classification":
            y_pred = self.model.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            
            # Direction accuracy
            return {
                "success": True,
                "model_type": self.model_type,
                "task": self.task,
                "train_accuracy": round(train_score * 100, 2),
                "test_accuracy": round(test_score * 100, 2),
                "direction_accuracy": round(accuracy * 100, 2),
                "n_features": len(self.feature_columns),
                "top_features": dict(list(self.feature_importance.items())[:10])
            }
        else:
            y_pred = self.model.predict(X_test)
            mse = np.mean((y_pred - y_test) ** 2)
            
            return {
                "success": True,
                "model_type": self.model_type,
                "task": self.task,
                "train_r2": round(train_score, 4),
                "test_r2": round(test_score, 4),
                "mse": round(mse, 6),
                "n_features": len(self.feature_columns),
                "top_features": dict(list(self.feature_importance.items())[:10])
            }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction on latest data."""
        if self.model is None:
            return {"success": False, "error": "Model not trained"}
        
        data = self._prepare_features(df)
        
        if len(data) < 1:
            return {"success": False, "error": "Insufficient data"}
        
        # Get latest features
        X = data[self.feature_columns].iloc[-1:].values
        
        if self.task == "classification":
            prob = self.model.predict_proba(X)[0]
            pred_class = self.model.predict(X)[0]
            
            direction = "UP" if pred_class == 1 else "DOWN"
            confidence = max(prob)
            
            return {
                "success": True,
                "direction": direction,
                "confidence": round(float(confidence), 4),
                "probabilities": {"down": round(float(prob[0]), 4), "up": round(float(prob[1]), 4)},
                "signal": "BUY" if direction == "UP" and confidence > 0.6 else (
                    "SELL" if direction == "DOWN" and confidence > 0.6 else "HOLD"
                )
            }
        else:
            pred_return = self.model.predict(X)[0]
            
            return {
                "success": True,
                "predicted_return": round(float(pred_return) * 100, 4),
                "direction": "UP" if pred_return > 0 else "DOWN",
                "signal": "BUY" if pred_return > 0.01 else (
                    "SELL" if pred_return < -0.01 else "HOLD"
                )
            }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance ranking."""
        return {
            "importance": self.feature_importance,
            "top_10": dict(list(self.feature_importance.items())[:10])
        }
    
    def save(self, symbol: str):
        """Save model to disk."""
        if self.model is None:
            return False
        
        path = os.path.join(self.model_dir, f"gb_{symbol}.pkl")
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'task': self.task,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance
            }, f)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load model from disk."""
        path = os.path.join(self.model_dir, f"gb_{symbol}.pkl")
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data['model_type']
            self.task = data['task']
            self.feature_columns = data['feature_columns']
            self.feature_importance = data['feature_importance']
        return True
