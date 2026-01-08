import os
import pickle
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Attempt sklearn import - it may not be installed
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. ML features will be disabled.")

class MLSignalGenerator:
    """
    Machine Learning based signal generator using Random Forest.
    Predicts next-day price direction (UP/DOWN).
    """
    
    MODELS_DIR = "models"
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features from price data."""
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['returns_5d'] = data['close'].pct_change(5)
        data['volatility_10d'] = data['returns'].rolling(10).std()
        
        # Moving Averages
        data['ma_5'] = data['close'].rolling(5).mean()
        data['ma_20'] = data['close'].rolling(20).mean()
        data['ma_ratio'] = data['ma_5'] / data['ma_20']
        
        # RSI
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = data['close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema_12 - ema_26
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma']
        else:
            data['volume_ratio'] = 1.0
        
        # Target: Next day direction (1 = up, 0 = down)
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        return data.dropna()
    
    def train(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Train the ML model on historical data."""
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'scikit-learn not installed'}
        
        data = self._generate_features(df)
        
        self.feature_columns = [
            'returns', 'returns_5d', 'volatility_10d', 
            'ma_ratio', 'rsi', 'macd', 'macd_hist', 'volume_ratio'
        ]
        
        X = data[self.feature_columns]
        y = data['target']
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        model_path = os.path.join(self.MODELS_DIR, f"{symbol}_rf.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'features': self.feature_columns
            }, f)
        
        return {
            'success': True,
            'accuracy': accuracy,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_path': model_path
        }
    
    def predict(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Predict signal for latest data point."""
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'scikit-learn not installed'}
        
        # Load model
        model_path = os.path.join(self.MODELS_DIR, f"{symbol}_rf.pkl")
        if not os.path.exists(model_path):
            return {'success': False, 'error': f'No trained model for {symbol}. Train first.'}
        
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved['model']
            self.feature_columns = saved['features']
        
        # Generate features
        data = self._generate_features(df)
        if len(data) == 0:
            return {'success': False, 'error': 'Insufficient data for prediction'}
        
        # Get latest row
        latest = data[self.feature_columns].iloc[-1:].values
        
        # Predict
        prediction = self.model.predict(latest)[0]
        probabilities = self.model.predict_proba(latest)[0]
        
        signal = 'BUY' if prediction == 1 else 'SELL'
        confidence = max(probabilities) * 100
        
        return {
            'success': True,
            'symbol': symbol,
            'signal': signal,
            'confidence': round(confidence, 2),
            'probabilities': {
                'down': round(probabilities[0] * 100, 2),
                'up': round(probabilities[1] * 100, 2)
            }
        }
    
    def get_feature_importance(self, symbol: str) -> Dict:
        """Get feature importance from trained model."""
        model_path = os.path.join(self.MODELS_DIR, f"{symbol}_rf.pkl")
        if not os.path.exists(model_path):
            return {'success': False, 'error': f'No trained model for {symbol}'}
        
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        
        importances = saved['model'].feature_importances_
        features = saved['features']
        
        importance_dict = {f: round(float(i), 4) for f, i in zip(features, importances)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {'success': True, 'importance': sorted_importance}
