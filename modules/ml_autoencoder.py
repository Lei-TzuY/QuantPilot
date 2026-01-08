"""
Autoencoder for Anomaly Detection
Detects unusual market behavior and potential trading opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AutoencoderModel(nn.Module if TORCH_AVAILABLE else object):
    """Autoencoder neural network for anomaly detection."""
    
    def __init__(self, input_size: int, encoding_dim: int = 8):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AnomalyDetector:
    """
    Autoencoder-based anomaly detector for market data.
    High reconstruction error indicates unusual market behavior.
    """
    
    def __init__(self, encoding_dim: int = 8):
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler_params = {}
        self.feature_columns = []
        self.threshold = None  # Anomaly threshold
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        data = df.copy()
        
        # Returns and volatility
        data['returns'] = data['close'].pct_change()
        data['returns_5d'] = data['close'].pct_change(5)
        data['volatility'] = data['returns'].rolling(10).std()
        
        # Volume changes
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
            data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Price range
        data['range'] = (data['high'] - data['low']) / data['close']
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # RSI deviation from mean
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))
        data['rsi_deviation'] = (rsi - 50) / 50
        
        # Bollinger Band width
        bb_mid = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_width'] = (4 * bb_std) / bb_mid
        
        # Price momentum
        data['momentum'] = data['close'] / data['close'].shift(10) - 1
        
        return data.dropna()
    
    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.scaler_params['mean'] = np.mean(data, axis=0)
            self.scaler_params['std'] = np.std(data, axis=0) + 1e-10
        return (data - self.scaler_params['mean']) / self.scaler_params['std']
    
    def train(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001,
              anomaly_percentile: float = 95) -> Dict:
        """
        Train autoencoder on normal market data.
        
        Args:
            df: OHLCV DataFrame
            epochs: Training epochs
            lr: Learning rate
            anomaly_percentile: Percentile for anomaly threshold
            
        Returns:
            Training results
        """
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        data = self._prepare_features(df)
        
        self.feature_columns = [c for c in data.columns 
                                if c not in ['open', 'high', 'low', 'close', 'volume', 'adj close']]
        
        X = data[self.feature_columns].values
        X = self._normalize(X, fit=True)
        
        X_tensor = torch.FloatTensor(X)
        loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=32, shuffle=True)
        
        input_size = len(self.feature_columns)
        self.model = AutoencoderModel(input_size, self.encoding_dim)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = []
            
            for batch_X, _ in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_X)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            
            history['loss'].append(np.mean(epoch_loss))
        
        # Calculate threshold from training data
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        self.threshold = np.percentile(errors, anomaly_percentile)
        
        return {
            "success": True,
            "epochs": epochs,
            "final_loss": history['loss'][-1],
            "threshold": round(float(self.threshold), 6),
            "n_features": input_size
        }
    
    def detect(self, df: pd.DataFrame, return_scores: bool = False) -> Dict:
        """
        Detect anomalies in market data.
        
        Args:
            df: OHLCV DataFrame
            return_scores: Whether to return all anomaly scores
            
        Returns:
            Detection results
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {"success": False, "error": "Model not ready"}
        
        data = self._prepare_features(df)
        X = data[self.feature_columns].values
        X = self._normalize(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # Identify anomalies
        is_anomaly = errors > self.threshold
        anomaly_indices = np.where(is_anomaly)[0]
        
        # Get anomaly dates
        anomaly_dates = data.index[anomaly_indices].tolist()
        anomaly_dates = [str(d)[:10] for d in anomaly_dates]
        
        # Latest status
        latest_error = errors[-1]
        latest_is_anomaly = bool(is_anomaly[-1])
        
        result = {
            "success": True,
            "latest_anomaly_score": round(float(latest_error), 6),
            "threshold": round(float(self.threshold), 6),
            "is_anomaly": latest_is_anomaly,
            "anomaly_level": self._get_anomaly_level(latest_error),
            "recent_anomalies": anomaly_dates[-10:],
            "total_anomalies": int(is_anomaly.sum()),
            "anomaly_rate": round(float(is_anomaly.mean()) * 100, 2)
        }
        
        if return_scores:
            result["all_scores"] = errors.tolist()
        
        return result
    
    def _get_anomaly_level(self, error: float) -> str:
        """Get anomaly severity level."""
        if self.threshold is None:
            return "unknown"
        
        ratio = error / self.threshold
        
        if ratio < 0.5:
            return "normal"
        elif ratio < 0.8:
            return "low"
        elif ratio < 1.0:
            return "moderate"
        elif ratio < 1.5:
            return "high"
        else:
            return "extreme"
    
    def get_market_state(self, df: pd.DataFrame) -> Dict:
        """
        Get current market state analysis.
        
        Returns:
            Market state with trading implications
        """
        detection = self.detect(df)
        
        if not detection.get("success"):
            return detection
        
        level = detection["anomaly_level"]
        is_anomaly = detection["is_anomaly"]
        
        # Trading interpretation
        if level == "extreme":
            interpretation = "Extreme market conditions. High risk. Consider reducing exposure."
            signal = "CAUTION"
        elif level == "high":
            interpretation = "Unusual market behavior. Potential opportunity or risk."
            signal = "ALERT"
        elif level in ["moderate", "low"]:
            interpretation = "Slightly unusual conditions. Monitor closely."
            signal = "WATCH"
        else:
            interpretation = "Normal market conditions."
            signal = "NORMAL"
        
        return {
            "success": True,
            "market_state": level,
            "signal": signal,
            "interpretation": interpretation,
            "anomaly_score": detection["latest_anomaly_score"],
            "threshold": detection["threshold"],
            "recent_anomaly_count": len(detection.get("recent_anomalies", []))
        }
    
    def save(self, symbol: str):
        """Save model."""
        if self.model is None:
            return False
        path = os.path.join(self.model_dir, f"autoencoder_{symbol}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_params': self.scaler_params,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'encoding_dim': self.encoding_dim
        }, path)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load model."""
        path = os.path.join(self.model_dir, f"autoencoder_{symbol}.pt")
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.scaler_params = checkpoint['scaler_params']
        self.feature_columns = checkpoint['feature_columns']
        self.threshold = checkpoint['threshold']
        self.encoding_dim = checkpoint['encoding_dim']
        
        self.model = AutoencoderModel(len(self.feature_columns), self.encoding_dim)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        return True
