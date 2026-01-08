"""
LSTM Time Series Prediction Module
Uses PyTorch for deep learning-based stock price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
import os

# Try to import torch, fall back to numpy-only implementation if not available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified numpy implementation.")


class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """LSTM neural network for sequence prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class LSTMPredictor:
    """
    LSTM-based stock price predictor.
    Predicts future returns based on historical sequences.
    """
    
    def __init__(self, seq_length: int = 20, hidden_size: int = 64, num_layers: int = 2):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler_params = {}  # Store mean/std for normalization
        self.feature_columns = []
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical indicator features."""
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20]:
            data[f'ma_{window}'] = data['close'].rolling(window).mean()
            data[f'ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
        
        # Volatility
        data['volatility_5'] = data['returns'].rolling(5).std()
        data['volatility_20'] = data['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Volume features
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(20).mean()
            data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1)
        
        # Bollinger Bands
        data['bb_mid'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_mid'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_mid'] - 2 * data['bb_std']
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
        
        return data.dropna()
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(target[i + self.seq_length])
        return np.array(X), np.array(y)
    
    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score."""
        if fit:
            self.scaler_params['mean'] = np.mean(data, axis=0)
            self.scaler_params['std'] = np.std(data, axis=0) + 1e-10
        
        return (data - self.scaler_params['mean']) / self.scaler_params['std']
    
    def train(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001, 
              target_days: int = 5, validation_split: float = 0.2) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            df: DataFrame with OHLCV data
            epochs: Number of training epochs
            lr: Learning rate
            target_days: Predict returns over this many days
            validation_split: Fraction of data for validation
            
        Returns:
            Training history with loss values
        """
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        # Prepare data
        data = self._prepare_features(df)
        
        # Select feature columns
        self.feature_columns = [
            'returns', 'log_returns', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
            'volatility_5', 'volatility_20', 'rsi', 'macd', 'macd_signal', 'bb_position'
        ]
        if 'volume_ratio' in data.columns:
            self.feature_columns.append('volume_ratio')
        
        # Create target: forward returns
        data['target'] = data['close'].pct_change(target_days).shift(-target_days)
        data = data.dropna()
        
        features = data[self.feature_columns].values
        target = data['target'].values
        
        # Normalize
        features = self._normalize(features, fit=True)
        
        # Create sequences
        X, y = self._create_sequences(features, target)
        
        # Train/validation split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = LSTMModel(
            input_size=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return {
            "success": True,
            "epochs": epochs,
            "final_train_loss": history['train_loss'][-1],
            "final_val_loss": history['val_loss'][-1],
            "history": history
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions on new data.
        
        Returns:
            Dictionary with prediction, direction, and confidence
        """
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        if self.model is None:
            return {"success": False, "error": "Model not trained"}
        
        # Prepare data
        data = self._prepare_features(df)
        
        if len(data) < self.seq_length:
            return {"success": False, "error": f"Need at least {self.seq_length} data points"}
        
        # Get last sequence
        features = data[self.feature_columns].values[-self.seq_length:]
        features = self._normalize(features)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            prediction = self.model(X).item()
        
        # Determine direction and confidence
        direction = "UP" if prediction > 0 else "DOWN"
        # Confidence based on magnitude of prediction
        confidence = min(abs(prediction) * 10, 1.0)  # Scale to 0-1
        
        return {
            "success": True,
            "predicted_return": round(prediction * 100, 4),  # As percentage
            "direction": direction,
            "confidence": round(confidence, 4),
            "signal": "BUY" if prediction > 0.005 else ("SELL" if prediction < -0.005 else "HOLD")
        }
    
    def save(self, symbol: str):
        """Save model to disk."""
        if self.model is None:
            return False
        
        path = os.path.join(self.model_dir, f"lstm_{symbol}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_params': self.scaler_params,
            'feature_columns': self.feature_columns,
            'seq_length': self.seq_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, path)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load model from disk."""
        path = os.path.join(self.model_dir, f"lstm_{symbol}.pt")
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.scaler_params = checkpoint['scaler_params']
        self.feature_columns = checkpoint['feature_columns']
        self.seq_length = checkpoint['seq_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        
        self.model = LSTMModel(
            input_size=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        return True
