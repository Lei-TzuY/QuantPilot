"""
Transformer-based Time Series Model
Uses attention mechanism for stock price prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import os
import math

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module if TORCH_AVAILABLE else object):
    """Transformer encoder for time series prediction."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use mean pooling over sequence
        x = x.mean(dim=1)
        return self.output_layer(x)


class TransformerPredictor:
    """
    Transformer-based stock price predictor.
    Uses self-attention for capturing long-range dependencies.
    """
    
    def __init__(self, seq_length: int = 30, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.model = None
        self.scaler_params = {}
        self.feature_columns = []
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features."""
        data = df.copy()
        
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for w in [5, 10, 20]:
            data[f'ma_ratio_{w}'] = data['close'] / data['close'].rolling(w).mean()
        
        # Volatility
        data['vol_5'] = data['returns'].rolling(5).std()
        data['vol_20'] = data['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        data['rsi_norm'] = data['rsi'] / 100
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = (ema12 - ema26) / data['close']
        
        # Volume
        if 'volume' in data.columns:
            data['vol_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Price range
        data['range'] = (data['high'] - data['low']) / data['close']
        
        return data.dropna()
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences."""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(target[i + self.seq_length])
        return np.array(X), np.array(y)
    
    def _normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.scaler_params['mean'] = np.mean(data, axis=0)
            self.scaler_params['std'] = np.std(data, axis=0) + 1e-10
        return (data - self.scaler_params['mean']) / self.scaler_params['std']
    
    def train(self, df: pd.DataFrame, epochs: int = 50, lr: float = 0.001,
              target_days: int = 5, validation_split: float = 0.2) -> Dict:
        """Train transformer model."""
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        data = self._prepare_features(df)
        
        self.feature_columns = [
            'returns', 'log_returns', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
            'vol_5', 'vol_20', 'rsi_norm', 'macd', 'range'
        ]
        if 'vol_ratio' in data.columns:
            self.feature_columns.append('vol_ratio')
        
        data['target'] = data['close'].pct_change(target_days).shift(-target_days)
        data = data.dropna()
        
        features = data[self.feature_columns].values
        target = data['target'].values
        
        features = self._normalize(features, fit=True)
        X, y = self._create_sequences(features, target)
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        
        self.model = TransformerModel(
            input_size=len(self.feature_columns),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val)
                val_loss = criterion(val_out, y_val).item()
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {history['train_loss'][-1]:.6f}, Val: {val_loss:.6f}")
        
        return {
            "success": True,
            "epochs": epochs,
            "final_train_loss": history['train_loss'][-1],
            "final_val_loss": history['val_loss'][-1],
            "model_type": "transformer"
        }
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction."""
        if not TORCH_AVAILABLE or self.model is None:
            return {"success": False, "error": "Model not ready"}
        
        data = self._prepare_features(df)
        if len(data) < self.seq_length:
            return {"success": False, "error": f"Need at least {self.seq_length} data points"}
        
        features = data[self.feature_columns].values[-self.seq_length:]
        features = self._normalize(features)
        
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            pred = self.model(X).item()
        
        return {
            "success": True,
            "predicted_return": round(pred * 100, 4),
            "direction": "UP" if pred > 0 else "DOWN",
            "signal": "BUY" if pred > 0.005 else ("SELL" if pred < -0.005 else "HOLD"),
            "confidence": round(min(abs(pred) * 20, 1.0), 4)
        }
    
    def save(self, symbol: str):
        """Save model."""
        if self.model is None:
            return False
        path = os.path.join(self.model_dir, f"transformer_{symbol}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler_params': self.scaler_params,
            'feature_columns': self.feature_columns,
            'config': {
                'seq_length': self.seq_length,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers
            }
        }, path)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load model."""
        path = os.path.join(self.model_dir, f"transformer_{symbol}.pt")
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.scaler_params = checkpoint['scaler_params']
        self.feature_columns = checkpoint['feature_columns']
        config = checkpoint['config']
        self.seq_length = config['seq_length']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        
        self.model = TransformerModel(
            input_size=len(self.feature_columns),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        return True
