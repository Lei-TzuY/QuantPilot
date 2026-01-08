"""
GAN for Synthetic Financial Data Generation
Generates realistic synthetic market data for training and augmentation.
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


class Generator(nn.Module if TORCH_AVAILABLE else object):
    """Generator network for GAN."""
    
    def __init__(self, latent_dim: int, output_dim: int, seq_length: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        self.seq_length = seq_length
        self.output_dim = output_dim
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, seq_length * output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.network(z)
        return out.view(-1, self.seq_length, self.output_dim)


class Discriminator(nn.Module if TORCH_AVAILABLE else object):
    """Discriminator network for GAN."""
    
    def __init__(self, input_dim: int, seq_length: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class FinancialGAN:
    """
    GAN for generating synthetic financial time series.
    Useful for data augmentation and scenario simulation.
    """
    
    def __init__(self, latent_dim: int = 32, seq_length: int = 20):
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.generator = None
        self.discriminator = None
        self.feature_dim = 0
        self.scaler_params = {}
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _prepare_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare training sequences."""
        data = df.copy()
        
        # Use returns-based features (stationary)
        data['returns'] = data['close'].pct_change()
        data['high_low'] = (data['high'] - data['low']) / data['close']
        data['close_open'] = (data['close'] - data['open']) / data['close']
        
        if 'volume' in data.columns:
            data['volume_change'] = data['volume'].pct_change()
        else:
            data['volume_change'] = 0
        
        data = data.dropna()
        
        features = data[['returns', 'high_low', 'close_open', 'volume_change']].values
        self.feature_dim = features.shape[1]
        
        # Normalize
        self.scaler_params['mean'] = np.mean(features, axis=0)
        self.scaler_params['std'] = np.std(features, axis=0) + 1e-10
        features = (features - self.scaler_params['mean']) / self.scaler_params['std']
        
        # Create sequences
        sequences = []
        for i in range(len(features) - self.seq_length):
            sequences.append(features[i:i + self.seq_length])
        
        return np.array(sequences)
    
    def train(self, df: pd.DataFrame, epochs: int = 100, lr: float = 0.0002) -> Dict:
        """Train the GAN."""
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        sequences = self._prepare_sequences(df)
        
        if len(sequences) < 50:
            return {"success": False, "error": "Insufficient data for training"}
        
        # Initialize models
        self.generator = Generator(self.latent_dim, self.feature_dim, self.seq_length)
        self.discriminator = Discriminator(self.feature_dim, self.seq_length)
        
        # Optimizers
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        # DataLoader
        real_data = torch.FloatTensor(sequences)
        loader = DataLoader(TensorDataset(real_data), batch_size=32, shuffle=True)
        
        history = {'g_loss': [], 'd_loss': []}
        
        for epoch in range(epochs):
            g_losses, d_losses = [], []
            
            for batch in loader:
                real_batch = batch[0]
                batch_size = real_batch.size(0)
                
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                real_output = self.discriminator(real_batch)
                d_real_loss = criterion(real_output, real_labels)
                
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data.detach())
                d_fake_loss = criterion(fake_output, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator(z)
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            history['g_loss'].append(np.mean(g_losses))
            history['d_loss'].append(np.mean(d_losses))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - G Loss: {history['g_loss'][-1]:.4f}, D Loss: {history['d_loss'][-1]:.4f}")
        
        return {
            "success": True,
            "epochs": epochs,
            "final_g_loss": history['g_loss'][-1],
            "final_d_loss": history['d_loss'][-1],
            "seq_length": self.seq_length,
            "feature_dim": self.feature_dim
        }
    
    def generate(self, n_samples: int = 100, base_price: float = 100.0) -> Dict:
        """
        Generate synthetic price series.
        
        Args:
            n_samples: Number of sequences to generate
            base_price: Starting price for reconstruction
            
        Returns:
            Generated data
        """
        if self.generator is None:
            return {"success": False, "error": "Model not trained"}
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim)
            fake_sequences = self.generator(z).numpy()
        
        # Denormalize
        fake_sequences = fake_sequences * self.scaler_params['std'] + self.scaler_params['mean']
        
        # Reconstruct prices from returns
        all_prices = []
        for seq in fake_sequences:
            prices = [base_price]
            for i in range(len(seq)):
                new_price = prices[-1] * (1 + seq[i, 0])  # returns column
                prices.append(new_price)
            all_prices.append(prices[1:])
        
        return {
            "success": True,
            "n_samples": n_samples,
            "seq_length": self.seq_length,
            "sample_prices": all_prices[:5],  # Return first 5 samples
            "statistics": {
                "mean_return": round(float(np.mean(fake_sequences[:, :, 0])) * 100, 4),
                "std_return": round(float(np.std(fake_sequences[:, :, 0])) * 100, 4)
            }
        }
    
    def augment_data(self, df: pd.DataFrame, augment_ratio: float = 0.5) -> pd.DataFrame:
        """
        Augment dataset with synthetic data.
        
        Args:
            df: Original DataFrame
            augment_ratio: Ratio of synthetic data to add
            
        Returns:
            Augmented DataFrame
        """
        if self.generator is None:
            return df
        
        n_synthetic = int(len(df) * augment_ratio / self.seq_length)
        generation = self.generate(n_synthetic, df['close'].iloc[0])
        
        if not generation.get("success"):
            return df
        
        # This is a simplified version - in practice you'd build full OHLCV
        return df  # Return original for safety
    
    def save(self, symbol: str):
        """Save models."""
        if self.generator is None:
            return False
        path = os.path.join(self.model_dir, f"gan_{symbol}.pt")
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'scaler_params': self.scaler_params,
            'latent_dim': self.latent_dim,
            'seq_length': self.seq_length,
            'feature_dim': self.feature_dim
        }, path)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load models."""
        path = os.path.join(self.model_dir, f"gan_{symbol}.pt")
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.scaler_params = checkpoint['scaler_params']
        self.latent_dim = checkpoint['latent_dim']
        self.seq_length = checkpoint['seq_length']
        self.feature_dim = checkpoint['feature_dim']
        
        self.generator = Generator(self.latent_dim, self.feature_dim, self.seq_length)
        self.discriminator = Discriminator(self.feature_dim, self.seq_length)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        return True
