"""
Trading Environment for Reinforcement Learning
Implements a Gym-compatible environment for stock trading.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from enum import IntEnum

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


class Actions(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnv:
    """
    Stock trading environment for reinforcement learning.
    Compatible with Gym interface but doesn't require gym import.
    """
    
    def __init__(
        self,
        df,
        initial_balance: float = 1_000_000,
        transaction_fee: float = 0.001425,
        tax_rate: float = 0.003,
        window_size: int = 20,
        reward_scaling: float = 1e-4
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Starting cash amount
            transaction_fee: Commission rate
            tax_rate: Transaction tax rate (sell only)
            window_size: Observation window size
            reward_scaling: Scale factor for rewards
        """
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.tax_rate = tax_rate
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # State variables
        self.balance = initial_balance
        self.shares = 0
        self.current_step = window_size
        self.done = False
        self.total_reward = 0
        
        # Precompute features
        self._precompute_features()
        
        # Action and observation spaces
        self.action_space_n = 3  # Hold, Buy, Sell
        self.observation_shape = (window_size, self.features.shape[1] + 2)  # +2 for position info
    
    def _precompute_features(self):
        """Precompute technical features for observations."""
        df = self.df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(5).mean() / df['close']
        df['ma_20'] = df['close'].rolling(20).mean() / df['close']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df['rsi'] = df['rsi'] / 100  # Normalize to 0-1
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / df['close']
        
        # Volume
        if 'volume' in df.columns:
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        else:
            df['volume_ratio'] = 1.0
        
        # Price ratios
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['close_open_ratio'] = (df['close'] - df['open']) / df['close']
        
        # Fill NaN
        df = df.fillna(0)
        
        feature_cols = ['returns', 'log_returns', 'ma_5', 'ma_20', 'volatility', 
                        'rsi', 'macd', 'volume_ratio', 'high_low_ratio', 'close_open_ratio']
        
        self.features = df[feature_cols].values
        self.prices = df['close'].values
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.window_size
        self.done = False
        self.total_reward = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get feature window
        start = self.current_step - self.window_size
        end = self.current_step
        features = self.features[start:end]
        
        # Add position information
        current_price = self.prices[self.current_step - 1]
        portfolio_value = self.balance + self.shares * current_price
        
        position_info = np.zeros((self.window_size, 2))
        position_info[:, 0] = self.shares / 1000  # Normalized shares
        position_info[:, 1] = (portfolio_value / self.initial_balance) - 1  # Return
        
        observation = np.concatenate([features, position_info], axis=1)
        return observation.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            
        Returns:
            observation, reward, done, info
        """
        current_price = self.prices[self.current_step]
        prev_portfolio_value = self.balance + self.shares * current_price
        
        # Execute action
        trade_info = {"action": action, "price": current_price}
        
        if action == Actions.BUY and self.balance > 0:
            # Buy with all available balance
            max_shares = int(self.balance / (current_price * (1 + self.transaction_fee)))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.shares += max_shares
                trade_info["shares_bought"] = max_shares
        
        elif action == Actions.SELL and self.shares > 0:
            # Sell all shares
            proceeds = self.shares * current_price * (1 - self.transaction_fee - self.tax_rate)
            self.balance += proceeds
            trade_info["shares_sold"] = self.shares
            self.shares = 0
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.prices) - 1:
            self.done = True
        
        # Calculate reward
        new_price = self.prices[self.current_step] if not self.done else current_price
        new_portfolio_value = self.balance + self.shares * new_price
        
        # Reward = change in portfolio value
        reward = (new_portfolio_value - prev_portfolio_value) * self.reward_scaling
        self.total_reward += reward
        
        # Info
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "shares": self.shares,
            "portfolio_value": new_portfolio_value,
            "trade": trade_info
        }
        
        observation = self._get_observation() if not self.done else np.zeros(self.observation_shape)
        
        return observation, reward, self.done, info
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        current_price = self.prices[min(self.current_step, len(self.prices) - 1)]
        return self.balance + self.shares * current_price
    
    def get_return(self) -> float:
        """Get current return percentage."""
        return (self.get_portfolio_value() / self.initial_balance - 1) * 100


# Gym-compatible wrapper
if GYM_AVAILABLE:
    class GymTradingEnv(gym.Env):
        """Gym-compatible trading environment."""
        
        metadata = {'render.modes': ['human']}
        
        def __init__(self, df, **kwargs):
            super().__init__()
            self.env = TradingEnv(df, **kwargs)
            
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=self.env.observation_shape,
                dtype=np.float32
            )
        
        def reset(self):
            return self.env.reset()
        
        def step(self, action):
            return self.env.step(action)
        
        def render(self, mode='human'):
            print(f"Step: {self.env.current_step}, Portfolio: {self.env.get_portfolio_value():.2f}, Return: {self.env.get_return():.2f}%")
