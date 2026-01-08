"""
Reinforcement Learning Trading Agent
Implements DQN and simple Q-learning for automated trading.
"""

import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from modules.trading_env import TradingEnv


class DQNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, input_shape: Tuple, n_actions: int = 3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        
        # Flatten input: (window_size, features)
        input_size = input_shape[0] * input_shape[1]
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Learning agent for stock trading.
    """
    
    def __init__(
        self,
        state_shape: Tuple,
        n_actions: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.training_step = 0
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Networks
            self.q_network = DQNetwork(state_shape, n_actions).to(self.device)
            self.target_network = DQNetwork(state_shape, n_actions).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
        
        self.model_dir = "data/models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randint(0, self.n_actions - 1)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if not TORCH_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss and update
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, df, episodes: int = 100, verbose: bool = True) -> Dict:
        """
        Train the agent on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            episodes: Number of training episodes
            verbose: Print progress
            
        Returns:
            Training history
        """
        if not TORCH_AVAILABLE:
            return {"success": False, "error": "PyTorch not installed"}
        
        env = TradingEnv(df)
        history = {
            "episode_rewards": [],
            "episode_returns": [],
            "losses": [],
            "epsilons": []
        }
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_losses = []
            
            while not env.done:
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Record history
            final_return = env.get_return()
            history["episode_rewards"].append(episode_reward)
            history["episode_returns"].append(final_return)
            history["losses"].append(np.mean(episode_losses) if episode_losses else 0)
            history["epsilons"].append(self.epsilon)
            
            if verbose and (episode + 1) % 10 == 0:
                avg_return = np.mean(history["episode_returns"][-10:])
                print(f"Episode {episode+1}/{episodes} - Return: {final_return:.2f}%, Avg Return (10): {avg_return:.2f}%, Epsilon: {self.epsilon:.3f}")
        
        return {
            "success": True,
            "episodes": episodes,
            "final_return": history["episode_returns"][-1],
            "best_return": max(history["episode_returns"]),
            "avg_return": np.mean(history["episode_returns"]),
            "history": history
        }
    
    def evaluate(self, df) -> Dict:
        """Evaluate agent on data without training."""
        env = TradingEnv(df)
        state = env.reset()
        
        trades = []
        while not env.done:
            action = self.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            
            if info['trade'].get('shares_bought') or info['trade'].get('shares_sold'):
                trades.append(info['trade'])
        
        return {
            "final_return": env.get_return(),
            "final_value": env.get_portfolio_value(),
            "total_trades": len(trades),
            "trades": trades
        }
    
    def get_action(self, state: np.ndarray) -> Dict:
        """Get trading action for current state."""
        action = self.select_action(state, training=False)
        action_names = ["HOLD", "BUY", "SELL"]
        
        # Get Q-values for confidence
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
            
            # Softmax for probabilities
            exp_q = np.exp(q_values - np.max(q_values))
            probs = exp_q / exp_q.sum()
            
            return {
                "action": action_names[action],
                "action_id": action,
                "confidence": float(probs[action]),
                "q_values": {name: float(q) for name, q in zip(action_names, q_values)}
            }
        else:
            return {"action": action_names[action], "action_id": action}
    
    def save(self, symbol: str):
        """Save model to disk."""
        if not TORCH_AVAILABLE:
            return False
        
        path = os.path.join(self.model_dir, f"dqn_{symbol}.pt")
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'epsilon': self.epsilon,
            'state_shape': self.state_shape
        }, path)
        return True
    
    def load(self, symbol: str) -> bool:
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            return False
        
        path = os.path.join(self.model_dir, f"dqn_{symbol}.pt")
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.epsilon = checkpoint['epsilon']
        return True
