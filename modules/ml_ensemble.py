"""
Ensemble Learning Module
Combines multiple ML models for robust trading signals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class EnsemblePredictor:
    """
    Ensemble model that combines multiple prediction sources.
    Supports voting, weighted averaging, and stacking.
    """
    
    def __init__(self, method: str = "weighted"):
        """
        Initialize ensemble predictor.
        
        Args:
            method: Combination method - "voting", "weighted", or "stacking"
        """
        self.method = method
        self.models = {}
        self.weights = {}
        self.performance_history = {}
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model identifier
            model: Model instance with predict() method
            weight: Initial weight for weighted averaging
        """
        self.models[name] = model
        self.weights[name] = weight
        self.performance_history[name] = []
    
    def predict(self, df, **kwargs) -> Dict:
        """
        Get ensemble prediction from all models.
        
        Args:
            df: Input DataFrame
            **kwargs: Additional arguments passed to models
            
        Returns:
            Ensemble prediction result
        """
        if not self.models:
            return {"error": "No models in ensemble"}
        
        predictions = {}
        signals = []
        confidences = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    result = model.predict(df, **kwargs)
                elif hasattr(model, 'get_action'):
                    # For RL agents
                    # Need to get observation first
                    from modules.trading_env import TradingEnv
                    env = TradingEnv(df)
                    state = env.reset()
                    result = model.get_action(state)
                elif hasattr(model, 'get_factor_signal'):
                    # For multi-factor model
                    result = model.get_factor_signal(df)
                else:
                    continue
                
                if not result.get('success', True):
                    continue
                
                predictions[name] = result
                
                # Normalize signal to numeric value
                signal = self._signal_to_numeric(result)
                signals.append(signal)
                
                # Get confidence
                conf = result.get('confidence', 0.5)
                if isinstance(conf, str):
                    conf = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf, 0.5)
                confidences.append(conf * self.weights[name])
                
            except Exception as e:
                predictions[name] = {"error": str(e)}
        
        if not signals:
            return {"error": "No valid predictions from models"}
        
        # Combine predictions
        if self.method == "voting":
            ensemble_signal = self._voting(signals)
        elif self.method == "weighted":
            ensemble_signal = self._weighted_average(signals, confidences)
        else:
            ensemble_signal = self._weighted_average(signals, confidences)
        
        # Convert to action
        action = self._numeric_to_signal(ensemble_signal)
        
        return {
            "signal": action,
            "score": round(ensemble_signal, 4),
            "confidence": round(np.mean(confidences) if confidences else 0, 4),
            "models_used": len(predictions),
            "model_predictions": predictions,
            "method": self.method
        }
    
    def _signal_to_numeric(self, result: Dict) -> float:
        """Convert signal to numeric value (-1 to 1)."""
        # Try different result formats
        
        # Direct score
        if 'score' in result:
            return np.clip(result['score'], -1, 1)
        
        # Predicted return
        if 'predicted_return' in result:
            # Scale return to signal
            ret = result['predicted_return']
            return np.clip(ret / 5, -1, 1)  # 5% = full signal
        
        # Signal string
        signal = result.get('signal', result.get('action', 'HOLD'))
        signal_map = {
            'STRONG_BUY': 1.0,
            'BUY': 0.5,
            'HOLD': 0.0,
            'SELL': -0.5,
            'STRONG_SELL': -1.0
        }
        return signal_map.get(signal.upper(), 0)
    
    def _numeric_to_signal(self, value: float) -> str:
        """Convert numeric value to signal string."""
        if value > 0.5:
            return "STRONG_BUY"
        elif value > 0.15:
            return "BUY"
        elif value < -0.5:
            return "STRONG_SELL"
        elif value < -0.15:
            return "SELL"
        else:
            return "HOLD"
    
    def _voting(self, signals: List[float]) -> float:
        """Majority voting."""
        # Convert to discrete votes
        votes = [1 if s > 0.1 else (-1 if s < -0.1 else 0) for s in signals]
        
        buy_votes = sum(1 for v in votes if v == 1)
        sell_votes = sum(1 for v in votes if v == -1)
        
        total = len(votes)
        if buy_votes > total / 2:
            return 0.5 * (buy_votes / total)
        elif sell_votes > total / 2:
            return -0.5 * (sell_votes / total)
        else:
            return 0
    
    def _weighted_average(self, signals: List[float], weights: List[float]) -> float:
        """Weighted average of signals."""
        if not weights or sum(weights) == 0:
            return np.mean(signals)
        
        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(signals, weights))
        return weighted_sum / total_weight
    
    def update_weights(self, performance: Dict[str, float]):
        """
        Update model weights based on performance.
        
        Args:
            performance: Dict mapping model name to performance score
        """
        for name, score in performance.items():
            if name in self.weights:
                self.performance_history[name].append(score)
                
                # Update weight based on recent performance
                recent = self.performance_history[name][-10:]  # Last 10
                if recent:
                    avg_performance = np.mean(recent)
                    # Scale weight: better performance = higher weight
                    self.weights[name] = max(0.1, min(2.0, 1 + avg_performance))
    
    def get_model_weights(self) -> Dict:
        """Get current model weights."""
        return {
            name: {
                "weight": round(self.weights[name], 4),
                "avg_performance": round(np.mean(self.performance_history[name][-10:]), 4) 
                    if self.performance_history[name] else None
            }
            for name in self.models.keys()
        }


class QuantEnsemble:
    """
    Pre-configured ensemble for quantitative trading.
    Combines RF, LSTM, DQN, Multi-Factor, and Sentiment.
    """
    
    def __init__(self):
        self.ensemble = EnsemblePredictor(method="weighted")
        self.models_loaded = []
    
    def load_models(self, symbol: str, data_fetcher=None):
        """Load all available models for a symbol."""
        models_info = []
        
        # Random Forest (existing ml_signal)
        try:
            from modules.ml_signal import MLSignalGenerator
            rf_model = MLSignalGenerator()
            if rf_model.load(symbol):
                self.ensemble.add_model("random_forest", rf_model, weight=1.0)
                models_info.append("random_forest")
        except Exception:
            pass
        
        # LSTM
        try:
            from modules.ml_lstm import LSTMPredictor
            lstm_model = LSTMPredictor()
            if lstm_model.load(symbol):
                self.ensemble.add_model("lstm", lstm_model, weight=1.2)
                models_info.append("lstm")
        except Exception:
            pass
        
        # DQN
        try:
            from modules.rl_agent import DQNAgent
            dqn_model = DQNAgent(state_shape=(20, 12))
            if dqn_model.load(symbol):
                self.ensemble.add_model("dqn", dqn_model, weight=0.8)
                models_info.append("dqn")
        except Exception:
            pass
        
        # Multi-Factor (doesn't need loading)
        try:
            from modules.multi_factor import MultiFactor
            factor_model = MultiFactor()
            self.ensemble.add_model("multi_factor", factor_model, weight=1.0)
            models_info.append("multi_factor")
        except Exception:
            pass
        
        # Sentiment (doesn't need loading)
        try:
            from modules.sentiment_analyzer import SentimentAnalyzer
            sentiment_model = SentimentAnalyzer()
            # Note: Sentiment needs news data, handled separately
            models_info.append("sentiment_available")
        except Exception:
            pass
        
        self.models_loaded = models_info
        return models_info
    
    def predict(self, df, news_items: List[Dict] = None) -> Dict:
        """
        Get ensemble prediction.
        
        Args:
            df: OHLCV DataFrame
            news_items: Optional news for sentiment analysis
            
        Returns:
            Ensemble prediction
        """
        result = self.ensemble.predict(df)
        
        # Add sentiment if news provided
        if news_items:
            try:
                from modules.sentiment_analyzer import SentimentAnalyzer
                sentiment = SentimentAnalyzer()
                sentiment_result = sentiment.analyze_news(news_items)
                sentiment_signal = sentiment.get_trading_signal(sentiment_result)
                
                result['sentiment'] = {
                    "score": sentiment_result['overall_score'],
                    "signal": sentiment_signal['signal'],
                    "news_count": sentiment_result['news_count']
                }
                
                # Blend sentiment into final signal
                sentiment_numeric = self.ensemble._signal_to_numeric(sentiment_signal)
                current_score = result.get('score', 0)
                blended_score = current_score * 0.8 + sentiment_numeric * 0.2
                result['score'] = round(blended_score, 4)
                result['signal'] = self.ensemble._numeric_to_signal(blended_score)
                
            except Exception as e:
                result['sentiment_error'] = str(e)
        
        result['models_loaded'] = self.models_loaded
        return result
