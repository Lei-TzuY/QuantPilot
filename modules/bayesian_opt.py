"""
Bayesian Hyperparameter Optimization Module
Efficient hyperparameter tuning using Bayesian optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GaussianProcess:
    """Simple Gaussian Process for Bayesian Optimization."""
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel."""
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / (self.length_scale ** 2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to data."""
        self.X_train = X
        self.y_train = y
        
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance."""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        K_star = self._kernel(X, self.X_train)
        K_star_star = self._kernel(X, X)
        
        mean = K_star @ self.K_inv @ self.y_train
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        var = np.clip(var, 1e-10, None)
        
        return mean, var


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    Uses Gaussian Process with Expected Improvement acquisition.
    """
    
    def __init__(self, param_space: Dict[str, Tuple[float, float]]):
        """
        Initialize optimizer.
        
        Args:
            param_space: Dict of {param_name: (min_value, max_value)}
        """
        self.param_space = param_space
        self.param_names = list(param_space.keys())
        self.bounds = np.array([param_space[k] for k in self.param_names])
        
        self.gp = GaussianProcess()
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
    
    def _to_unit_cube(self, X: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _from_unit_cube(self, X: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1]."""
        return X * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement."""
        if len(self.X_observed) == 0:
            return np.ones(len(X))
        
        X_norm = self._to_unit_cube(X)
        mean, var = self.gp.predict(X_norm)
        std = np.sqrt(var)
        
        best_y = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            imp = mean - best_y - xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std < 1e-10] = 0.0
        
        return ei
    
    def suggest(self) -> Dict[str, float]:
        """Suggest next point to evaluate."""
        if len(self.X_observed) < 5:
            # Random exploration for first few points
            point = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        else:
            # Optimize acquisition function
            best_ei = -np.inf
            best_point = None
            
            # Multi-start optimization
            for _ in range(20):
                x0 = np.random.uniform(0, 1, len(self.param_names))
                
                def neg_ei(x):
                    return -self._expected_improvement(self._from_unit_cube(x.reshape(1, -1)))[0]
                
                result = minimize(neg_ei, x0, method='L-BFGS-B',
                                  bounds=[(0, 1)] * len(self.param_names))
                
                if -result.fun > best_ei:
                    best_ei = -result.fun
                    best_point = result.x
            
            point = self._from_unit_cube(best_point)
        
        return {name: float(point[i]) for i, name in enumerate(self.param_names)}
    
    def register(self, params: Dict[str, float], score: float):
        """Register an observation."""
        point = np.array([params[name] for name in self.param_names])
        
        self.X_observed.append(point)
        self.y_observed.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
        
        # Refit GP
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        X_norm = self._to_unit_cube(X)
        self.gp.fit(X_norm, y)
    
    def optimize(self, objective_func: Callable, n_iterations: int = 20,
                 verbose: bool = True) -> Dict:
        """
        Run optimization.
        
        Args:
            objective_func: Function that takes params dict and returns score
            n_iterations: Number of iterations
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        history = []
        
        for i in range(n_iterations):
            params = self.suggest()
            
            try:
                score = objective_func(params)
            except Exception as e:
                score = -np.inf
                if verbose:
                    print(f"Iteration {i+1}: Error - {e}")
                continue
            
            self.register(params, score)
            
            history.append({
                "iteration": i + 1,
                "params": params,
                "score": score,
                "best_score": self.best_score
            })
            
            if verbose and (i + 1) % 5 == 0:
                print(f"Iteration {i+1}/{n_iterations} - Score: {score:.4f}, Best: {self.best_score:.4f}")
        
        return {
            "success": True,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_iterations": n_iterations,
            "history": history
        }
    
    def get_importance(self) -> Dict[str, float]:
        """Estimate parameter importance."""
        if len(self.X_observed) < 10:
            return {name: 1.0 / len(self.param_names) for name in self.param_names}
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Correlation-based importance
        importances = {}
        for i, name in enumerate(self.param_names):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            importances[name] = float(corr) if not np.isnan(corr) else 0
        
        # Normalize
        total = sum(importances.values()) + 1e-10
        return {k: round(v / total, 4) for k, v in importances.items()}


def optimize_strategy(backtester, df, strategy: str, param_ranges: Dict,
                      n_iterations: int = 20, metric: str = "sharpe_ratio") -> Dict:
    """
    Optimize trading strategy parameters using Bayesian optimization.
    
    Args:
        backtester: Backtester instance
        df: DataFrame with stock data
        strategy: Strategy name
        param_ranges: Dict of {param: (min, max)}
        n_iterations: Number of optimization iterations
        metric: Metric to optimize ("sharpe_ratio", "return_pct", etc.)
        
    Returns:
        Optimization results
    """
    # Convert int params
    int_params = {'short_window', 'long_window', 'rsi_period', 'rsi_lower', 'rsi_upper'}
    
    def objective(params):
        # Convert to int where needed
        clean_params = {}
        for k, v in params.items():
            if k in int_params:
                clean_params[k] = int(round(v))
            else:
                clean_params[k] = v
        
        result = backtester.run(df, strategy, params=clean_params)
        
        if metric == "sharpe_ratio":
            return result.get("sharpe_ratio", -np.inf)
        elif metric == "return_pct":
            return result.get("return_pct", -np.inf)
        elif metric == "profit_factor":
            return result.get("profit_factor", -np.inf)
        else:
            return result.get(metric, -np.inf)
    
    optimizer = BayesianOptimizer(param_ranges)
    result = optimizer.optimize(objective, n_iterations, verbose=False)
    
    # Clean up best params
    if result["best_params"]:
        for k in result["best_params"]:
            if k in int_params:
                result["best_params"][k] = int(round(result["best_params"][k]))
    
    result["param_importance"] = optimizer.get_importance()
    return result
