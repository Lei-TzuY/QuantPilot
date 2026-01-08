"""
SHAP Model Explainability Module
Provides interpretable explanations for ML model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelExplainer:
    """
    Model explainability using SHAP (SHapley Additive exPlanations).
    Provides feature importance and prediction explanations.
    """
    
    def __init__(self):
        self.explainer = None
        self.expected_value = None
        self.feature_names = []
    
    def create_explainer(self, model, X_background: np.ndarray, 
                         feature_names: List[str], model_type: str = "tree") -> Dict:
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model (sklearn-compatible)
            X_background: Background dataset for SHAP
            feature_names: List of feature names
            model_type: "tree" for tree-based, "kernel" for any model
            
        Returns:
            Status dictionary
        """
        if not SHAP_AVAILABLE:
            return {"success": False, "error": "SHAP not installed. Run: pip install shap"}
        
        self.feature_names = feature_names
        
        try:
            if model_type == "tree":
                # For XGBoost, LightGBM, Random Forest
                self.explainer = shap.TreeExplainer(model)
            else:
                # For any model (slower)
                self.explainer = shap.KernelExplainer(model.predict, X_background[:100])
            
            self.expected_value = self.explainer.expected_value
            
            return {
                "success": True,
                "explainer_type": model_type,
                "n_features": len(feature_names),
                "expected_value": float(self.expected_value) if not isinstance(self.expected_value, np.ndarray) else float(self.expected_value[0])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def explain_prediction(self, X: np.ndarray, top_n: int = 10) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            X: Single sample to explain (1, n_features)
            top_n: Number of top features to return
            
        Returns:
            Explanation dictionary
        """
        if self.explainer is None:
            return {"error": "Explainer not created. Call create_explainer first."}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class or binary
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            values = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Create feature importance dict
            feature_effects = dict(zip(self.feature_names, values))
            
            # Sort by absolute impact
            sorted_effects = sorted(feature_effects.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Separate positive and negative contributors
            positive_contributors = [(f, v) for f, v in sorted_effects if v > 0][:top_n]
            negative_contributors = [(f, v) for f, v in sorted_effects if v < 0][:top_n]
            
            # Calculate prediction contribution
            base_value = float(self.expected_value) if not isinstance(self.expected_value, np.ndarray) else float(self.expected_value[0])
            total_effect = sum(values)
            
            return {
                "base_value": round(base_value, 4),
                "total_effect": round(total_effect, 4),
                "predicted_value": round(base_value + total_effect, 4),
                "top_positive_features": {f: round(v, 4) for f, v in positive_contributors},
                "top_negative_features": {f: round(v, 4) for f, v in negative_contributors},
                "all_effects": {f: round(v, 4) for f, v in sorted_effects}
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_global_importance(self, X: np.ndarray, sample_size: int = 100) -> Dict:
        """
        Get global feature importance across multiple samples.
        
        Args:
            X: Dataset to analyze
            sample_size: Number of samples to use
            
        Returns:
            Global importance dictionary
        """
        if self.explainer is None:
            return {"error": "Explainer not created"}
        
        try:
            # Sample if too large
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            shap_values = self.explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Mean absolute SHAP value per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            importance = dict(zip(self.feature_names, mean_abs_shap))
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            # Normalize to percentages
            total = sum(sorted_importance.values())
            normalized = {f: round(v / total * 100, 2) for f, v in sorted_importance.items()}
            
            return {
                "importance": sorted_importance,
                "importance_pct": normalized,
                "top_10": dict(list(normalized.items())[:10]),
                "samples_analyzed": len(X_sample)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def feature_dependence(self, X: np.ndarray, feature: str, 
                          interaction_feature: str = None) -> Dict:
        """
        Analyze how a feature affects predictions.
        
        Args:
            X: Dataset to analyze
            feature: Main feature to analyze
            interaction_feature: Optional interaction feature
            
        Returns:
            Dependence data
        """
        if self.explainer is None or feature not in self.feature_names:
            return {"error": "Explainer not ready or feature not found"}
        
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            feature_idx = self.feature_names.index(feature)
            feature_values = X[:, feature_idx]
            shap_effects = shap_values[:, feature_idx]
            
            # Create bins for summary
            n_bins = 10
            bins = np.percentile(feature_values, np.linspace(0, 100, n_bins + 1))
            bin_indices = np.digitize(feature_values, bins[1:-1])
            
            bin_summary = {}
            for i in range(n_bins):
                mask = bin_indices == i
                if mask.sum() > 0:
                    bin_summary[f"bin_{i}"] = {
                        "range": f"{bins[i]:.4f} - {bins[i+1]:.4f}",
                        "mean_value": round(float(feature_values[mask].mean()), 4),
                        "mean_shap": round(float(shap_effects[mask].mean()), 4),
                        "count": int(mask.sum())
                    }
            
            # Overall correlation
            correlation = np.corrcoef(feature_values, shap_effects)[0, 1]
            
            return {
                "feature": feature,
                "correlation_with_impact": round(correlation, 4),
                "direction": "positive" if correlation > 0 else "negative",
                "bin_analysis": bin_summary,
                "value_range": {"min": float(feature_values.min()), "max": float(feature_values.max())}
            }
        except Exception as e:
            return {"error": str(e)}


def explain_model_simple(model, X: np.ndarray, feature_names: List[str], 
                         sample_to_explain: np.ndarray = None) -> Dict:
    """
    Simple function to explain any model without creating explainer object.
    Falls back to permutation importance if SHAP unavailable.
    
    Args:
        model: Trained model
        X: Training/background data
        feature_names: Feature names
        sample_to_explain: Optional specific sample to explain
        
    Returns:
        Explanation dictionary
    """
    result = {"method": "unknown", "importance": {}}
    
    # Try tree-based feature importance first
    if hasattr(model, 'feature_importances_'):
        result["method"] = "native_importance"
        importance = dict(zip(feature_names, model.feature_importances_))
        result["importance"] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        result["top_10"] = dict(list(result["importance"].items())[:10])
        return result
    
    # Try SHAP
    if SHAP_AVAILABLE:
        try:
            explainer = ModelExplainer()
            explainer.create_explainer(model, X, feature_names, "tree")
            
            if sample_to_explain is not None:
                result = explainer.explain_prediction(sample_to_explain.reshape(1, -1))
                result["method"] = "shap_local"
            else:
                result = explainer.get_global_importance(X)
                result["method"] = "shap_global"
            return result
        except Exception:
            pass
    
    # Fallback: coefficient-based (for linear models)
    if hasattr(model, 'coef_'):
        result["method"] = "coefficients"
        coefs = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        importance = dict(zip(feature_names, np.abs(coefs)))
        result["importance"] = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return result
    
    result["method"] = "unavailable"
    result["error"] = "Could not explain model. Install SHAP or use tree-based models."
    return result
