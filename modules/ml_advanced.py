"""
高級機器學習管理器
Advanced Machine Learning Manager with Multiple Models
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入機器學習庫
try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    
    from sklearn.model_selection import (
        train_test_split, 
        cross_val_score, 
        GridSearchCV,
        TimeSeriesSplit
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        roc_auc_score,
        confusion_matrix,
        classification_report
    )
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. ML features disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class AdvancedMLManager:
    """
    高級機器學習管理器
    支持多種模型、特徵選擇、超參數優化、模型評估等
    """
    
    MODELS_DIR = "models/ml"
    RESULTS_DIR = "data/ml_results"
    
    # 可用模型配置
    MODEL_CONFIGS = {
        'random_forest': {
            'class': RandomForestClassifier if SKLEARN_AVAILABLE else None,
            'default_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'class': GradientBoostingClassifier if SKLEARN_AVAILABLE else None,
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'xgboost': {
            'class': xgb.XGBClassifier if XGBOOST_AVAILABLE else None,
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'eval_metric': 'logloss'
            },
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'lightgbm': {
            'class': lgb.LGBMClassifier if LIGHTGBM_AVAILABLE else None,
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'random_state': 42,
                'verbose': -1
            },
            'tune_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [15, 31, 63]
            }
        },
        'logistic_regression': {
            'class': LogisticRegression if SKLEARN_AVAILABLE else None,
            'default_params': {
                'max_iter': 1000,
                'random_state': 42
            },
            'tune_params': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        },
        'svm': {
            'class': SVC if SKLEARN_AVAILABLE else None,
            'default_params': {
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            },
            'tune_params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear']
            }
        },
        'neural_network': {
            'class': MLPClassifier if SKLEARN_AVAILABLE else None,
            'default_params': {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'random_state': 42
            },
            'tune_params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }
    
    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_type = None
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'random_forest',
        tune_hyperparams: bool = False,
        use_feature_selection: bool = True,
        n_features: int = 30,
        scale_features: bool = True
    ) -> Dict:
        """
        訓練機器學習模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            model_type: 模型類型
            tune_hyperparams: 是否進行超參數調優
            use_feature_selection: 是否使用特徵選擇
            n_features: 選擇的特徵數量
            scale_features: 是否標準化特徵
        
        Returns:
            訓練結果字典
        """
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self.MODEL_CONFIGS[model_type]
        if model_config['class'] is None:
            raise ValueError(f"Model {model_type} not available. Install required library.")
        
        self.model_type = model_type
        
        # 特徵選擇
        if use_feature_selection and len(X_train.columns) > n_features:
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            selected_features = X_train.columns[selector.get_support()].tolist()
            X_train = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            self.feature_names = selected_features
        else:
            self.feature_names = X_train.columns.tolist()
        
        # 特徵標準化
        if scale_features:
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        # 創建模型
        if tune_hyperparams:
            # 網格搜索調優
            base_model = model_config['class']()
            param_grid = model_config['tune_params']
            
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # 使用默認參數
            self.model = model_config['class'](**model_config['default_params'])
            self.model.fit(X_train, y_train)
            best_params = model_config['default_params']
        
        # 交叉驗證評估
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy')
        
        return {
            'model_type': model_type,
            'n_features': len(self.feature_names),
            'selected_features': self.feature_names,
            'best_params': best_params,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std())
        }
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        評估模型性能
        
        Args:
            X_test: 測試特徵
            y_test: 測試標籤
        
        Returns:
            評估指標字典
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # 選擇特徵
        X_test = X_test[self.feature_names]
        
        # 標準化
        if self.scaler is not None:
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        # 預測
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 計算各種指標
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
        
        # 分類報告
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """獲取特徵重要性"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # 不同模型獲取特徵重要性的方法不同
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {'error': 'Model does not support feature importance'}
        
        # 創建特徵重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 取前N個
        top_features = importance_df.head(top_n).to_dict('records')
        
        return {
            'top_features': top_features,
            'all_features': importance_df.to_dict('records')
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True
    ) -> Dict:
        """
        進行預測
        
        Args:
            X: 特徵DataFrame
            return_proba: 是否返回概率
        
        Returns:
            預測結果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # 選擇特徵
        X_selected = X[self.feature_names]
        
        # 標準化
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_selected)
            X_selected = pd.DataFrame(X_scaled, columns=X_selected.columns, index=X_selected.index)
        
        # 預測
        predictions = self.model.predict(X_selected)
        
        result = {
            'predictions': predictions.tolist(),
            'signals': ['BUY' if p == 1 else 'SELL' for p in predictions]
        }
        
        if return_proba:
            probabilities = self.model.predict_proba(X_selected)
            result['probabilities'] = probabilities.tolist()
            result['confidence'] = [max(p) * 100 for p in probabilities]
        
        return result
    
    def save_model(self, symbol: str, version: str = 'v1') -> str:
        """保存模型"""
        if self.model is None:
            raise ValueError("No model to save")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{self.model_type}_{version}_{timestamp}.pkl"
        filepath = os.path.join(self.MODELS_DIR, filename)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'timestamp': timestamp,
            'version': version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        return filepath
    
    def load_model(self, filepath: str):
        """加載模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
    
    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models: List[str] = None
    ) -> Dict:
        """
        比較多個模型的性能
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_test: 測試特徵
            y_test: 測試標籤
            models: 要比較的模型列表
        
        Returns:
            比較結果
        """
        if models is None:
            models = ['random_forest', 'gradient_boosting', 'logistic_regression']
        
        results = []
        
        for model_type in models:
            if model_type not in self.MODEL_CONFIGS:
                continue
            
            if self.MODEL_CONFIGS[model_type]['class'] is None:
                continue
            
            print(f"Training {model_type}...")
            
            try:
                # 訓練模型
                train_result = self.train_model(
                    X_train.copy(),
                    y_train.copy(),
                    model_type=model_type,
                    tune_hyperparams=False
                )
                
                # 評估模型
                eval_result = self.evaluate_model(X_test.copy(), y_test.copy())
                
                # 合併結果
                result = {
                    'model_type': model_type,
                    'train_metrics': train_result,
                    'test_metrics': eval_result
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error training {model_type}: {e}")
                continue
        
        # 按準確率排序
        results.sort(key=lambda x: x['test_metrics']['accuracy'], reverse=True)
        
        return {
            'comparison': results,
            'best_model': results[0]['model_type'] if results else None
        }
    
    @staticmethod
    def create_target(df: pd.DataFrame, strategy: str = 'next_day_direction', **kwargs) -> pd.Series:
        """
        創建目標變量
        
        Args:
            df: 包含價格數據的DataFrame
            strategy: 目標策略
                - 'next_day_direction': 下一日漲跌方向
                - 'next_n_day_direction': 未來N日漲跌方向
                - 'threshold_return': 超過閾值的收益
        
        Returns:
            目標變量Series
        """
        if strategy == 'next_day_direction':
            return (df['close'].shift(-1) > df['close']).astype(int)
        
        elif strategy == 'next_n_day_direction':
            n = kwargs.get('n', 5)
            return (df['close'].shift(-n) > df['close']).astype(int)
        
        elif strategy == 'threshold_return':
            threshold = kwargs.get('threshold', 0.02)  # 2%
            returns = df['close'].pct_change().shift(-1)
            return (returns > threshold).astype(int)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
