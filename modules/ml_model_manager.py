"""
機器學習模型管理器
ML Model Manager
"""
import os
import json
import pickle
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


class MLModelManager:
    """
    ML模型管理器
    處理模型版本控制、存儲、載入
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化模型管理器
        
        Args:
            models_dir: 模型存儲目錄
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(models_dir, "model_registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """載入模型註冊表"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """保存模型註冊表"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metadata: Optional[Dict] = None,
        version: Optional[str] = None
    ) -> str:
        """
        保存模型
        
        Args:
            model: 模型對象
            model_name: 模型名稱
            model_type: 模型類型
            metadata: 額外元數據
            version: 版本號（如果不提供則自動生成）
        
        Returns:
            模型ID
        """
        # 生成版本號
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
        
        # 生成模型ID
        model_id = f"{model_name}_{version}"
        
        # 模型文件路徑
        model_file = os.path.join(self.models_dir, f"{model_id}.pkl")
        
        # 保存模型
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # 計算模型文件哈希
        with open(model_file, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # 更新註冊表
        self.registry[model_id] = {
            'model_name': model_name,
            'model_type': model_type,
            'version': version,
            'file_path': model_file,
            'file_hash': file_hash,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self._save_registry()
        
        return model_id
    
    def load_model(self, model_id: str) -> Any:
        """
        載入模型
        
        Args:
            model_id: 模型ID
        
        Returns:
            模型對象
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.registry[model_id]
        model_file = model_info['file_path']
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file {model_file} not found")
        
        # 驗證文件哈希
        with open(model_file, 'rb') as f:
            current_hash = hashlib.md5(f.read()).hexdigest()
        
        if current_hash != model_info['file_hash']:
            raise ValueError(f"Model file {model_file} has been corrupted")
        
        # 載入模型
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """
        獲取模型的最新版本
        
        Args:
            model_name: 模型名稱
        
        Returns:
            最新版本的模型ID
        """
        matching_models = [
            (model_id, info) for model_id, info in self.registry.items()
            if info['model_name'] == model_name
        ]
        
        if not matching_models:
            return None
        
        # 按創建時間排序
        matching_models.sort(key=lambda x: x[1]['created_at'], reverse=True)
        
        return matching_models[0][0]
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> List[Dict]:
        """
        列出模型
        
        Args:
            model_name: 過濾模型名稱
            model_type: 過濾模型類型
        
        Returns:
            模型列表
        """
        models = []
        
        for model_id, info in self.registry.items():
            if model_name and info['model_name'] != model_name:
                continue
            
            if model_type and info['model_type'] != model_type:
                continue
            
            models.append({
                'model_id': model_id,
                **info
            })
        
        # 按創建時間排序
        models.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models
    
    def delete_model(self, model_id: str):
        """
        刪除模型
        
        Args:
            model_id: 模型ID
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.registry[model_id]
        model_file = model_info['file_path']
        
        # 刪除模型文件
        if os.path.exists(model_file):
            os.remove(model_file)
        
        # 從註冊表中移除
        del self.registry[model_id]
        self._save_registry()
    
    def compare_models(
        self,
        model_ids: List[str],
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict:
        """
        比較多個模型的性能
        
        Args:
            model_ids: 模型ID列表
            X_test: 測試特徵
            y_test: 測試標籤
        
        Returns:
            比較結果
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        results = []
        
        for model_id in model_ids:
            try:
                model = self.load_model(model_id)
                model_info = self.registry[model_id]
                
                # 預測
                y_pred = model.predict(X_test)
                
                # 如果支持概率預測
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    auc = None
                
                # 計算指標
                metrics = {
                    'model_id': model_id,
                    'model_name': model_info['model_name'],
                    'model_type': model_info['model_type'],
                    'version': model_info['version'],
                    'created_at': model_info['created_at'],
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0),
                    'auc': auc
                }
                
                results.append(metrics)
                
            except Exception as e:
                print(f"Error evaluating model {model_id}: {e}")
        
        # 按準確率排序
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return {
            'num_models': len(results),
            'best_model': results[0]['model_id'] if results else None,
            'comparisons': results
        }
    
    def export_model_report(self, model_id: str, output_file: str):
        """
        導出模型報告
        
        Args:
            model_id: 模型ID
            output_file: 輸出文件路徑
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")
        
        model_info = self.registry[model_id]
        
        report = {
            'model_id': model_id,
            'model_info': model_info,
            'export_date': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class ModelEnsemble:
    """
    模型集成器
    組合多個模型進行預測
    """
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        """
        初始化集成器
        
        Args:
            models: 模型列表
            weights: 模型權重（如果不提供則使用平均權重）
        """
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # 歸一化權重
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        集成預測概率
        
        Args:
            X: 特徵數據
        
        Returns:
            預測概率
        """
        predictions = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                # 如果模型不支持概率預測，轉換為概率形式
                pred_labels = model.predict(X)
                pred = np.zeros((len(pred_labels), 2))
                pred[np.arange(len(pred_labels)), pred_labels] = 1
            
            predictions.append(pred)
        
        # 加權平均
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        集成預測
        
        Args:
            X: 特徵數據
        
        Returns:
            預測標籤
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def voting_predict(self, X: pd.DataFrame, method: str = 'hard') -> np.ndarray:
        """
        投票預測
        
        Args:
            X: 特徵數據
            method: 'hard' (多數投票) 或 'soft' (加權概率)
        
        Returns:
            預測標籤
        """
        if method == 'soft':
            return self.predict(X)
        
        # 硬投票
        votes = []
        for model in self.models:
            pred = model.predict(X)
            votes.append(pred)
        
        votes = np.array(votes)
        
        # 多數投票
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=votes
        )
