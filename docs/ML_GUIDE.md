# 機器學習功能完整指南

## 目錄
1. [功能概述](#功能概述)
2. [新增ML模組](#新增ml模組)
3. [API端點](#api端點)
4. [使用示例](#使用示例)
5. [最佳實踐](#最佳實踐)

---

## 功能概述

系統現在包含完整的機器學習功能，支持：

### 核心功能
- ✅ **300+ 特徵工程** - 自動生成技術指標、統計特徵、時間特徵
- ✅ **7種ML模型** - RandomForest、GradientBoosting、XGBoost、LightGBM、LogisticRegression、SVM、NeuralNetwork
- ✅ **超參數調優** - GridSearchCV自動尋找最優參數
- ✅ **特徵選擇** - SelectKBest自動篩選重要特徵
- ✅ **時間序列交叉驗證** - TimeSeriesSplit防止數據洩漏
- ✅ **模型版本管理** - 自動追蹤、保存、載入模型
- ✅ **模型集成** - 組合多個模型提升預測準確度
- ✅ **ML策略回測** - 基於ML預測的完整回測系統
- ✅ **滾動視窗分析** - Walk-Forward Analysis評估模型穩定性

---

## 新增ML模組

### 1. ml_feature_engineering.py
**功能**: 生成300+個機器學習特徵

**特徵類型**:
```python
# 價格特徵
- returns (1日-20日)
- log_returns
- price_ranges
- gaps

# 技術指標
- MA (5, 10, 20, 50, 100, 200日)
- RSI (14, 21日)
- MACD
- Bollinger Bands
- ATR
- ADX
- CCI
- Stochastic
- Williams %R

# 統計特徵
- volatility (5-30日)
- skewness, kurtosis
- momentum
- ROC

# 成交量特徵
- volume_ratio
- OBV
- VWAP
- MFI
- price_volume_corr

# 時間特徵
- day_of_week
- month
- quarter
- is_month_start/end

# 進階特徵
- trend_strength
- support/resistance
- breakout signals
```

**使用方法**:
```python
from modules.ml_feature_engineering import FeatureEngineering

fe = FeatureEngineering()
features = fe.generate_all_features(df)
print(f"生成了 {len(features.columns)} 個特徵")
```

### 2. ml_advanced.py
**功能**: 進階機器學習管理器

**支持的模型**:
| 模型 | 類型 | 適用場景 |
|------|------|----------|
| RandomForest | 集成學習 | 通用、穩定 |
| GradientBoosting | 集成學習 | 高準確度 |
| XGBoost | 梯度提升 | 競賽級性能 |
| LightGBM | 梯度提升 | 大數據、快速 |
| LogisticRegression | 線性模型 | 簡單、可解釋 |
| SVM | 支持向量機 | 小數據集 |
| NeuralNetwork | 深度學習 | 複雜模式 |

**使用方法**:
```python
from modules.ml_advanced import AdvancedMLManager

# 創建管理器
ml = AdvancedMLManager(model_type='xgboost')

# 訓練模型
result = ml.train_model(
    X_train, y_train,
    tune_hyperparams=True,  # 自動調優
    feature_selection=True,  # 特徵選擇
    cv_folds=5              # 5折交叉驗證
)

# 評估模型
eval_result = ml.evaluate_model(X_test, y_test)

# 獲取特徵重要性
importance = ml.get_feature_importance(top_n=20)

# 預測
pred = ml.predict(X_new, return_proba=True)
```

### 3. ml_backtester.py
**功能**: ML策略回測引擎

**特點**:
- 基於ML預測的買賣信號
- 信心度閾值過濾
- 止損止盈機制
- 滑價和手續費模擬
- 完整績效指標

**使用方法**:
```python
from modules.ml_backtester import MLBacktester

backtester = MLBacktester()

# 回測ML策略
result = backtester.backtest_ml_strategy(
    df=price_df,
    predictions=ml_predictions,
    probabilities=ml_probabilities,
    initial_capital=1_000_000,
    confidence_threshold=0.6,  # 只在信心度>60%時交易
    stop_loss_pct=0.05,        # 5%止損
    take_profit_pct=0.10       # 10%止盈
)

print(f"總收益率: {result['total_return_pct']:.2f}%")
print(f"交易次數: {result['num_trades']}")
print(f"勝率: {result['metrics']['win_rate']:.2f}%")
```

### 4. ml_model_manager.py
**功能**: 模型版本管理和集成

**模型管理**:
```python
from modules.ml_model_manager import MLModelManager

manager = MLModelManager(models_dir="models")

# 保存模型
model_id = manager.save_model(
    model=trained_model,
    model_name="AAPL_predictor",
    model_type="xgboost",
    metadata={
        'accuracy': 0.85,
        'train_date': '2024-01-01'
    }
)

# 載入模型
model = manager.load_model(model_id)

# 列出所有模型
models = manager.list_models(model_name="AAPL_predictor")

# 比較多個模型
comparison = manager.compare_models(
    model_ids=['model1', 'model2', 'model3'],
    X_test=X_test,
    y_test=y_test
)
```

**模型集成**:
```python
from modules.ml_model_manager import ModelEnsemble

# 創建集成模型
ensemble = ModelEnsemble(
    models=[model1, model2, model3],
    weights=[0.5, 0.3, 0.2]  # 可選權重
)

# 預測
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)

# 投票預測
voted_pred = ensemble.voting_predict(X_test, method='hard')
```

---

## API端點

### 1. 生成ML特徵
```http
POST /api/ml/features/generate
Content-Type: application/json

{
  "symbol": "AAPL",
  "period": "2y"
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "stats": {
    "num_features": 312,
    "num_samples": 504,
    "feature_names": ["return_1d", "return_5d", ...],
    "missing_values": {...}
  }
}
```

### 2. 訓練進階ML模型
```http
POST /api/ml/train/advanced
Content-Type: application/json

{
  "symbol": "AAPL",
  "period": "2y",
  "model_type": "xgboost",
  "tune_hyperparams": true,
  "test_size": 0.2
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "model_id": "AAPL_ml_v20240101_120000",
  "model_type": "xgboost",
  "train_result": {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.87,
    "f1_score": 0.85
  },
  "test_result": {
    "accuracy": 0.82,
    "precision": 0.80,
    "recall": 0.84,
    "f1_score": 0.82,
    "auc": 0.88
  },
  "feature_importance": {
    "return_5d": 0.15,
    "rsi_14": 0.12,
    ...
  }
}
```

### 3. ML預測
```http
POST /api/ml/predict/advanced
Content-Type: application/json

{
  "model_id": "AAPL_ml_v20240101_120000",
  "symbol": "AAPL",
  "period": "3mo"
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "model_id": "AAPL_ml_v20240101_120000",
  "latest_prediction": {
    "signal": "BUY",
    "prediction": 1,
    "probability": [0.35, 0.65],
    "confidence": 0.65
  },
  "num_predictions": 63
}
```

### 4. ML策略回測
```http
POST /api/ml/backtest/ml_strategy
Content-Type: application/json

{
  "model_id": "AAPL_ml_v20240101_120000",
  "symbol": "AAPL",
  "period": "2y",
  "initial_capital": 1000000,
  "confidence_threshold": 0.6
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "model_id": "AAPL_ml_v20240101_120000",
  "backtest_result": {
    "initial_capital": 1000000,
    "final_value": 1250000,
    "total_return": 250000,
    "total_return_pct": 25.0,
    "num_trades": 15,
    "metrics": {
      "win_rate": 66.67,
      "num_winning_trades": 10,
      "num_losing_trades": 5,
      "avg_win": 35000,
      "avg_loss": -15000,
      "profit_factor": 2.33,
      "max_drawdown": 8.5,
      "sharpe_ratio": 1.8,
      "annual_return_pct": 22.5
    }
  }
}
```

### 5. 列出所有ML模型
```http
GET /api/ml/models?model_name=AAPL_ml&model_type=xgboost
```

**響應**:
```json
{
  "success": true,
  "num_models": 3,
  "models": [
    {
      "model_id": "AAPL_ml_v20240101_120000",
      "model_name": "AAPL_ml",
      "model_type": "xgboost",
      "version": "v20240101_120000",
      "created_at": "2024-01-01T12:00:00",
      "metadata": {
        "symbol": "AAPL",
        "train_accuracy": 0.85,
        "test_accuracy": 0.82
      }
    }
  ]
}
```

### 6. 刪除ML模型
```http
DELETE /api/ml/models/AAPL_ml_v20240101_120000
```

### 7. 比較多個ML模型
```http
POST /api/ml/compare
Content-Type: application/json

{
  "model_ids": ["model1", "model2", "model3"],
  "symbol": "AAPL",
  "period": "1y"
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "comparison": {
    "num_models": 3,
    "best_model": "model2",
    "comparisons": [
      {
        "model_id": "model2",
        "model_type": "xgboost",
        "accuracy": 0.85,
        "precision": 0.83,
        "recall": 0.87,
        "f1_score": 0.85,
        "auc": 0.90
      },
      ...
    ]
  }
}
```

### 8. 滾動視窗分析
```http
POST /api/ml/walk_forward
Content-Type: application/json

{
  "symbol": "AAPL",
  "period": "3y",
  "model_type": "random_forest",
  "train_window": 252,
  "test_window": 63,
  "step_size": 63
}
```

**響應**:
```json
{
  "success": true,
  "symbol": "AAPL",
  "model_type": "random_forest",
  "walk_forward_result": {
    "num_periods": 12,
    "periods": [
      {
        "period": "2021-01-01 to 2021-04-01",
        "return_pct": 15.5,
        "num_trades": 8,
        "metrics": {...}
      },
      ...
    ],
    "summary": {
      "avg_return_pct": 12.3,
      "std_return_pct": 8.5,
      "win_rate": 75.0
    }
  }
}
```

---

## 使用示例

### 完整ML交易流程

#### 步驟1: 訓練模型
```python
import requests

# 訓練XGBoost模型
response = requests.post('http://localhost:5000/api/ml/train/advanced', json={
    'symbol': 'AAPL',
    'period': '2y',
    'model_type': 'xgboost',
    'tune_hyperparams': True
})

result = response.json()
model_id = result['model_id']
print(f"模型已訓練: {model_id}")
print(f"測試準確率: {result['test_result']['accuracy']:.2%}")
```

#### 步驟2: 回測策略
```python
# 回測ML策略
response = requests.post('http://localhost:5000/api/ml/backtest/ml_strategy', json={
    'model_id': model_id,
    'symbol': 'AAPL',
    'period': '2y',
    'initial_capital': 1_000_000,
    'confidence_threshold': 0.6
})

backtest = response.json()['backtest_result']
print(f"總收益: {backtest['total_return_pct']:.2f}%")
print(f"勝率: {backtest['metrics']['win_rate']:.2f}%")
print(f"夏普比率: {backtest['metrics']['sharpe_ratio']:.2f}")
```

#### 步驟3: 實時預測
```python
# 使用訓練好的模型進行預測
response = requests.post('http://localhost:5000/api/ml/predict/advanced', json={
    'model_id': model_id,
    'symbol': 'AAPL',
    'period': '3mo'
})

prediction = response.json()['latest_prediction']
print(f"預測信號: {prediction['signal']}")
print(f"信心度: {prediction['confidence']:.2%}")
```

### 模型比較流程

```python
# 訓練多個不同類型的模型
model_types = ['random_forest', 'xgboost', 'lightgbm']
model_ids = []

for model_type in model_types:
    response = requests.post('http://localhost:5000/api/ml/train/advanced', json={
        'symbol': 'AAPL',
        'period': '2y',
        'model_type': model_type
    })
    model_ids.append(response.json()['model_id'])

# 比較所有模型
response = requests.post('http://localhost:5000/api/ml/compare', json={
    'model_ids': model_ids,
    'symbol': 'AAPL',
    'period': '1y'
})

comparison = response.json()['comparison']
best = comparison['comparisons'][0]
print(f"最佳模型: {best['model_type']}")
print(f"準確率: {best['accuracy']:.2%}")
print(f"F1分數: {best['f1_score']:.2%}")
```

---

## 最佳實踐

### 1. 數據準備
- ✅ **足夠的歷史數據**: 至少2年數據用於訓練
- ✅ **數據清洗**: 處理缺失值和異常值
- ✅ **特徵縮放**: 自動進行StandardScaler

### 2. 模型訓練
- ✅ **超參數調優**: 首次訓練時啟用 `tune_hyperparams=True`
- ✅ **交叉驗證**: 使用TimeSeriesSplit防止數據洩漏
- ✅ **特徵選擇**: 減少過擬合，提升性能

### 3. 模型評估
- ✅ **多指標評估**: 不只看準確率，也要看精確率、召回率、F1分數、AUC
- ✅ **滾動視窗分析**: 評估模型在不同市場環境的穩定性
- ✅ **實時回測**: 在歷史數據上驗證策略表現

### 4. 模型部署
- ✅ **版本管理**: 保存所有訓練過的模型版本
- ✅ **模型更新**: 定期用最新數據重新訓練
- ✅ **監控性能**: 追蹤模型在實時數據上的表現

### 5. 風險管理
- ✅ **信心度閾值**: 只在高信心度預測時交易
- ✅ **止損止盈**: 設置合理的止損和止盈點
- ✅ **倉位管理**: 不要一次性投入全部資金

### 6. 模型選擇建議

| 場景 | 推薦模型 | 原因 |
|------|----------|------|
| 初學者 | RandomForest | 穩定、易理解 |
| 追求性能 | XGBoost | 競賽級準確度 |
| 大數據 | LightGBM | 訓練速度快 |
| 可解釋性 | LogisticRegression | 簡單透明 |
| 複雜模式 | NeuralNetwork | 捕捉非線性 |

### 7. 常見問題

**Q: 模型準確率低怎麼辦？**
A: 
1. 增加訓練數據量
2. 啟用超參數調優
3. 嘗試不同模型類型
4. 檢查特徵質量

**Q: 回測表現好但實盤差？**
A:
1. 可能過擬合，使用交叉驗證
2. 進行滾動視窗分析
3. 增加信心度閾值
4. 加入交易成本和滑價

**Q: 如何選擇信心度閾值？**
A:
- 0.5: 平衡點（默認）
- 0.6-0.7: 保守策略
- 0.8+: 非常保守，交易次數少

**Q: 多久更新一次模型？**
A:
- 建議: 每月或每季度
- 當市場環境發生重大變化時
- 當模型性能明顯下降時

---

## 技術架構

```
ML功能架構
├── 數據層
│   ├── DataFetcher (獲取價格數據)
│   └── FeatureEngineering (生成300+特徵)
│
├── 模型層
│   ├── AdvancedMLManager (7種模型支持)
│   ├── MLModelManager (版本管理)
│   └── ModelEnsemble (模型集成)
│
├── 評估層
│   ├── MLBacktester (策略回測)
│   └── WalkForwardAnalysis (滾動驗證)
│
└── API層
    └── Flask REST API (10個ML端點)
```

---

## 下一步

### 可以擴展的功能
1. **深度學習模型** - LSTM, GRU, Transformer
2. **強化學習** - DQN, PPO用於交易策略
3. **特徵工程優化** - 自動特徵生成和選擇
4. **模型解釋** - SHAP, LIME提升可解釋性
5. **實時預測API** - WebSocket推送實時信號
6. **自動化交易** - 連接券商API執行交易

---

## 參考資源

- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/
- scikit-learn: https://scikit-learn.org/
- 時間序列交叉驗證: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split

---

**完成日期**: 2024-01-01
**版本**: v2.0
**作者**: QuantPilot Team
