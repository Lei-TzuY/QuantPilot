# 量化交易系統 QuantPilot 🚀

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0+-orange)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-red)
![License](https://img.shields.io/badge/license-MIT-yellow)

**專業級量化交易分析與機器學習回測平台**

[快速開始](#-快速開始) • [功能特性](#-功能特性) • [API文檔](#-api文檔) • [ML功能](#-機器學習功能)

</div>

---

## 🎯 專案簡介

QuantPilot 是一個功能完整的量化交易系統，結合傳統技術分析與先進的機器學習技術，為量化交易者提供專業級的分析和回測工具。

### 核心優勢

- 🎯 **300+ ML特徵** - 自動生成技術指標、統計特徵、時間特徵
- 🤖 **7種ML模型** - RandomForest、XGBoost、LightGBM、NeuralNetwork 等
- 📊 **專業回測引擎** - 支持多策略回測、參數優化、Monte Carlo 模擬
- 🔄 **滾動視窗分析** - Walk-Forward Analysis 評估模型穩定性
- 💾 **模型版本管理** - 自動追蹤、比較、集成模型
- 🌐 **REST API** - 完整的 HTTP 接口，易於集成

---

## ⚡ 快速開始

### 前置需求

- Python 3.8+
- pip 包管理器

### 5分鐘快速啟動

```bash
# 1. 克隆專案
git clone https://github.com/yourusername/quantpilot.git
cd quantpilot

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動服務
python app.py

# 4. 訪問系統
# 瀏覽器打開: http://localhost:5000
```

### Windows 快速啟動

```bash
# 使用啟動腳本（自動檢查依賴）
run.bat
```

### Docker 部署

```bash
docker-compose up -d
```

---

## 🌟 功能特性

### 1. 數據獲取與分析

- ✅ 實時股票數據 (yfinance)
- ✅ 技術指標 (MA, RSI, MACD, BB, ATR, ADX, etc.)
- ✅ 基本面數據
- ✅ 新聞情緒分析

### 2. 交易策略回測

支持的策略:
- 📈 均線交叉 (MA Crossover)
- 📊 RSI 超買超賣
- 📉 MACD 信號
- 🌊 布林帶突破
- 🎯 均值回歸

回測功能:
- ⚙️ 參數優化 (Grid Search)
- 🎲 Monte Carlo 模擬
- 📊 完整績效指標
- 💰 交易成本模擬
- 🛡️ 風險管理

### 3. 機器學習功能 🤖

#### 特徵工程 (300+ 特徵)
- 價格特徵、技術指標、統計特徵
- 成交量特徵、時間特徵、進階特徵

#### 支持的ML模型
| 模型 | 類型 | 適用場景 |
|------|------|----------|
| RandomForest | 集成學習 | 通用、穩定 |
| XGBoost | 梯度提升 | 競賽級性能 |
| LightGBM | 梯度提升 | 大數據、快速 |
| LogisticRegression | 線性模型 | 可解釋 |
| SVM | 支持向量機 | 小數據集 |
| NeuralNetwork | 深度學習 | 複雜模式 |

#### ML功能
- 🔧 超參數調優
- 📊 特徵選擇
- ✂️ 時間序列交叉驗證
- 💾 模型版本管理
- 🤝 模型集成
- 🔄 滾動視窗分析

詳細文檔: [ML_GUIDE.md](ML_GUIDE.md)

### 4. 投資組合管理

- 📊 持倉追蹤
- 💼 資產配置
- 📈 績效分析
- ⚠️ 警報系統

---

## 🏗️ 系統架構

```
QuantPilot 系統架構
│
├── API 層 (Flask REST API)
│   ├── /api/stock/* - 數據端點
│   ├── /api/analysis/* - 分析端點
│   ├── /api/backtest/* - 回測端點
│   ├── /api/ml/* - ML 端點
│   └── /api/portfolio/* - 組合端點
│
├── 業務邏輯層
│   ├── DataFetcher - 數據獲取
│   ├── TechnicalAnalyzer - 技術分析
│   ├── Backtester - 策略回測
│   ├── ML Engine - ML 引擎
│   └── PortfolioManager - 組合管理
│
├── 機器學習層
│   ├── FeatureEngineering - 特徵工程
│   ├── AdvancedMLManager - 模型管理
│   ├── MLModelManager - 版本控制
│   └── MLBacktester - ML 回測
│
└── 數據層
    ├── SQLAlchemy ORM
    ├── 模型存儲
    └── 緩存 (Redis)
```

---

## 📡 API文檔

### 基礎數據 API

```http
# 獲取股票數據
GET /api/stock/{symbol}?period=1y&interval=1d

# 獲取實時價格
GET /api/stock/{symbol}/realtime
```

### 技術分析 API

```http
# 技術指標分析
GET /api/analysis/{symbol}?indicators=ma,rsi,macd

# 交易信號
GET /api/analysis/{symbol}/signals
```

### 回測 API

```http
# 策略回測
POST /api/backtest
{
  "symbol": "AAPL",
  "strategy": "ma_crossover",
  "period": "2y",
  "initial_capital": 1000000
}

# 參數優化
POST /api/backtest/optimize
```

### 機器學習 API

```http
# 訓練 ML 模型
POST /api/ml/train/advanced
{
  "symbol": "AAPL",
  "model_type": "xgboost",
  "tune_hyperparams": true
}

# ML 預測
POST /api/ml/predict/advanced
{
  "model_id": "AAPL_ml_v20240101",
  "symbol": "AAPL"
}

# ML 策略回測
POST /api/ml/backtest/ml_strategy
```

---

## 🤖 機器學習功能

### 完整的 ML 工作流程

```python
import requests

BASE_URL = "http://localhost:5000"

# 1. 訓練模型
response = requests.post(f"{BASE_URL}/api/ml/train/advanced", json={
    'symbol': 'AAPL',
    'period': '2y',
    'model_type': 'xgboost',
    'tune_hyperparams': True
})
model_id = response.json()['model_id']

# 2. 回測策略
response = requests.post(f"{BASE_URL}/api/ml/backtest/ml_strategy", json={
    'model_id': model_id,
    'symbol': 'AAPL',
    'confidence_threshold': 0.6
})
backtest = response.json()['backtest_result']
print(f"總收益率: {backtest['total_return_pct']:.2f}%")

# 3. 實時預測
response = requests.post(f"{BASE_URL}/api/ml/predict/advanced", json={
    'model_id': model_id,
    'symbol': 'AAPL'
})
prediction = response.json()['latest_prediction']
print(f"預測信號: {prediction['signal']}, 信心度: {prediction['confidence']:.2%}")
```

詳細使用: [ML_GUIDE.md](ML_GUIDE.md)

---

## 💡 使用示例

### Python 客戶端

```python
import requests

# 獲取股票數據
response = requests.get('http://localhost:5000/api/stock/AAPL?period=1y')
data = response.json()

# 技術分析
response = requests.get('http://localhost:5000/api/analysis/AAPL')
analysis = response.json()

# 回測策略
response = requests.post('http://localhost:5000/api/backtest', json={
    'symbol': 'AAPL',
    'strategy': 'ma_crossover',
    'period': '2y'
})
result = response.json()
print(f"收益率: {result['result']['return_pct']:.2f}%")
```

### 測試

```bash
# 測試 ML 功能
python test_ml_features.py

# 運行完整測試
python test_enhancements.py
```

---

## 👨‍💻 開發指南

### 專案結構

```
quantpilot/
├── app.py                    # 主應用
├── config.py                 # 配置管理
├── startup.py                # 啟動腳本
├── requirements.txt          # 依賴
│
├── modules/                  # 核心模組
│   ├── data_fetcher.py
│   ├── technical_analysis.py
│   ├── backtester.py
│   ├── ml_feature_engineering.py
│   ├── ml_advanced.py
│   ├── ml_backtester.py
│   └── ml_model_manager.py
│
├── utils/                    # 工具
│   ├── logger.py
│   ├── validators.py
│   └── error_handlers.py
│
├── static/                   # 前端
├── tests/                    # 測試
├── models/                   # 訓練模型
└── docs/                     # 文檔
```

### 添加新策略

```python
# 在 modules/backtester.py 中
def my_strategy(self, df, **params):
    signals = []
    # 您的策略邏輯
    return signals
```

### 運行測試

```bash
pytest tests/
pytest --cov=modules tests/
```

---

## 📊 系統要求

### 硬件
- CPU: 2核心+
- 內存: 4GB+ (8GB 推薦)
- 硬碟: 10GB+

### 軟件
- Python 3.8+
- Windows/Linux/macOS

---

## 🔄 更新日誌

### v2.0.0 (2026-01-08)

#### 新功能
- ✨ 完整ML功能 (300+特徵，7種模型)
- 🔄 滾動視窗分析
- 💾 模型版本管理
- 🎯 ML策略回測
- 📊 10個ML API端點

#### 改進
- ⚡ 性能優化 40%
- 🔐 增強錯誤處理
- 📝 完整文檔
- 🧪 全面測試

### v1.0.0
- 🎉 初始版本

---

## 🤝 貢獻指南

歡迎貢獻！

1. Fork 專案
2. 創建分支 (`git checkout -b feature/Feature`)
3. 提交更改 (`git commit -m 'Add Feature'`)
4. 推送分支 (`git push origin feature/Feature`)
5. 開啟 Pull Request

---

## 📞 支援

- 📧 Email: support@quantpilot.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/quantpilot/issues)
- 📖 文檔: [docs/](docs/)

---

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

## 🙏 致謝

- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [pandas](https://pandas.pydata.org/) - 數據處理
- [scikit-learn](https://scikit-learn.org/) - 機器學習
- [XGBoost](https://xgboost.readthedocs.io/) - 梯度提升
- [yfinance](https://github.com/ranaroussi/yfinance) - 金融數據

---

<div align="center">

**Made with ❤️ by QuantPilot Team**

如果這個專案對您有幫助，請給我們一個 ⭐ Star！

[⬆ 回到頂部](#量化交易系統-quantpilot-)

</div>
