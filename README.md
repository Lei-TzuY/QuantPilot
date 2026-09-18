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

---

## 🛡️ 交易模式與安全規範 (Execution Modes & Safety)

QuantPilot 支援三種執行模式，相同的策略邏輯無需任何修改即可在三種模式間無縫切換：

1. **BACKTEST (歷史回測)**:
   - 採用嚴格的事件驅動時間語義，徹底杜絕未來資訊洩漏（Look-Ahead Bias）。
   - Bar $t$ 收盤完成計算特徵並產生訊號後，訂單嚴格於 Bar $t+1$ 起始才具備成交資格。
2. **PAPER (模擬交易 - 預設模式)**:
   - 透過 `PaperBrokerAdapter` 進行高真實度即時撮合，完整模擬市價單/限價單、部分成交（Partial Fills）、手續費（0.1425%）、台灣證券交易稅（0.3%）與滑價。
3. **LIVE (實盤交易 - 嚴格受限)**:
   - 透過隔離的 `ShioajiBrokerAdapter` 對接永豐金證券（Sinopac Shioaji）。

> [!CAUTION]
> **重要風險聲明與安全防線 (Risk Disclosure & Safety Invariant):**
> 1. **實盤交易預設全面鎖定 (Opt-in Only)**：系統預設 `TRADING_MODE=paper`。除非在環境變數中明確設定 `TRADING_MODE=live`，否則任何實盤委託送單將被系統底層直接拒絕拋出異常。
> 2. **實盤具備虧損風險 (Capital Risk)**：金融量化交易旨在尋求統計優勢，絕無保證獲利。實盤交易可能導致本金損失。
> 3. **全訂單強制風控閘門 (Risk Engine Gate)**：所有委託單（包括手動 API 下單）在送達券商前，必須通過 `RiskEngine` 前置檢核（單筆金額、總曝險、單檔上限、每日累積虧損、行情資料過期、異常偏離與重覆防護）。
> 4. **全域緊急熔斷機制 (Kill Switch)**：當日虧損達限或人工觸發時立即進入 `HALTED` 狀態，拒絕所有新委託。熔斷狀態持久化於硬碟，重啟後**絕不自動恢復交易**，必須由指定操作員明確授權恢復。

---

## 🏗️ 事件驅動系統架構 (Event-Driven Architecture)

```
行情資料 (Market Ticks / Streams)
       │
       ▼
K 線聚合器 BarBuilder (產生 BarEvent)
       │
       ▼
交易策略 Strategy (on_bar -> 發出 SignalEvent)
       │
       ▼
風控引擎 RiskEngine & 緊急熔斷 KillSwitch (輸出 RiskDecision: 允許/拒絕理由)
       │ (通過)
       ▼
訂單管理系統 OrderManager / OMS (生命週期狀態機、唯一 ID、防重覆送單)
       │
       ▼
券商抽象介面 BrokerAdapter
       ├─────────────────────────────────┐
       ▼                                 ▼
模擬券商 PaperBrokerAdapter      台灣永豐 ShioajiBrokerAdapter
(模擬滑價、稅費、部分成交)        (隔離憑證、Live Mode 嚴格鎖定)
       │                                 │
       └────────────────┬────────────────┘
                        │ 成交回報 (FillEvent) / 委託狀態
                        ▼
            部位與委託對帳系統 Reconciler
            (偵測帳務差異、缺失回報、孤兒委託)
                        │
                        ▼
            狀態持久化與審計日誌 (Crash Recovery & Audit Trail)
```

---

## 📡 即時交易與風控 API (Trading APIs)

| 端點 | 方法 | 說明 |
| :--- | :---: | :--- |
| `/api/trading/status` | `GET` | 查詢引擎執行狀態、交易模式、市場時段、熔斷狀態、持倉與未實現損益 |
| `/api/trading/positions` | `GET` | 查詢即時對帳部位、持倉均價、成本與即時浮動盈虧 |
| `/api/trading/orders` | `GET` | 查詢 OMS 追蹤的所有委託單（進行中、完全成交、已取消、已拒絕） |
| `/api/trading/pnl` | `GET` | 查詢當日已實現損益、未實現損益與成交筆數 |
| `/api/trading/order` | `POST` | 手動下單（**強制受 RiskEngine 風控閘門檢驗**，禁止越權） |
| `/api/trading/halt` | `POST` | 緊急停機熔斷（切換 KillSwitch 至 `HALTED` 並持久化） |
| `/api/trading/resume` | `POST` | 恢復交易（需填寫 `operator_id` 與授權理由） |
| `/api/trading/reconcile`| `POST` | 執行內部部位與券商權威帳務的全量對帳 |

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
