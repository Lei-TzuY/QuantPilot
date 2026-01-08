# 量化交易系統 Quantitative Trading System

<div align="center">

🚀 **專業級量化交易分析平台** 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📖 項目簡介

這是一個功能完善的量化交易系統，提供股票數據分析、技術指標計算、回測引擎、機器學習預測、紙上交易等功能。系統採用前後端分離架構，後端使用 Flask 框架，前端使用原生 JavaScript + Chart.js。

## ✨ 核心功能

### 📊 數據分析
- **實時行情**: 獲取股票實時價格和基本信息
- **歷史數據**: 支持多種時間週期和間隔的歷史數據查詢
- **技術指標**: MA、EMA、RSI、MACD、布林帶、ATR、OBV等
- **批次處理**: 同時分析多支股票

### 📈 交易策略
- **均線交叉**: 經典的短期長期均線交叉策略
- **RSI策略**: 超買超賣指標策略
- **MACD策略**: 趨勢跟蹤策略
- **布林帶策略**: 波動性突破策略
- **突破策略**: 價格突破新高/新低
- **VWAP回歸**: 成交量加權平均價格策略
- **成交量突破**: 基於成交量異常的策略

### 🔬 回測引擎
- **完整回測**: 支持多種策略的歷史回測
- **參數優化**: 自動尋找最佳策略參數
- **風險管理**: 止損、止盈、追蹤止損、最大回撤控制
- **績效指標**: 收益率、夏普比率、最大回撤、勝率等
- **交易記錄**: 詳細的買賣信號和持倉記錄

### 🤖 機器學習
- **模型訓練**: 支持隨機森林、梯度提升等模型
- **信號預測**: 預測買賣信號
- **特徵重要性**: 分析哪些特徵最重要

### 💼 投資組合管理
- **持倉管理**: 添加、刪除、更新持倉
- **績效追蹤**: 實時計算收益、虧損、總值
- **多元化分析**: 投資組合風險分析

### 📰 新聞與情緒
- **新聞獲取**: 自動獲取相關股票新聞
- **情緒分析**: 分析新聞情緒傾向

### 🔔 智能警報
- **價格警報**: 價格突破設定值時通知
- **變化警報**: 價格變化達到百分比時通知
- **成交量警報**: 成交量異常時通知
- **後台監控**: 自動檢查並觸發警報

### 📝 紙上交易
- **模擬交易**: 無風險模擬真實交易
- **交易記錄**: 完整的買賣歷史
- **績效追蹤**: 實時追蹤模擬投資組合表現

## 🏗️ 技術架構

### 後端技術棧
- **Web框架**: Flask 3.0+
- **API文檔**: Flask-RESTX (Swagger)
- **速率限制**: Flask-Limiter
- **跨域支持**: Flask-CORS
- **數據處理**: Pandas, NumPy
- **數據源**: yfinance
- **技術分析**: TA-Lib, ta
- **機器學習**: scikit-learn
- **數據庫**: SQLAlchemy (SQLite/PostgreSQL)
- **驗證**: Pydantic
- **日誌**: Python logging
- **快取**: Redis (可選)

### 前端技術棧
- **純JavaScript**: 無框架依賴
- **圖表庫**: Chart.js
- **UI**: 響應式設計

## 📦 安裝與運行

### 前置要求
- Python 3.8 或更高版本
- pip 包管理器

### 快速開始

#### Windows 用戶

1. **克隆或下載專案**
```bash
git clone <repository-url>
cd 量化金融
```

2. **運行啟動腳本**
```bash
run.bat
```

這個腳本會自動：
- 創建虛擬環境
- 安裝所有依賴
- 初始化資料庫
- 啟動服務器

#### Linux/Mac 用戶

1. **克隆或下載專案**
```bash
git clone <repository-url>
cd 量化金融
```

2. **創建虛擬環境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

3. **安裝依賴**
```bash
pip install -r requirements.txt
```

4. **配置環境變數**
```bash
cp .env.example .env
# 編輯 .env 文件設置您的配置
```

5. **初始化並運行**
```bash
python startup.py --enhanced
```

### 訪問應用

打開瀏覽器訪問：
```
http://localhost:5000
```

API 文檔：
```
http://localhost:5000/api/docs  (即將推出)
```

## 📚 API 文檔

### 股票數據 API

#### 獲取股票歷史數據
```http
GET /api/stock/{symbol}?period=1y&interval=1d
```

#### 獲取股票信息
```http
GET /api/stock/{symbol}/info
```

#### 獲取實時價格
```http
GET /api/stock/{symbol}/realtime
```

### 技術分析 API

#### 獲取技術分析
```http
GET /api/analysis/{symbol}?period=1y&indicators=ma,rsi,macd
```

#### 獲取交易信號
```http
GET /api/analysis/{symbol}/signals?period=6mo
```

### 回測 API

#### 運行回測
```http
POST /api/backtest
Content-Type: application/json

{
  "symbol": "2330",
  "strategy": "ma_crossover",
  "period": "2y",
  "initial_capital": 1000000,
  "params": {
    "short_window": 20,
    "long_window": 60
  }
}
```

#### 優化策略
```http
POST /api/backtest/optimize
Content-Type: application/json

{
  "symbol": "2330",
  "strategy": "ma_crossover",
  "period": "2y",
  "param_ranges": {
    "short_window": [10, 30, 5],
    "long_window": [40, 80, 10]
  }
}
```

### 投資組合 API

#### 獲取投資組合
```http
GET /api/portfolio
```

#### 添加持倉
```http
POST /api/portfolio/add
Content-Type: application/json

{
  "symbol": "2330",
  "shares": 1000,
  "buy_price": 500
}
```

### 警報 API

#### 創建警報
```http
POST /api/alerts
Content-Type: application/json

{
  "symbol": "2330",
  "condition": "above",
  "target_value": 600,
  "note": "突破600元"
}
```

## 🔧 配置說明

### 環境變數配置

編輯 `.env` 文件：

```env
# 基本配置
FLASK_ENV=development
DEBUG=True
HOST=0.0.0.0
PORT=5000

# 速率限制
RATELIMIT_ENABLED=True
RATELIMIT_DEFAULT=200 per hour

# 數據庫
DATABASE_URL=sqlite:///data/trading.db

# 日誌
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# 交易參數
DEFAULT_SUFFIX=.TW
DEFAULT_FEE_PCT=0.001425
```

### 高級配置

#### 使用 Redis 快取
```env
CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0
```

#### 使用 PostgreSQL
```env
DATABASE_URL=postgresql://user:password@localhost:5432/trading_db
```

## 🎯 使用範例

### Python 範例

```python
import requests

# 獲取股票數據
response = requests.get('http://localhost:5000/api/stock/2330?period=1y')
data = response.json()

# 運行回測
backtest_data = {
    "symbol": "2330",
    "strategy": "ma_crossover",
    "period": "2y",
    "initial_capital": 1000000
}
response = requests.post('http://localhost:5000/api/backtest', json=backtest_data)
result = response.json()
print(f"總收益: {result['result']['total_return_pct']:.2f}%")
```

### JavaScript 範例

```javascript
// 獲取實時價格
async function getRealtimePrice(symbol) {
  const response = await fetch(`/api/stock/${symbol}/realtime`);
  const data = await response.json();
  return data.price;
}

// 創建警報
async function createAlert(symbol, targetPrice) {
  const response = await fetch('/api/alerts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      symbol: symbol,
      condition: 'above',
      target_value: targetPrice
    })
  });
  return await response.json();
}
```

## 📊 系統特性

### 性能優化
- ✅ **請求快取**: 減少重複數據獲取
- ✅ **速率限制**: 防止API濫用
- ✅ **異步處理**: 批次操作不阻塞
- ✅ **數據庫索引**: 快速查詢優化

### 安全性
- ✅ **輸入驗證**: Pydantic 數據驗證
- ✅ **錯誤處理**: 完善的異常捕獲
- ✅ **請求追蹤**: 唯一請求ID
- ✅ **CORS配置**: 跨域請求控制

### 可維護性
- ✅ **模塊化設計**: 清晰的代碼結構
- ✅ **日誌記錄**: 完整的操作日誌
- ✅ **配置管理**: 環境變數配置
- ✅ **錯誤追蹤**: 詳細的錯誤堆棧

## 🔍 項目結構

```
量化金融/
├── app.py                  # 原始應用入口
├── app_enhanced.py         # 增強版應用入口
├── startup.py              # 啟動腳本
├── run.bat                 # Windows 啟動腳本
├── config.py               # 配置文件
├── requirements.txt        # 依賴清單
├── .env.example            # 環境變數範例
├── README.md               # 本文件
│
├── modules/                # 業務模組
│   ├── data_fetcher.py     # 數據獲取
│   ├── technical_analysis.py  # 技術分析
│   ├── backtester.py       # 回測引擎
│   ├── signal_generator.py # 信號生成
│   ├── portfolio_manager.py # 投資組合管理
│   ├── paper_trader.py     # 紙上交易
│   ├── ml_signal.py        # 機器學習
│   ├── alert_manager.py    # 警報管理
│   └── ...
│
├── models/                 # 數據模型
│   ├── __init__.py
│   └── database.py         # SQLAlchemy 模型
│
├── utils/                  # 工具模組
│   ├── __init__.py
│   ├── logger.py           # 日誌工具
│   ├── validators.py       # 輸入驗證
│   └── error_handlers.py   # 錯誤處理
│
├── static/                 # 靜態文件
│   ├── index.html
│   ├── css/
│   └── js/
│
├── data/                   # 數據目錄
│   └── trading.db          # SQLite 數據庫
│
└── logs/                   # 日誌目錄
    └── app.log
```

## 🛠️ 開發指南

### 添加新策略

1. 在 `modules/backtester.py` 中添加策略邏輯
2. 更新 `get_available_strategies()` 方法
3. 添加策略參數驗證

### 添加新的技術指標

1. 在 `modules/technical_analysis.py` 中實現指標計算
2. 更新 `analyze()` 方法支持新指標
3. 在前端添加圖表顯示

### 擴展 API

1. 在 `app_enhanced.py` 中添加新路由
2. 實現業務邏輯
3. 添加輸入驗證和錯誤處理
4. 更新 API 文檔

## 🤝 貢獻指南

歡迎貢獻！請遵循以下步驟：

1. Fork 本倉庫
2. 創建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📝 更新日誌

### v2.0.0 (2026-01-08)
- ✨ 新增增強版應用架構
- ✨ 添加完整的錯誤處理和日誌系統
- ✨ 實現速率限制和請求追蹤
- ✨ 添加 Pydantic 數據驗證
- ✨ 實現 SQLAlchemy 數據持久化
- ✨ 添加健康檢查端點
- ✨ 優化配置管理
- 🐛 修復多個已知問題

### v1.0.0
- 🎉 初始版本發布
- 基本功能實現

## 📄 許可證

本項目採用 MIT 許可證 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- [yfinance](https://github.com/ranaroussi/yfinance) - 股票數據獲取
- [TA-Lib](https://github.com/mrjbq7/ta-lib) - 技術分析
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [Chart.js](https://www.chartjs.org/) - 圖表庫

## 📧 聯繫方式

如有問題或建議，請開啟 Issue 或 Pull Request。

---

<div align="center">

**⭐ 如果這個項目對你有幫助，請給個星標 Star！ ⭐**

Made with ❤️ by Quantitative Trading Team

</div>
