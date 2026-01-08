# 🚀 QuantPilot 快速參考

## 📦 一鍵啟動

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動服務
python app.py
```

訪問：http://localhost:5000

---

## 📁 項目結構速覽

```
QuantPilot/
├── app.py                 # 主應用入口 ⭐
├── requirements.txt       # Python 依賴 📦
├── .gitignore            # Git 配置 🔒
├── LICENSE               # MIT 授權 📜
├── README.md             # 完整文檔 📖
│
├── modules/              # 核心模組 🧩
├── static/               # 前端文件 🌐
├── tests/                # 測試文件 🧪
├── docs/                 # 文檔目錄 📚
├── data/                 # 數據存儲 💾
└── models/               # ML 模型 🤖
```

---

## 🎯 核心功能

| 功能 | 描述 | 文件 |
|------|------|------|
| 📊 實時數據 | Yahoo Finance API | `modules/data_fetcher.py` |
| 📈 技術分析 | 300+ 指標 | `modules/technical_analysis.py` |
| 🤖 機器學習 | 7 種模型 | `modules/ml_signal.py` |
| 📰 新聞分析 | 情感分析 | `modules/news_fetcher.py` |
| 🔔 警報系統 | 多種觸發條件 | `modules/alert_manager.py` |
| 💼 投資組合 | 管理與優化 | `modules/portfolio_manager.py` |
| 🔄 回測系統 | 歷史驗證 | `modules/backtester.py` |
| 📊 批次分析 | 風險評估 | `modules/batch_processor.py` |

---

## 🔧 主要 API 端點

```
GET  /api/v2/data/realtime/{symbol}       # 實時數據
GET  /api/v2/analysis/{symbol}            # 技術分析
POST /api/v2/ml/predict                   # ML 預測
GET  /api/v2/news/{symbol}                # 新聞情感
POST /api/v2/alerts                       # 創建警報
POST /api/v2/portfolio/optimize           # 投資組合優化
POST /api/v2/backtest/run                 # 運行回測
POST /api/v2/batch/risk                   # 批次風險分析
```

---

## 📊 支持的技術指標

### 趨勢類
- SMA, EMA, WMA
- ADX, Aroon
- Parabolic SAR

### 動量類
- RSI, Stochastic
- MACD, CCI
- Williams %R

### 波動率
- ATR, Bollinger Bands
- Keltner Channels

### 成交量
- OBV, VWAP
- Volume Oscillator

---

## 🤖 機器學習模型

1. Random Forest
2. XGBoost
3. LightGBM
4. LSTM
5. Gradient Boosting
6. SVM
7. Ensemble

---

## 📚 文檔索引

| 文檔 | 描述 |
|------|------|
| [README.md](README.md) | 主文檔 |
| [GITHUB_SETUP.md](GITHUB_SETUP.md) | GitHub 上傳指南 |
| [PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md) | 整理總結 |
| [docs/ML_GUIDE.md](docs/ML_GUIDE.md) | ML 使用指南 |
| [docs/ML_IMPLEMENTATION_SUMMARY.md](docs/ML_IMPLEMENTATION_SUMMARY.md) | ML 實現細節 |

---

## ⚙️ 環境變量

創建 `.env` 文件：

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///quantpilot.db
API_RATE_LIMIT=100
```

---

## 🐳 Docker 部署

```bash
# 構建
docker-compose build

# 啟動
docker-compose up -d

# 停止
docker-compose down
```

---

## 🧪 運行測試

```bash
# 所有測試
pytest

# 特定測試
pytest tests/test_backend.py

# 覆蓋率
pytest --cov=modules
```

---

## 📦 依賴更新

```bash
# 更新所有包
pip install -U -r requirements.txt

# 生成新的 requirements.txt
pip freeze > requirements.txt
```

---

## 🔒 安全檢查清單

- [ ] 檢查 `.gitignore` 是否排除敏感文件
- [ ] 不要提交 `.env` 文件
- [ ] 不要提交 API 密鑰
- [ ] 不要提交數據庫文件
- [ ] 不要提交訓練模型（大文件）

---

## 🚀 GitHub 上傳（快速版）

```bash
# 1. 安裝 Git
# 從 https://git-scm.com 下載

# 2. 初始化
git init
git add .
git commit -m "Initial commit: QuantPilot v2.0"

# 3. 連接 GitHub
git remote add origin https://github.com/YOUR_USERNAME/QuantPilot.git
git push -u origin main
```

詳細步驟見 [GITHUB_SETUP.md](GITHUB_SETUP.md)

---

## 💡 快速命令

```bash
# 啟動開發服務器
python app.py

# 啟動（帶檢查）
python startup.py

# 清理緩存
python cleanup.py

# 運行測試
pytest

# 檢查代碼風格
flake8 modules/

# 格式化代碼
black modules/
```

---

## 🆘 常見問題

### Q: 端口 5000 被占用？
```python
# 在 app.py 中修改
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # 改為 5001
```

### Q: 找不到模組？
```bash
pip install -r requirements.txt
```

### Q: 數據庫錯誤？
```bash
# 刪除舊數據庫
rm quantpilot.db
# 重新啟動應用
python app.py
```

---

## 📞 獲取幫助

- 📖 查看完整文檔：[README.md](README.md)
- 🐛 報告問題：GitHub Issues
- 💬 討論：GitHub Discussions
- 📧 郵件：support@quantpilot.com

---

## 🎉 開始使用

```bash
# 克隆項目（從 GitHub）
git clone https://github.com/YOUR_USERNAME/QuantPilot.git
cd QuantPilot

# 安裝依賴
pip install -r requirements.txt

# 啟動
python app.py

# 打開瀏覽器
# http://localhost:5000
```

**就這麼簡單！** 🚀
