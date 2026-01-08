# 快速入門指南 Quick Start Guide

## 🚀 5分鐘快速啟動

### Windows 用戶

1. **雙擊運行 `run.bat`**
   ```
   只需雙擊 run.bat 文件，腳本會自動處理一切！
   ```

2. **等待安裝完成**
   - 創建虛擬環境
   - 安裝依賴包
   - 初始化數據庫
   - 啟動服務器

3. **打開瀏覽器**
   ```
   訪問: http://localhost:5000
   ```

### Linux/Mac 用戶

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 運行啟動腳本
python startup.py --enhanced

# 3. 打開瀏覽器訪問 http://localhost:5000
```

## 📖 第一次使用

### 1. 查看熱門股票

進入首頁後，你會看到熱門台股列表，包括：
- 台積電 (2330)
- 鴻海 (2317)
- 聯發科 (2454)
等等...

### 2. 查詢股票數據

在搜索框輸入股票代碼（例如：2330），即可查看：
- 📊 歷史價格走勢圖
- 📈 技術指標（MA, RSI, MACD等）
- 📰 相關新聞
- 💡 交易信號

### 3. 運行回測

點擊"回測"標籤：
1. 輸入股票代碼
2. 選擇策略（如：均線交叉）
3. 設定初始資金
4. 點擊"運行回測"

系統會顯示：
- 總收益率
- 夏普比率
- 最大回撤
- 交易次數
- 勝率
- 資金曲線圖

### 4. 管理投資組合

在"投資組合"標籤：
1. 添加持倉（股票代碼、股數、買入價）
2. 查看總收益
3. 追蹤每支股票表現

### 5. 設置價格警報

在"警報"標籤：
1. 輸入股票代碼
2. 選擇條件（高於/低於）
3. 設定目標價格
4. 系統會自動監控並在觸發時通知

### 6. 紙上交易

在"紙上交易"標籤：
1. 初始有 100萬 虛擬資金
2. 模擬買入/賣出股票
3. 追蹤交易記錄和績效
4. 無風險練習交易策略

## 🎯 常見使用案例

### 案例1: 分析台積電近期表現

```
1. 首頁搜索框輸入: 2330
2. 選擇時間週期: 3個月
3. 查看技術指標: MA, RSI, MACD
4. 查看交易信號: 買入/賣出建議
```

### 案例2: 測試均線交叉策略

```
1. 進入"回測"頁面
2. 股票代碼: 2330
3. 策略: ma_crossover
4. 參數: 短期=20, 長期=60
5. 初始資金: 1,000,000
6. 運行並查看結果
```

### 案例3: 優化策略參數

```
1. 進入"回測"頁面
2. 選擇"參數優化"
3. 設定參數範圍:
   - 短期: 10-30 (步長5)
   - 長期: 40-80 (步長10)
4. 運行優化
5. 查看最佳參數組合
```

### 案例4: 批次分析多支股票

```
API調用範例:
POST /api/batch/backtest
{
  "symbols": ["2330", "2317", "2454"],
  "strategy": "ma_crossover",
  "period": "1y",
  "initial_capital": 1000000
}
```

## 🔍 API 使用範例

### Python 腳本範例

```python
import requests

BASE_URL = "http://localhost:5000/api"

# 1. 獲取股票數據
def get_stock_data(symbol):
    response = requests.get(f"{BASE_URL}/stock/{symbol}")
    return response.json()

# 2. 獲取技術分析
def get_analysis(symbol):
    response = requests.get(
        f"{BASE_URL}/analysis/{symbol}",
        params={"period": "6mo", "indicators": "ma,rsi,macd"}
    )
    return response.json()

# 3. 運行回測
def run_backtest(symbol, strategy="ma_crossover"):
    data = {
        "symbol": symbol,
        "strategy": strategy,
        "period": "2y",
        "initial_capital": 1000000,
        "params": {
            "short_window": 20,
            "long_window": 60
        }
    }
    response = requests.post(f"{BASE_URL}/backtest", json=data)
    return response.json()

# 4. 創建警報
def create_alert(symbol, target_price):
    data = {
        "symbol": symbol,
        "condition": "above",
        "target_value": target_price,
        "note": f"目標價 {target_price}"
    }
    response = requests.post(f"{BASE_URL}/alerts", json=data)
    return response.json()

# 使用範例
if __name__ == "__main__":
    # 分析台積電
    result = get_analysis("2330")
    print("RSI:", result['analysis']['rsi'][-1])
    
    # 運行回測
    backtest = run_backtest("2330")
    print("總收益:", backtest['result']['total_return_pct'], "%")
    
    # 設置警報
    alert = create_alert("2330", 600)
    print("警報已創建:", alert['alert']['alert_id'])
```

### JavaScript 範例

```javascript
// 獲取股票數據並顯示
async function displayStockData(symbol) {
  try {
    const response = await fetch(`/api/stock/${symbol}`);
    const data = await response.json();
    
    if (data.success) {
      console.log('價格數據:', data.data);
      // 繪製圖表...
    }
  } catch (error) {
    console.error('錯誤:', error);
  }
}

// 運行回測
async function runBacktest(symbol) {
  const backtestData = {
    symbol: symbol,
    strategy: 'ma_crossover',
    period: '2y',
    initial_capital: 1000000
  };
  
  const response = await fetch('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(backtestData)
  });
  
  const result = await response.json();
  console.log('回測結果:', result.result);
}
```

## ⚙️ 配置說明

### 基本配置

編輯 `.env` 文件：

```env
# 台灣股票使用 .TW 後綴
DEFAULT_SUFFIX=.TW

# 手續費率 (台股一般為 0.1425%)
DEFAULT_FEE_PCT=0.001425

# 初始資金
DEFAULT_INITIAL_CAPITAL=1000000
```

### 高級配置

```env
# 啟用速率限制
RATELIMIT_ENABLED=True
RATELIMIT_DEFAULT=200 per hour

# 日誌級別 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# 警報檢查間隔（秒）
ALERT_CHECK_INTERVAL=60
```

## 🐛 常見問題

### 問題1: 無法獲取股票數據

**可能原因:**
- 股票代碼錯誤
- 網絡連接問題
- Yahoo Finance API 暫時不可用

**解決方案:**
- 確認股票代碼正確（台股需加 .TW 後綴）
- 檢查網絡連接
- 稍後重試

### 問題2: 回測運行緩慢

**可能原因:**
- 數據量過大
- 參數優化範圍太大

**解決方案:**
- 減小時間週期
- 縮小優化參數範圍
- 使用更快的時間間隔（如1d而非1h）

### 問題3: 警報沒有觸發

**可能原因:**
- 警報檢查線程未啟動
- 目標價格設置不合理

**解決方案:**
- 重啟應用
- 檢查警報條件設置
- 查看日誌文件

### 問題4: 依賴包安裝失敗

**解決方案:**
```bash
# 升級 pip
python -m pip install --upgrade pip

# 逐個安裝核心包
pip install flask pandas yfinance ta

# 如果 TA-Lib 安裝失敗，可以跳過
# 系統會使用 ta 庫替代
```

## 📚 下一步

- 📖 閱讀完整文檔: [README_ENHANCED.md](README_ENHANCED.md)
- 🔍 查看 API 文檔
- 💡 探索更多策略
- 🤝 參與貢獻

## 💬 獲取幫助

如有問題：
1. 查看日誌文件 `logs/app.log`
2. 檢查控制台輸出
3. 開啟 GitHub Issue
4. 查看常見問題文檔

---

<div align="center">

**祝交易順利！Happy Trading! 📈**

</div>
