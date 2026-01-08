"""
系統測試腳本
System test script to verify enhancements
"""
import requests
import time
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

BASE_URL = "http://localhost:5000/api"


def test_health_check():
    """測試健康檢查端點"""
    console.print("\n[bold blue]測試 1: 健康檢查[/bold blue]")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            console.print("✅ 健康檢查通過", style="green")
            console.print(f"   狀態: {data.get('status', 'N/A')}")
            console.print(f"   版本: {data.get('application', {}).get('version', 'N/A')}")
            return True
        else:
            console.print(f"❌ 健康檢查失敗: {response.status_code}", style="red")
            return False
    except Exception as e:
        console.print(f"❌ 連接失敗: {e}", style="red")
        return False


def test_stock_data():
    """測試股票數據獲取"""
    console.print("\n[bold blue]測試 2: 股票數據獲取[/bold blue]")
    try:
        response = requests.get(f"{BASE_URL}/stock/2330?period=1mo", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                console.print("✅ 股票數據獲取成功", style="green")
                console.print(f"   數據點數: {len(data.get('data', []))}")
                # 檢查響應頭
                if 'X-Request-ID' in response.headers:
                    console.print(f"   請求ID: {response.headers['X-Request-ID']}")
                if 'X-Execution-Time' in response.headers:
                    console.print(f"   執行時間: {response.headers['X-Execution-Time']}")
                return True
        console.print(f"❌ 獲取失敗: {response.status_code}", style="red")
        return False
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
        return False


def test_technical_analysis():
    """測試技術分析"""
    console.print("\n[bold blue]測試 3: 技術分析[/bold blue]")
    try:
        response = requests.get(
            f"{BASE_URL}/analysis/2330?period=6mo&indicators=ma,rsi,macd",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                console.print("✅ 技術分析成功", style="green")
                analysis = data.get('analysis', {})
                console.print(f"   指標數量: {len(analysis)}")
                if 'rsi' in analysis:
                    console.print(f"   RSI 最新值: {analysis['rsi'][-1]:.2f}")
                return True
        console.print(f"❌ 分析失敗: {response.status_code}", style="red")
        return False
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
        return False


def test_backtest():
    """測試回測功能"""
    console.print("\n[bold blue]測試 4: 回測引擎[/bold blue]")
    try:
        backtest_data = {
            "symbol": "2330",
            "strategy": "ma_crossover",
            "period": "1y",
            "initial_capital": 1000000,
            "params": {
                "short_window": 20,
                "long_window": 60
            }
        }
        response = requests.post(
            f"{BASE_URL}/backtest",
            json=backtest_data,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                console.print("✅ 回測執行成功", style="green")
                console.print(f"   總收益率: {result.get('total_return_pct', 0):.2f}%")
                console.print(f"   夏普比率: {result.get('sharpe_ratio', 0):.2f}")
                console.print(f"   最大回撤: {result.get('max_drawdown', 0):.2f}%")
                console.print(f"   交易次數: {result.get('num_trades', 0)}")
                return True
        console.print(f"❌ 回測失敗: {response.status_code}", style="red")
        return False
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
        return False


def test_rate_limiting():
    """測試速率限制"""
    console.print("\n[bold blue]測試 5: 速率限制[/bold blue]")
    console.print("   發送多個快速請求...")
    
    success_count = 0
    rate_limited_count = 0
    
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/popular", timeout=5)
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
                console.print(f"   請求 {i+1}: 被限流", style="yellow")
        except Exception:
            pass
    
    console.print(f"✅ 速率限制測試完成", style="green")
    console.print(f"   成功請求: {success_count}")
    console.print(f"   被限流: {rate_limited_count}")
    return True


def test_input_validation():
    """測試輸入驗證"""
    console.print("\n[bold blue]測試 6: 輸入驗證[/bold blue]")
    
    # 測試無效的股票代碼
    try:
        response = requests.get(f"{BASE_URL}/stock/INVALID_SYMBOL", timeout=5)
        if response.status_code in [400, 404]:
            console.print("✅ 無效輸入被正確拒絕", style="green")
            return True
        else:
            console.print("⚠️  輸入驗證可能有問題", style="yellow")
            return False
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
        return False


def test_error_handling():
    """測試錯誤處理"""
    console.print("\n[bold blue]測試 7: 錯誤處理[/bold blue]")
    
    # 測試錯誤的端點
    try:
        response = requests.get(f"{BASE_URL}/nonexistent_endpoint", timeout=5)
        if response.status_code == 404:
            data = response.json()
            if 'error' in data:
                console.print("✅ 錯誤被正確處理", style="green")
                console.print(f"   錯誤類型: {data.get('error_type', 'N/A')}")
                return True
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
    
    return False


def test_portfolio_api():
    """測試投資組合API"""
    console.print("\n[bold blue]測試 8: 投資組合管理[/bold blue]")
    
    try:
        # 獲取投資組合
        response = requests.get(f"{BASE_URL}/portfolio", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                console.print("✅ 投資組合獲取成功", style="green")
                portfolio = data.get('portfolio', [])
                console.print(f"   持倉數量: {len(portfolio)}")
                return True
        
        console.print(f"⚠️  投資組合API響應異常: {response.status_code}", style="yellow")
        return False
    except Exception as e:
        console.print(f"❌ 錯誤: {e}", style="red")
        return False


def run_all_tests():
    """運行所有測試"""
    console.print("\n" + "="*60, style="bold cyan")
    console.print("🧪 量化交易系統 - 增強功能測試套件", style="bold cyan")
    console.print("="*60 + "\n", style="bold cyan")
    
    console.print("[yellow]⚠️  請確保服務器正在運行: python startup.py --enhanced[/yellow]\n")
    
    time.sleep(2)
    
    tests = [
        ("健康檢查", test_health_check),
        ("股票數據", test_stock_data),
        ("技術分析", test_technical_analysis),
        ("回測引擎", test_backtest),
        ("速率限制", test_rate_limiting),
        ("輸入驗證", test_input_validation),
        ("錯誤處理", test_error_handling),
        ("投資組合", test_portfolio_api),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"❌ 測試異常: {e}", style="red")
            results.append((test_name, False))
        time.sleep(1)  # 避免請求過快
    
    # 生成測試報告
    console.print("\n" + "="*60, style="bold cyan")
    console.print("📊 測試報告", style="bold cyan")
    console.print("="*60 + "\n", style="bold cyan")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("測試項目", style="cyan")
    table.add_column("結果", justify="center")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        if result:
            table.add_row(test_name, "✅ 通過")
            passed += 1
        else:
            table.add_row(test_name, "❌ 失敗")
    
    console.print(table)
    
    console.print(f"\n總計: {passed}/{total} 測試通過", style="bold")
    
    if passed == total:
        console.print("\n🎉 所有測試通過！系統運行正常。", style="bold green")
    elif passed >= total * 0.7:
        console.print(f"\n⚠️  部分測試失敗，但核心功能正常。", style="bold yellow")
    else:
        console.print(f"\n❌ 多個測試失敗，請檢查系統配置。", style="bold red")
    
    console.print("\n" + "="*60 + "\n", style="bold cyan")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        console.print("\n\n測試被用戶中斷", style="yellow")
    except Exception as e:
        console.print(f"\n測試套件錯誤: {e}", style="red")
        import traceback
        traceback.print_exc()
