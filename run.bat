@echo off
chcp 65001 >nul
echo ====================================
echo 量化交易系統啟動器
echo Quantitative Trading System Launcher
echo ====================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 未找到 Python，請先安裝 Python 3.8+
    pause
    exit /b 1
)

echo [信息] Python 版本:
python --version
echo.

REM 檢查是否需要安裝依賴
if not exist "venv\" (
    echo [信息] 首次運行，正在創建虛擬環境...
    python -m venv venv
    echo [完成] 虛擬環境已創建
    echo.
)

REM 啟動虛擬環境
call venv\Scripts\activate.bat

REM 安裝/更新依賴
echo [信息] 檢查並安裝依賴套件...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [警告] 某些套件安裝失敗，請檢查 requirements.txt
)
echo.

REM 運行啟動腳本
echo [信息] 啟動應用程式...
echo.
python startup.py --enhanced

REM 如果出錯，暫停以便查看錯誤信息
if errorlevel 1 (
    echo.
    echo [錯誤] 應用啟動失敗
    pause
)
