@echo off
chcp 65001 >nul
echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║                 QuantPilot Git 設置腳本                    ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo [1/6] 檢查 Git 安裝...
git --version
if errorlevel 1 (
    echo.
    echo ❌ Git 未找到！請確認：
    echo    1. Git 已完成安裝
    echo    2. 已重啟終端
    echo.
    pause
    exit /b 1
)
echo ✅ Git 已安裝
echo.

echo [2/6] 配置 Git 用戶信息...
set /p USERNAME="請輸入您的 GitHub 用戶名: "
set /p EMAIL="請輸入您的 GitHub 郵箱: "
git config --global user.name "%USERNAME%"
git config --global user.email "%EMAIL%"
echo ✅ Git 配置完成
echo.

echo [3/6] 初始化 Git 倉庫...
git init
echo ✅ Git 倉庫已初始化
echo.

echo [4/6] 添加所有文件...
git add .
echo ✅ 文件已添加
echo.

echo [5/6] 創建初始提交...
git commit -m "Initial commit: QuantPilot v2.0 - 量化交易系統"
echo ✅ 提交已創建
echo.

echo [6/6] 準備推送到 GitHub...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 下一步：在 GitHub 上創建倉庫
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 1. 前往: https://github.com/new
echo 2. Repository name: QuantPilot
echo 3. Description: 量化交易系統 v2.0 - 基於機器學習的股票分析平台
echo 4. 選擇 Public 或 Private
echo 5. 不要勾選 "Initialize this repository with a README"
echo 6. 點擊 "Create repository"
echo.
echo 創建完成後，運行以下命令（替換 YOUR_USERNAME）:
echo.
echo git remote add origin https://github.com/YOUR_USERNAME/QuantPilot.git
echo git branch -M main
echo git push -u origin main
echo.
pause
