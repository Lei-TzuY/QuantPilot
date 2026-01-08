# GitHub 上傳指南

## 📋 專案整理狀態

✅ **已完成的工作：**
- 刪除重複文件夾 (QuantPilot, quantlib)
- 清理所有 `__pycache__` 目錄和 `.pyc` 文件
- 創建專業的 `.gitignore` 文件
- 重寫 README.md 文檔
- 創建 MIT 授權文件
- 整理所有文檔到 `docs/` 目錄
- 創建 `data/` 和 `models/` 目錄結構

---

## 🔧 步驟 1: 安裝 Git

如果您還沒有安裝 Git，請按照以下步驟操作：

### Windows:
1. 前往 [Git for Windows](https://git-scm.com/download/win)
2. 下載並安裝
3. 安裝完成後，重新啟動終端

### 驗證安裝:
```bash
git --version
```

---

## 🚀 步驟 2: 初始化 Git 倉庫

在專案根目錄執行：

```bash
# 初始化 Git 倉庫
git init

# 配置用戶信息（首次使用）
git config --global user.name "您的名字"
git config --global user.email "您的郵箱"

# 添加所有文件
git add .

# 創建初始提交
git commit -m "Initial commit: QuantPilot v2.0 - 量化交易系統"
```

---

## 📦 步驟 3: 在 GitHub 上創建倉庫

1. 前往 [GitHub](https://github.com)
2. 點擊右上角 **"+"** → **"New repository"**
3. 填寫倉庫信息：
   - **Repository name**: `QuantPilot`
   - **Description**: `量化交易系統 v2.0 - 基於機器學習的股票分析平台`
   - **Visibility**: 選擇 Public 或 Private
   - ⚠️ **不要** 勾選 "Initialize this repository with a README"
4. 點擊 **"Create repository"**

---

## 🌐 步驟 4: 推送到 GitHub

在創建倉庫後，GitHub 會顯示一組命令。執行以下命令（替換為您的倉庫 URL）：

```bash
# 添加遠程倉庫（替換為您的 GitHub 用戶名）
git remote add origin https://github.com/YOUR_USERNAME/QuantPilot.git

# 推送到 GitHub
git push -u origin main
```

**注意：** 如果默認分支是 `master` 而不是 `main`，請使用：
```bash
git branch -M main
git push -u origin main
```

---

## 🔑 步驟 5: 配置 GitHub 認證（可選）

### 使用 Personal Access Token (推薦):

1. 前往 GitHub Settings → Developer settings → Personal access tokens
2. 點擊 "Generate new token (classic)"
3. 勾選必要權限：
   - `repo` (完整倉庫訪問)
   - `workflow` (如果使用 GitHub Actions)
4. 生成並保存 token
5. 推送時使用 token 作為密碼

---

## 📝 步驟 6: 後續更新

當您對專案進行修改後，使用以下命令更新：

```bash
# 查看修改狀態
git status

# 添加修改的文件
git add .

# 提交修改
git commit -m "描述您的修改"

# 推送到 GitHub
git push
```

---

## 📂 當前專案結構

```
QuantPilot/
├── app.py                  # Flask 主應用
├── requirements.txt        # Python 依賴
├── .gitignore              # Git 忽略文件
├── LICENSE                 # MIT 授權
├── README.md               # 主文檔
├── Dockerfile              # Docker 配置
├── docker-compose.yml      # Docker Compose 配置
├── research_demo.ipynb     # 研究示例
│
├── modules/                # 核心模組
│   ├── data_fetcher.py     # 數據獲取
│   ├── technical_analysis.py  # 技術分析
│   ├── ml_signal.py        # ML 信號生成
│   ├── portfolio_manager.py   # 投資組合管理
│   ├── alert_manager.py    # 警報管理
│   ├── backtester.py       # 回測引擎
│   └── ...
│
├── static/                 # 前端文件
│   ├── index.html
│   ├── css/
│   └── js/
│
├── tests/                  # 測試文件
│   ├── test_backend.py
│   └── test_risk_batch.py
│
├── docs/                   # 文檔目錄
│   ├── ML_GUIDE.md
│   ├── ML_IMPLEMENTATION_SUMMARY.md
│   └── ...
│
├── data/                   # 數據存儲（已排除 Git）
└── models/                 # 模型存儲（已排除 Git）
```

---

## 🛡️ 安全建議

1. **不要提交敏感信息**：
   - API 密鑰
   - 數據庫密碼
   - 個人數據

2. **使用 .env 文件**：
   - 創建 `.env` 文件存儲配置
   - 已在 `.gitignore` 中排除

3. **檢查提交內容**：
   ```bash
   git diff
   ```

---

## 📧 需要幫助？

- GitHub 文檔: https://docs.github.com/
- Git 教程: https://git-scm.com/book/zh/v2

---

## ✅ 檢查清單

- [ ] 安裝 Git
- [ ] 初始化本地倉庫
- [ ] 在 GitHub 上創建倉庫
- [ ] 配置遠程倉庫
- [ ] 首次推送
- [ ] 驗證 GitHub 上的內容

---

**注意：** 所有重複文件和緩存已清理完畢，專案結構已優化，可以安全推送到 GitHub！
