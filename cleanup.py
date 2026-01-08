"""
專案清理腳本
Project Cleanup Script
"""
import os
import shutil

def cleanup_project():
    """清理專案中的重複文件和文件夾"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 要刪除的文件夾
    folders_to_remove = [
        "QuantPilot",      # 舊版本文件夾
        "quantlib",        # 重複的庫
        "__pycache__",     # Python 緩存
        ".pytest_cache"    # Pytest 緩存
    ]
    
    # 要刪除的文件
    files_to_remove = [
        "差異化功能",       # 未使用的文件
        "app_enhanced.py",  # 合併到 app.py
    ]
    
    print("🧹 開始清理專案...\n")
    
    # 刪除文件夾
    for folder in folders_to_remove:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"✓ 已刪除文件夾: {folder}")
            except Exception as e:
                print(f"✗ 無法刪除 {folder}: {e}")
    
    # 刪除文件
    for file in files_to_remove:
        file_path = os.path.join(base_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✓ 已刪除文件: {file}")
            except Exception as e:
                print(f"✗ 無法刪除 {file}: {e}")
    
    # 清理所有 __pycache__ 文件夾
    print("\n🔍 搜尋並清理所有 __pycache__ 文件夾...")
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            cache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_path)
                print(f"✓ 已刪除: {cache_path}")
            except Exception as e:
                print(f"✗ 無法刪除 {cache_path}: {e}")
    
    # 清理所有 .pyc 文件
    print("\n🔍 搜尋並清理所有 .pyc 文件...")
    pyc_count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                try:
                    os.remove(pyc_path)
                    pyc_count += 1
                except Exception as e:
                    print(f"✗ 無法刪除 {pyc_path}: {e}")
    
    if pyc_count > 0:
        print(f"✓ 已刪除 {pyc_count} 個 .pyc 文件")
    
    print("\n✨ 清理完成！")
    print("\n📁 當前專案結構:")
    print("quantpilot/")
    print("├── app.py              # 主應用")
    print("├── config.py           # 配置")
    print("├── startup.py          # 啟動腳本")
    print("├── requirements.txt    # 依賴")
    print("├── .gitignore          # Git 忽略")
    print("├── LICENSE             # 授權")
    print("├── README.md           # 主文檔")
    print("├── QUICKSTART.md       # 快速開始")
    print("│")
    print("├── modules/            # 核心模組")
    print("├── utils/              # 工具函數")
    print("├── static/             # 前端文件")
    print("├── tests/              # 測試文件")
    print("├── models/             # 訓練模型")
    print("├── data/               # 數據存儲")
    print("└── docs/               # 文檔")

if __name__ == "__main__":
    cleanup_project()
