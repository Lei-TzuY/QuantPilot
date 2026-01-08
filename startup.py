#!/usr/bin/env python
"""
啟動腳本 - 初始化並運行量化交易系統
Startup script - Initialize and run the quantitative trading system
"""
import os
import sys
import argparse
from pathlib import Path

def setup_directories():
    """創建必要的目錄"""
    directories = ['data', 'logs', 'static', 'modules', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ 目錄已確認: {directory}/")

def init_database():
    """初始化資料庫"""
    try:
        from models import init_database
        db = init_database()
        print("✓ 資料庫初始化成功")
        return True
    except Exception as e:
        print(f"✗ 資料庫初始化失敗: {e}")
        return False

def check_dependencies():
    """檢查依賴套件"""
    required_packages = [
        'flask', 'flask_cors', 'flask_limiter', 
        'pandas', 'numpy', 'yfinance', 'sqlalchemy',
        'pydantic', 'ta'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("✗ 缺少以下依賴套件:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n請執行: pip install -r requirements.txt")
        return False
    
    print("✓ 所有依賴套件已安裝")
    return True

def create_env_file():
    """創建 .env 文件（如果不存在）"""
    if not Path('.env').exists():
        if Path('.env.example').exists():
            import shutil
            shutil.copy('.env.example', '.env')
            print("✓ 已從 .env.example 創建 .env 文件")
        else:
            print("! 警告: .env 文件不存在，將使用默認配置")

def main():
    parser = argparse.ArgumentParser(description='量化交易系統啟動器')
    parser.add_argument('--env', default='development', 
                       choices=['development', 'production', 'testing'],
                       help='運行環境')
    parser.add_argument('--host', default='0.0.0.0', help='主機地址')
    parser.add_argument('--port', type=int, default=5000, help='端口號')
    parser.add_argument('--enhanced', action='store_true', 
                       help='使用增強版應用 (app_enhanced.py)')
    parser.add_argument('--init-only', action='store_true',
                       help='僅初始化，不啟動服務器')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 量化交易系統 Quantitative Trading System")
    print("=" * 70)
    
    # 設置環境變數
    os.environ['FLASK_ENV'] = args.env
    os.environ['HOST'] = args.host
    os.environ['PORT'] = str(args.port)
    
    # 步驟1: 檢查依賴
    print("\n📦 步驟 1/4: 檢查依賴套件...")
    if not check_dependencies():
        sys.exit(1)
    
    # 步驟2: 創建目錄
    print("\n📁 步驟 2/4: 設置目錄結構...")
    setup_directories()
    
    # 步驟3: 創建環境配置
    print("\n⚙️  步驟 3/4: 配置環境...")
    create_env_file()
    
    # 步驟4: 初始化資料庫
    print("\n💾 步驟 4/4: 初始化資料庫...")
    if not init_database():
        print("! 警告: 資料庫初始化失敗，但將繼續運行...")
    
    print("\n" + "=" * 70)
    print("✅ 初始化完成！")
    print("=" * 70)
    
    if args.init_only:
        print("\n僅執行初始化，服務器未啟動。")
        print("要啟動服務器，請執行: python startup.py")
        return
    
    # 啟動應用
    print(f"\n🌐 環境: {args.env}")
    print(f"🌐 地址: http://{args.host}:{args.port}")
    print(f"📝 應用: {'增強版 Enhanced' if args.enhanced else '標準版 Standard'}")
    print("=" * 70)
    print("\n按 Ctrl+C 停止服務器\n")
    
    try:
        if args.enhanced:
            # 使用增強版應用
            if not Path('app_enhanced.py').exists():
                print("✗ 找不到 app_enhanced.py")
                sys.exit(1)
            from app_enhanced import create_app
            app = create_app(args.env)
        else:
            # 使用標準應用
            if Path('app.py').exists():
                import app as app_module
                app = app_module.app
            else:
                print("✗ 找不到 app.py")
                sys.exit(1)
        
        app.run(
            debug=(args.env == 'development'),
            host=args.host,
            port=args.port
        )
    
    except KeyboardInterrupt:
        print("\n\n👋 服務器已停止")
    except Exception as e:
        print(f"\n✗ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
