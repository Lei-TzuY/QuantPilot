"""
應用程式配置文件
Configuration file for the trading application
"""
import os
from datetime import timedelta


class Config:
    """基礎配置"""
    # Flask設定
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5000"))
    
    # CORS設定
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # 速率限制設定
    RATELIMIT_ENABLED = os.getenv("RATELIMIT_ENABLED", "True").lower() == "true"
    RATELIMIT_STORAGE_URL = os.getenv("RATELIMIT_STORAGE_URL", "memory://")
    RATELIMIT_DEFAULT = os.getenv("RATELIMIT_DEFAULT", "200 per hour")
    RATELIMIT_STRATEGY = "fixed-window"
    
    # 快取設定
    CACHE_TYPE = os.getenv("CACHE_TYPE", "simple")  # simple, redis, memcached
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", "300"))  # 5分鐘
    CACHE_KEY_PREFIX = "quantapp:"
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # 數據庫設定
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        "sqlite:///data/trading.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    
    # 交易參數設定
    DEFAULT_PERIOD = "1y"
    DEFAULT_INTERVAL = "1d"
    DEFAULT_SUFFIX = ".TW"
    DEFAULT_INITIAL_CAPITAL = 1_000_000
    DEFAULT_FEE_PCT = 0.001425  # 0.1425%
    DEFAULT_SLIPPAGE_PCT = 0.001
    
    # WebSocket設定
    SOCKETIO_MESSAGE_QUEUE = os.getenv("SOCKETIO_MESSAGE_QUEUE", None)
    SOCKETIO_ASYNC_MODE = "threading"
    
    # API設定
    API_VERSION = "v1"
    API_TITLE = "QuantPilot Trading API"
    API_DESCRIPTION = "量化交易系統API - Quantitative Trading System API"
    
    # 日誌設定
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.getenv("LOG_FILE", "data/app.log")
    
    # 驗證參數
    ALLOWED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    ALLOWED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    
    # 批次處理設定
    BATCH_MAX_SYMBOLS = int(os.getenv("BATCH_MAX_SYMBOLS", "50"))
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", "300"))  # 5分鐘
    
    # 警報設定
    ALERT_CHECK_INTERVAL = int(os.getenv("ALERT_CHECK_INTERVAL", "60"))  # 秒
    MAX_ALERTS_PER_USER = int(os.getenv("MAX_ALERTS_PER_USER", "100"))
    
    # 紙上交易設定
    PAPER_TRADING_INITIAL_BALANCE = float(os.getenv("PAPER_TRADING_INITIAL_BALANCE", "1000000"))
    PAPER_TRADING_FEE_PCT = float(os.getenv("PAPER_TRADING_FEE_PCT", "0.001425"))


class DevelopmentConfig(Config):
    """開發環境配置"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """生產環境配置"""
    DEBUG = False
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    """測試環境配置"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    RATELIMIT_ENABLED = False
    WTF_CSRF_ENABLED = False


# 根據環境變數選擇配置
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig
}


def get_config(env=None):
    """獲取配置對象"""
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
