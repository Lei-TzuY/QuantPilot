"""
日誌記錄工具
Logging utilities for the application
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import uuid
from flask import request, g
from functools import wraps
import time


def setup_logger(name, log_file=None, level=logging.INFO, format_string=None):
    """設置記錄器
    
    Args:
        name: 記錄器名稱
        log_file: 日誌文件路徑
        level: 日誌級別
        format_string: 日誌格式字符串
    
    Returns:
        配置好的記錄器
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    formatter = logging.Formatter(format_string)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重複添加處理器
    if logger.handlers:
        return logger
    
    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件處理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用輪轉文件處理器（按大小）
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_request_id():
    """獲取或生成請求ID"""
    if not hasattr(g, 'request_id'):
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    return g.request_id


class RequestIDFilter(logging.Filter):
    """添加請求ID到日誌記錄"""
    def filter(self, record):
        try:
            record.request_id = get_request_id()
        except RuntimeError:
            # 不在請求上下文中
            record.request_id = 'N/A'
        return True


def add_request_id_filter(logger):
    """為記錄器添加請求ID過濾器"""
    request_id_filter = RequestIDFilter()
    for handler in logger.handlers:
        handler.addFilter(request_id_filter)


def log_execution_time(logger=None):
    """記錄函數執行時間的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                _logger.info(
                    f"Function '{func.__name__}' executed in {execution_time:.4f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.4f}s: {str(e)}"
                )
                raise
        return wrapper
    return decorator


def log_api_call(logger=None):
    """記錄API調用的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            start_time = time.time()
            request_id = get_request_id()
            
            _logger.info(
                f"API Call [{request_id}]: {request.method} {request.path} "
                f"from {request.remote_addr}"
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                status_code = 200
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, tuple) and len(result) > 1:
                    status_code = result[1]
                
                _logger.info(
                    f"API Response [{request_id}]: {status_code} "
                    f"in {execution_time:.4f}s"
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"API Error [{request_id}]: {str(e)} "
                    f"after {execution_time:.4f}s",
                    exc_info=True
                )
                raise
                
        return wrapper
    return decorator


class APILogger:
    """API請求日誌記錄器"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化Flask應用"""
        
        @app.before_request
        def before_request():
            g.start_time = time.time()
            g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
        @app.after_request
        def after_request(response):
            if hasattr(g, 'start_time'):
                execution_time = time.time() - g.start_time
                response.headers['X-Request-ID'] = g.request_id
                response.headers['X-Execution-Time'] = f"{execution_time:.4f}s"
            return response


def configure_app_logging(app):
    """配置Flask應用的日誌"""
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    log_file = app.config.get('LOG_FILE')
    
    # 配置應用記錄器
    app.logger.handlers.clear()
    logger = setup_logger(
        'trading_app',
        log_file=log_file,
        level=log_level
    )
    
    # 添加請求ID過濾器
    add_request_id_filter(logger)
    
    # 設置Flask應用使用相同的記錄器
    app.logger = logger
    
    # 初始化API日誌記錄器
    api_logger = APILogger(app)
    
    return logger
