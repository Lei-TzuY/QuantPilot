"""
錯誤處理模組
Error handling utilities and custom exceptions
"""
from flask import jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    """API基礎錯誤類"""
    status_code = 400
    
    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        rv = dict(self.payload or ())
        rv['success'] = False
        rv['error'] = self.message
        rv['error_type'] = self.__class__.__name__
        return rv


class ValidationError(APIError):
    """驗證錯誤"""
    status_code = 400


class ResourceNotFoundError(APIError):
    """資源未找到錯誤"""
    status_code = 404


class DataFetchError(APIError):
    """數據獲取錯誤"""
    status_code = 502


class BacktestError(APIError):
    """回測錯誤"""
    status_code = 500


class PortfolioError(APIError):
    """投資組合錯誤"""
    status_code = 400


class RateLimitError(APIError):
    """速率限制錯誤"""
    status_code = 429


class AuthenticationError(APIError):
    """認證錯誤"""
    status_code = 401


class AuthorizationError(APIError):
    """授權錯誤"""
    status_code = 403


def register_error_handlers(app):
    """註冊錯誤處理器到Flask應用"""
    
    @app.errorhandler(APIError)
    def handle_api_error(error):
        """處理自定義API錯誤"""
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        logger.error(
            f"API Error: {error.message} | "
            f"Path: {request.path} | "
            f"Method: {request.method}"
        )
        return response
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """處理HTTP異常"""
        response = jsonify({
            'success': False,
            'error': error.description,
            'error_type': 'HTTPException',
            'status_code': error.code
        })
        response.status_code = error.code
        logger.warning(
            f"HTTP Exception: {error.code} {error.description} | "
            f"Path: {request.path}"
        )
        return response
    
    @app.errorhandler(ValueError)
    def handle_value_error(error):
        """處理值錯誤"""
        response = jsonify({
            'success': False,
            'error': str(error),
            'error_type': 'ValueError'
        })
        response.status_code = 400
        logger.error(f"ValueError: {str(error)} | Path: {request.path}")
        return response
    
    @app.errorhandler(KeyError)
    def handle_key_error(error):
        """處理鍵錯誤"""
        response = jsonify({
            'success': False,
            'error': f"Missing required field: {str(error)}",
            'error_type': 'KeyError'
        })
        response.status_code = 400
        logger.error(f"KeyError: {str(error)} | Path: {request.path}")
        return response
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """處理未預期的錯誤"""
        error_trace = traceback.format_exc()
        logger.critical(
            f"Unexpected error: {str(error)} | "
            f"Path: {request.path} | "
            f"Traceback: {error_trace}"
        )
        
        # 在生產環境不返回詳細錯誤信息
        if app.config.get('DEBUG'):
            response = jsonify({
                'success': False,
                'error': str(error),
                'error_type': type(error).__name__,
                'traceback': error_trace
            })
        else:
            response = jsonify({
                'success': False,
                'error': 'An unexpected error occurred',
                'error_type': 'InternalServerError'
            })
        
        response.status_code = 500
        return response
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """處理404錯誤"""
        response = jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'error_type': 'NotFound',
            'path': request.path
        })
        response.status_code = 404
        return response
    
    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """處理405錯誤"""
        response = jsonify({
            'success': False,
            'error': f'Method {request.method} not allowed for this endpoint',
            'error_type': 'MethodNotAllowed',
            'allowed_methods': error.valid_methods
        })
        response.status_code = 405
        return response


def error_response(message, status_code=400, **kwargs):
    """創建標準錯誤響應"""
    response_data = {
        'success': False,
        'error': message,
        **kwargs
    }
    return jsonify(response_data), status_code


def success_response(data=None, message=None, **kwargs):
    """創建標準成功響應"""
    response_data = {
        'success': True,
        **kwargs
    }
    
    if message:
        response_data['message'] = message
    
    if data is not None:
        if isinstance(data, dict):
            response_data.update(data)
        else:
            response_data['data'] = data
    
    return jsonify(response_data)


class ErrorLogger:
    """錯誤日誌記錄器"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化Flask應用"""
        register_error_handlers(app)
