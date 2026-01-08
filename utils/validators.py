"""
輸入驗證模組
Input validation module using Pydantic
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, ValidationError
from datetime import datetime


class StockQuerySchema(BaseModel):
    """股票查詢參數驗證"""
    symbol: str = Field(..., min_length=1, max_length=10, description="股票代碼")
    period: str = Field(default="1y", description="時間週期")
    interval: str = Field(default="1d", description="時間間隔")
    suffix: Optional[str] = Field(default=None, description="股票代碼後綴")
    
    @validator('period')
    def validate_period(cls, v):
        allowed = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if v not in allowed:
            raise ValueError(f"period must be one of {allowed}")
        return v
    
    @validator('interval')
    def validate_interval(cls, v):
        allowed = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in allowed:
            raise ValueError(f"interval must be one of {allowed}")
        return v


class TechnicalAnalysisSchema(BaseModel):
    """技術分析參數驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    period: str = Field(default="1y")
    interval: str = Field(default="1d")
    indicators: List[str] = Field(default=["ma", "rsi", "macd"])
    suffix: Optional[str] = None
    
    @validator('indicators')
    def validate_indicators(cls, v):
        allowed = ["ma", "ema", "rsi", "macd", "bbands", "atr", "obv", "adx", "cci", "stoch"]
        for indicator in v:
            if indicator not in allowed:
                raise ValueError(f"indicator '{indicator}' not supported. Allowed: {allowed}")
        return v


class BacktestSchema(BaseModel):
    """回測參數驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    strategy: str = Field(default="ma_crossover")
    period: str = Field(default="2y")
    interval: str = Field(default="1d")
    initial_capital: float = Field(default=1_000_000, gt=0, le=1_000_000_000)
    params: Dict[str, Any] = Field(default_factory=dict)
    risk_params: Dict[str, Any] = Field(default_factory=dict)
    suffix: Optional[str] = None
    
    @validator('strategy')
    def validate_strategy(cls, v):
        allowed = [
            "ma_crossover", "ema_crossover", "rsi", "macd", 
            "bollinger", "breakout", "vwap_revert", "volume_breakout"
        ]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v
    
    @validator('initial_capital')
    def validate_capital(cls, v):
        if v < 10000:
            raise ValueError("initial_capital must be at least 10,000")
        return v


class BatchBacktestSchema(BaseModel):
    """批次回測參數驗證"""
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    strategy: str = Field(default="ma_crossover")
    period: str = Field(default="1y")
    interval: str = Field(default="1d")
    initial_capital: float = Field(default=1_000_000, gt=0)
    params: Dict[str, Any] = Field(default_factory=dict)
    risk_params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if len(v) > 50:
            raise ValueError("Maximum 50 symbols allowed in batch processing")
        unique_symbols = set(v)
        if len(unique_symbols) < len(v):
            raise ValueError("Duplicate symbols detected")
        return v


class PortfolioPositionSchema(BaseModel):
    """投資組合持倉驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    shares: float = Field(..., gt=0)
    buy_price: float = Field(..., gt=0)
    
    @validator('shares')
    def validate_shares(cls, v):
        if v <= 0:
            raise ValueError("shares must be positive")
        if v > 1_000_000:
            raise ValueError("shares exceeds reasonable limit")
        return v


class AlertSchema(BaseModel):
    """警報參數驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    condition: str = Field(..., description="警報條件: above, below, change_pct")
    target_value: float = Field(..., gt=0)
    note: Optional[str] = Field(default="", max_length=500)
    
    @validator('condition')
    def validate_condition(cls, v):
        allowed = ["above", "below", "change_pct", "volume_spike"]
        if v not in allowed:
            raise ValueError(f"condition must be one of {allowed}")
        return v


class PaperTradeSchema(BaseModel):
    """紙上交易驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    shares: int = Field(..., gt=0, le=1_000_000)
    action: str = Field(..., description="buy or sell")
    
    @validator('action')
    def validate_action(cls, v):
        if v not in ["buy", "sell"]:
            raise ValueError("action must be 'buy' or 'sell'")
        return v


class MLTrainingSchema(BaseModel):
    """機器學習訓練參數驗證"""
    symbol: str = Field(..., min_length=1, max_length=10)
    period: str = Field(default="2y")
    model_type: str = Field(default="random_forest")
    features: Optional[List[str]] = None
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed = ["random_forest", "gradient_boosting", "svm", "neural_network"]
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v


def validate_request_data(schema_class):
    """驗證請求數據的裝飾器"""
    def decorator(func):
        from functools import wraps
        from flask import request, jsonify
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 從請求中獲取數據
                if request.method == "GET":
                    data = request.args.to_dict()
                else:
                    data = request.get_json() or {}
                
                # 合併URL參數
                data.update(kwargs)
                
                # 驗證數據
                validated_data = schema_class(**data)
                
                # 將驗證後的數據傳遞給函數
                kwargs['validated_data'] = validated_data
                return func(*args, **kwargs)
                
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = " -> ".join(str(x) for x in error['loc'])
                    message = error['msg']
                    errors.append(f"{field}: {message}")
                
                return jsonify({
                    "success": False,
                    "error": "Validation error",
                    "details": errors
                }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 400
        
        return wrapper
    return decorator


def sanitize_symbol(symbol: str) -> str:
    """清理和驗證股票代碼"""
    if not symbol:
        raise ValueError("Symbol cannot be empty")
    
    # 移除空白字符
    symbol = symbol.strip().upper()
    
    # 移除特殊字符（除了點和連字符）
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    symbol = ''.join(c for c in symbol if c in allowed_chars)
    
    if not symbol:
        raise ValueError("Invalid symbol format")
    
    if len(symbol) > 10:
        raise ValueError("Symbol too long")
    
    return symbol


def validate_date_range(start_date: str, end_date: str) -> tuple:
    """驗證日期範圍"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start >= end:
            raise ValueError("start_date must be before end_date")
        
        if (end - start).days < 1:
            raise ValueError("Date range must be at least 1 day")
        
        if (end - start).days > 3650:  # 10 years
            raise ValueError("Date range cannot exceed 10 years")
        
        return start, end
    except ValueError as e:
        raise ValueError(f"Invalid date format or range: {str(e)}")
