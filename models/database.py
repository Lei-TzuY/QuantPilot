"""
資料庫模型
Database models for persistence layer
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()


class Portfolio(Base):
    """投資組合持倉模型"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    shares = Column(Float, nullable=False)
    buy_price = Column(Float, nullable=False)
    current_price = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    profit_loss = Column(Float, default=0.0)
    profit_loss_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'shares': self.shares,
            'buy_price': self.buy_price,
            'current_price': self.current_price,
            'total_cost': self.total_cost,
            'current_value': self.current_value,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Alert(Base):
    """價格警報模型"""
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    alert_id = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    condition = Column(String(20), nullable=False)  # above, below, change_pct, volume_spike
    target_value = Column(Float, nullable=False)
    current_value = Column(Float, default=0.0)
    note = Column(Text, default='')
    is_triggered = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'alert_id': self.alert_id,
            'symbol': self.symbol,
            'condition': self.condition,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'note': self.note,
            'is_triggered': self.is_triggered,
            'is_active': self.is_active,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class Trade(Base):
    """交易記錄模型"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    action = Column(String(10), nullable=False)  # buy, sell
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    trade_type = Column(String(20), default='paper')  # paper, real
    status = Column(String(20), default='completed')  # pending, completed, cancelled
    notes = Column(Text, default='')
    executed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'total_amount': self.total_amount,
            'fee': self.fee,
            'trade_type': self.trade_type,
            'status': self.status,
            'notes': self.notes,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class BacktestResult(Base):
    """回測結果模型"""
    __tablename__ = 'backtest_results'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    period = Column(String(10), nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_value = Column(Float, nullable=False)
    total_return = Column(Float, default=0.0)
    total_return_pct = Column(Float, default=0.0)
    annual_return_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    parameters = Column(Text, default='{}')  # JSON string
    metrics = Column(Text, default='{}')  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'result_id': self.result_id,
            'symbol': self.symbol,
            'strategy': self.strategy,
            'period': self.period,
            'initial_capital': self.initial_capital,
            'final_value': self.final_value,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annual_return_pct': self.annual_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'parameters': json.loads(self.parameters) if self.parameters else {},
            'metrics': json.loads(self.metrics) if self.metrics else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class WatchList(Base):
    """觀察清單模型"""
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, default='')
    symbols = Column(Text, default='[]')  # JSON array of symbols
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'symbols': json.loads(self.symbols) if self.symbols else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class MLModel(Base):
    """機器學習模型記錄"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(String(36), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    training_samples = Column(Integer, default=0)
    features = Column(Text, default='[]')  # JSON array
    parameters = Column(Text, default='{}')  # JSON object
    model_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'symbol': self.symbol,
            'model_type': self.model_type,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_samples': self.training_samples,
            'features': json.loads(self.features) if self.features else [],
            'parameters': json.loads(self.parameters) if self.parameters else {},
            'model_path': self.model_path,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 資料庫工具類
class Database:
    """資料庫管理類"""
    
    def __init__(self, database_url='sqlite:///data/trading.db'):
        self.engine = create_engine(database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_all_tables(self):
        """創建所有表"""
        Base.metadata.create_all(self.engine)
    
    def drop_all_tables(self):
        """刪除所有表"""
        Base.metadata.drop_all(self.engine)
    
    def get_session(self):
        """獲取資料庫會話"""
        return self.Session()
    
    def close(self):
        """關閉資料庫連接"""
        self.engine.dispose()


# 初始化資料庫
def init_database(database_url='sqlite:///data/trading.db'):
    """初始化資料庫"""
    db = Database(database_url)
    db.create_all_tables()
    return db


if __name__ == "__main__":
    # 測試資料庫創建
    import os
    os.makedirs("data", exist_ok=True)
    
    db = init_database()
    print("✅ 資料庫表創建成功！")
    print("📊 可用的表:")
    for table in Base.metadata.tables.keys():
        print(f"   - {table}")
