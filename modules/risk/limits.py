"""
Risk Limits Configuration
"""
from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_value_per_symbol: float = 500_000.0
    max_total_exposure: float = 2_000_000.0
    max_order_value: float = 300_000.0
    max_open_positions: int = 5
    max_trades_per_day: int = 50
    max_daily_realized_loss: float = 50_000.0
    max_daily_total_loss: float = 100_000.0
    max_price_deviation_pct: float = 0.08  # Max 8% deviation between order price and market price
    max_stale_data_seconds: float = 60.0   # Reject orders if market tick older than 60s
    duplicate_window_seconds: float = 5.0
    enable_intraday_flattening: bool = True
