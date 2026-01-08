import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

class AlertManager:
    """
    Enhanced Price Alert Manager - supports complex trigger conditions.
    """
    
    ALERTS_FILE = "data/alerts.json"
    
    # Supported condition types
    CONDITION_TYPES = {
        'above': 'Price Above',
        'below': 'Price Below',
        'volatility_spike': 'Volatility Spike (ATR)',
        'volume_surge': 'Volume Surge',
        'rsi_oversold': 'RSI Oversold',
        'rsi_overbought': 'RSI Overbought'
    }
    
    def __init__(self):
        self.alerts: List[Dict] = []
        self._load_alerts()
    
    def _load_alerts(self):
        """Load alerts from file."""
        if os.path.exists(self.ALERTS_FILE):
            try:
                with open(self.ALERTS_FILE, 'r') as f:
                    self.alerts = json.load(f)
            except Exception as e:
                print(f"Failed to load alerts: {e}")
                self.alerts = []
    
    def _save_alerts(self):
        """Save alerts to file."""
        os.makedirs(os.path.dirname(self.ALERTS_FILE), exist_ok=True)
        with open(self.ALERTS_FILE, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def create_alert(self, symbol: str, condition: str, target_value: float, note: str = "") -> Dict:
        """
        Create a new price alert.
        
        Args:
            symbol: Stock symbol
            condition: 'above', 'below', 'volatility_spike', 'volume_surge', 'rsi_oversold', 'rsi_overbought'
            target_value: Target price or threshold
            note: Optional note
        """
        if condition not in self.CONDITION_TYPES:
            raise ValueError(f"Unsupported condition: {condition}. Supported: {list(self.CONDITION_TYPES.keys())}")
        
        alert = {
            'id': str(uuid.uuid4())[:8],
            'symbol': symbol.upper(),
            'condition': condition,
            'target_value': target_value,
            'note': note,
            'created_at': datetime.now().isoformat(),
            'triggered': False,
            'triggered_at': None,
            'triggered_price': None,
            'active': True
        }
        self.alerts.append(alert)
        self._save_alerts()
        return alert
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert by ID."""
        for i, alert in enumerate(self.alerts):
            if alert['id'] == alert_id:
                self.alerts.pop(i)
                self._save_alerts()
                return True
        return False
    
    def get_alerts(self, active_only: bool = True) -> List[Dict]:
        """Get all alerts."""
        if active_only:
            return [a for a in self.alerts if a['active'] and not a['triggered']]
        return self.alerts
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return atr
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _check_condition(self, alert: Dict, data_fetcher) -> tuple:
        """
        Check if alert condition is met.
        Returns (should_trigger, current_value)
        """
        symbol = alert['symbol']
        condition = alert['condition']
        target = alert['target_value']
        
        try:
            # Get current price
            price_data = data_fetcher.get_realtime_price(symbol)
            current_price = price_data.get('price', 0)
            
            # Simple price conditions
            if condition == 'above':
                return current_price >= target, current_price
            elif condition == 'below':
                return current_price <= target, current_price
            
            # Complex conditions need historical data
            try:
                df = data_fetcher.get_stock_data_df(symbol, '1mo')
                if df is None or len(df) < 20:
                    return False, current_price
                
                if condition == 'volatility_spike':
                    atr = self._calculate_atr(df)
                    avg_atr = df['close'].diff().abs().rolling(14).mean().mean()
                    atr_ratio = atr / avg_atr if avg_atr > 0 else 0
                    return atr_ratio >= target, round(atr_ratio, 2)
                
                elif condition == 'volume_surge':
                    if 'volume' not in df.columns:
                        return False, 0
                    avg_volume = df['volume'].rolling(10).mean().iloc[-2]
                    current_volume = df['volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    return volume_ratio >= target, round(volume_ratio, 2)
                
                elif condition == 'rsi_oversold':
                    rsi = self._calculate_rsi(df)
                    return rsi <= target, round(rsi, 2)
                
                elif condition == 'rsi_overbought':
                    rsi = self._calculate_rsi(df)
                    return rsi >= target, round(rsi, 2)
                    
            except Exception as e:
                print(f"Failed to get historical data for {symbol}: {e}")
                return False, current_price
                
        except Exception as e:
            print(f"Failed to check alert for {symbol}: {e}")
            return False, 0
        
        return False, 0
    
    def check_alerts(self, data_fetcher) -> List[Dict]:
        """
        Check all active alerts against current conditions.
        Returns list of triggered alerts.
        """
        triggered = []
        
        for alert in self.alerts:
            if not alert['active'] or alert['triggered']:
                continue
            
            should_trigger, current_value = self._check_condition(alert, data_fetcher)
            
            if should_trigger:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
                alert['triggered_price'] = current_value
                triggered.append(alert)
                
                condition_label = self.CONDITION_TYPES.get(alert['condition'], alert['condition'])
                print(f"🔔 ALERT TRIGGERED: {alert['symbol']} - {condition_label} {alert['target_value']} (current: {current_value})")
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def get_triggered_alerts(self) -> List[Dict]:
        """Get all triggered alerts."""
        return [a for a in self.alerts if a['triggered']]
    
    def clear_triggered(self):
        """Clear all triggered alerts."""
        self.alerts = [a for a in self.alerts if not a['triggered']]
        self._save_alerts()
    
    def get_condition_types(self) -> Dict:
        """Get available condition types."""
        return self.CONDITION_TYPES.copy()

