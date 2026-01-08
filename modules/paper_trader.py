import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class PaperTrader:
    """
    Paper Trading Engine for simulated order execution.
    All trades are virtual - no real money is involved.
    """
    
    PORTFOLIO_FILE = "data/paper_portfolio.json"
    
    def __init__(self, initial_cash: float = 1_000_000):
        self.positions: Dict[str, Dict] = {}  # {symbol: {shares, avg_price}}
        self.cash: float = initial_cash
        self.trade_log: List[Dict] = []
        self._load_state()
    
    def _load_state(self):
        """Load portfolio state from file if exists."""
        if os.path.exists(self.PORTFOLIO_FILE):
            try:
                with open(self.PORTFOLIO_FILE, 'r') as f:
                    state = json.load(f)
                    self.positions = state.get('positions', {})
                    self.cash = state.get('cash', 1_000_000)
                    self.trade_log = state.get('trade_log', [])
            except Exception as e:
                print(f"Failed to load paper portfolio: {e}")
    
    def _save_state(self):
        """Persist portfolio state to file."""
        os.makedirs(os.path.dirname(self.PORTFOLIO_FILE), exist_ok=True)
        state = {
            'positions': self.positions,
            'cash': self.cash,
            'trade_log': self.trade_log
        }
        with open(self.PORTFOLIO_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def execute_buy(self, symbol: str, shares: int, price: float) -> Dict:
        """Execute a paper buy order."""
        cost = shares * price
        if cost > self.cash:
            return {'success': False, 'error': 'Insufficient cash'}
        
        self.cash -= cost
        
        if symbol in self.positions:
            # Average up
            pos = self.positions[symbol]
            total_shares = pos['shares'] + shares
            total_cost = (pos['shares'] * pos['avg_price']) + cost
            pos['avg_price'] = total_cost / total_shares
            pos['shares'] = total_shares
        else:
            self.positions[symbol] = {'shares': shares, 'avg_price': price}
        
        trade = {
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'timestamp': datetime.now().isoformat()
        }
        self.trade_log.append(trade)
        self._save_state()
        
        return {'success': True, 'trade': trade, 'cash_remaining': self.cash}
    
    def execute_sell(self, symbol: str, shares: int, price: float) -> Dict:
        """Execute a paper sell order."""
        if symbol not in self.positions:
            return {'success': False, 'error': f'No position in {symbol}'}
        
        pos = self.positions[symbol]
        if shares > pos['shares']:
            return {'success': False, 'error': 'Insufficient shares'}
        
        proceeds = shares * price
        pnl = (price - pos['avg_price']) * shares
        
        self.cash += proceeds
        pos['shares'] -= shares
        
        if pos['shares'] == 0:
            del self.positions[symbol]
        
        trade = {
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        }
        self.trade_log.append(trade)
        self._save_state()
        
        return {'success': True, 'trade': trade, 'pnl': pnl, 'cash_remaining': self.cash}
    
    def get_portfolio_value(self, data_fetcher) -> Dict:
        """Calculate current portfolio value using live prices."""
        holdings = []
        total_value = self.cash
        
        for symbol, pos in self.positions.items():
            try:
                price_data = data_fetcher.get_realtime_price(symbol)
                current_price = price_data.get('price', pos['avg_price'])
            except:
                current_price = pos['avg_price']
            
            market_value = pos['shares'] * current_price
            unrealized_pnl = (current_price - pos['avg_price']) * pos['shares']
            total_value += market_value
            
            holdings.append({
                'symbol': symbol,
                'shares': pos['shares'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl
            })
        
        return {
            'cash': self.cash,
            'holdings': holdings,
            'total_value': total_value,
            'trade_count': len(self.trade_log)
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Return recent trade log."""
        return self.trade_log[-20:]  # Last 20 trades
    
    def reset(self, initial_cash: float = 1_000_000):
        """Reset paper portfolio."""
        self.positions = {}
        self.cash = initial_cash
        self.trade_log = []
        self._save_state()
        return {'success': True, 'message': 'Portfolio reset'}
