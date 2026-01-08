"""
機器學習策略回測器
ML Strategy Backtester
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class MLBacktester:
    """
    機器學習策略回測器
    整合ML預測與回測引擎
    """
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
    
    def backtest_ml_strategy(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        initial_capital: float = 1_000_000,
        confidence_threshold: float = 0.6,
        fee_rate: float = 0.001425,
        slippage_pct: float = 0.001,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10
    ) -> Dict:
        """
        使用ML預測進行回測
        
        Args:
            df: 價格數據
            predictions: ML預測結果 (0=SELL, 1=BUY)
            probabilities: 預測概率 [prob_sell, prob_buy]
            initial_capital: 初始資金
            confidence_threshold: 信心閾值
            fee_rate: 手續費率
            slippage_pct: 滑價百分比
            stop_loss_pct: 止損百分比
            take_profit_pct: 止盈百分比
        
        Returns:
            回測結果
        """
        # 重置
        self.trades = []
        self.equity_curve = []
        
        capital = initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        
        # 確保數據對齊
        if len(df) != len(predictions):
            raise ValueError("Data and predictions length mismatch")
        
        # 逐日回測
        for i in range(len(df)):
            date = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
            price = df['close'].iloc[i]
            pred = predictions[i]
            prob = probabilities[i]
            
            # 獲取預測信心度
            confidence = max(prob)
            
            # 如果有持倉，檢查止損止盈
            if position > 0:
                # 計算收益率
                pnl_pct = (price - entry_price) / entry_price
                
                # 止損
                if pnl_pct <= -stop_loss_pct:
                    sell_price = price * (1 - slippage_pct)
                    sell_value = position * sell_price
                    fee = sell_value * fee_rate
                    capital += sell_value - fee
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': sell_price,
                        'shares': position,
                        'pnl': sell_value - position * entry_price - fee,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': 'stop_loss'
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
                
                # 止盈
                elif pnl_pct >= take_profit_pct:
                    sell_price = price * (1 - slippage_pct)
                    sell_value = position * sell_price
                    fee = sell_value * fee_rate
                    capital += sell_value - fee
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': sell_price,
                        'shares': position,
                        'pnl': sell_value - position * entry_price - fee,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': 'take_profit'
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
                
                # ML預測賣出信號
                elif pred == 0 and confidence >= confidence_threshold:
                    sell_price = price * (1 - slippage_pct)
                    sell_value = position * sell_price
                    fee = sell_value * fee_rate
                    capital += sell_value - fee
                    
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': sell_price,
                        'shares': position,
                        'pnl': sell_value - position * entry_price - fee,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': 'ml_signal',
                        'confidence': confidence
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_date = None
            
            # 如果沒有持倉且ML預測買入
            elif position == 0 and pred == 1 and confidence >= confidence_threshold:
                buy_price = price * (1 + slippage_pct)
                shares_to_buy = int(capital * 0.95 / buy_price)  # 使用95%的資金
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * buy_price
                    fee = cost * fee_rate
                    total_cost = cost + fee
                    
                    if total_cost <= capital:
                        capital -= total_cost
                        position = shares_to_buy
                        entry_price = buy_price
                        entry_date = date
            
            # 記錄資產曲線
            current_value = capital
            if position > 0:
                current_value += position * price
            
            self.equity_curve.append({
                'date': date,
                'value': current_value,
                'capital': capital,
                'position_value': position * price if position > 0 else 0
            })
        
        # 如果最後還有持倉，平倉
        if position > 0:
            final_price = df['close'].iloc[-1]
            sell_value = position * final_price
            fee = sell_value * fee_rate
            capital += sell_value - fee
            
            pnl_pct = (final_price - entry_price) / entry_price
            
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl': sell_value - position * entry_price - fee,
                'pnl_pct': pnl_pct * 100,
                'exit_reason': 'final_close'
            })
        
        # 計算績效指標
        final_value = self.equity_curve[-1]['value']
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # 計算其他指標
        metrics = self._calculate_metrics(initial_capital, df)
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(self.trades),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, initial_capital: float, df: pd.DataFrame) -> Dict:
        """計算績效指標"""
        if not self.trades:
            return {}
        
        # 勝率
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        # 平均盈虧
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # 盈虧比
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # 最大回撤
        equity_values = [e['value'] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 夏普比率
        equity_series = pd.Series(equity_values)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 年化收益率
        trading_days = len(df)
        years = trading_days / 252
        final_value = equity_values[-1]
        
        if years > 0:
            annual_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100
        else:
            annual_return = 0
        
        return {
            'win_rate': round(win_rate * 100, 2),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'annual_return_pct': round(annual_return, 2)
        }
    
    def walk_forward_analysis(
        self,
        df: pd.DataFrame,
        ml_model,
        feature_engineer,
        train_window: int = 252,  # 1年
        test_window: int = 63,    # 3個月
        step_size: int = 63       # 每次前進3個月
    ) -> Dict:
        """
        滾動視窗分析 (Walk-Forward Analysis)
        
        Args:
            df: 完整數據
            ml_model: ML模型實例
            feature_engineer: 特徵工程實例
            train_window: 訓練窗口大小
            test_window: 測試窗口大小
            step_size: 步進大小
        
        Returns:
            滾動分析結果
        """
        results = []
        
        start_idx = 0
        while start_idx + train_window + test_window <= len(df):
            # 分割訓練和測試數據
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_df = df.iloc[start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()
            
            # 生成特徵
            train_features = feature_engineer.generate_all_features(train_df)
            test_features = feature_engineer.generate_all_features(test_df)
            
            # 創建目標變量
            train_features['target'] = (train_features['close'].shift(-1) > train_features['close']).astype(int)
            
            # 移除NaN
            train_features = train_features.dropna()
            
            if len(train_features) < 50:  # 至少需要50個樣本
                start_idx += step_size
                continue
            
            # 準備訓練數據
            feature_cols = [col for col in train_features.columns 
                          if col not in ['target', 'open', 'high', 'low', 'close', 'volume']]
            
            X_train = train_features[feature_cols]
            y_train = train_features['target']
            
            # 訓練模型
            try:
                ml_model.train_model(X_train, y_train, tune_hyperparams=False)
                
                # 在測試集上預測
                test_features_clean = test_features[feature_cols].dropna()
                
                if len(test_features_clean) == 0:
                    start_idx += step_size
                    continue
                
                pred_result = ml_model.predict(test_features_clean, return_proba=True)
                
                # 對齊預測和測試數據
                test_df_aligned = test_df.loc[test_features_clean.index]
                
                # 回測
                backtest_result = self.backtest_ml_strategy(
                    test_df_aligned,
                    np.array(pred_result['predictions']),
                    np.array(pred_result['probabilities']),
                    initial_capital=1_000_000
                )
                
                results.append({
                    'period': f"{train_df.index[0]} to {test_df.index[-1]}",
                    'train_start': str(train_df.index[0]),
                    'train_end': str(train_df.index[-1]),
                    'test_start': str(test_df.index[0]),
                    'test_end': str(test_df.index[-1]),
                    'return_pct': backtest_result['total_return_pct'],
                    'num_trades': backtest_result['num_trades'],
                    'metrics': backtest_result['metrics']
                })
                
            except Exception as e:
                print(f"Error in walk-forward period: {e}")
            
            start_idx += step_size
        
        # 計算總體統計
        if results:
            avg_return = np.mean([r['return_pct'] for r in results])
            std_return = np.std([r['return_pct'] for r in results])
            win_rate = len([r for r in results if r['return_pct'] > 0]) / len(results) * 100
        else:
            avg_return = std_return = win_rate = 0
        
        return {
            'periods': results,
            'num_periods': len(results),
            'summary': {
                'avg_return_pct': round(avg_return, 2),
                'std_return_pct': round(std_return, 2),
                'win_rate': round(win_rate, 2)
            }
        }
