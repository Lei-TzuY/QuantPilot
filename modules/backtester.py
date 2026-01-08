from typing import Dict, List, Optional, Any

import pandas as pd


class Backtester:
    """Simple rule-based backtester for common strategies."""

    def __init__(self):
        self._strategies = {
            "ma_crossover": "Moving average crossover (short above long -> buy, below -> sell)",
            "rsi": "RSI mean-reversion (buy <30, sell >70)",
            "macd": "MACD signal crossover",
        }

    def get_available_strategies(self) -> List[Dict[str, str]]:
        return [{"name": name, "description": desc} for name, desc in self._strategies.items()]

    def run(
        self,
        stock_df: pd.DataFrame,
        strategy: str = "ma_crossover",
        initial_capital: float = 1_000_000,
        params: Optional[Dict] = None,
        risk_params: Optional[Dict] = None,
    ) -> Dict:
        params = params or {}
        risk_params = risk_params or {}
        df = stock_df.copy()
        df = df.sort_index()

        if strategy == "ma_crossover":
            df = self._ma_crossover(df, params)
        elif strategy == "rsi":
            df = self._rsi_strategy(df, params)
        elif strategy == "macd":
            df = self._macd_strategy(df, params)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")


        if "signal" not in df.columns:
            raise ValueError("Strategy did not produce signals.")

        trades, final_value, equity_curve, fee_summary = self._simulate(df, initial_capital, risk_params)
        start_date = df.index.min().strftime("%Y-%m-%d")
        end_date = df.index.max().strftime("%Y-%m-%d")
        return_pct = (final_value - initial_capital) / initial_capital * 100

        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=df.index)
        
        # Max Drawdown
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100  # Percentage

        # Sharpe Ratio (Ann) Assuming 0% risk free for simplicity
        daily_returns = equity_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0.0

        # CAGR - Compound Annual Growth Rate
        trading_days = len(df)
        years = trading_days / 252
        if years > 0 and final_value > 0:
            cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0.0

        # Win Rate, Profit Factor, Average Holding Days
        sell_trades = [t for t in trades if t['type'] == 'sell']
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        win_rate = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0.0
        
        total_profit = sum(t.get('pnl', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0.0

        # Average Holding Days calculation
        avg_holding_days = 0.0
        if len(trades) >= 2:
            holding_periods = []
            buy_date = None
            for t in trades:
                if t['type'] == 'buy':
                    buy_date = pd.to_datetime(t['date'])
                elif t['type'] == 'sell' and buy_date is not None:
                    sell_date = pd.to_datetime(t['date'])
                    holding_periods.append((sell_date - buy_date).days)
                    buy_date = None
            if holding_periods:
                avg_holding_days = sum(holding_periods) / len(holding_periods)

        return {
            "strategy": strategy,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "return_pct": return_pct,
            "cagr": cagr,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_holding_days,
            "total_trades": len(sell_trades),
            "trades": trades,
            "equity_curve": equity_curve,
            "risk_settings": risk_params,
            "fee_summary": fee_summary
        }

    def _simulate(self, df: pd.DataFrame, initial_capital: float, risk_params: Dict):
        cash = initial_capital
        position = 0
        entry_price = 0.0
        trades: List[Dict] = []
        equity_curve = []
        
        # Fee and slippage tracking
        total_commission = 0.0
        total_slippage = 0.0

        # Risk parameters
        stop_loss_pct = float(risk_params.get("stop_loss_pct", 0)) if risk_params else 0
        take_profit_pct = float(risk_params.get("take_profit_pct", 0)) if risk_params else 0
        position_size_pct = float(risk_params.get("position_size_pct", 1.0)) if risk_params else 1.0
        
        # Fee parameters (Taiwan stock market defaults)
        commission_rate = float(risk_params.get("commission_rate", 0.001425)) if risk_params else 0.001425  # 0.1425%
        tax_rate = float(risk_params.get("tax_rate", 0.003)) if risk_params else 0.003  # 0.3% sell tax
        slippage_pct = float(risk_params.get("slippage_pct", 0.001)) if risk_params else 0.001  # 0.1% slippage

        for date, row in df.iterrows():
            price = float(row["close"])
            high = float(row.get("high", price))
            low = float(row.get("low", price))
            open_price = float(row.get("open", price))
            
            signal = int(row["signal"])
            
            # 1. Check Risk Exits (Stop Loss / Take Profit) FIRST
            if position > 0:
                exit_price = None
                exit_reason = ""
                
                # Check Stop Loss
                if stop_loss_pct > 0:
                    sl_price = entry_price * (1 - stop_loss_pct)
                    if low <= sl_price:
                        # Gap down check: if open is already below SL, we exit at Open
                        exit_price = open_price if open_price < sl_price else sl_price
                        exit_reason = "Stop Loss"

                # Check Take Profit
                if take_profit_pct > 0 and exit_price is None:
                    tp_price = entry_price * (1 + take_profit_pct)
                    if high >= tp_price:
                        # Gap up check
                        exit_price = open_price if open_price > tp_price else tp_price
                        exit_reason = "Take Profit"
                
                if exit_price is not None:
                    proceeds = position * exit_price
                    cash += proceeds
                    trades.append({
                        "type": "sell",
                        "date": date.strftime("%Y-%m-%d"),
                        "price": exit_price,
                        "shares": position,
                        "cash_after": cash,
                        "reason": exit_reason,
                        "pnl": (exit_price - entry_price) * position
                    })
                    position = 0
                    entry_price = 0.0

            # 2. Check Signals (Entry/Exit)
            # Only enter if we have cash and no position (assuming simple 1 shot strategy for now)
            # Or if signal flips from -1 to 1 (reversal)
            
            if position == 0 and signal == 1:
                # Buy Logic
                # Position Sizing
                allocatable_cash = cash * position_size_pct
                
                # Apply slippage to execution price (buy at higher price)
                exec_price = price * (1 + slippage_pct)
                shares = int(allocatable_cash // exec_price)
                
                if shares > 0:
                    gross_cost = shares * exec_price
                    commission = gross_cost * commission_rate
                    slippage_cost = shares * price * slippage_pct
                    
                    total_cost = gross_cost + commission
                    cash -= total_cost
                    position += shares
                    entry_price = exec_price
                    
                    # Track cumulative fees
                    total_commission += commission
                    total_slippage += slippage_cost
                    
                    trades.append({
                        "type": "buy",
                        "date": date.strftime("%Y-%m-%d"),
                        "price": exec_price,
                        "market_price": price,
                        "shares": shares,
                        "commission": round(commission, 2),
                        "slippage": round(slippage_cost, 2),
                        "cash_after": cash,
                        "reason": "Signal"
                    })
            
            elif position > 0 and signal == -1:
                # Sell Logic (Signal Exit)
                # Apply slippage to execution price (sell at lower price)
                exec_price = price * (1 - slippage_pct)
                gross_proceeds = position * exec_price
                
                # Calculate fees: commission + tax (only on sell in Taiwan)
                commission = gross_proceeds * commission_rate
                tax = gross_proceeds * tax_rate
                slippage_cost = position * price * slippage_pct
                
                net_proceeds = gross_proceeds - commission - tax
                cash += net_proceeds
                
                # Track cumulative fees
                total_commission += commission + tax
                total_slippage += slippage_cost
                
                gross_pnl = (exec_price - entry_price) * position
                net_pnl = gross_pnl - commission - tax
                
                trades.append({
                    "type": "sell",
                    "date": date.strftime("%Y-%m-%d"),
                    "price": exec_price,
                    "market_price": price,
                    "shares": position,
                    "commission": round(commission, 2),
                    "tax": round(tax, 2),
                    "slippage": round(slippage_cost, 2),
                    "cash_after": cash,
                    "reason": "Signal",
                    "gross_pnl": round(gross_pnl, 2),
                    "pnl": round(net_pnl, 2)
                })
                position = 0
                entry_price = 0.0
            
            # Track daily equity
            daily_value = cash + (position * price)
            equity_curve.append(daily_value)

        # Force Close at end
        # last_price = float(df["close"].iloc[-1]) # Already have price variable from loop
        # final_value = cash + position * last_price # Already calculated in equity_curve[-1]
        
        final_value = equity_curve[-1] if equity_curve else initial_capital
        
        # Return fee summary along with trades
        return trades, final_value, equity_curve, {
            "total_commission": round(total_commission, 2),
            "total_slippage": round(total_slippage, 2),
            "total_fees": round(total_commission + total_slippage, 2)
        }


    def optimize(
        self,
        stock_df: pd.DataFrame,
        strategy: str,
        initial_capital: float,
        param_ranges: Dict[str, List[Any]],
        risk_params: Optional[Dict] = None,
    ) -> List[Dict]:
        import itertools
        risk_params = risk_params or {}
        
        # Generate all combinations of parameters
        keys = param_ranges.keys()
        values = param_ranges.values()
        combinations = list(itertools.product(*values))
        
        results = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                res = self.run(stock_df, strategy, initial_capital, params, risk_params)
                results.append({
                    "params": params,
                    "return_pct": res["return_pct"],
                    "sharpe_ratio": res["sharpe_ratio"],
                    "max_drawdown_pct": res["max_drawdown_pct"]
                })
            except Exception:
                continue
                
        # Sort by return_pct descending
        results.sort(key=lambda x: x["return_pct"], reverse=True)
        return results[:10] # Return top 10

    def _ma_crossover(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        short_window = int(params.get("short_window", 20))
        long_window = int(params.get("long_window", 60))
        data = df.copy()
        data["ma_short"] = data["close"].rolling(window=short_window).mean()
        data["ma_long"] = data["close"].rolling(window=long_window).mean()
        data["signal_raw"] = (data["ma_short"] > data["ma_long"]).astype(int)
        data["signal"] = data["signal_raw"].diff().fillna(0)
        data.loc[data["signal"] > 0, "signal"] = 1
        data.loc[data["signal"] < 0, "signal"] = -1
        data["signal"] = data["signal"].fillna(0)
        return data

    def _rsi_strategy(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))

        data = df.copy()
        delta = data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data["rsi"] = rsi

        data["signal"] = 0
        data.loc[data["rsi"] < oversold, "signal"] = 1
        data.loc[data["rsi"] > overbought, "signal"] = -1
        return data

    def _macd_strategy(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal_span = int(params.get("signal", 9))

        data = df.copy()
        ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
        data["macd"] = macd_line
        data["signal_line"] = signal_line
        data["signal_raw"] = (macd_line > signal_line).astype(int)
        data["signal"] = data["signal_raw"].diff().fillna(0)
        data.loc[data["signal"] > 0, "signal"] = 1
        data.loc[data["signal"] < 0, "signal"] = -1
        return data
