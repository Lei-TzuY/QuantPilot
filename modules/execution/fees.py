"""
Taiwan Equity Fee and Taxation Model
Implements official Taiwan Stock Exchange (TWSE/TPEx) fee and securities transaction tax rules.
Distinguishes between ordinary stock sales (0.3%) and qualifying cash-stock day trades (0.15%).
"""
from dataclasses import dataclass, field
from datetime import date, datetime
import math
import threading
from typing import Dict, Optional, Tuple

from modules.execution.order import OrderSide


@dataclass(frozen=True)
class FeeBreakdown:
    gross_value: float
    commission: float
    tax: float
    slippage: float
    day_trade_quantity: int
    ordinary_quantity: int
    tax_rebate: float = 0.0  # For sell-first same-day buy cover adjustments

    @property
    def total_cost(self) -> float:
        return self.commission + self.tax + self.slippage - self.tax_rebate

    @property
    def net_cash_impact(self) -> float:
        """Net cash impact of the execution."""
        # For BUY: cash outflow = -(gross_value + commission + slippage) + tax_rebate
        # For SELL: cash inflow = (gross_value - commission - tax - slippage)
        pass


class TaiwanFeeModel:
    """
    Centralized Taiwan Equity Fee and Tax Calculator.
    
    Rules:
    1. Brokerage Commission:
       - 0.1425% (0.001425) charged on BOTH Buy and Sell.
       - Optional broker discount (e.g. 0.6 for 40% discount).
       - Minimum commission: TWD 20 (configurable).
    2. Securities Transaction Tax (證券交易稅):
       - Charged ONLY on SELL transactions.
       - Ordinary stock sales: 0.3% (0.003).
       - Qualifying cash-stock day-trading offset (現股當沖): 0.15% (0.0015).
       - Mutual / ETF funds: 0.1% (0.001) if symbol indicates ETF.
    3. Day-Trading Offset Rules:
       - Buy then same-day Sell: The sell quantity up to the same-day bought quantity
         qualifies for 0.15% tax. The excess sell quantity is taxed at 0.3%.
       - Sell then same-day Buy: When an intraday short-sell is covered on the same day,
         the tax difference (0.15%) is rebated.
    """

    def __init__(
        self,
        commission_rate: float = 0.001425,
        commission_discount: float = 1.0,
        min_commission: float = 20.0,
        ordinary_tax_rate: float = 0.003,
        day_trade_tax_rate: float = 0.0015,
        etf_tax_rate: float = 0.001,
        slippage_pct: float = 0.001,
        enable_day_trade_tax: bool = True,
        enable_short_day_trade: bool = True,
    ):
        self.commission_rate = commission_rate
        self.commission_discount = commission_discount
        self.min_commission = min_commission
        self.ordinary_tax_rate = ordinary_tax_rate
        self.day_trade_tax_rate = day_trade_tax_rate
        self.etf_tax_rate = etf_tax_rate
        self.slippage_pct = slippage_pct
        self.enable_day_trade_tax = enable_day_trade_tax
        self.enable_short_day_trade = enable_short_day_trade

        self._lock = threading.RLock()
        # Daily tracking: {trade_date: {symbol: {"buys": int, "sells": int, "offset_buys": int, "unhedged_sells": [(qty, price)]}}}
        self._daily_activity: Dict[date, Dict[str, Dict]] = {}

    def _is_etf(self, symbol: str) -> bool:
        clean = symbol.replace(".TW", "").replace(".TWO", "")
        # Common TW ETF prefixes (0050, 0056, 00878, etc. starting with 00)
        return clean.startswith("00")

    def _get_symbol_tracker(self, trade_date: date, symbol: str) -> Dict:
        if trade_date not in self._daily_activity:
            self._daily_activity[trade_date] = {}
        if symbol not in self._daily_activity[trade_date]:
            self._daily_activity[trade_date][symbol] = {
                "buys": 0,
                "sells": 0,
                "offset_buys": 0,
                "unhedged_sells": [],  # List of [qty, price] for sell-first day trading
            }
        return self._daily_activity[trade_date][symbol]

    def calculate_commission(self, gross_value: float) -> float:
        raw_comm = gross_value * self.commission_rate * self.commission_discount
        return max(self.min_commission, math.floor(raw_comm))

    def calculate_execution_costs(
        self,
        side: OrderSide,
        quantity: int,
        price: float,
        symbol: str,
        trade_date: Optional[date] = None,
    ) -> FeeBreakdown:
        """
        Calculates commission, securities transaction tax, slippage, and day-trade offsets.
        Mutates daily activity state atomically.
        """
        with self._lock:
            dt = trade_date or datetime.now().date()
            tracker = self._get_symbol_tracker(dt, symbol)
            gross_value = quantity * price
            commission = self.calculate_commission(gross_value)
            slippage = gross_value * self.slippage_pct

            day_trade_qty = 0
            ordinary_qty = 0
            tax = 0.0
            tax_rebate = 0.0

            is_etf_symbol = self._is_etf(symbol)
            ord_tax_rate = self.etf_tax_rate if is_etf_symbol else self.ordinary_tax_rate

            if side == OrderSide.BUY:
                tracker["buys"] += quantity

                # Check if this buy covers an earlier sell on the same day (先賣後買沖銷)
                if self.enable_day_trade_tax and self.enable_short_day_trade and tracker["unhedged_sells"]:
                    remaining_buy_to_offset = quantity
                    new_unhedged = []
                    for sell_entry in tracker["unhedged_sells"]:
                        sell_qty, sell_price = sell_entry
                        if remaining_buy_to_offset > 0:
                            matched = min(sell_qty, remaining_buy_to_offset)
                            # Rebate the tax difference: (ordinary - day_trade) = 0.15%
                            rebate_rate = ord_tax_rate - self.day_trade_tax_rate
                            tax_rebate += matched * sell_price * rebate_rate
                            remaining_buy_to_offset -= matched
                            rem_sell = sell_qty - matched
                            if rem_sell > 0:
                                new_unhedged.append([rem_sell, sell_price])
                        else:
                            new_unhedged.append(sell_entry)
                    tracker["unhedged_sells"] = new_unhedged

            elif side == OrderSide.SELL:
                tracker["sells"] += quantity

                if self.enable_day_trade_tax and not is_etf_symbol:
                    # Available same-day bought shares that have not been offset yet
                    available_same_day_buys = tracker["buys"] - tracker["offset_buys"]

                    if available_same_day_buys > 0:
                        # 先買後賣當沖
                        day_trade_qty = min(quantity, available_same_day_buys)
                        ordinary_qty = quantity - day_trade_qty
                        tracker["offset_buys"] += day_trade_qty

                        tax_dt = day_trade_qty * price * self.day_trade_tax_rate
                        tax_ord = ordinary_qty * price * ord_tax_rate
                        tax = math.floor(tax_dt + tax_ord)
                    else:
                        # No same-day buys available -> ordinary sale (or potential sell-first day trade)
                        ordinary_qty = quantity
                        tax = math.floor(quantity * price * ord_tax_rate)
                        if self.enable_short_day_trade:
                            tracker["unhedged_sells"].append([quantity, price])
                else:
                    ordinary_qty = quantity
                    tax = math.floor(quantity * price * ord_tax_rate)

            return FeeBreakdown(
                gross_value=gross_value,
                commission=commission,
                tax=tax,
                slippage=slippage,
                day_trade_quantity=day_trade_qty,
                ordinary_quantity=ordinary_qty,
                tax_rebate=tax_rebate,
            )

    def reset_daily_activity(self) -> None:
        with self._lock:
            self._daily_activity.clear()
