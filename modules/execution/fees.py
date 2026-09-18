"""
Taiwan Equity Fee and Taxation Model
Implements official Taiwan Stock Exchange (TWSE/TPEx) fee and securities transaction tax rules.
Distinguishes between ordinary stock sales (0.3%) and qualifying cash-stock day trades (0.15%).
"""
from dataclasses import dataclass, field
from datetime import date, datetime
import math
import threading
from typing import Dict, List, Optional, Tuple

from modules.execution.order import OrderSide


@dataclass(frozen=True)
class FeeBreakdown:
    gross_value: float
    commission: float
    tax: float
    slippage: float
    day_trade_quantity: int
    ordinary_quantity: int
    tax_rebate: float = 0.0  # Labeled simulation convenience for intraday cash estimation
    is_simulation_rebate: bool = False

    @property
    def total_cost(self) -> float:
        return self.commission + self.tax + self.slippage - self.tax_rebate

    @property
    def net_cash_impact(self) -> float:
        """Net cash impact of the execution."""
        return self.gross_value - self.total_cost


@dataclass
class TaiwanLot:
    """Represents an execution lot for tax and clearing settlement."""
    lot_id: str
    symbol: str
    side: OrderSide
    quantity: int
    remaining_quantity: int
    price: float
    timestamp: datetime
    is_overnight: bool = False


@dataclass(frozen=True)
class MatchedDayTrade:
    """A matched same-day offset under Taiwan equity day-trading rules."""
    symbol: str
    buy_lot_id: str
    sell_lot_id: str
    matched_quantity: int
    buy_price: float
    sell_price: float
    tax_rate: float = 0.0015  # Direct 0.15% statutory tax rate on sell proceeds
    tax: float = 0.0
    gross_pnl: float = 0.0


@dataclass
class SettlementReport:
    """End-of-day or real-time settlement matching report."""
    trade_date: date
    symbol: str
    matched_day_trades: List[MatchedDayTrade] = field(default_factory=list)
    ordinary_sales: List[TaiwanLot] = field(default_factory=list)
    remaining_overnight_inventory: List[TaiwanLot] = field(default_factory=list)
    remaining_intraday_buys: List[TaiwanLot] = field(default_factory=list)
    total_commission: float = 0.0
    total_statutory_tax: float = 0.0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0


class TaiwanSettlementModel:
    """
    Deterministic FIFO Matching and Settlement Engine for Taiwan Equities.
    
    Principles:
    1. Direct 0.15% Tax: Matched qualifying day-trade quantities have a statutory
       tax liability of directly 0.15% on the sell proceeds.
    2. Inventory Separation: Overnight inventory (prior sessions) is tracked strictly
       separate from same-day intraday executions.
    3. Deterministic FIFO Matching:
       - Sales first match available same-day intraday buys (0.15% day-trade rate).
       - Excess sales match available overnight inventory (0.30% ordinary rate).
       - Unhedged sales (sell-first day trades) are matched by subsequent same-day buys,
         with the final settled tax liability directly established at 0.15%.
    4. Independent Broker Commission: Commission schedules (rates, discounts, min fee)
       are calculated independently from statutory taxes.
    """

    def __init__(
        self,
        commission_rate: float = 0.001425,
        commission_discount: float = 1.0,
        min_commission: float = 0.0,  # 0.0 TWD default: not all TW brokers enforce 20 TWD minimum
        ordinary_tax_rate: float = 0.003,
        day_trade_tax_rate: float = 0.0015,
        etf_tax_rate: float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.commission_discount = commission_discount
        self.min_commission = min_commission
        self.ordinary_tax_rate = ordinary_tax_rate
        self.day_trade_tax_rate = day_trade_tax_rate
        self.etf_tax_rate = etf_tax_rate

        self._lock = threading.RLock()
        self._overnight_pools: Dict[str, List[TaiwanLot]] = {}
        self._intraday_buys: Dict[str, List[TaiwanLot]] = {}
        self._intraday_sells: Dict[str, List[TaiwanLot]] = {}
        self._matched_day_trades: Dict[str, List[MatchedDayTrade]] = {}
        self._ordinary_sales: Dict[str, List[TaiwanLot]] = {}
        self._lot_counter = 0

    def _next_lot_id(self, prefix: str = "LOT") -> str:
        self._lot_counter += 1
        return f"{prefix}-{self._lot_counter:06d}"

    def _is_etf(self, symbol: str) -> bool:
        clean = symbol.replace(".TW", "").replace(".TWO", "")
        return clean.startswith("00")

    def add_overnight_inventory(self, symbol: str, quantity: int, avg_price: float, timestamp: Optional[datetime] = None) -> None:
        """Seeds overnight inventory carried from prior trading days."""
        with self._lock:
            if symbol not in self._overnight_pools:
                self._overnight_pools[symbol] = []
            lot = TaiwanLot(
                lot_id=self._next_lot_id("OVN"),
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                remaining_quantity=quantity,
                price=avg_price,
                timestamp=timestamp or datetime.now(),
                is_overnight=True,
            )
            self._overnight_pools[symbol].append(lot)

    def calculate_commission(self, gross_value: float) -> float:
        """Calculates broker commission independently from tax."""
        raw_comm = gross_value * self.commission_rate * self.commission_discount
        if self.min_commission > 0:
            return max(self.min_commission, math.floor(raw_comm))
        return float(math.floor(raw_comm))

    def process_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[List[MatchedDayTrade], List[TaiwanLot], float, float]:
        """
        Processes an intraday fill lot using deterministic FIFO matching.
        Returns:
            (new_matched_day_trades, ordinary_sales, commission, immediate_tax)
        """
        with self._lock:
            ts = timestamp or datetime.now()
            gross_value = quantity * price
            commission = self.calculate_commission(gross_value)
            is_etf = self._is_etf(symbol)
            ord_tax_rate = self.etf_tax_rate if is_etf else self.ordinary_tax_rate
            dt_tax_rate = self.etf_tax_rate if is_etf else self.day_trade_tax_rate

            if symbol not in self._intraday_buys:
                self._intraday_buys[symbol] = []
                self._intraday_sells[symbol] = []
                self._matched_day_trades[symbol] = []
                self._ordinary_sales[symbol] = []
                self._overnight_pools.setdefault(symbol, [])

            new_matches: List[MatchedDayTrade] = []
            ordinary_sales: List[TaiwanLot] = []
            tax_liability = 0.0

            if side == OrderSide.BUY:
                buy_lot = TaiwanLot(
                    lot_id=self._next_lot_id("BUY"),
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    remaining_quantity=quantity,
                    price=price,
                    timestamp=ts,
                    is_overnight=False,
                )

                # Check if this buy covers existing unhedged intraday short sells (Sell-first day trade)
                for sell_lot in self._intraday_sells[symbol]:
                    if sell_lot.remaining_quantity <= 0 or buy_lot.remaining_quantity <= 0:
                        continue
                    matched_qty = min(buy_lot.remaining_quantity, sell_lot.remaining_quantity)
                    # For sell-first day trade, final tax liability for matched portion is DIRECTLY 0.15% on sell proceeds
                    matched_tax = math.floor(matched_qty * sell_lot.price * dt_tax_rate)
                    match = MatchedDayTrade(
                        symbol=symbol,
                        buy_lot_id=buy_lot.lot_id,
                        sell_lot_id=sell_lot.lot_id,
                        matched_quantity=matched_qty,
                        buy_price=buy_lot.price,
                        sell_price=sell_lot.price,
                        tax_rate=dt_tax_rate,
                        tax=matched_tax,
                        gross_pnl=matched_qty * (sell_lot.price - buy_lot.price),
                    )
                    new_matches.append(match)
                    self._matched_day_trades[symbol].append(match)
                    buy_lot.remaining_quantity -= matched_qty
                    sell_lot.remaining_quantity -= matched_qty

                if buy_lot.remaining_quantity > 0:
                    self._intraday_buys[symbol].append(buy_lot)

            elif side == OrderSide.SELL:
                sell_lot = TaiwanLot(
                    lot_id=self._next_lot_id("SELL"),
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    remaining_quantity=quantity,
                    price=price,
                    timestamp=ts,
                    is_overnight=False,
                )

                # 1. First match against available same-day intraday buys (Buy-then-Sell day trade)
                for buy_lot in self._intraday_buys[symbol]:
                    if buy_lot.remaining_quantity <= 0 or sell_lot.remaining_quantity <= 0:
                        continue
                    matched_qty = min(sell_lot.remaining_quantity, buy_lot.remaining_quantity)
                    matched_tax = math.floor(matched_qty * sell_lot.price * dt_tax_rate)
                    match = MatchedDayTrade(
                        symbol=symbol,
                        buy_lot_id=buy_lot.lot_id,
                        sell_lot_id=sell_lot.lot_id,
                        matched_quantity=matched_qty,
                        buy_price=buy_lot.price,
                        sell_price=sell_lot.price,
                        tax_rate=dt_tax_rate,
                        tax=matched_tax,
                        gross_pnl=matched_qty * (sell_lot.price - buy_lot.price),
                    )
                    new_matches.append(match)
                    self._matched_day_trades[symbol].append(match)
                    tax_liability += matched_tax
                    sell_lot.remaining_quantity -= matched_qty
                    buy_lot.remaining_quantity -= matched_qty

                # 2. Next, match against overnight inventory (Ordinary sale, 0.30% tax)
                if sell_lot.remaining_quantity > 0:
                    for ovn_lot in self._overnight_pools[symbol]:
                        if ovn_lot.remaining_quantity <= 0 or sell_lot.remaining_quantity <= 0:
                            continue
                        matched_qty = min(sell_lot.remaining_quantity, ovn_lot.remaining_quantity)
                        ovn_tax = math.floor(matched_qty * sell_lot.price * ord_tax_rate)
                        ord_sale = TaiwanLot(
                            lot_id=sell_lot.lot_id,
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=matched_qty,
                            remaining_quantity=0,
                            price=sell_lot.price,
                            timestamp=ts,
                            is_overnight=True,
                        )
                        ordinary_sales.append(ord_sale)
                        self._ordinary_sales[symbol].append(ord_sale)
                        tax_liability += ovn_tax
                        sell_lot.remaining_quantity -= matched_qty
                        ovn_lot.remaining_quantity -= matched_qty

                # 3. If shares still remain, this is an unhedged sell (potential sell-first day trade)
                if sell_lot.remaining_quantity > 0:
                    self._intraday_sells[symbol].append(sell_lot)
                    # Provisional ordinary tax estimate (collected by broker until same-day buy cover)
                    tax_liability += math.floor(sell_lot.remaining_quantity * sell_lot.price * ord_tax_rate)

            return new_matches, ordinary_sales, commission, tax_liability

    def generate_settlement_report(self, symbol: str, trade_date: Optional[date] = None) -> SettlementReport:
        """Generates the official daily settlement and tax report for a symbol."""
        with self._lock:
            dt = trade_date or datetime.now().date()
            matches = list(self._matched_day_trades.get(symbol, []))
            ord_sales = list(self._ordinary_sales.get(symbol, []))
            rem_ovn = [l for l in self._overnight_pools.get(symbol, []) if l.remaining_quantity > 0]
            rem_buys = [l for l in self._intraday_buys.get(symbol, []) if l.remaining_quantity > 0]

            is_etf = self._is_etf(symbol)
            ord_tax_rate = self.etf_tax_rate if is_etf else self.ordinary_tax_rate

            # Total statutory tax = (matched day trades * 0.15%) + (ordinary sales * 0.30%)
            dt_tax = sum(m.tax for m in matches)
            ord_tax = sum(math.floor(s.quantity * s.price * ord_tax_rate) for s in ord_sales)
            total_tax = dt_tax + ord_tax
            gross_pnl = sum(m.gross_pnl for m in matches)

            return SettlementReport(
                trade_date=dt,
                symbol=symbol,
                matched_day_trades=matches,
                ordinary_sales=ord_sales,
                remaining_overnight_inventory=rem_ovn,
                remaining_intraday_buys=rem_buys,
                total_statutory_tax=total_tax,
                total_gross_pnl=gross_pnl,
                total_net_pnl=gross_pnl - total_tax,
            )


class TaiwanFeeModel:
    """
    Taiwan Equity Fee and Taxation Model.
    
    Provides both:
    1. Real-time FeeBreakdown for intraday order validation & cash accounting.
    2. Deterministic TaiwanSettlementModel for FIFO lot matching and direct 0.15% tax clearing.
    """

    def __init__(
        self,
        commission_rate: float = 0.001425,
        commission_discount: float = 1.0,
        min_commission: float = 0.0,        # Configurable, default 0.0 (no universal 20 TWD rule)
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
        if self.min_commission > 0:
            return max(self.min_commission, math.floor(raw_comm))
        return float(math.floor(raw_comm))

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
                is_simulation_rebate=(tax_rebate > 0),
            )

    def reset_daily_activity(self) -> None:
        with self._lock:
            self._daily_activity.clear()
