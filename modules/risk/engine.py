"""
Risk Engine
Enforces pre-trade risk controls, loss limits, stale data checks, and kill switch integration.
"""
from datetime import datetime
import threading
from typing import Dict, Optional

from modules.execution.order import OrderRequest, OrderSide
from modules.execution.position import Position
from modules.execution.events import RiskDecision
from modules.risk.limits import RiskLimits
from modules.risk.kill_switch import KillSwitch


class RiskEngine:
    """
    Institutional Pre-Trade Risk Engine.
    Gates EVERY order before submission to any BrokerAdapter.
    """

    def __init__(self, limits: Optional[RiskLimits] = None, kill_switch: Optional[KillSwitch] = None):
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()
        self._lock = threading.RLock()

        # Daily tracking metrics
        self._daily_trades_count = 0
        self._daily_realized_loss = 0.0
        self._current_date = datetime.now().date()

    def _check_and_reset_daily_counters(self) -> None:
        today = datetime.now().date()
        if today != self._current_date:
            self._current_date = today
            self._daily_trades_count = 0
            self._daily_realized_loss = 0.0

    def record_trade_execution(self, realized_pnl: float = 0.0) -> None:
        """Records trade completion and accumulates daily loss tracking."""
        with self._lock:
            self._check_and_reset_daily_counters()
            self._daily_trades_count += 1
            if realized_pnl < 0:
                self._daily_realized_loss += abs(realized_pnl)

            # Auto-halt if daily loss limit breached
            if self._daily_realized_loss >= self.limits.max_daily_realized_loss:
                self.kill_switch.halt(
                    reason=f"DAILY_LOSS_LIMIT_BREACH: Realized loss {self._daily_realized_loss:.2f} "
                           f"exceeded limit {self.limits.max_daily_realized_loss:.2f}"
                )

    def evaluate_order(
        self,
        request: OrderRequest,
        current_positions: Dict[str, Position],
        market_price: Optional[float] = None,
        market_price_timestamp: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
        strategy_ready: Optional[bool] = None,
        market_data_healthy: Optional[bool] = None,
        bidask_healthy: Optional[bool] = None,
    ) -> RiskDecision:
        """
        Pre-trade risk gate. Evaluates an OrderRequest against all configured limits.
        """
        with self._lock:
            self._check_and_reset_daily_counters()
            eval_time = current_time or datetime.now()

            # 0a. Strategy Warmup Check
            if strategy_ready is False:
                return RiskDecision(
                    allowed=False,
                    reason=f"STRATEGY_NOT_WARMED_UP: Strategy {request.strategy_id} has not completed historical warmup lookback window",
                    rule_violated="STRATEGY_NOT_WARMED_UP",
                )

            # 0b. Market Data Integrity / Health Check
            if market_data_healthy is False:
                return RiskDecision(
                    allowed=False,
                    reason=f"UNHEALTHY_MARKET_DATA: Market data stream for {request.symbol} is not in HEALTHY state",
                    rule_violated="UNHEALTHY_MARKET_DATA",
                )

            # 0c. Stale BidAsk Spread Check
            if bidask_healthy is False:
                return RiskDecision(
                    allowed=False,
                    reason=f"STALE_BIDASK_SPREAD: Market BidAsk stream for {request.symbol} is stale or degraded",
                    rule_violated="STALE_BIDASK_SPREAD",
                )

            # 1. Kill Switch Check
            if self.kill_switch.is_halted():
                status = self.kill_switch.get_status()
                return RiskDecision(
                    allowed=False,
                    reason=f"KILL_SWITCH_HALTED: {status.reason}",
                    rule_violated="KILL_SWITCH",
                )

            # 2. Daily Trade Count Check
            if self._daily_trades_count >= self.limits.max_trades_per_day:
                return RiskDecision(
                    allowed=False,
                    reason=f"DAILY_TRADE_LIMIT_EXCEEDED: count {self._daily_trades_count} >= max {self.limits.max_trades_per_day}",
                    rule_violated="MAX_TRADES_PER_DAY",
                )

            # 3. Daily Loss Limit Check
            if self._daily_realized_loss >= self.limits.max_daily_realized_loss:
                return RiskDecision(
                    allowed=False,
                    reason=f"DAILY_LOSS_LIMIT_EXCEEDED: loss {self._daily_realized_loss:.2f} >= max {self.limits.max_daily_realized_loss:.2f}",
                    rule_violated="MAX_DAILY_REALIZED_LOSS",
                )

            # Reference price determination
            ref_price = request.price or market_price
            if ref_price is None or ref_price <= 0:
                return RiskDecision(
                    allowed=False,
                    reason="NO_VALID_PRICE: Cannot determine order value without market or limit price",
                    rule_violated="INVALID_PRICE",
                )

            # 4. Stale Market Data Check
            if market_price_timestamp is not None:
                age_seconds = (eval_time - market_price_timestamp).total_seconds()
                if age_seconds > self.limits.max_stale_data_seconds:
                    return RiskDecision(
                        allowed=False,
                        reason=f"STALE_MARKET_DATA: Data age {age_seconds:.1f}s exceeds threshold {self.limits.max_stale_data_seconds}s",
                        rule_violated="STALE_MARKET_DATA",
                    )

            # 5. Price Deviation Check (if both limit and market price provided)
            if request.price is not None and market_price is not None and market_price > 0:
                deviation = abs(request.price - market_price) / market_price
                if deviation > self.limits.max_price_deviation_pct:
                    return RiskDecision(
                        allowed=False,
                        reason=f"PRICE_DEVIATION_TOO_HIGH: {deviation*100:.2f}% exceeds limit {self.limits.max_price_deviation_pct*100:.2f}%",
                        rule_violated="PRICE_DEVIATION",
                    )

            order_value = request.quantity * ref_price

            # 6. Maximum Order Value Check
            if order_value > self.limits.max_order_value:
                return RiskDecision(
                    allowed=False,
                    reason=f"ORDER_VALUE_EXCEEDS_LIMIT: {order_value:.2f} > max {self.limits.max_order_value:.2f}",
                    rule_violated="MAX_ORDER_VALUE",
                )

            # 7. Position Limits (only for BUY orders adding to exposure)
            if request.side == OrderSide.BUY:
                current_pos = current_positions.get(request.symbol)
                current_qty = current_pos.quantity if current_pos else 0
                new_qty = current_qty + request.quantity
                new_pos_value = new_qty * ref_price

                # Max position value per symbol
                if new_pos_value > self.limits.max_position_value_per_symbol:
                    return RiskDecision(
                        allowed=False,
                        reason=f"SYMBOL_EXPOSURE_EXCEEDED: projected {new_pos_value:.2f} > max {self.limits.max_position_value_per_symbol:.2f} for {request.symbol}",
                        rule_violated="MAX_SYMBOL_EXPOSURE",
                    )

                # Max open positions count
                is_new_symbol = current_pos is None or current_pos.quantity == 0
                active_symbols_count = sum(1 for p in current_positions.values() if p.quantity > 0)
                if is_new_symbol and active_symbols_count >= self.limits.max_open_positions:
                    return RiskDecision(
                        allowed=False,
                        reason=f"MAX_OPEN_POSITIONS_REACHED: {active_symbols_count} >= max {self.limits.max_open_positions}",
                        rule_violated="MAX_OPEN_POSITIONS",
                    )

                # Max total portfolio exposure
                current_total_exposure = sum(
                    p.quantity * (market_price if p.symbol == request.symbol else p.last_price or p.avg_price)
                    for p in current_positions.values()
                )
                if (current_total_exposure + order_value) > self.limits.max_total_exposure:
                    return RiskDecision(
                        allowed=False,
                        reason=f"TOTAL_EXPOSURE_EXCEEDED: projected {current_total_exposure + order_value:.2f} > max {self.limits.max_total_exposure:.2f}",
                        rule_violated="MAX_TOTAL_EXPOSURE",
                    )

            # All risk checks passed
            return RiskDecision(allowed=True, reason="APPROVED")
