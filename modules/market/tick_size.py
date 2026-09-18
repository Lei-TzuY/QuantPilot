"""
Taiwan Equity Market Tick-Size Model (臺灣證券交易所股票升降單位模型)
Authoritative implementation conforming to TWSE Operating Rules Article 62 (營業細則第六十二條).
"""
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING
import math
from typing import Optional, Union
from modules.execution.order import OrderSide


class TaiwanTickSizeModel:
    """
    Implements the Taiwan Stock Exchange price bracket tick-size rules for ordinary equities.

    Price brackets:
    - P < 10.00: 0.01 TWD (includes boundaries P < 5 and 5 <= P < 10)
    - 10.00 <= P < 50.00: 0.05 TWD
    - 50.00 <= P < 100.00: 0.10 TWD
    - 100.00 <= P < 500.00: 0.50 TWD (includes boundaries 100 <= P < 150 and 150 <= P < 500)
    - 500.00 <= P < 1000.00: 1.00 TWD
    - P >= 1000.00: 5.00 TWD (configurable for future TWSE 1.00 rule)
    """

    def __init__(self, high_price_tick: float = 5.0):
        self.high_price_tick = high_price_tick

    @classmethod
    def get_tick_size(cls, price: float) -> float:
        """Returns the mandatory minimum price movement for a given equity price in TWD."""
        if price < 10.0:
            return 0.01
        elif price < 50.0:
            return 0.05
        elif price < 100.0:
            return 0.10
        elif price < 500.0:
            return 0.50
        elif price < 1000.0:
            return 1.00
        else:
            return 5.00

    @classmethod
    def is_valid_tick(cls, price: float, tolerance: float = 1e-5) -> bool:
        """Determines whether a price strictly conforms to an exchange tick boundary."""
        if price <= 0:
            return False
        tick = cls.get_tick_size(price)
        remainder = round((price / tick) % 1.0, 6)
        return remainder < tolerance or abs(remainder - 1.0) < tolerance

    @classmethod
    def round_to_tick(cls, price: float, mode: str = "nearest") -> float:
        """
        Rounds an arbitrary price to the nearest, floor, or ceiling exchange tick.
        Uses Decimal arithmetic to prevent IEEE-754 floating-point inaccuracies.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        tick = cls.get_tick_size(price)
        d_price = Decimal(str(round(price, 6)))
        d_tick = Decimal(str(tick))

        quotient = d_price / d_tick
        if mode == "nearest":
            rounded_units = quotient.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        elif mode == "floor":
            rounded_units = quotient.quantize(Decimal("1"), rounding=ROUND_FLOOR)
        elif mode == "ceil":
            rounded_units = quotient.quantize(Decimal("1"), rounding=ROUND_CEILING)
        else:
            raise ValueError(f"Unknown rounding mode: {mode}")

        result = float(rounded_units * d_tick)
        # Handle boundary crossover if result changed tick bracket
        new_tick = cls.get_tick_size(result)
        if new_tick != tick and not cls.is_valid_tick(result):
            return cls.round_to_tick(result, mode=mode)
        return round(result, 4)

    @classmethod
    def normalize_limit_price(cls, price: float, side: Union[str, OrderSide]) -> float:
        """
        Side-aware price normalization for limit orders:
        - BUY limit: floors to tick (never exceeds maximum willing purchase price).
        - SELL limit: ceils to tick (never undercuts minimum acceptable selling price).
        """
        side_val = side.value if isinstance(side, OrderSide) else str(side).upper()
        if side_val == "BUY":
            return cls.round_to_tick(price, mode="floor")
        elif side_val == "SELL":
            return cls.round_to_tick(price, mode="ceil")
        return cls.round_to_tick(price, mode="nearest")

    @classmethod
    def apply_execution_slippage(
        cls,
        base_price: float,
        side: Union[str, OrderSide],
        slippage_pct: float = 0.0,
        slippage_bps: Optional[float] = None,
    ) -> float:
        """
        Applies slippage and rounds conservatively to the next valid exchange tick:
        - BUY execution: price slips UP -> round UP (ceil) to legal tick.
        - SELL execution: price slips DOWN -> round DOWN (floor) to legal tick.
        """
        if slippage_bps is not None:
            slippage_pct = slippage_bps / 10000.0

        side_val = side.value if isinstance(side, OrderSide) else str(side).upper()
        if side_val == "BUY":
            raw_slipped = base_price * (1.0 + max(0.0, slippage_pct))
            return cls.round_to_tick(raw_slipped, mode="ceil")
        else:
            raw_slipped = base_price * (1.0 - max(0.0, slippage_pct))
            return cls.round_to_tick(raw_slipped, mode="floor")

    @classmethod
    def next_tick(cls, price: float, steps: int = 1) -> float:
        """Moves forward by N ticks."""
        p = price
        for _ in range(steps):
            tick = cls.get_tick_size(p)
            p = round(p + tick, 4)
        return p

    @classmethod
    def prev_tick(cls, price: float, steps: int = 1) -> float:
        """Moves backward by N ticks."""
        p = price
        for _ in range(steps):
            tick = cls.get_tick_size(p)
            # If at exact boundary, previous tick may be from lower bracket
            if p in (10.0, 50.0, 100.0, 500.0, 1000.0):
                tick = cls.get_tick_size(round(p - 0.001, 4))
            p = max(0.01, round(p - tick, 4))
        return p


# Global singleton instance with standard TWSE parameters
default_tick_model = TaiwanTickSizeModel()
