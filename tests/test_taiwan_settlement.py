from datetime import date, datetime
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.execution.fees import (
    MatchedDayTrade,
    TaiwanLot,
    TaiwanSettlementModel,
)
from modules.execution.order import OrderSide


class TestTaiwanSettlementModel(unittest.TestCase):
    """
    Tests the canonical TaiwanSettlementModel:
    - Direct 0.15% statutory tax on matched day-trade quantity
    - Separation of overnight inventory from intraday inventory
    - Deterministic FIFO matching policy
    - Partial matching & multiple sequential fills
    - Broker commission independent from statutory tax
    - Configurable minimum commission (no universal 20 TWD assumption)
    """

    def setUp(self):
        self.today = date(2026, 9, 18)
        self.settlement = TaiwanSettlementModel(
            commission_rate=0.001425,
            commission_discount=1.0,
            min_commission=0.0,         # Default 0 min commission
            ordinary_tax_rate=0.003,    # 0.30%
            day_trade_tax_rate=0.0015,  # 0.15%
        )

    def test_buy_then_sell_direct_day_trade_tax(self):
        """Buy then Sell matches intraday lots directly with 0.15% statutory tax on sell proceeds."""
        # 1. Buy 1,000 shares @ 100.0
        matches, ord_sales, comm_buy, tax_buy = self.settlement.process_fill(
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=100.0,
        )
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(ord_sales), 0)
        self.assertEqual(tax_buy, 0.0)  # No tax on buy
        self.assertEqual(comm_buy, 142.0)

        # 2. Sell 1,000 shares @ 105.0 same day
        matches, ord_sales, comm_sell, tax_sell = self.settlement.process_fill(
            symbol="2330",
            side=OrderSide.SELL,
            quantity=1000,
            price=105.0,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(ord_sales), 0)
        m = matches[0]
        self.assertEqual(m.matched_quantity, 1000)
        self.assertEqual(m.buy_price, 100.0)
        self.assertEqual(m.sell_price, 105.0)
        # Direct statutory tax = 1,000 * 105.0 * 0.0015 = 157.5 -> 157.0
        self.assertEqual(m.tax, 157.0)
        self.assertEqual(tax_sell, 157.0)
        self.assertEqual(m.gross_pnl, 5000.0)

        # Final settlement report
        report = self.settlement.generate_settlement_report("2330", self.today)
        self.assertEqual(report.total_statutory_tax, 157.0)
        self.assertEqual(report.total_gross_pnl, 5000.0)
        self.assertEqual(report.total_net_pnl, 5000.0 - 157.0)
        self.assertEqual(len(report.remaining_intraday_buys), 0)
        self.assertEqual(len(report.remaining_overnight_inventory), 0)

    def test_sell_then_buy_short_day_trade_settlement(self):
        """Sell then Buy matches intraday lots with final statutory tax liability directly 0.15%."""
        # 1. Sell 1,000 shares @ 110.0 (Sell-first intraday trade)
        matches1, ord_sales1, comm1, _ = self.settlement.process_fill(
            symbol="2330",
            side=OrderSide.SELL,
            quantity=1000,
            price=110.0,
        )
        self.assertEqual(len(matches1), 0)

        # 2. Buy 1,000 shares @ 105.0 (Cover trade)
        matches2, ord_sales2, comm2, _ = self.settlement.process_fill(
            symbol="2330",
            side=OrderSide.BUY,
            quantity=1000,
            price=105.0,
        )
        self.assertEqual(len(matches2), 1)
        m = matches2[0]
        self.assertEqual(m.matched_quantity, 1000)
        self.assertEqual(m.sell_price, 110.0)
        self.assertEqual(m.buy_price, 105.0)
        # Final direct tax liability on sell proceeds: 1,000 * 110 * 0.0015 = 165
        self.assertEqual(m.tax, 165.0)
        self.assertEqual(m.gross_pnl, 5000.0)  # (110 - 105) * 1000

        # Official settlement report shows statutory tax is directly 165.0 (not 0.3% - refund)
        report = self.settlement.generate_settlement_report("2330", self.today)
        self.assertEqual(report.total_statutory_tax, 165.0)
        self.assertEqual(report.total_gross_pnl, 5000.0)
        self.assertEqual(report.total_net_pnl, 5000.0 - 165.0)

    def test_overnight_inventory_separation_and_fifo(self):
        """
        Overnight inventory is strictly separated from intraday inventory.
        Sales first match same-day buys (0.15%), then overnight inventory (0.30%).
        """
        # Seed 1,000 shares of overnight inventory carried from yesterday @ 90.0
        self.settlement.add_overnight_inventory("2330", quantity=1000, avg_price=90.0)

        # Intraday: Buy 500 shares @ 100.0
        self.settlement.process_fill("2330", OrderSide.BUY, 500, 100.0)

        # Intraday: Sell 1,200 shares @ 110.0
        # FIFO matching:
        # - 500 shares match intraday buy (day trade @ 0.15% = 500 * 110 * 0.0015 = 82)
        # - 700 shares match overnight inventory (ordinary sale @ 0.30% = 700 * 110 * 0.003 = 231)
        matches, ord_sales, comm, tax = self.settlement.process_fill("2330", OrderSide.SELL, 1200, 110.0)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].matched_quantity, 500)
        self.assertEqual(matches[0].tax, 82.0)

        self.assertEqual(len(ord_sales), 1)
        self.assertEqual(ord_sales[0].quantity, 700)

        report = self.settlement.generate_settlement_report("2330", self.today)
        # Total statutory tax = 82 + 231 = 313
        self.assertEqual(report.total_statutory_tax, 82.0 + 231.0)
        # Remaining overnight inventory: 1000 - 700 = 300 shares
        self.assertEqual(len(report.remaining_overnight_inventory), 1)
        self.assertEqual(report.remaining_overnight_inventory[0].remaining_quantity, 300)
        # Intraday buys exhausted
        self.assertEqual(len(report.remaining_intraday_buys), 0)

    def test_multiple_sequential_buys_and_sells(self):
        """Supports multiple sequential fills throughout the trading session."""
        # Buy 300 @ 100, Buy 300 @ 102
        self.settlement.process_fill("2330", OrderSide.BUY, 300, 100.0)
        self.settlement.process_fill("2330", OrderSide.BUY, 300, 102.0)

        # Sell 400 @ 105
        # Matches: 300 against first lot @ 100, 100 against second lot @ 102
        matches, _, _, _ = self.settlement.process_fill("2330", OrderSide.SELL, 400, 105.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].matched_quantity, 300)
        self.assertEqual(matches[0].buy_price, 100.0)
        self.assertEqual(matches[1].matched_quantity, 100)
        self.assertEqual(matches[1].buy_price, 102.0)

        # Remaining intraday buys should be 200 shares @ 102.0
        report = self.settlement.generate_settlement_report("2330", self.today)
        self.assertEqual(len(report.remaining_intraday_buys), 1)
        self.assertEqual(report.remaining_intraday_buys[0].remaining_quantity, 200)
        self.assertEqual(report.remaining_intraday_buys[0].price, 102.0)

    def test_independent_broker_commission_configurations(self):
        """Broker commission is configurable and independent of statutory tax."""
        # Case 1: Zero minimum commission with 60% discount (0.4 multiplier)
        discount_settlement = TaiwanSettlementModel(
            commission_rate=0.001425,
            commission_discount=0.4,
            min_commission=0.0,
        )
        # Small trade: 100 shares @ 50 = 5,000 gross value
        # raw comm = 5,000 * 0.001425 * 0.4 = 2.85 -> floor = 2
        comm1 = discount_settlement.calculate_commission(5000.0)
        self.assertEqual(comm1, 2.0)

        # Case 2: Broker with 20 TWD minimum commission
        min_comm_settlement = TaiwanSettlementModel(
            commission_rate=0.001425,
            commission_discount=1.0,
            min_commission=20.0,
        )
        comm2 = min_comm_settlement.calculate_commission(5000.0)
        # raw comm = 7.125 -> min_commission = 20
        self.assertEqual(comm2, 20.0)


if __name__ == "__main__":
    unittest.main()
