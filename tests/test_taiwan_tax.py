from datetime import date
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.execution.fees import TaiwanFeeModel
from modules.execution.order import OrderSide


class TestTaiwanTaxModel(unittest.TestCase):

    def setUp(self):
        self.fee_model = TaiwanFeeModel(
            commission_rate=0.001425,
            commission_discount=1.0,
            min_commission=20.0,
            ordinary_tax_rate=0.003,    # 0.3%
            day_trade_tax_rate=0.0015,  # 0.15%
            slippage_pct=0.001,         # 0.1%
        )
        self.today = date(2026, 9, 18)
        self.tomorrow = date(2026, 9, 19)

    def test_ordinary_stock_sale_tax(self):
        """Sale without any same-day purchase is taxed at standard 0.3%."""
        # Sell 1,000 shares of 2330 @ 100.0 on self.today
        breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )

        self.assertEqual(breakdown.gross_value, 100_000.0)
        self.assertEqual(breakdown.day_trade_quantity, 0)
        self.assertEqual(breakdown.ordinary_quantity, 1000)
        # 100,000 * 0.003 = 300
        self.assertEqual(breakdown.tax, 300.0)
        # Commission = max(20, floor(100,000 * 0.001425)) = 142
        self.assertEqual(breakdown.commission, 142.0)
        # Slippage = 100,000 * 0.001 = 100
        self.assertEqual(breakdown.slippage, 100.0)

    def test_buy_then_same_day_sell(self):
        """Buying and selling same-day qualifies for 0.15% day-trade tax rate."""
        # Buy 1,000 shares @ 100.0 on self.today
        self.fee_model.calculate_execution_costs(
            side=OrderSide.BUY,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )

        # Sell 1,000 shares @ 105.0 on self.today
        sell_breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=1000,
            price=105.0,
            symbol="2330",
            trade_date=self.today,
        )

        self.assertEqual(sell_breakdown.gross_value, 105_000.0)
        self.assertEqual(sell_breakdown.day_trade_quantity, 1000)
        self.assertEqual(sell_breakdown.ordinary_quantity, 0)
        # Day-trade tax: 105,000 * 0.0015 = 157.5 -> floor = 157
        self.assertEqual(sell_breakdown.tax, 157.0)

    def test_partial_same_day_offset(self):
        """
        Selling more than the same-day buy quantity:
        Same-day buy quantity gets 0.15% tax, remaining overnight shares get 0.3% tax.
        """
        # Buy 1,000 shares on self.today
        self.fee_model.calculate_execution_costs(
            side=OrderSide.BUY,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )

        # Sell 1,500 shares on self.today (1,000 same-day + 500 overnight)
        sell_breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=1500,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )

        self.assertEqual(sell_breakdown.day_trade_quantity, 1000)
        self.assertEqual(sell_breakdown.ordinary_quantity, 500)
        # Tax = (1000 * 100 * 0.0015) + (500 * 100 * 0.003) = 150 + 150 = 300
        self.assertEqual(sell_breakdown.tax, 300.0)

    def test_remaining_overnight_quantity_next_day(self):
        """
        Shares bought today but sold tomorrow do NOT qualify for day-trading tax.
        They are taxed at the full ordinary 0.3% rate.
        """
        # Buy 1,000 shares on self.today
        self.fee_model.calculate_execution_costs(
            side=OrderSide.BUY,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )

        # Sell 1,000 shares tomorrow
        sell_breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.tomorrow,
        )

        self.assertEqual(sell_breakdown.day_trade_quantity, 0)
        self.assertEqual(sell_breakdown.ordinary_quantity, 1000)
        # Standard tax = 100,000 * 0.003 = 300
        self.assertEqual(sell_breakdown.tax, 300.0)

    def test_multiple_fills_day_trade_offset(self):
        """Day-trade offset tracking works accurately across multiple partial fills."""
        # Morning Buy 1: 300 shares
        self.fee_model.calculate_execution_costs(OrderSide.BUY, 300, 100.0, "2330", trade_date=self.today)
        # Morning Buy 2: 700 shares (Total bought = 1000)
        self.fee_model.calculate_execution_costs(OrderSide.BUY, 700, 100.0, "2330", trade_date=self.today)

        # Afternoon Sell 1: 400 shares (offsets 400 of 1000)
        s1 = self.fee_model.calculate_execution_costs(OrderSide.SELL, 400, 102.0, "2330", trade_date=self.today)
        self.assertEqual(s1.day_trade_quantity, 400)
        self.assertEqual(s1.ordinary_quantity, 0)
        self.assertEqual(s1.tax, 61.0)  # 40,800 * 0.0015 = 61.2 -> 61

        # Afternoon Sell 2: 800 shares (offsets remaining 600 of 1000, 200 overnight)
        s2 = self.fee_model.calculate_execution_costs(OrderSide.SELL, 800, 102.0, "2330", trade_date=self.today)
        self.assertEqual(s2.day_trade_quantity, 600)
        self.assertEqual(s2.ordinary_quantity, 200)
        # Tax = (600 * 102 * 0.0015) + (200 * 102 * 0.003) = 91.8 + 61.2 = 153.0
        self.assertEqual(s2.tax, 153.0)

    def test_sell_then_same_day_buy_rebate(self):
        """
        When selling first and covering with a same-day buy,
        the model rebates the 0.15% tax difference.
        """
        # Sell 1,000 shares @ 100.0 first (charged 0.3% initially = 300)
        sell_breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=1000,
            price=100.0,
            symbol="2330",
            trade_date=self.today,
        )
        self.assertEqual(sell_breakdown.tax, 300.0)

        # Cover later with Buy 1,000 shares @ 95.0 on the same day
        buy_breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.BUY,
            quantity=1000,
            price=95.0,
            symbol="2330",
            trade_date=self.today,
        )
        # Tax rebate = 1,000 * 100.0 * (0.003 - 0.0015) = 150.0
        self.assertAlmostEqual(buy_breakdown.tax_rebate, 150.0, places=2)

    def test_combined_commission_tax_slippage(self):
        """Tests combined commission, tax, and slippage cost breakdown."""
        breakdown = self.fee_model.calculate_execution_costs(
            side=OrderSide.SELL,
            quantity=2000,
            price=50.0,
            symbol="2330",
            trade_date=self.today,
        )
        gross = 100_000.0
        self.assertEqual(breakdown.gross_value, gross)
        self.assertEqual(breakdown.commission, 142.0)
        self.assertEqual(breakdown.tax, 300.0)
        self.assertEqual(breakdown.slippage, 100.0)
        self.assertEqual(breakdown.total_cost, 142.0 + 300.0 + 100.0)


if __name__ == "__main__":
    unittest.main()
