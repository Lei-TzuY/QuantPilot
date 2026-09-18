"""
Unit tests for TaiwanTickSizeModel.
Verifies TWSE Article 62 equity price brackets, boundary transitions,
side-aware limit normalization, and slippage calculations.
"""
import unittest

from modules.market.tick_size import TaiwanTickSizeModel


class TestTaiwanTickSizeModel(unittest.TestCase):
    def test_tick_sizes_by_bracket(self):
        # Under 10 TWD -> 0.01 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(0.01), 0.01)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(4.99), 0.01)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(5.00), 0.01)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(9.99), 0.01)

        # 10 to 50 TWD -> 0.05 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(10.00), 0.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(25.50), 0.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(49.95), 0.05)

        # 50 to 100 TWD -> 0.10 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(50.00), 0.10)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(75.50), 0.10)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(99.90), 0.10)

        # 100 to 500 TWD -> 0.50 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(100.00), 0.50)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(149.50), 0.50)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(150.00), 0.50)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(499.50), 0.50)

        # 500 to 1000 TWD -> 1.00 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(500.00), 1.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(950.00), 1.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(999.00), 1.00)

        # 1000+ TWD -> 5.00 TWD
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(1000.00), 5.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(1200.00), 5.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.get_tick_size(2500.00), 5.00)

    def test_is_valid_tick(self):
        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(9.99))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(9.995))

        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(10.05))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(10.02))

        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(50.10))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(50.05))

        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(100.50))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(100.25))

        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(501.00))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(500.50))

        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(1005.00))
        self.assertFalse(TaiwanTickSizeModel.is_valid_tick(1002.00))

    def test_rounding_modes(self):
        # 10.02 in 10-50 bracket (tick=0.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(10.02, "nearest"), 10.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(10.03, "nearest"), 10.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(10.04, "floor"), 10.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(10.01, "ceil"), 10.05)

        # 950.40 in 500-1000 bracket (tick=1.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(950.40, "nearest"), 950.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(950.60, "nearest"), 951.00)

        # 1002.00 in >=1000 bracket (tick=5.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(1002.00, "nearest"), 1000.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(1003.00, "nearest"), 1005.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(1004.00, "floor"), 1000.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.round_to_tick(1001.00, "ceil"), 1005.00)

    def test_limit_order_normalization(self):
        # BUY limit price should floor (never cross above intended max price)
        self.assertAlmostEqual(TaiwanTickSizeModel.normalize_limit_price(10.04, "BUY"), 10.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.normalize_limit_price(1003.00, "BUY"), 1000.00)

        # SELL limit price should ceil (never cross below intended min price)
        self.assertAlmostEqual(TaiwanTickSizeModel.normalize_limit_price(10.01, "SELL"), 10.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.normalize_limit_price(1002.00, "SELL"), 1005.00)

    def test_next_and_prev_tick_across_boundaries(self):
        # Transition at 10 TWD: 9.99 + 0.01 = 10.00; 10.00 + 0.05 = 10.05
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(9.99), 10.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(10.00), 10.05)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(10.05), 10.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(10.00), 9.99)

        # Transition at 50 TWD: 49.95 + 0.05 = 50.00; 50.00 + 0.10 = 50.10
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(49.95), 50.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(50.00), 50.10)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(50.10), 50.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(50.00), 49.95)

        # Transition at 100 TWD: 99.90 + 0.10 = 100.00; 100.00 + 0.50 = 100.50
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(99.90), 100.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(100.00), 100.50)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(100.50), 100.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(100.00), 99.90)

        # Transition at 500 TWD: 499.50 + 0.50 = 500.00; 500.00 + 1.00 = 501.00
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(499.50), 500.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(500.00), 501.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(501.00), 500.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(500.00), 499.50)

        # Transition at 1000 TWD: 999.00 + 1.00 = 1000.00; 1000.00 + 5.00 = 1005.00
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(999.00), 1000.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.next_tick(1000.00), 1005.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(1005.00), 1000.00)
        self.assertAlmostEqual(TaiwanTickSizeModel.prev_tick(1000.00), 999.00)

    def test_apply_execution_slippage(self):
        # BUY fills at or above market price, on valid tick
        fill_buy = TaiwanTickSizeModel.apply_execution_slippage(1000.00, "BUY", slippage_bps=5.0)
        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(fill_buy))
        self.assertGreaterEqual(fill_buy, 1000.00)

        # SELL fills at or below market price, on valid tick
        fill_sell = TaiwanTickSizeModel.apply_execution_slippage(1000.00, "SELL", slippage_bps=5.0)
        self.assertTrue(TaiwanTickSizeModel.is_valid_tick(fill_sell))
        self.assertLessEqual(fill_sell, 1000.00)


if __name__ == "__main__":
    unittest.main()
