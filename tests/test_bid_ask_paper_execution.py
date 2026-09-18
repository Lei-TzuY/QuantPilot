"""
Bid/Ask Aware Paper Execution Tests.
Verifies realistic simulated execution against top-of-book quotes:
BUY market orders execute against ask price (+ slippage).
SELL market orders execute against bid price (- slippage).
Limit order crossing semantics against executable top-of-book bid/ask quotes.
"""
from datetime import datetime
import unittest

from modules.brokers.paper import PaperBrokerAdapter
from modules.execution.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce


class TestBidAskPaperExecution(unittest.TestCase):

    def setUp(self):
        self.broker = PaperBrokerAdapter(
            initial_cash=10_000_000.0,
            commission_rate=0.001425,
            tax_rate=0.003,
            slippage_pct=0.0005,  # 5 bps slippage
        )
        self.broker.connect()

    def test_market_buy_executes_against_ask_plus_slippage(self):
        fills = []
        self.broker.register_fill_callback(fills.append)

        # Set market quote: Bid 950.0, Ask 952.0, Last 951.0
        self.broker.set_market_quote(symbol="2330", price=951.0, bid_price=950.0, ask_price=952.0)

        order = Order(
            order_id="ORD-BUY-MKT",
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        submitted = self.broker.submit_order(order)
        self.assertEqual(submitted.status, OrderStatus.FILLED)
        self.assertEqual(len(fills), 1)

        fill = fills[0]
        expected_price = round(952.0 * (1 + 0.0005), 2)
        self.assertEqual(fill.price, expected_price)
        self.assertEqual(submitted.average_fill_price, expected_price)
        self.assertGreaterEqual(fill.price, 952.0)

    def test_market_sell_executes_against_bid_minus_slippage(self):
        fills = []
        self.broker.register_fill_callback(fills.append)

        # First buy shares so position exists
        self.broker.set_market_quote(symbol="2330", price=950.0, bid_price=950.0, ask_price=950.0)
        buy_order = Order(
            order_id="ORD-BUY-PRE",
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000,
        )
        self.broker.submit_order(buy_order)
        fills.clear()

        # Set market quote: Bid 948.0, Ask 951.0, Last 950.0
        self.broker.set_market_quote(symbol="2330", price=950.0, bid_price=948.0, ask_price=951.0)

        sell_order = Order(
            order_id="ORD-SELL-MKT",
            symbol="2330",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1000,
        )

        submitted = self.broker.submit_order(sell_order)
        self.assertEqual(submitted.status, OrderStatus.FILLED)
        self.assertEqual(len(fills), 1)

        fill = fills[0]
        expected_price = round(948.0 * (1 - 0.0005), 2)
        self.assertEqual(fill.price, expected_price)
        self.assertEqual(submitted.average_fill_price, expected_price)
        self.assertLessEqual(fill.price, 948.0)

    def test_limit_buy_crossing_semantics(self):
        fills = []
        self.broker.register_fill_callback(fills.append)

        # Quote: Bid 948.0, Ask 952.0. Limit BUY at 950.0
        self.broker.set_market_quote(symbol="2330", price=950.0, bid_price=948.0, ask_price=952.0)

        limit_buy = Order(
            order_id="ORD-LIMIT-BUY-1",
            symbol="2330",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=950.0,
        )

        # Ask is 952.0 > Limit 950.0 -> Cannot fill yet! Must stay ACCEPTED in book
        submitted = self.broker.submit_order(limit_buy)
        self.assertEqual(submitted.status, OrderStatus.ACCEPTED)
        self.assertEqual(submitted.filled_quantity, 0)
        self.assertEqual(len(fills), 0)

        # Market ask drops to 949.0 (crosses below limit of 950.0)
        self.broker.set_market_quote(symbol="2330", price=949.0, bid_price=948.0, ask_price=949.0)

        # Re-evaluate open orders against new quote
        self.broker._process_open_orders("2330")
        updated = self.broker.get_order("ORD-LIMIT-BUY-1")
        self.assertEqual(updated.status, OrderStatus.FILLED)
        self.assertEqual(len(fills), 1)
        self.assertLessEqual(fills[0].price, 950.0)

    def test_limit_sell_crossing_semantics(self):
        fills = []
        self.broker.register_fill_callback(fills.append)

        # Pre-seed position
        self.broker.set_market_quote(symbol="2330", price=950.0, bid_price=950.0, ask_price=950.0)
        self.broker.submit_order(Order(order_id="B1", symbol="2330", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000))
        fills.clear()

        # Quote: Bid 948.0, Ask 952.0. Limit SELL at 955.0
        self.broker.set_market_quote(symbol="2330", price=950.0, bid_price=948.0, ask_price=952.0)

        limit_sell = Order(
            order_id="ORD-LIMIT-SELL-1",
            symbol="2330",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=1000,
            price=955.0,
        )

        # Bid is 948.0 < Limit 955.0 -> Cannot fill yet
        submitted = self.broker.submit_order(limit_sell)
        self.assertEqual(submitted.status, OrderStatus.ACCEPTED)
        self.assertEqual(submitted.filled_quantity, 0)
        self.assertEqual(len(fills), 0)

        # Bid rises to 956.0 (crosses above limit 955.0)
        self.broker.set_market_quote(symbol="2330", price=956.0, bid_price=956.0, ask_price=958.0)
        self.broker._process_open_orders("2330")

        updated = self.broker.get_order("ORD-LIMIT-SELL-1")
        self.assertEqual(updated.status, OrderStatus.FILLED)
        self.assertEqual(len(fills), 1)
        self.assertGreaterEqual(fills[0].price, 955.0)


if __name__ == "__main__":
    unittest.main()
