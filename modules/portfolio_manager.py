from typing import Dict, List


class PortfolioManager:
    """In-memory portfolio tracker."""

    def __init__(self):
        self.positions: Dict[str, Dict] = {}

    def add_position(self, symbol: str, shares: float, buy_price: float) -> Dict:
        if shares <= 0:
            raise ValueError("Shares must be positive")
        if buy_price <= 0:
            raise ValueError("Buy price must be positive")

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_cost = pos["avg_price"] * pos["shares"] + buy_price * shares
            total_shares = pos["shares"] + shares
            pos["shares"] = total_shares
            pos["avg_price"] = total_cost / total_shares
        else:
            self.positions[symbol] = {
                "symbol": symbol,
                "shares": shares,
                "avg_price": buy_price,
            }
        return self.positions[symbol]

    def remove_position(self, symbol: str) -> Dict:
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} not found")
        return self.positions.pop(symbol)

    def get_portfolio(self) -> List[Dict]:
        return list(self.positions.values())

    def get_performance(self, data_fetcher) -> Dict:
        total_cost = 0.0
        total_value = 0.0
        breakdown: List[Dict] = []

        for symbol, pos in self.positions.items():
            try:
                price = data_fetcher.get_realtime_price(symbol)
            except Exception:
                # If realtime fetch fails, skip price update for this symbol.
                price = None

            market_value = pos["shares"] * price if price else None
            cost = pos["shares"] * pos["avg_price"]
            total_cost += cost
            if market_value is not None:
                total_value += market_value

            breakdown.append(
                {
                    "symbol": symbol,
                    "shares": pos["shares"],
                    "avg_price": pos["avg_price"],
                    "last_price": price,
                    "market_value": market_value,
                    "pnl": market_value - cost if market_value is not None else None,
                }
            )

        total_pnl = total_value - total_cost if total_value else None
        return {
            "positions": breakdown,
            "total_cost": total_cost,
            "total_value": total_value,
            "total_pnl": total_pnl,
        }
