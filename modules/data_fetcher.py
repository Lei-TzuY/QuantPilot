import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


class DataFetcher:
    """Fetch stock data and basic info using yfinance."""

    def _format_symbol(self, symbol: str) -> str:
        # Append .TW for numeric Taiwan stock codes; otherwise return as-is.
        if symbol.isdigit():
            return f"{symbol}.TW"
        return symbol

    def get_stock_data_df(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        ticker = self._format_symbol(symbol)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No price data found for {symbol}")
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df.index = pd.to_datetime(df.index)
        return df

    def get_stock_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> List[Dict]:
        df = self.get_stock_data_df(symbol, period=period, interval=interval)
        df = df.reset_index().rename(columns={"Date": "date"})
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.to_dict(orient="records")

    def get_stock_info(self, symbol: str) -> Dict:
        ticker = yf.Ticker(self._format_symbol(symbol))

        info: Dict[str, Optional[str]] = {}
        try:
            fast = ticker.fast_info
            info.update(
                {
                    "symbol": symbol,
                    "currency": getattr(fast, "currency", None),
                    "exchange": getattr(fast, "exchange", None),
                    "last_price": float(fast.last_price)
                    if getattr(fast, "last_price", None)
                    else None,
                    "fifty_two_week_low": float(fast.year_low)
                    if getattr(fast, "year_low", None)
                    else None,
                    "fifty_two_week_high": float(fast.year_high)
                    if getattr(fast, "year_high", None)
                    else None,
                }
            )
        except Exception:
            pass

        # Fall back to .info for extra fields if available.
        try:
            raw_info = ticker.info
            for key in [
                "shortName",
                "longName",
                "sector",
                "industry",
                "marketCap",
                "trailingPE",
                "forwardPE",
                "dividendYield",
            ]:
                if key in raw_info:
                    info[key] = raw_info.get(key)
        except Exception:
            # .info can be slow or throttled; ignore failures.
            pass
        return info

    def get_realtime_price(self, symbol: str) -> float:
        ticker = yf.Ticker(self._format_symbol(symbol))
        # Try fast_info first.
        try:
            price = float(ticker.fast_info.last_price)
            if price:
                return price
        except Exception:
            pass

        # Fallback: last close from 1-day history.
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty:
            raise ValueError(f"Realtime price unavailable for {symbol}")
        return float(hist["Close"].iloc[-1])
