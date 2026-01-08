import pandas as pd
from typing import Dict, List, Any


class TechnicalAnalyzer:
    """Compute common technical indicators."""

    def analyze(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        data = df.copy()

        if "ma" in indicators:
            results["ma"] = self._moving_averages(data)

        if "rsi" in indicators:
            results["rsi"] = self._rsi(data, period=14)

        if "macd" in indicators:
            results["macd"] = self._macd(data)

        if "bollinger" in indicators:
            results["bollinger"] = self._bollinger(data)

        if "kdj" in indicators:
            results["kdj"] = self._kdj(data)

        return results

    def _moving_averages(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        periods = [5, 20, 60]
        data = df.copy()
        for p in periods:
            data[f"ma_{p}"] = data["close"].rolling(window=p).mean()
        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")

        return {
            "periods": periods,
            "data": data[["date"] + [f"ma_{p}" for p in periods]].to_dict(
                orient="records"
            ),
        }

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, List[Dict]]:
        data = df.copy()
        delta = data["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        data["rsi"] = rsi
        return {"period": period, "data": data[["date", "rsi"]].to_dict("records")}

    def _macd(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        data = df.copy()
        ema_fast = data["close"].ewm(span=12, adjust=False).mean()
        ema_slow = data["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal

        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        data["macd"] = macd_line
        data["signal"] = signal
        data["histogram"] = histogram

        return {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "data": data[["date", "macd", "signal", "histogram"]].to_dict("records"),
        }

    def _bollinger(self, df: pd.DataFrame, period: int = 20, num_std: float = 2.0):
        data = df.copy()
        ma = data["close"].rolling(window=period).mean()
        std = data["close"].rolling(window=period).std()
        upper = ma + num_std * std
        lower = ma - num_std * std

        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        data["middle"] = ma
        data["upper"] = upper
        data["lower"] = lower

        return {
            "period": period,
            "num_std": num_std,
            "data": data[["date", "upper", "middle", "lower"]].to_dict("records"),
        }

    def _kdj(self, df: pd.DataFrame, period: int = 9) -> Dict[str, List[Dict]]:
        data = df.copy()
        low_min = data["low"].rolling(window=period).min()
        high_max = data["high"].rolling(window=period).max()
        rsv = (data["close"] - low_min) / (high_max - low_min) * 100

        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        data = data.reset_index().rename(columns={"index": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
        data["k"] = k
        data["d"] = d
        data["j"] = j

        return {
            "period": period,
            "data": data[["date", "k", "d", "j"]].to_dict(orient="records"),
        }
