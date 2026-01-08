from typing import List, Dict

import pandas as pd


class SignalGenerator:
    """Generate basic trading signals."""

    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        data = df.copy()
        data = data.sort_index()
        data["ma_short"] = data["close"].rolling(window=20).mean()
        data["ma_long"] = data["close"].rolling(window=60).mean()
        data["ma_cross"] = data["ma_short"] - data["ma_long"]
        data["cross_signal"] = data["ma_cross"].diff()

        signals: List[Dict] = []
        for date, row in data.iterrows():
            direction = None
            reason = ""

            if row.get("cross_signal", 0) > 0:
                direction = "BUY"
                reason = "MA20 crossed above MA60"
            elif row.get("cross_signal", 0) < 0:
                direction = "SELL"
                reason = "MA20 crossed below MA60"

            if direction:
                signals.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "price": float(row["close"]),
                        "signal": direction,
                        "reason": reason,
                    }
                )

        return signals
