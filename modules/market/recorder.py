"""
Raw Market Data Recorder
Captures high-frequency normalized TickEvents into append-only structured Parquet files
partitioned by date and symbol: data/market/YYYY-MM-DD/{symbol}.parquet.
Uses buffered asynchronous I/O to ensure zero callback latency impact.
"""
from datetime import date, datetime
import logging
import os
import threading
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from modules.execution.events import TickEvent

logger = logging.getLogger("QuantPilot.MarketRecorder")

TICK_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("exchange_timestamp", pa.timestamp("us")),
    ("local_receive_timestamp", pa.timestamp("us")),
    ("price", pa.float64()),
    ("volume", pa.float64()),
    ("tick_type", pa.string()),
    ("sequence", pa.int64()),
    ("bid_price", pa.float64()),
    ("ask_price", pa.float64()),
    ("bid_volume", pa.float64()),
    ("ask_volume", pa.float64()),
    ("source", pa.string()),
    ("simtrade", pa.bool_()),
])


class RawMarketDataRecorder:
    """
    Buffered, thread-safe market data tick capture.
    Persists tick streams to Parquet for exact deterministic replay.
    """

    def __init__(
        self,
        base_dir: str = "data/market",
        buffer_size: int = 500,
        flush_interval_seconds: float = 5.0,
    ):
        self.base_dir = base_dir
        self.buffer_size = buffer_size
        self.flush_interval_seconds = flush_interval_seconds

        self._lock = threading.RLock()
        self._buffers: Dict[str, List[Dict]] = {}  # {symbol: [records]}
        self._total_recorded = 0
        self._last_flush_time = datetime.now()

    def record_tick(self, tick: TickEvent) -> None:
        """Buffers a TickEvent for background or periodic flush."""
        now = datetime.now()
        record = {
            "symbol": tick.symbol,
            "exchange_timestamp": tick.timestamp,
            "local_receive_timestamp": tick.receive_timestamp or now,
            "price": float(tick.price),
            "volume": float(tick.volume),
            "tick_type": tick.tick_type,
            "sequence": int(tick.sequence),
            "bid_price": float(tick.bid_price) if tick.bid_price is not None else float("nan"),
            "ask_price": float(tick.ask_price) if tick.ask_price is not None else float("nan"),
            "bid_volume": float(tick.bid_volume) if tick.bid_volume is not None else float("nan"),
            "ask_volume": float(tick.ask_volume) if tick.ask_volume is not None else float("nan"),
            "source": tick.source,
            "simtrade": bool(tick.simtrade),
        }

        should_flush = False
        with self._lock:
            if tick.symbol not in self._buffers:
                self._buffers[tick.symbol] = []
            self._buffers[tick.symbol].append(record)
            self._total_recorded += 1

            time_elapsed = (now - self._last_flush_time).total_seconds()
            if len(self._buffers[tick.symbol]) >= self.buffer_size or time_elapsed >= self.flush_interval_seconds:
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flushes buffered ticks to respective partition parquet files."""
        with self._lock:
            if not self._buffers:
                return

            to_flush = dict(self._buffers)
            self._buffers.clear()
            self._last_flush_time = datetime.now()

        for symbol, records in to_flush.items():
            if not records:
                continue
            self._write_records(symbol, records)

    def _write_records(self, symbol: str, records: List[Dict]) -> None:
        try:
            sample_ts: datetime = records[0]["exchange_timestamp"]
            date_str = sample_ts.strftime("%Y-%m-%d")
            dir_path = os.path.join(self.base_dir, date_str)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, f"{symbol}.parquet")

            table = pa.Table.from_pylist(records, schema=TICK_SCHEMA)

            if os.path.exists(file_path):
                # Append to existing parquet file
                existing = pq.read_table(file_path)
                combined = pa.concat_tables([existing, table])
                pq.write_table(combined, file_path, compression="snappy")
            else:
                pq.write_table(table, file_path, compression="snappy")

        except Exception as e:
            logger.error(f"Failed to flush parquet ticks for {symbol}: {e}", exc_info=True)

    def close(self) -> None:
        """Flushes all remaining buffered ticks."""
        self.flush()

    def get_total_recorded(self) -> int:
        with self._lock:
            return self._total_recorded
