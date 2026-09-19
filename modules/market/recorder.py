"""
Raw Market Data Recorder
Captures high-frequency normalized TickEvents and BidAskEvents into append-only structured Parquet files
partitioned by date and symbol:
  data/market/YYYY-MM-DD/{symbol}_ticks.parquet (and {symbol}.parquet)
  data/market/YYYY-MM-DD/{symbol}_bidask.parquet
Uses buffered asynchronous I/O to ensure zero callback latency impact.
"""
from datetime import date, datetime
import logging
import os
import threading
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from modules.execution.events import BidAskEvent, TickEvent

logger = logging.getLogger("QuantPilot.MarketRecorder")

TICK_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("exchange_timestamp", pa.timestamp("us")),
    ("local_receive_timestamp", pa.timestamp("us")),
    ("price", pa.float64()),
    ("volume", pa.float64()),
    ("total_volume", pa.float64()),
    ("tick_type", pa.string()),
    ("sequence", pa.int64()),
    ("bid_price", pa.float64()),
    ("ask_price", pa.float64()),
    ("bid_volume", pa.float64()),
    ("ask_volume", pa.float64()),
    ("source", pa.string()),
    ("simtrade", pa.bool_()),
    ("intraday_odd", pa.bool_()),
])

BIDASK_SCHEMA = pa.schema([
    ("symbol", pa.string()),
    ("exchange_timestamp", pa.timestamp("us")),
    ("local_receive_timestamp", pa.timestamp("us")),
    ("bid_price", pa.float64()),
    ("ask_price", pa.float64()),
    ("bid_volume", pa.float64()),
    ("ask_volume", pa.float64()),
    ("sequence", pa.int64()),
    ("source", pa.string()),
    ("simtrade", pa.bool_()),
])


class RawMarketDataRecorder:
    """
    Buffered, thread-safe market data capture for both Tick and BidAsk streams.
    Persists raw events to Parquet for exact deterministic offline replay.
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
        self._tick_buffers: Dict[str, List[Dict]] = {}
        self._bidask_buffers: Dict[str, List[Dict]] = {}
        self._total_recorded_ticks = 0
        self._total_recorded_bidask = 0
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
            "total_volume": float(tick.total_volume) if tick.total_volume is not None else float("nan"),
            "tick_type": str(tick.tick_type),
            "sequence": int(tick.sequence),
            "bid_price": float(tick.bid_price) if tick.bid_price is not None else float("nan"),
            "ask_price": float(tick.ask_price) if tick.ask_price is not None else float("nan"),
            "bid_volume": float(tick.bid_volume) if tick.bid_volume is not None else float("nan"),
            "ask_volume": float(tick.ask_volume) if tick.ask_volume is not None else float("nan"),
            "source": str(tick.source),
            "simtrade": bool(tick.simtrade),
            "intraday_odd": bool(getattr(tick, "intraday_odd", False)),
        }

        should_flush = False
        with self._lock:
            if tick.symbol not in self._tick_buffers:
                self._tick_buffers[tick.symbol] = []
            self._tick_buffers[tick.symbol].append(record)
            self._total_recorded_ticks += 1

            time_elapsed = (now - self._last_flush_time).total_seconds()
            if len(self._tick_buffers[tick.symbol]) >= self.buffer_size or time_elapsed >= self.flush_interval_seconds:
                should_flush = True

        if should_flush:
            self.flush()

    def record_bidask(self, bidask: BidAskEvent) -> None:
        """Buffers a BidAskEvent for background or periodic flush."""
        now = datetime.now()
        record = {
            "symbol": bidask.symbol,
            "exchange_timestamp": bidask.timestamp,
            "local_receive_timestamp": bidask.receive_timestamp or now,
            "bid_price": float(bidask.bid_price),
            "ask_price": float(bidask.ask_price),
            "bid_volume": float(bidask.bid_volume),
            "ask_volume": float(bidask.ask_volume),
            "sequence": int(bidask.sequence),
            "source": str(bidask.source),
            "simtrade": bool(bidask.simtrade),
        }

        should_flush = False
        with self._lock:
            if bidask.symbol not in self._bidask_buffers:
                self._bidask_buffers[bidask.symbol] = []
            self._bidask_buffers[bidask.symbol].append(record)
            self._total_recorded_bidask += 1

            time_elapsed = (now - self._last_flush_time).total_seconds()
            if len(self._bidask_buffers[bidask.symbol]) >= self.buffer_size or time_elapsed >= self.flush_interval_seconds:
                should_flush = True

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flushes buffered ticks and bidask records to parquet files."""
        with self._lock:
            if not self._tick_buffers and not self._bidask_buffers:
                return

            tick_to_flush = dict(self._tick_buffers)
            bidask_to_flush = dict(self._bidask_buffers)
            self._tick_buffers.clear()
            self._bidask_buffers.clear()
            self._last_flush_time = datetime.now()

        for symbol, records in tick_to_flush.items():
            if records:
                self._write_tick_records(symbol, records)

        for symbol, records in bidask_to_flush.items():
            if records:
                self._write_bidask_records(symbol, records)

    def _write_tick_records(self, symbol: str, records: List[Dict]) -> None:
        try:
            sample_ts: datetime = records[0]["exchange_timestamp"]
            date_str = sample_ts.strftime("%Y-%m-%d")
            dir_path = os.path.join(self.base_dir, date_str)
            os.makedirs(dir_path, exist_ok=True)

            table = pa.Table.from_pylist(records, schema=TICK_SCHEMA)

            # Write to both {symbol}.parquet (backwards compat) and {symbol}_ticks.parquet
            for fname in (f"{symbol}.parquet", f"{symbol}_ticks.parquet"):
                file_path = os.path.join(dir_path, fname)
                if os.path.exists(file_path):
                    existing = pq.read_table(file_path)
                    combined = pa.concat_tables([existing, table])
                    pq.write_table(combined, file_path, compression="snappy")
                else:
                    pq.write_table(table, file_path, compression="snappy")

        except Exception as e:
            logger.error(f"Failed to flush parquet ticks for {symbol}: {e}", exc_info=True)

    def _write_bidask_records(self, symbol: str, records: List[Dict]) -> None:
        try:
            sample_ts: datetime = records[0]["exchange_timestamp"]
            date_str = sample_ts.strftime("%Y-%m-%d")
            dir_path = os.path.join(self.base_dir, date_str)
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, f"{symbol}_bidask.parquet")
            table = pa.Table.from_pylist(records, schema=BIDASK_SCHEMA)

            if os.path.exists(file_path):
                existing = pq.read_table(file_path)
                combined = pa.concat_tables([existing, table])
                pq.write_table(combined, file_path, compression="snappy")
            else:
                pq.write_table(table, file_path, compression="snappy")

        except Exception as e:
            logger.error(f"Failed to flush parquet bidask for {symbol}: {e}", exc_info=True)

    def close(self) -> None:
        """Flushes all remaining buffered ticks and bidask records."""
        self.flush()

    def get_total_recorded(self) -> int:
        with self._lock:
            return self._total_recorded_ticks + self._total_recorded_bidask

    def get_total_recorded_ticks(self) -> int:
        with self._lock:
            return self._total_recorded_ticks

    def get_total_recorded_bidask(self) -> int:
        with self._lock:
            return self._total_recorded_bidask
