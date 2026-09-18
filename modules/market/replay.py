"""
Deterministic Market Replay Engine
Replays recorded raw market tick datasets through the event-driven execution platform
with identical interface to Shioaji live feeds.
Guarantees 100% reproducible bars, signals, risk evaluations, orders, fills, and PnL.
"""
from datetime import datetime
import glob
import logging
import math
import os
import threading
import time
from typing import Callable, List, Optional

import pyarrow.parquet as pq

from modules.execution.events import TickEvent

logger = logging.getLogger("QuantPilot.Replay")


class ReplayMode:
    REAL_TIME = "real_time"
    ACCELERATED = "accelerated"
    MAX_SPEED = "max_speed"


ReplaySpeed = ReplayMode


class ReplayMarketDataSource:
    """
    Market data source that reads recorded ticks from Parquet files or memory,
    replaying them deterministically to registered tick listeners.
    """

    def __init__(
        self,
        mode: str = ReplayMode.MAX_SPEED,
        speed_factor: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        data_dir: Optional[str] = None,
        speed: Optional[str] = None,
    ):
        self.mode = speed or mode
        self.speed_factor = max(0.01, speed_factor)
        self.start_time = start_time
        self.end_time = end_time
        self.data_dir = data_dir

        self._tick_callbacks: List[Callable[[TickEvent], None]] = []
        self._lock = threading.RLock()
        self._is_running = False
        self._is_paused = False
        self._pause_cond = threading.Condition(self._lock)
        self._replay_thread: Optional[threading.Thread] = None

        self._ticks: List[TickEvent] = []
        self._current_index = 0
        self._total_replayed = 0

    def register_tick_callback(self, callback: Callable[[TickEvent], None]) -> None:
        """Registers a listener for normalized streaming ticks (same interface as Shioaji)."""
        with self._lock:
            self._tick_callbacks.append(callback)

    def load_from_memory(self, ticks: List[TickEvent]) -> None:
        """Loads ticks directly from in-memory list and sorts chronologically."""
        with self._lock:
            filtered = []
            for t in ticks:
                if self.start_time and t.timestamp < self.start_time:
                    continue
                if self.end_time and t.timestamp > self.end_time:
                    continue
                filtered.append(t)
            # Sort deterministically by timestamp, then sequence
            self._ticks = sorted(filtered, key=lambda x: (x.timestamp, x.sequence))
            self._current_index = 0

    def load_from_parquet(self, file_path_or_glob: str) -> None:
        """Loads and sorts ticks from one or more Parquet files."""
        matched_files = glob.glob(file_path_or_glob)
        if not matched_files and os.path.exists(file_path_or_glob):
            matched_files = [file_path_or_glob]

        loaded_ticks: List[TickEvent] = []
        for fpath in matched_files:
            try:
                table = pq.read_table(fpath)
                pydict = table.to_pydict()
                num_rows = len(pydict["symbol"])
                for i in range(num_rows):
                    ts: datetime = pydict["exchange_timestamp"][i]
                    if self.start_time and ts < self.start_time:
                        continue
                    if self.end_time and ts > self.end_time:
                        continue

                    bid = pydict["bid_price"][i]
                    ask = pydict["ask_price"][i]
                    bid_vol = pydict["bid_volume"][i]
                    ask_vol = pydict["ask_volume"][i]

                    tick = TickEvent(
                        timestamp=ts,
                        symbol=pydict["symbol"][i],
                        price=float(pydict["price"][i]),
                        volume=float(pydict["volume"][i]),
                        bid_price=float(bid) if not math.isnan(bid) else None,
                        ask_price=float(ask) if not math.isnan(ask) else None,
                        bid_volume=float(bid_vol) if not math.isnan(bid_vol) else None,
                        ask_volume=float(ask_vol) if not math.isnan(ask_vol) else None,
                        receive_timestamp=pydict["local_receive_timestamp"][i],
                        sequence=int(pydict["sequence"][i]),
                        tick_type=pydict["tick_type"][i],
                        source="replay",
                        simtrade=bool(pydict["simtrade"][i]),
                    )
                    loaded_ticks.append(tick)
            except Exception as e:
                logger.error(f"Error loading parquet file {fpath}: {e}", exc_info=True)

        self.load_from_memory(loaded_ticks)
        logger.info(f"Loaded {len(self._ticks)} ticks for replay from {len(matched_files)} files.")

    def start(
        self,
        symbols: Optional[List[str]] = None,
        date_str: Optional[str] = None,
        blocking: bool = True,
    ) -> None:
        """Starts replaying ticks."""
        with self._lock:
            if not self._ticks and self.data_dir:
                files_to_load = []
                date_target = date_str or "*"
                if symbols:
                    for s in symbols:
                        fpath = os.path.join(self.data_dir, date_target, f"{s}.parquet")
                        files_to_load.extend(glob.glob(fpath))
                else:
                    files_to_load.extend(glob.glob(os.path.join(self.data_dir, date_target, "*.parquet")))

                for fp in files_to_load:
                    self.load_from_parquet(fp)

            if self._is_running:
                return
            self._is_running = True
            self._is_paused = False

        if blocking:
            self._replay_loop()
        else:
            self._replay_thread = threading.Thread(
                target=self._replay_loop,
                name="ReplayEngineWorker",
                daemon=True,
            )
            self._replay_thread.start()

    def pause(self) -> None:
        with self._lock:
            self._is_paused = True

    def resume(self) -> None:
        with self._lock:
            self._is_paused = False
            self._pause_cond.notify_all()

    def stop(self) -> None:
        with self._lock:
            self._is_running = False
            self._is_paused = False
            self._pause_cond.notify_all()
        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=3.0)

    def _notify(self, tick: TickEvent) -> None:
        for cb in list(self._tick_callbacks):
            try:
                cb(tick)
            except Exception as e:
                logger.error(f"Error in replay tick callback: {e}", exc_info=True)

    def _replay_loop(self) -> None:
        last_tick_time: Optional[datetime] = None

        while self._current_index < len(self._ticks):
            with self._lock:
                if not self._is_running:
                    break
                while self._is_paused and self._is_running:
                    self._pause_cond.wait(timeout=0.1)

                tick = self._ticks[self._current_index]
                self._current_index += 1
                self._total_replayed += 1

            # Timing delay simulation if not max_speed
            if self.mode in (ReplayMode.REAL_TIME, ReplayMode.ACCELERATED) and last_tick_time:
                delta_sec = (tick.timestamp - last_tick_time).total_seconds()
                if delta_sec > 0:
                    delay = delta_sec / self.speed_factor
                    # Interruptible sleep in micro-slices to maintain responsiveness without capping delay
                    remaining = delay
                    while remaining > 0 and self._is_running and not self._is_paused:
                        step = min(0.05, remaining)
                        time.sleep(step)
                        remaining -= step

            last_tick_time = tick.timestamp
            self._notify(tick)

        with self._lock:
            self._is_running = False
        logger.info(f"Replay completed. Total ticks replayed: {self._total_replayed}")

    def get_progress(self) -> float:
        with self._lock:
            if not self._ticks:
                return 1.0
            return self._current_index / len(self._ticks)
