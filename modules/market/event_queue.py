"""
Market Data Event Queue
Decouples high-frequency Shioaji market data callbacks from downstream trading,
feature calculation, risk evaluation, and order management logic.
"""
from dataclasses import dataclass, field
from datetime import datetime
import logging
import queue
import threading
import time
from typing import Callable, List, Optional

from modules.execution.events import TickEvent

logger = logging.getLogger("QuantPilot.EventQueue")


@dataclass
class QueueMetrics:
    capacity: int
    current_depth: int
    max_depth: int
    total_enqueued: int
    total_dequeued: int
    overflow_count: int
    dropped_count: int
    is_healthy: bool


class MarketDataEventQueue:
    """
    Bounded, thread-safe asynchronous queue for streaming tick ingestion.
    
    Guarantees:
    - Ingestion callback returns in microseconds without blocking.
    - Monotonically increasing sequence numbers per event.
    - Timestamp preservation (exchange, receive, enqueue, dequeue).
    - Overflow detection: never silently drops market data.
    - Graceful worker thread shutdown.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        name: str = "MarketDataQueue",
        synchronous: bool = False,
        test_only_synchronous: bool = False,
        on_overflow: Optional[Callable[[TickEvent, QueueMetrics], None]] = None,
    ):
        self.capacity = capacity
        self.name = name
        self.synchronous = test_only_synchronous or synchronous
        self.on_overflow = on_overflow

        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._lock = threading.RLock()
        self._seq_counter = 0
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._max_depth = 0
        self._overflow_count = 0
        self._dropped_count = 0
        self._is_healthy = True

        self._consumer_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._subscribers: List[Callable[[TickEvent], None]] = []

    def subscribe(self, callback: Callable[[TickEvent], None]) -> None:
        """Subscribes downstream processor to dequeued ticks."""
        with self._lock:
            self._subscribers.append(callback)

    def enqueue(self, tick: TickEvent) -> bool:
        """
        Non-blocking enqueue for market callbacks.
        Attaches monotonic sequence number, enqueue wall timestamp, and monotonic nanoseconds.
        """
        now = datetime.now()
        now_ns = time.perf_counter_ns()
        with self._lock:
            self._seq_counter += 1
            seq = self._seq_counter

        # Create updated TickEvent with sequence and enqueue timestamps
        stamped_tick = TickEvent(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            price=tick.price,
            volume=tick.volume,
            bid_price=tick.bid_price,
            ask_price=tick.ask_price,
            bid_volume=tick.bid_volume,
            ask_volume=tick.ask_volume,
            receive_timestamp=tick.receive_timestamp or now,
            enqueue_timestamp=now,
            enqueue_ns=now_ns,
            sequence=seq,
            tick_type=tick.tick_type,
            source=tick.source,
            simtrade=tick.simtrade,
            is_replay=getattr(tick, "is_replay", False),
        )

        if self.synchronous:
            with self._lock:
                self._total_enqueued += 1
                self._total_dequeued += 1
            processed_tick = TickEvent(
                timestamp=stamped_tick.timestamp,
                symbol=stamped_tick.symbol,
                price=stamped_tick.price,
                volume=stamped_tick.volume,
                bid_price=stamped_tick.bid_price,
                ask_price=stamped_tick.ask_price,
                bid_volume=stamped_tick.bid_volume,
                ask_volume=stamped_tick.ask_volume,
                receive_timestamp=stamped_tick.receive_timestamp,
                enqueue_timestamp=stamped_tick.enqueue_timestamp,
                dequeue_timestamp=datetime.now(),
                enqueue_ns=stamped_tick.enqueue_ns,
                dequeue_ns=time.perf_counter_ns(),
                sequence=stamped_tick.sequence,
                tick_type=stamped_tick.tick_type,
                source=stamped_tick.source,
                simtrade=stamped_tick.simtrade,
                is_replay=stamped_tick.is_replay,
            )
            for subscriber in list(self._subscribers):
                try:
                    subscriber(processed_tick)
                except Exception as e:
                    logger.error(f"Error in queue subscriber callback: {e}", exc_info=True)
            return True

        try:
            self._queue.put_nowait(stamped_tick)
            with self._lock:
                self._total_enqueued += 1
                # Enforce accounting: measure queue depth AFTER successful enqueue!
                current_qsize = self._queue.qsize()
                if current_qsize > self._max_depth:
                    self._max_depth = current_qsize
            return True
        except queue.Full:
            with self._lock:
                self._overflow_count += 1
                self._dropped_count += 1
                self._is_healthy = False
                metrics = self.get_metrics()

            logger.critical(
                f"[QUEUE OVERFLOW] MarketDataEventQueue '{self.name}' exceeded capacity {self.capacity}! "
                f"Tick {tick.symbol}@{tick.price} dropped. Data stream marked UNHEALTHY."
            )
            if self.on_overflow:
                try:
                    self.on_overflow(stamped_tick, metrics)
                except Exception as e:
                    logger.error(f"Error in on_overflow callback: {e}")
            return False

    def start(self) -> None:
        """Starts background consumer worker thread."""
        with self._lock:
            if self._is_running:
                return
            self._is_running = True
            self._consumer_thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self.name}-Worker",
                daemon=True,
            )
            self._consumer_thread.start()
            logger.info(f"MarketDataEventQueue worker thread started (capacity={self.capacity}).")

    def stop(self, drain: bool = True, timeout: float = 5.0) -> None:
        """Stops worker thread gracefully, optionally draining remaining items."""
        with self._lock:
            if not self._is_running:
                return
            self._is_running = False

        if drain:
            self.drain(timeout=timeout)

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=timeout)
        logger.info(f"MarketDataEventQueue stopped. Total enqueued={self._total_enqueued}, dequeued={self._total_dequeued}")

    def drain(self, timeout: float = 10.0) -> None:
        """Waits until all enqueued items are processed or timeout is reached."""
        start_t = time.time()
        while getattr(self._queue, "unfinished_tasks", 0) > 0 or not self._queue.empty():
            if time.time() - start_t > timeout:
                logger.warning("MarketDataEventQueue drain timeout reached before queue emptied.")
                break
            time.sleep(0.01)

    def join(self, timeout: Optional[float] = None) -> None:
        """Blocks until all items in the queue have been gotten and processed."""
        if self.synchronous:
            return
        if timeout is None:
            self._queue.join()
        else:
            self.drain(timeout=timeout)

    def _worker_loop(self) -> None:
        """Consumer worker loop dispatching ticks to downstream subscribers."""
        while self._is_running or not self._queue.empty():
            try:
                tick: TickEvent = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            dequeue_time = datetime.now()
            dequeue_ns = time.perf_counter_ns()
            with self._lock:
                self._total_dequeued += 1

            # Attach dequeue timestamp and monotonic nanoseconds
            processed_tick = TickEvent(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                bid_price=tick.bid_price,
                ask_price=tick.ask_price,
                bid_volume=tick.bid_volume,
                ask_volume=tick.ask_volume,
                receive_timestamp=tick.receive_timestamp,
                enqueue_timestamp=tick.enqueue_timestamp,
                dequeue_timestamp=dequeue_time,
                enqueue_ns=tick.enqueue_ns,
                dequeue_ns=dequeue_ns,
                sequence=tick.sequence,
                tick_type=tick.tick_type,
                source=tick.source,
                simtrade=tick.simtrade,
                is_replay=getattr(tick, "is_replay", False),
            )

            # Dispatch to subscribers
            for subscriber in list(self._subscribers):
                try:
                    subscriber(processed_tick)
                except Exception as e:
                    logger.error(f"Error in queue subscriber callback: {e}", exc_info=True)

            self._queue.task_done()

    def get_metrics(self) -> QueueMetrics:
        with self._lock:
            return QueueMetrics(
                capacity=self.capacity,
                current_depth=self._queue.qsize(),
                max_depth=self._max_depth,
                total_enqueued=self._total_enqueued,
                total_dequeued=self._total_dequeued,
                overflow_count=self._overflow_count,
                dropped_count=self._dropped_count,
                is_healthy=self._is_healthy,
            )

    def reset_health(self) -> None:
        with self._lock:
            self._is_healthy = True
            self._overflow_count = 0
            self._dropped_count = 0
