"""
Pipeline Latency Instrumentation
Tracks microsecond-accurate latency across each stage of the event-driven execution pipeline:
Exchange Tick -> Local Receive -> Enqueue -> Dequeue -> Bar Finalized -> Strategy -> Risk -> OMS -> Broker Fill.
Computes rolling p50 / p95 / p99 percentile distributions.
"""
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import threading
from typing import Dict, List, Optional

logger = logging.getLogger("QuantPilot.Latency")


@dataclass
class LatencyPercentiles:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class LatencySnapshot:
    timestamp: datetime
    stages: Dict[str, LatencyPercentiles] = field(default_factory=dict)


class LatencyTracker:
    """
    Records and summarizes latency metrics across event-driven execution stages.
    Maintains bounded rolling reservoirs to compute accurate p50/p95/p99 without unbounded memory usage.
    """

    STAGES = [
        "market_data",    # local_receive - exchange_timestamp
        "queue",          # dequeue - enqueue
        "bar_processing", # bar finalized - tick dequeue
        "strategy",       # strategy on_bar duration
        "risk_evaluation",# risk evaluate_order duration
        "execution",      # fill - order submission
        "end_to_end",     # fill - exchange_timestamp
    ]

    def __init__(self, reservoir_size: int = 5000):
        self.reservoir_size = reservoir_size
        self._lock = threading.RLock()
        self._samples: Dict[str, List[float]] = {s: [] for s in self.STAGES}

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        """Records a single stage duration in milliseconds."""
        if latency_ms < 0:
            return
        with self._lock:
            if stage not in self._samples:
                self._samples[stage] = []
            samples = self._samples[stage]
            samples.append(float(latency_ms))
            if len(samples) > self.reservoir_size:
                samples.pop(0)

    def record_tick_latencies(
        self,
        exchange_ts: datetime,
        receive_ts: Optional[datetime],
        enqueue_ts: Optional[datetime],
        dequeue_ts: Optional[datetime],
        enqueue_ns: Optional[int] = None,
        dequeue_ns: Optional[int] = None,
        is_replay: bool = False,
    ) -> None:
        """
        Records market data feed latency and queue dwell latency.
        Guarantees:
        - NEVER compares historical replay exchange timestamps against machine wall clock.
        - Uses monotonic high-resolution nanosecond timers for internal queue duration.
        """
        # Market data feed latency (exchange -> local receive)
        if receive_ts and exchange_ts:
            mkt_lat = (receive_ts - exchange_ts).total_seconds() * 1000.0
            if 0.0 <= mkt_lat < 300_000.0:  # Valid only if within realistic 5-minute window
                self.record_stage_latency("market_data", mkt_lat)
        elif not is_replay and exchange_ts:
            # Only for live feeds when local receive time was not explicitly stamped
            mkt_lat = (datetime.now() - exchange_ts).total_seconds() * 1000.0
            if 0.0 <= mkt_lat < 300_000.0:
                self.record_stage_latency("market_data", mkt_lat)

        # Queue latency (enqueue -> dequeue) - Prefer monotonic nanoseconds
        if enqueue_ns is not None and dequeue_ns is not None:
            q_lat = max(0.0, (dequeue_ns - enqueue_ns) / 1_000_000.0)
            self.record_stage_latency("queue", q_lat)
        elif enqueue_ts and dequeue_ts:
            q_lat = (dequeue_ts - enqueue_ts).total_seconds() * 1000.0
            if q_lat >= 0:
                self.record_stage_latency("queue", q_lat)

    def get_percentiles(self, stage: str) -> LatencyPercentiles:
        with self._lock:
            samples = self._samples.get(stage, [])
            if not samples:
                return LatencyPercentiles(
                    count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, min_ms=0.0, max_ms=0.0
                )

            sorted_s = sorted(samples)
            n = len(sorted_s)
            mean_v = sum(sorted_s) / n

            def _percentile(p: float) -> float:
                k = (n - 1) * p
                f = math.floor(k)
                c = math.ceil(k)
                if f == c:
                    return sorted_s[int(k)]
                return sorted_s[int(f)] * (c - k) + sorted_s[int(c)] * (k - f)

            return LatencyPercentiles(
                count=n,
                mean_ms=round(mean_v, 3),
                p50_ms=round(_percentile(0.50), 3),
                p95_ms=round(_percentile(0.95), 3),
                p99_ms=round(_percentile(0.99), 3),
                min_ms=round(sorted_s[0], 3),
                max_ms=round(sorted_s[-1], 3),
            )

    def get_snapshot(self) -> LatencySnapshot:
        with self._lock:
            stages_dict = {s: self.get_percentiles(s) for s in self._samples.keys()}
            return LatencySnapshot(
                timestamp=datetime.now(),
                stages=stages_dict,
            )

    def reset(self) -> None:
        with self._lock:
            for s in self._samples.keys():
                self._samples[s].clear()
