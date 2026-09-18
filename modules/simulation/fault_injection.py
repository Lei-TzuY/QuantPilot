"""
Fault-Injection Simulation Framework
Provides deterministic, controllable failure simulation hooks to verify fail-safe behavior
across market-data ingestion, event queue, risk evaluation, broker callbacks, and execution journal.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from modules.execution.events import BarEvent, SignalEvent, TickEvent
from modules.execution.order import Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger("QuantPilot.FaultInjection")


class FaultType:
    DISCONNECT = "DISCONNECT"
    RECONNECT = "RECONNECT"
    DUPLICATE_TICK = "DUPLICATE_TICK"
    DELAYED_CALLBACK = "DELAYED_CALLBACK"
    OUT_OF_ORDER_TICK = "OUT_OF_ORDER_TICK"
    QUEUE_OVERFLOW = "QUEUE_OVERFLOW"
    JOURNAL_WRITE_FAILURE = "JOURNAL_WRITE_FAILURE"
    BROKER_CALLBACK_DELAY = "BROKER_CALLBACK_DELAY"
    REJECTED_ORDER = "REJECTED_ORDER"
    STRATEGY_EXCEPTION = "STRATEGY_EXCEPTION"


@dataclass
class FaultConfig:
    target_symbol: Optional[str] = None
    delay_seconds: float = 0.0
    drop_rate: float = 0.0
    price_multiplier: float = 1.0


class FaultInjector:
    """
    Simulation harness to inject real-world faults and verify that QuantPilot
    consistently fails safe (never fails open).
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._active_faults: Dict[str, Any] = {}
        self._fault_history: List[Dict[str, Any]] = []

    def enable_fault(self, fault_type: str, config: Optional[Any] = None) -> None:
        """Enables a specific fault."""
        with self._lock:
            self._active_faults[fault_type] = config or {}
            self._record(fault_type, "ENABLED", config)

    def disable_fault(self, fault_type: str) -> None:
        """Disables a specific fault."""
        with self._lock:
            if fault_type in self._active_faults:
                del self._active_faults[fault_type]
                self._record(fault_type, "DISABLED", None)

    def is_fault_active(self, fault_type: str) -> bool:
        with self._lock:
            return fault_type in self._active_faults

    def get_fault_config(self, fault_type: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._active_faults.get(fault_type, {}))

    def _record(self, fault_type: str, action: str, details: Any) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "fault_type": fault_type,
            "action": action,
            "details": details,
        }
        self._fault_history.append(entry)
        logger.info(f"[FAULT INJECTION] {fault_type} -> {action} | Details: {details}")

    def intercept_tick(self, tick: TickEvent, downstream: Callable[[TickEvent], None]) -> None:
        """Applies active tick-level faults (delay, duplicate, out-of-order)."""
        with self._lock:
            if self.is_fault_active(FaultType.DISCONNECT):
                # Drops tick silently (simulating socket disconnect)
                logger.warning(f"[FAULT] Dropped tick {tick.symbol} due to simulated DISCONNECT.")
                return

            delay_cfg = self.get_fault_config(FaultType.DELAYED_CALLBACK)
            dup_cfg = self.get_fault_config(FaultType.DUPLICATE_TICK)
            ooo_cfg = self.get_fault_config(FaultType.OUT_OF_ORDER_TICK)

        # 1. Delayed callback
        if delay_cfg:
            delay_sec = delay_cfg.get("delay_seconds", 0.05)
            time.sleep(delay_sec)

        # 2. Out-of-order timestamp manipulation
        if ooo_cfg:
            retro_seconds = ooo_cfg.get("regression_seconds", 60)
            tick = TickEvent(
                timestamp=tick.timestamp - timedelta(seconds=retro_seconds),
                symbol=tick.symbol,
                price=tick.price,
                volume=tick.volume,
                bid_price=tick.bid_price,
                ask_price=tick.ask_price,
                bid_volume=tick.bid_volume,
                ask_volume=tick.ask_volume,
                receive_timestamp=tick.receive_timestamp,
                sequence=tick.sequence,
                tick_type=tick.tick_type,
                source=tick.source,
                simtrade=tick.simtrade,
            )

        # Dispatch regular tick
        downstream(tick)

        # 3. Duplicate tick injection
        if dup_cfg:
            dup_count = dup_cfg.get("count", 1)
            for _ in range(dup_count):
                downstream(tick)

    def intercept_journal_write(self) -> None:
        """Simulates journal disk failure / permission error."""
        with self._lock:
            if self.is_fault_active(FaultType.JOURNAL_WRITE_FAILURE):
                logger.critical("[FAULT] Raising simulated IOError for Journal write!")
                raise IOError("Simulated disk write failure: DISK_FULL / EACCES")
