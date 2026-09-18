"""
Market Clock Abstractions and High-Resolution Monotonic Timers
Canonical timezone: Asia/Taipei
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
import time
from typing import Optional
from zoneinfo import ZoneInfo

try:
    TAIPEI_TZ = ZoneInfo("Asia/Taipei")
except Exception:
    TAIPEI_TZ = timezone(timedelta(hours=8))


def ensure_taipei_tz(dt: datetime) -> datetime:
    """Ensures a datetime is timezone-aware in Asia/Taipei."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TAIPEI_TZ)
    return dt.astimezone(TAIPEI_TZ)


def elapsed_ms(start_ns: int, end_ns: Optional[int] = None) -> float:
    """Calculates elapsed milliseconds from monotonic nanosecond counter."""
    if end_ns is None:
        end_ns = time.perf_counter_ns()
    return max(0.0, (end_ns - start_ns) / 1_000_000.0)


class IClock(ABC):
    """Abstract Clock interface decoupling wall time, replay virtual time, and monotonic duration."""

    @abstractmethod
    def now(self) -> datetime:
        """Returns the current market time, guaranteed timezone-aware in Asia/Taipei."""
        pass

    @abstractmethod
    def now_ns(self) -> int:
        """Returns monotonic timestamp in nanoseconds for stage duration telemetry."""
        pass

    def elapsed_ms(self, start_ns: int, end_ns: Optional[int] = None) -> float:
        """Calculates elapsed milliseconds from monotonic nanosecond counter."""
        if end_ns is None:
            end_ns = self.now_ns()
        return max(0.0, (end_ns - start_ns) / 1_000_000.0)

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Pauses execution according to the clock implementation."""
        pass

    @property
    @abstractmethod
    def is_virtual(self) -> bool:
        """Whether this clock represents a simulated/virtual timeline."""
        pass


class SystemClock(IClock):
    """
    Standard host-system wall clock.
    Used for live feeds, shadow burn-in, and operational soak testing.
    """

    def __init__(self, tz: ZoneInfo = TAIPEI_TZ):
        self.tz = tz

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        # Interruptible sleep in micro-slices for responsive shutdown
        remaining = seconds
        while remaining > 0:
            step = min(0.05, remaining)
            time.sleep(step)
            remaining -= step

    @property
    def is_virtual(self) -> bool:
        return False


class VirtualClock(IClock):
    """
    Deterministic virtual clock for backtesting and historical replay.
    Advances strictly when instructed by market data events.
    """

    def __init__(self, initial_time: Optional[datetime] = None, tz: ZoneInfo = TAIPEI_TZ):
        self.tz = tz
        if initial_time is not None:
            self._current_time = ensure_taipei_tz(initial_time)
        else:
            self._current_time = datetime(2026, 9, 18, 9, 0, 0, tzinfo=self.tz)
        self._virtual_ns_offset = 0
        self._start_perf_ns = time.perf_counter_ns()

    def now(self) -> datetime:
        return self._current_time

    def set_time(self, new_time: datetime) -> None:
        """Sets the virtual clock time."""
        self._current_time = ensure_taipei_tz(new_time)

    def advance_to(self, new_time: datetime) -> None:
        """Advances virtual clock time monotonically forward."""
        aware = ensure_taipei_tz(new_time)
        if aware < self._current_time:
            # Allow equal or forward only
            return
        delta = aware - self._current_time
        self._virtual_ns_offset += int(delta.total_seconds() * 1_000_000_000)
        self._current_time = aware

    def advance_by(self, delta: timedelta) -> None:
        """Advances virtual clock by timedelta."""
        self.advance_to(self._current_time + delta)

    advance = advance_by

    def now_ns(self) -> int:
        """Returns monotonic nanoseconds for internal duration calculations."""
        return time.perf_counter_ns()

    def sleep(self, seconds: float) -> None:
        """In virtual mode, sleep advances virtual time rather than blocking CPU."""
        if seconds > 0:
            self.advance_by(timedelta(seconds=seconds))

    @property
    def is_virtual(self) -> bool:
        return True
