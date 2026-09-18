"""
Shadow Session Supervisor
Governs the operational lifecycle of QuantPilot in SHADOW mode.
Enforces institutional checklists before permitting automated or manual trading.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("QuantPilot.Supervisor")


class SupervisorState(str, Enum):
    INITIALIZING = "INITIALIZING"
    CONNECTING = "CONNECTING"
    SYNCING = "SYNCING"
    READY = "READY"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class ReadinessChecklist:
    broker_connected: bool = False
    market_clock_valid: bool = False
    journal_available: bool = False
    strategies_warmed_up: bool = False
    risk_engine_ready: bool = False
    kill_switch_active: bool = False  # False means safe to trade
    reconciliation_clean: bool = False

    @property
    def is_all_passed(self) -> bool:
        return (
            self.broker_connected
            and self.market_clock_valid
            and self.journal_available
            and self.strategies_warmed_up
            and self.risk_engine_ready
            and not self.kill_switch_active
            and self.reconciliation_clean
        )


class ShadowSessionSupervisor:
    """
    Session supervisor governing lifecycle state transitions and pre-running invariants.
    """

    LEGAL_TRANSITIONS = {
        SupervisorState.INITIALIZING: [SupervisorState.CONNECTING, SupervisorState.HALTED, SupervisorState.CLOSED],
        SupervisorState.CONNECTING: [SupervisorState.SYNCING, SupervisorState.HALTED, SupervisorState.DEGRADED, SupervisorState.CLOSED],
        SupervisorState.SYNCING: [SupervisorState.READY, SupervisorState.HALTED, SupervisorState.DEGRADED, SupervisorState.CLOSED],
        SupervisorState.READY: [SupervisorState.RUNNING, SupervisorState.HALTED, SupervisorState.CLOSING, SupervisorState.CLOSED],
        SupervisorState.RUNNING: [SupervisorState.DEGRADED, SupervisorState.HALTED, SupervisorState.CLOSING],
        SupervisorState.DEGRADED: [SupervisorState.RUNNING, SupervisorState.HALTED, SupervisorState.CLOSING],
        SupervisorState.HALTED: [SupervisorState.READY, SupervisorState.CLOSING, SupervisorState.CLOSED],
        SupervisorState.CLOSING: [SupervisorState.CLOSED],
        SupervisorState.CLOSED: [SupervisorState.INITIALIZING],
    }

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"SHADOW-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._state = SupervisorState.INITIALIZING
        self._lock = threading.RLock()
        self._state_history: List[Dict[str, Any]] = []
        self._checklist = ReadinessChecklist()
        self._transition_listeners: List[Callable[[SupervisorState, SupervisorState, str], None]] = []

        self._record_transition(SupervisorState.CLOSED, SupervisorState.INITIALIZING, "Supervisor initialized")

    @property
    def current_state(self) -> SupervisorState:
        with self._lock:
            return self._state

    def register_transition_listener(self, listener: Callable[[SupervisorState, SupervisorState, str], None]) -> None:
        with self._lock:
            self._transition_listeners.append(listener)

    def _record_transition(self, from_state: SupervisorState, to_state: SupervisorState, reason: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason,
        }
        self._state_history.append(entry)
        logger.info(f"[SUPERVISOR] {from_state.value} -> {to_state.value} | Reason: {reason}")
        for listener in list(self._transition_listeners):
            try:
                listener(from_state, to_state, reason)
            except Exception as e:
                logger.error(f"Error in supervisor listener: {e}", exc_info=True)

    def can_transition_to(self, new_state: SupervisorState) -> bool:
        """Checks if a transition to new_state is currently legal and satisfied."""
        with self._lock:
            allowed = self.LEGAL_TRANSITIONS.get(self._state, [])
            if new_state not in allowed:
                return False
            if new_state == SupervisorState.RUNNING:
                return self._checklist.is_all_passed
            return True

    def transition_to(self, new_state: SupervisorState, reason: str = "") -> bool:
        """Transitions to a new state if legal."""
        with self._lock:
            allowed = self.LEGAL_TRANSITIONS.get(self._state, [])
            if new_state not in allowed:
                msg = f"Illegal supervisor transition: {self._state.value} -> {new_state.value}. Allowed: {[s.value for s in allowed]}"
                logger.error(msg)
                raise ValueError(msg)

            # Invariant check before entering RUNNING
            if new_state == SupervisorState.RUNNING:
                if not self._checklist.is_all_passed:
                    reasons = []
                    if not self._checklist.broker_connected: reasons.append("Broker disconnected")
                    if not self._checklist.market_clock_valid: reasons.append("Market clock invalid")
                    if not self._checklist.journal_available: reasons.append("Journal unavailable")
                    if not self._checklist.strategies_warmed_up: reasons.append("Strategies not warmed up")
                    if not self._checklist.risk_engine_ready: reasons.append("Risk engine not ready")
                    if self._checklist.kill_switch_active: reasons.append("Kill switch halted")
                    if not self._checklist.reconciliation_clean: reasons.append("Reconciliation not clean")
                    err = f"Cannot transition to RUNNING: Checklist failed: {', '.join(reasons)}"
                    logger.critical(err)
                    raise RuntimeError(err)

            prev_state = self._state
            self._state = new_state
            self._record_transition(prev_state, new_state, reason)
            return True

    def update_checklist(
        self,
        broker_connected: Optional[bool] = None,
        market_clock_valid: Optional[bool] = None,
        journal_available: Optional[bool] = None,
        strategies_warmed_up: Optional[bool] = None,
        risk_engine_ready: Optional[bool] = None,
        kill_switch_active: Optional[bool] = None,
        reconciliation_clean: Optional[bool] = None,
    ) -> ReadinessChecklist:
        """Updates readiness checklist components."""
        with self._lock:
            if broker_connected is not None: self._checklist.broker_connected = broker_connected
            if market_clock_valid is not None: self._checklist.market_clock_valid = market_clock_valid
            if journal_available is not None: self._checklist.journal_available = journal_available
            if strategies_warmed_up is not None: self._checklist.strategies_warmed_up = strategies_warmed_up
            if risk_engine_ready is not None: self._checklist.risk_engine_ready = risk_engine_ready
            if kill_switch_active is not None: self._checklist.kill_switch_active = kill_switch_active
            if reconciliation_clean is not None: self._checklist.reconciliation_clean = reconciliation_clean
            return self._checklist

    def trigger_degraded(self, reason: str) -> None:
        """Transitions RUNNING to DEGRADED on data or queue incidents."""
        with self._lock:
            if self._state == SupervisorState.RUNNING:
                self.transition_to(SupervisorState.DEGRADED, reason=reason)

    def recover_to_running(self, reason: str = "Conditions restored") -> None:
        """Recovers DEGRADED back to RUNNING."""
        with self._lock:
            if self._state == SupervisorState.DEGRADED:
                self.transition_to(SupervisorState.RUNNING, reason=reason)

    def trigger_emergency_halt(self, reason: str) -> None:
        """Transitions any active state to HALTED."""
        with self._lock:
            if self._state not in (SupervisorState.HALTED, SupervisorState.CLOSED):
                self.transition_to(SupervisorState.HALTED, reason=reason)

    def get_status_report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "current_state": self._state.value,
                "checklist": {
                    "broker_connected": self._checklist.broker_connected,
                    "market_clock_valid": self._checklist.market_clock_valid,
                    "journal_available": self._checklist.journal_available,
                    "strategies_warmed_up": self._checklist.strategies_warmed_up,
                    "risk_engine_ready": self._checklist.risk_engine_ready,
                    "kill_switch_active": self._checklist.kill_switch_active,
                    "reconciliation_clean": self._checklist.reconciliation_clean,
                    "all_passed": self._checklist.is_all_passed,
                },
                "history_length": len(self._state_history),
                "latest_transition": self._state_history[-1] if self._state_history else None,
            }
