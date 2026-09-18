"""
Global Kill Switch
Provides emergency trading halt with persistent safety state across system restarts.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("QuantPilot.KillSwitch")


class KillSwitchStatus(str, Enum):
    ACTIVE = "ACTIVE"
    HALTED = "HALTED"


@dataclass
class KillSwitchRecord:
    status: KillSwitchStatus
    reason: str
    updated_at: datetime = field(default_factory=datetime.now)
    operator_id: Optional[str] = None


class KillSwitch:
    """
    Global Kill Switch for QuantPilot.
    Persists status to disk so that process restart does NOT accidentally resume trading.
    """

    def __init__(self, state_file: str = "data/kill_switch.json"):
        self.state_file = state_file
        self._lock = threading.RLock()
        self._status: KillSwitchStatus = KillSwitchStatus.ACTIVE
        self._reason: str = "Initial system startup"
        self._operator_id: Optional[str] = None
        self._updated_at: datetime = datetime.now()
        self._load_state()

    def _load_state(self) -> None:
        with self._lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._status = KillSwitchStatus(data.get("status", KillSwitchStatus.ACTIVE.value))
                        self._reason = data.get("reason", "Restored from file")
                        self._operator_id = data.get("operator_id")
                        self._updated_at = datetime.fromisoformat(data["updated_at"])
                except Exception as e:
                    # In case of corruption, fail-safe to HALTED
                    print(f"Warning: Failed to load kill switch state ({e}). Defaulting to HALTED for safety.")
                    self._status = KillSwitchStatus.HALTED
                    self._reason = f"Corrupt state file fallback: {e}"

    def _save_state(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            payload = {
                "status": self._status.value,
                "reason": self._reason,
                "operator_id": self._operator_id,
                "updated_at": self._updated_at.isoformat(),
            }
            tmp = f"{self.state_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self.state_file)

    def halt(self, reason: str, operator_id: Optional[str] = None) -> None:
        """Halts all trading operations immediately."""
        with self._lock:
            self._status = KillSwitchStatus.HALTED
            self._reason = reason
            self._operator_id = operator_id
            self._updated_at = datetime.now()
            self._save_state()
            logger.critical(f"[KILL SWITCH ENGAGED] Reason: {reason} by {operator_id or 'System'}")

    def resume(self, operator_id: str, reason: str) -> None:
        """Explicit manual resumption requiring operator identity."""
        if not operator_id:
            raise ValueError("Operator ID is required to resume trading after a halt")
        with self._lock:
            self._status = KillSwitchStatus.ACTIVE
            self._reason = f"Resumed by {operator_id}: {reason}"
            self._operator_id = operator_id
            self._updated_at = datetime.now()
            self._save_state()
            logger.info(f"[KILL SWITCH DISENGAGED] Operator: {operator_id}, Reason: {reason}")

    def is_halted(self) -> bool:
        with self._lock:
            return self._status == KillSwitchStatus.HALTED

    def get_status(self) -> KillSwitchRecord:
        with self._lock:
            return KillSwitchRecord(
                status=self._status,
                reason=self._reason,
                updated_at=self._updated_at,
                operator_id=self._operator_id,
            )
