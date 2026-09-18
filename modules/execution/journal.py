"""
Durable Append-Only Execution Journal
SQLite WAL mode implementation for crash-safe execution state persistence.
"""
import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.execution.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from modules.execution.fills import Fill


SCHEMA_VERSION = 1

INIT_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    target_quantity INTEGER,
    target_price REAL,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS risk_decisions (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    approved INTEGER NOT NULL,
    reason TEXT,
    metrics_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS order_requests (
    order_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL,
    strategy_id TEXT,
    signal_id TEXT,
    correlation_id TEXT
);

CREATE TABLE IF NOT EXISTS order_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    broker_order_id TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL,
    tax REAL NOT NULL,
    slippage REAL NOT NULL,
    broker_order_id TEXT
);

CREATE TABLE IF NOT EXISTS reconciliation_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    is_clean INTEGER NOT NULL,
    critical_count INTEGER NOT NULL,
    warning_count INTEGER NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS kill_switch_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    reason TEXT
);
"""


class ExecutionJournal:
    """
    Append-only durable execution journal backed by SQLite in WAL mode.
    Guarantees crash-safe persistence and deterministic state restoration.
    """

    def __init__(self, db_path: str = "data/execution_journal.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.executescript(INIT_SQL)
                cur = conn.cursor()
                cur.execute("SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,))
                if not cur.fetchone():
                    cur.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (SCHEMA_VERSION, datetime.now().isoformat()),
                    )
                conn.commit()

    def record_signal(
        self,
        event_id: str,
        timestamp: datetime,
        strategy_id: str,
        symbol: str,
        signal_type: str,
        target_quantity: Optional[int] = None,
        target_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a signal event idempotently."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO signals 
                    (event_id, timestamp, strategy_id, symbol, signal_type, target_quantity, target_price, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        timestamp.isoformat(),
                        strategy_id,
                        symbol,
                        signal_type,
                        target_quantity,
                        target_price,
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()

    def record_risk_decision(
        self,
        decision_id: str,
        timestamp: datetime,
        order_id: str,
        approved: bool,
        reason: str = "",
        metrics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a pre-trade risk decision idempotently."""
        try:
            snapshot_str = json.dumps(metrics_snapshot or {}, default=str)
        except Exception:
            snapshot_str = "{}"

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO risk_decisions
                    (decision_id, timestamp, order_id, approved, reason, metrics_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        timestamp.isoformat(),
                        order_id,
                        1 if approved else 0,
                        reason,
                        snapshot_str,
                    ),
                )
                conn.commit()

    def record_order_request(
        self,
        order_id: str,
        timestamp: datetime,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: Optional[float] = None,
        strategy_id: str = "default",
        signal_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Appends an order request into the journal."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO order_requests
                    (order_id, timestamp, symbol, side, order_type, quantity, price, strategy_id, signal_id, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_id,
                        timestamp.isoformat(),
                        symbol,
                        side,
                        order_type,
                        quantity,
                        price,
                        strategy_id,
                        signal_id,
                        correlation_id,
                    ),
                )
                conn.commit()

    def record_order_transition(
        self,
        order_id: str,
        timestamp: datetime,
        from_status: str,
        to_status: str,
        broker_order_id: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """Appends an order lifecycle transition."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO order_transitions
                    (timestamp, order_id, from_status, to_status, broker_order_id, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp.isoformat(),
                        order_id,
                        from_status,
                        to_status,
                        broker_order_id,
                        reason,
                    ),
                )
                conn.commit()

    def record_fill(self, fill: Fill) -> None:
        """Appends a fill event idempotently."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO fills
                    (fill_id, timestamp, order_id, symbol, side, quantity, price, commission, tax, slippage, broker_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill.fill_id,
                        fill.timestamp.isoformat(),
                        fill.order_id,
                        fill.symbol,
                        fill.side.value,
                        fill.quantity,
                        fill.price,
                        fill.commission,
                        fill.tax,
                        fill.slippage,
                        fill.broker_order_id,
                    ),
                )
                conn.commit()

    def record_reconciliation(
        self,
        event_id: str,
        timestamp: datetime,
        is_clean: bool,
        critical_count: int,
        warning_count: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Appends a reconciliation event."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO reconciliation_events
                    (event_id, timestamp, is_clean, critical_count, warning_count, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        timestamp.isoformat(),
                        1 if is_clean else 0,
                        critical_count,
                        warning_count,
                        json.dumps(details or {}),
                    ),
                )
                conn.commit()

    def record_kill_switch(
        self,
        event_id: str,
        timestamp: datetime,
        action: str,
        operator_id: str,
        reason: str,
    ) -> None:
        """Appends a kill switch state change."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO kill_switch_events
                    (event_id, timestamp, action, operator_id, reason)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        timestamp.isoformat(),
                        action.upper(),
                        operator_id,
                        reason,
                    ),
                )
                conn.commit()

    def restore_state(self) -> Dict[str, Any]:
        """
        Reconstructs internal state from the append-only journal:
        - Orders (with accumulated fills, VWAP prices, and latest status)
        - Fills
        - Kill switch halted state
        """
        with self._lock:
            with self._get_connection() as conn:
                # 1. Rebuild Orders
                orders: Dict[str, Order] = {}
                cur = conn.cursor()
                cur.execute("SELECT * FROM order_requests ORDER BY timestamp ASC")
                for row in cur.fetchall():
                    order_id = row["order_id"]
                    orders[order_id] = Order(
                        order_id=order_id,
                        symbol=row["symbol"],
                        side=OrderSide(row["side"]),
                        order_type=OrderType(row["order_type"]),
                        quantity=row["quantity"],
                        price=row["price"],
                        time_in_force=TimeInForce.ROD,
                        status=OrderStatus.NEW,
                        strategy_id=row["strategy_id"],
                        signal_id=row["signal_id"],
                        created_at=datetime.fromisoformat(row["timestamp"]),
                        updated_at=datetime.fromisoformat(row["timestamp"]),
                    )

                # Apply Transitions
                cur.execute("SELECT * FROM order_transitions ORDER BY id ASC")
                for row in cur.fetchall():
                    order_id = row["order_id"]
                    if order_id in orders:
                        to_status = OrderStatus(row["to_status"])
                        order = orders[order_id]
                        order.status = to_status
                        if row["broker_order_id"]:
                            order.broker_order_id = row["broker_order_id"]
                        if row["reason"]:
                            order.rejection_reason = row["reason"]
                        order.updated_at = datetime.fromisoformat(row["timestamp"])

                # Apply Fills to Orders
                fills: List[Fill] = []
                cur.execute("SELECT * FROM fills ORDER BY timestamp ASC")
                for row in cur.fetchall():
                    fill = Fill(
                        fill_id=row["fill_id"],
                        order_id=row["order_id"],
                        symbol=row["symbol"],
                        side=OrderSide(row["side"]),
                        quantity=row["quantity"],
                        price=row["price"],
                        commission=row["commission"],
                        tax=row["tax"],
                        slippage=row["slippage"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        broker_order_id=row["broker_order_id"],
                    )
                    fills.append(fill)

                    order_id = fill.order_id
                    if order_id in orders:
                        order = orders[order_id]
                        prev_filled = order.filled_quantity
                        prev_cost = order.average_fill_price * prev_filled
                        new_cost = prev_cost + (fill.price * fill.quantity)
                        total_filled = prev_filled + fill.quantity
                        order.filled_quantity = total_filled
                        order.remaining_quantity = order.quantity - total_filled
                        order.average_fill_price = new_cost / total_filled if total_filled > 0 else 0.0

                # 2. Rebuild Kill Switch State
                cur.execute("SELECT action FROM kill_switch_events ORDER BY rowid DESC LIMIT 1")
                last_ks = cur.fetchone()
                is_halted = False
                if last_ks and last_ks["action"] == "HALT":
                    is_halted = True

                return {
                    "orders": orders,
                    "fills": fills,
                    "is_halted": is_halted,
                }
