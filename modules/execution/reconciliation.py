"""
Position and Order Reconciliation Engine
Detects discrepancies between internal state and broker authoritative state.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from modules.execution.order import Order, OrderStatus
from modules.execution.position import Position


class DiscrepancyType(str, Enum):
    POSITION_QUANTITY_MISMATCH = "POSITION_QUANTITY_MISMATCH"
    POSITION_MISSING_INTERNALLY = "POSITION_MISSING_INTERNALLY"
    POSITION_MISSING_AT_BROKER = "POSITION_MISSING_AT_BROKER"
    ORDER_STATUS_MISMATCH = "ORDER_STATUS_MISMATCH"
    ORPHAN_BROKER_ORDER = "ORPHAN_BROKER_ORDER"
    ORPHAN_INTERNAL_ORDER = "ORPHAN_INTERNAL_ORDER"


class DiscrepancySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class ReconciliationDiscrepancy:
    discrepancy_type: DiscrepancyType
    severity: DiscrepancySeverity
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    internal_value: Optional[str] = None
    broker_value: Optional[str] = None
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReconciliationReport:
    timestamp: datetime
    is_clean: bool
    discrepancies: List[ReconciliationDiscrepancy]
    critical_count: int = 0
    warning_count: int = 0


class Reconciler:
    """
    Reconciliation engine comparing internal OMS state with broker authoritative state.
    """

    def reconcile_positions(
        self,
        internal_positions: Dict[str, Position],
        broker_positions: Dict[str, Position],
    ) -> List[ReconciliationDiscrepancy]:
        discrepancies: List[ReconciliationDiscrepancy] = []
        all_symbols = set(internal_positions.keys()) | set(broker_positions.keys())

        for sym in all_symbols:
            int_pos = internal_positions.get(sym)
            brk_pos = broker_positions.get(sym)

            int_qty = int_pos.quantity if int_pos else 0
            brk_qty = brk_pos.quantity if brk_pos else 0

            if int_pos is None and brk_qty > 0:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNALLY,
                        severity=DiscrepancySeverity.CRITICAL,
                        symbol=sym,
                        internal_value="None",
                        broker_value=str(brk_qty),
                        description=f"Broker holds {brk_qty} shares of {sym} not recognized internally.",
                    )
                )
            elif brk_pos is None and int_qty > 0:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_AT_BROKER,
                        severity=DiscrepancySeverity.CRITICAL,
                        symbol=sym,
                        internal_value=str(int_qty),
                        broker_value="None",
                        description=f"Internal state records {int_qty} shares of {sym} but broker has 0.",
                    )
                )
            elif int_qty != brk_qty:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        discrepancy_type=DiscrepancyType.POSITION_QUANTITY_MISMATCH,
                        severity=DiscrepancySeverity.CRITICAL,
                        symbol=sym,
                        internal_value=str(int_qty),
                        broker_value=str(brk_qty),
                        description=f"Position mismatch for {sym}: internal={int_qty}, broker={brk_qty}.",
                    )
                )

        return discrepancies

    def reconcile_orders(
        self,
        internal_open_orders: List[Order],
        broker_open_orders: List[Order],
    ) -> List[ReconciliationDiscrepancy]:
        discrepancies: List[ReconciliationDiscrepancy] = []

        int_by_broker_id: Dict[str, Order] = {
            o.broker_order_id: o for o in internal_open_orders if o.broker_order_id
        }
        brk_by_id: Dict[str, Order] = {
            (o.broker_order_id or o.order_id): o for o in broker_open_orders
        }

        # Check broker orders missing internally
        for brk_id, brk_order in brk_by_id.items():
            if brk_id not in int_by_broker_id:
                discrepancies.append(
                    ReconciliationDiscrepancy(
                        discrepancy_type=DiscrepancyType.ORPHAN_BROKER_ORDER,
                        severity=DiscrepancySeverity.WARNING,
                        symbol=brk_order.symbol,
                        broker_order_id=brk_id,
                        description=f"Broker open order {brk_id} ({brk_order.symbol}) is not tracked in internal OMS.",
                    )
                )

        # Check internal open orders missing at broker
        for int_order in internal_open_orders:
            if int_order.status in {OrderStatus.SUBMITTED, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}:
                if int_order.broker_order_id and int_order.broker_order_id not in brk_by_id:
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            discrepancy_type=DiscrepancyType.ORPHAN_INTERNAL_ORDER,
                            severity=DiscrepancySeverity.WARNING,
                            symbol=int_order.symbol,
                            order_id=int_order.order_id,
                            broker_order_id=int_order.broker_order_id,
                            description=f"Internal open order {int_order.order_id} not found in broker open orders.",
                        )
                    )

        return discrepancies

    def run_full_reconciliation(
        self,
        internal_positions: Dict[str, Position],
        broker_positions: Dict[str, Position],
        internal_open_orders: List[Order],
        broker_open_orders: List[Order],
    ) -> ReconciliationReport:
        pos_discrepancies = self.reconcile_positions(internal_positions, broker_positions)
        ord_discrepancies = self.reconcile_orders(internal_open_orders, broker_open_orders)
        all_discrepancies = pos_discrepancies + ord_discrepancies

        crit_count = sum(1 for d in all_discrepancies if d.severity == DiscrepancySeverity.CRITICAL)
        warn_count = sum(1 for d in all_discrepancies if d.severity == DiscrepancySeverity.WARNING)

        return ReconciliationReport(
            timestamp=datetime.now(),
            is_clean=(len(all_discrepancies) == 0),
            discrepancies=all_discrepancies,
            critical_count=crit_count,
            warning_count=warn_count,
        )
