"""
Execution State Persistence and Crash Recovery
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from modules.execution.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from modules.execution.fills import Fill
from modules.execution.position import Position


class ExecutionStatePersistence:
    """
    Persists and recovers orders, fills, positions, and risk metrics to survive process crashes.
    """

    def __init__(self, storage_path: str = "data/execution_state.json"):
        self.storage_path = storage_path

    def save_state(
        self,
        positions: Dict[str, Position],
        orders: List[Order],
        fills: List[Fill],
        realized_pnl: float,
        daily_trades: int,
        session_status: str,
    ) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "saved_at": datetime.now().isoformat(),
            "realized_pnl": realized_pnl,
            "daily_trades": daily_trades,
            "session_status": session_status,
            "positions": {
                sym: {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "cost_basis": p.cost_basis,
                    "realized_pnl": p.realized_pnl,
                    "total_commission": p.total_commission,
                    "total_tax": p.total_tax,
                    "last_price": p.last_price,
                    "updated_at": p.updated_at.isoformat(),
                }
                for sym, p in positions.items()
            },
            "orders": [
                {
                    "order_id": o.order_id,
                    "broker_order_id": o.broker_order_id,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "order_type": o.order_type.value,
                    "quantity": o.quantity,
                    "price": o.price,
                    "time_in_force": o.time_in_force.value,
                    "status": o.status.value,
                    "filled_quantity": o.filled_quantity,
                    "remaining_quantity": o.remaining_quantity,
                    "average_fill_price": o.average_fill_price,
                    "strategy_id": o.strategy_id,
                    "signal_id": o.signal_id,
                    "rejection_reason": o.rejection_reason,
                    "created_at": o.created_at.isoformat(),
                    "updated_at": o.updated_at.isoformat(),
                }
                for o in orders
            ],
            "fills": [
                {
                    "fill_id": f.fill_id,
                    "order_id": f.order_id,
                    "broker_order_id": f.broker_order_id,
                    "symbol": f.symbol,
                    "side": f.side.value,
                    "quantity": f.quantity,
                    "price": f.price,
                    "commission": f.commission,
                    "tax": f.tax,
                    "slippage": f.slippage,
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in fills
            ],
        }

        tmp_file = f"{self.storage_path}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.storage_path)

    def load_state(self) -> Optional[Dict]:
        if not os.path.exists(self.storage_path):
            return None

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            positions: Dict[str, Position] = {}
            for sym, p_data in raw.get("positions", {}).items():
                positions[sym] = Position(
                    symbol=p_data["symbol"],
                    quantity=p_data["quantity"],
                    avg_price=p_data["avg_price"],
                    cost_basis=p_data["cost_basis"],
                    realized_pnl=p_data["realized_pnl"],
                    total_commission=p_data.get("total_commission", 0.0),
                    total_tax=p_data.get("total_tax", 0.0),
                    last_price=p_data.get("last_price", 0.0),
                    updated_at=datetime.fromisoformat(p_data["updated_at"]),
                )

            orders: List[Order] = []
            for o_data in raw.get("orders", []):
                order = Order(
                    order_id=o_data["order_id"],
                    symbol=o_data["symbol"],
                    side=OrderSide(o_data["side"]),
                    order_type=OrderType(o_data["order_type"]),
                    quantity=o_data["quantity"],
                    price=o_data.get("price"),
                    time_in_force=TimeInForce(o_data.get("time_in_force", "ROD")),
                    status=OrderStatus(o_data["status"]),
                    broker_order_id=o_data.get("broker_order_id"),
                    filled_quantity=o_data.get("filled_quantity", 0),
                    remaining_quantity=o_data.get("remaining_quantity", o_data["quantity"]),
                    average_fill_price=o_data.get("average_fill_price", 0.0),
                    strategy_id=o_data.get("strategy_id", "default"),
                    signal_id=o_data.get("signal_id"),
                    rejection_reason=o_data.get("rejection_reason"),
                    created_at=datetime.fromisoformat(o_data["created_at"]),
                    updated_at=datetime.fromisoformat(o_data["updated_at"]),
                )
                orders.append(order)

            order_map = {o.order_id: o for o in orders}
            fills: List[Fill] = []
            for f_data in raw.get("fills", []):
                fill = Fill(
                    fill_id=f_data["fill_id"],
                    order_id=f_data["order_id"],
                    broker_order_id=f_data.get("broker_order_id"),
                    symbol=f_data["symbol"],
                    side=OrderSide(f_data["side"]),
                    quantity=f_data["quantity"],
                    price=f_data["price"],
                    commission=f_data.get("commission", 0.0),
                    tax=f_data.get("tax", 0.0),
                    slippage=f_data.get("slippage", 0.0),
                    timestamp=datetime.fromisoformat(f_data["timestamp"]),
                )
                fills.append(fill)
                if fill.order_id in order_map and fill not in order_map[fill.order_id].fills:
                    order_map[fill.order_id].fills.append(fill)

            return {
                "saved_at": raw.get("saved_at"),
                "realized_pnl": raw.get("realized_pnl", 0.0),
                "daily_trades": raw.get("daily_trades", 0),
                "session_status": raw.get("session_status", "HALTED"),
                "positions": positions,
                "orders": orders,
                "fills": fills,
            }
        except Exception as e:
            print(f"Error restoring execution state: {e}")
            return None
