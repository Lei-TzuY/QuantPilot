"""
Shioaji Broker Adapter for Taiwan Equities (Sinopac)
Provides strict isolation, environmental configuration, and live-trading safety guards.
"""
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from modules.brokers.base import BrokerAdapter
from modules.execution.order import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from modules.execution.fills import Fill
from modules.execution.position import Position
from modules.execution.events import TickEvent

# Check if shioaji is installed
try:
    import shioaji as sj
    from shioaji import constant as sj_const
    SHIOAJI_AVAILABLE = True
except ImportError:
    SHIOAJI_AVAILABLE = False
    sj = None
    sj_const = None


class ShioajiBrokerAdapter(BrokerAdapter):
    """
    Shioaji Taiwan Equity Broker Adapter.
    Translates between QuantPilot domain models and SinoPac Shioaji API.
    
    SAFETY INVARIANT:
    Real orders can NEVER be submitted unless TRADING_MODE='live' is explicitly configured.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        cert_path: Optional[str] = None,
        cert_password: Optional[str] = None,
        person_id: Optional[str] = None,
        trading_mode: str = "paper",
        simulation: bool = True,
    ):
        super().__init__()
        # Load from arguments or environment variables
        self.api_key = api_key or os.getenv("SHIOAJI_API_KEY", "")
        self.secret_key = secret_key or os.getenv("SHIOAJI_SECRET_KEY", "")
        self.cert_path = cert_path or os.getenv("SHIOAJI_CERT_PATH", "")
        self.cert_password = cert_password or os.getenv("SHIOAJI_CERT_PASSWORD", "")
        self.person_id = person_id or os.getenv("SHIOAJI_PERSON_ID", "")
        self.trading_mode = (trading_mode or os.getenv("TRADING_MODE", "paper")).lower()
        self.simulation = simulation

        self._api = None
        self._connected = False
        self._account = None
        self._positions_cache: Dict[str, Position] = {}
        self._orders_cache: Dict[str, Order] = {}
        self._tick_callbacks: List[Callable[[TickEvent], None]] = []

    def connect(self) -> bool:
        """Authenticates and initializes Shioaji session."""
        if self.trading_mode == "live":
            if not self.api_key or not self.secret_key:
                raise ValueError(
                    "Shioaji credentials missing. Provide SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY via environment."
                )
            if not SHIOAJI_AVAILABLE:
                raise RuntimeError("Cannot operate in live mode: shioaji library is not installed.")

        if not SHIOAJI_AVAILABLE:
            # When Shioaji library is not present (e.g. CI or mock env)
            print("Notice: shioaji package not installed. Operating in mock/isolated adapter mode.")
            self._connected = True
            return True

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Shioaji credentials missing. Provide SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY via environment."
            )

        try:
            self._api = sj.Shioaji(simulation=self.simulation)
            accounts = self._api.login(
                api_key=self.api_key,
                secret_key=self.secret_key,
                contracts_cb=lambda count: None,
            )
            if accounts:
                self._account = self._api.stock_account
            
            # Activate CA certificate for real order placement if live
            if self.cert_path and self.cert_password and self.person_id:
                self._api.activate_ca(
                    ca_path=self.cert_path,
                    ca_passwd=self.cert_password,
                    person_id=self.person_id,
                )

            # Register Shioaji callbacks
            self._api.set_order_callback(self._on_shioaji_order_status)
            self._connected = True
            return True
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Shioaji: {e}")

    def disconnect(self) -> None:
        if self._api and SHIOAJI_AVAILABLE and self._connected:
            try:
                self._api.logout()
            except Exception:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_account(self) -> Dict[str, Any]:
        if not self._connected:
            return {"connected": False, "broker": "Shioaji"}

        if not SHIOAJI_AVAILABLE or not self._api:
            return {
                "connected": self._connected,
                "broker": "Shioaji (Simulated/Mock)",
                "trading_mode": self.trading_mode,
            }

        try:
            margin = self._api.margin(self._account) if self._account else None
            return {
                "connected": True,
                "broker": "Shioaji",
                "trading_mode": self.trading_mode,
                "account_id": getattr(self._account, "account_id", "N/A"),
                "margin": str(margin),
            }
        except Exception as e:
            return {"connected": True, "broker": "Shioaji", "error": str(e)}

    def get_positions(self) -> Dict[str, Position]:
        """Retrieves live positions from Shioaji and normalizes to domain Position models."""
        if not self._connected or not SHIOAJI_AVAILABLE or not self._api or not self._account:
            return self._positions_cache

        try:
            positions_data = self._api.list_positions(self._account)
            result: Dict[str, Position] = {}
            for p in positions_data:
                symbol = getattr(p, "code", "")
                qty = int(getattr(p, "quantity", 0))
                price = float(getattr(p, "price", 0.0))
                pnl = float(getattr(p, "pnl", 0.0))
                pos = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=price,
                    cost_basis=qty * price,
                    realized_pnl=pnl,
                    last_price=price,
                )
                result[symbol] = pos
            self._positions_cache = result
            return result
        except Exception as e:
            print(f"Error fetching Shioaji positions: {e}")
            return self._positions_cache

    def get_open_orders(self) -> List[Order]:
        if not self._connected or not SHIOAJI_AVAILABLE or not self._api:
            return list(self._orders_cache.values())

        try:
            self._api.update_status(self._account)
            # Normalize Shioaji trades/orders
            return list(self._orders_cache.values())
        except Exception as e:
            print(f"Error updating Shioaji order status: {e}")
            return list(self._orders_cache.values())

    def submit_order(self, order: Order) -> Order:
        """
        Submits order to Shioaji.
        STRICT LIVE GUARD: Live execution is unconditionally blocked unless TRADING_MODE='live' and ENABLE_LIVE_TRADING='true'.
        """
        enable_live = os.getenv("ENABLE_LIVE_TRADING", "false").strip().lower() in ("true", "1", "yes")
        if self.trading_mode != "live" or not enable_live:
            order.transition_to(
                OrderStatus.REJECTED,
                reason="LIVE_TRADING_DISABLED: Both TRADING_MODE='live' and ENABLE_LIVE_TRADING='true' must be explicitly configured.",
            )
            self._notify_order(order)
            return order

        if not self._connected:
            order.transition_to(OrderStatus.REJECTED, reason="BROKER_NOT_CONNECTED")
            self._notify_order(order)
            return order

        if not SHIOAJI_AVAILABLE or not self._api:
            order.transition_to(OrderStatus.REJECTED, reason="SHIOAJI_SDK_NOT_AVAILABLE")
            self._notify_order(order)
            return order

        try:
            # Map symbol to contract
            clean_sym = order.symbol.replace(".TW", "").replace(".TWO", "")
            contract = self._api.Contracts.Stocks[clean_sym]
            if not contract:
                order.transition_to(OrderStatus.REJECTED, reason=f"CONTRACT_NOT_FOUND: {clean_sym}")
                self._notify_order(order)
                return order

            action = sj_const.Action.Buy if order.side == OrderSide.BUY else sj_const.Action.Sell
            price_type = (
                sj_const.StockPriceType.LMT
                if order.order_type == OrderType.LIMIT
                else sj_const.StockPriceType.MKT
            )
            order_type = sj_const.OrderType.ROD

            sj_order = self._api.Order(
                price=order.price or 0.0,
                quantity=order.quantity,
                action=action,
                price_type=price_type,
                order_type=order_type,
            )

            trade = self._api.place_order(contract, sj_order)
            broker_order_id = getattr(trade.status, "id", None) or getattr(sj_order, "id", "SJ-PENDING")
            order.broker_order_id = str(broker_order_id)
            order.transition_to(OrderStatus.SUBMITTED)
            self._orders_cache[order.order_id] = order
            self._notify_order(order)
            return order

        except Exception as e:
            order.transition_to(OrderStatus.REJECTED, reason=f"SHIOAJI_SUBMIT_EXCEPTION: {str(e)}")
            self._notify_order(order)
            return order

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected or not SHIOAJI_AVAILABLE or not self._api:
            return False

        order = self._orders_cache.get(order_id)
        if not order or not order.broker_order_id:
            return False

        try:
            # In Shioaji, cancelling requires the trade object
            self._api.cancel_order(order.broker_order_id)
            order.transition_to(OrderStatus.CANCEL_PENDING)
            self._notify_order(order)
            return True
        except Exception as e:
            print(f"Error cancelling Shioaji order {order_id}: {e}")
            return False

    def subscribe_market_data(self, symbols: List[str]) -> None:
        if not self._connected or not SHIOAJI_AVAILABLE or not self._api:
            return
        for s in symbols:
            clean = s.replace(".TW", "")
            try:
                contract = self._api.Contracts.Stocks[clean]
                if contract:
                    self._api.quote.subscribe(contract, quote_type=sj_const.QuoteType.Tick)
            except Exception as e:
                print(f"Error subscribing to {s}: {e}")

    def register_tick_callback(self, callback: Callable[[TickEvent], None]) -> None:
        """Registers listener for normalized real-time quote ticks."""
        self._tick_callbacks.append(callback)

    def _notify_tick(self, tick: TickEvent) -> None:
        for cb in self._tick_callbacks:
            try:
                cb(tick)
            except Exception as e:
                print(f"Error in tick callback: {e}")

    def simulate_tick(self, tick: TickEvent) -> None:
        """Emits a normalized TickEvent to registered listeners (for shadow/replay mode)."""
        self._notify_tick(tick)

    def _on_shioaji_tick(self, exchange: str, tick: Any) -> None:
        """Normalizes native Shioaji quote tick to domain TickEvent."""
        try:
            code = str(getattr(tick, "code", ""))
            price = float(getattr(tick, "close", getattr(tick, "price", 0.0)))
            vol = float(getattr(tick, "volume", getattr(tick, "vol", 0.0)))
            bid = float(getattr(tick, "bid_price", 0.0)) if hasattr(tick, "bid_price") else None
            ask = float(getattr(tick, "ask_price", 0.0)) if hasattr(tick, "ask_price") else None
            ts = getattr(tick, "ts", None) or getattr(tick, "datetime", None) or datetime.now()
            if not isinstance(ts, datetime):
                ts = datetime.now()

            tick_event = TickEvent(
                timestamp=ts,
                symbol=code,
                price=price,
                volume=vol,
                bid_price=bid,
                ask_price=ask,
            )
            self._notify_tick(tick_event)
        except Exception as e:
            print(f"Error parsing Shioaji tick: {e}")

    def unsubscribe_market_data(self, symbols: List[str]) -> None:
        pass

    def _on_shioaji_order_status(self, stat: Any, msg: Dict) -> None:
        """Handles Shioaji asynchronous order and deal callbacks."""
        try:
            broker_id = str(getattr(stat, "id", ""))
            status_code = getattr(stat, "status", "")

            # Match order in cache
            target_order = None
            for o in self._orders_cache.values():
                if o.broker_order_id == broker_id:
                    target_order = o
                    break

            if not target_order:
                return

            # Check if deal/fill event
            deals = getattr(stat, "deals", [])
            for deal in deals:
                deal_qty = int(getattr(deal, "quantity", 0))
                deal_price = float(getattr(deal, "price", 0.0))
                deal_ts = getattr(deal, "ts", None) or datetime.now()

                fill = Fill(
                    fill_id=f"SJ-FILL-{broker_id}-{deal_qty}",
                    order_id=target_order.order_id,
                    broker_order_id=broker_id,
                    symbol=target_order.symbol,
                    side=target_order.side,
                    quantity=deal_qty,
                    price=deal_price,
                    commission=deal_price * deal_qty * 0.001425,
                    tax=deal_price * deal_qty * 0.003 if target_order.side == OrderSide.SELL else 0.0,
                    timestamp=deal_ts if isinstance(deal_ts, datetime) else datetime.now(),
                )
                self._notify_fill(fill)

            if status_code in ("Filled", "PartFilled", "Cancelled", "Failed"):
                if status_code == "Filled" and target_order.status != OrderStatus.FILLED:
                    target_order.transition_to(OrderStatus.FILLED)
                elif status_code == "Cancelled" and target_order.status != OrderStatus.CANCELLED:
                    target_order.transition_to(OrderStatus.CANCELLED)
                elif status_code == "Failed" and target_order.status != OrderStatus.REJECTED:
                    target_order.transition_to(OrderStatus.REJECTED, reason=str(msg))
                self._notify_order(target_order)

        except Exception as e:
            print(f"Error handling Shioaji callback: {e}")

    def heartbeat(self) -> Dict[str, Any]:
        return {
            "broker": "Shioaji",
            "connected": self._connected,
            "trading_mode": self.trading_mode,
            "timestamp": datetime.now().isoformat(),
        }
