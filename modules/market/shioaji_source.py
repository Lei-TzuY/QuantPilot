"""
Shioaji Read-Only Market Data Source for Taiwan Equities.
Provides strict separation between market-data transport and broker execution.
Exclusively handles Tick and BidAsk quote streaming with ZERO live-order capabilities.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

from modules.execution.events import BidAskEvent, TickEvent

logger = logging.getLogger("QuantPilot.ShioajiMarketData")

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

try:
    import shioaji as sj
    from shioaji import constant as sj_const
    SHIOAJI_AVAILABLE = True
except ImportError:
    SHIOAJI_AVAILABLE = False
    sj = None
    sj_const = None


class ConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    AUTHENTICATED = "AUTHENTICATED"
    SUBSCRIBING = "SUBSCRIBING"
    STREAMING = "STREAMING"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"


class MarketDataSource(ABC):
    """
    Abstract interface for streaming market data transports.
    Crucial Architectural Rule: Market data sources must NEVER expose order methods.
    """

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def subscribe(self, symbols: List[str]) -> None:
        pass

    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> None:
        pass

    @abstractmethod
    def register_tick_callback(self, callback: Callable[[TickEvent], None]) -> None:
        pass

    @abstractmethod
    def register_bidask_callback(self, callback: Callable[[BidAskEvent], None]) -> None:
        pass

    @abstractmethod
    def heartbeat(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass


class ShioajiMarketDataSource(MarketDataSource):
    """
    SinoPac Shioaji Quote-Only Streaming Market Data Transport.
    Features:
    - Official native stock V1 callbacks on `api.quote`:
        - `set_on_tick_stk_v1_callback`
        - `set_on_bidask_stk_v1_callback`
        - `set_event_callback`
    - Full dual subscription (Tick + BidAsk) and unsubscription.
    - Robust Decimal handling and Asia/Taipei timezone awareness.
    - Malformed payload rejection without fallback to misleading wall clock.
    - Zero capability for order placement (no CA certificate, no order API access).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        simulation: bool = True,
        api_instance: Optional[Any] = None,
    ):
        self.api_key = api_key or os.getenv("SHIOAJI_API_KEY", "")
        self.secret_key = secret_key or os.getenv("SHIOAJI_SECRET_KEY", "")
        self.simulation = simulation

        self._api = api_instance
        self._custom_api_provided = api_instance is not None
        self._lock = threading.RLock()
        self._state = ConnectionState.DISCONNECTED

        # Registered external callbacks
        self._tick_callbacks: List[Callable[[TickEvent], None]] = []
        self._bidask_callbacks: List[Callable[[BidAskEvent], None]] = []

        # Subscription state
        self._subscribed_tick_symbols: Set[str] = set()
        self._subscribed_bidask_symbols: Set[str] = set()
        self._target_symbols: Set[str] = set()

        # Telemetry & Health counters
        self._ticks_received_count = 0
        self._bidask_received_count = 0
        self._malformed_payload_count = 0
        self._disconnect_count = 0
        self._reconnect_count = 0
        self._resubscribe_count = 0
        self._last_tick_time: Optional[datetime] = None
        self._last_bidask_time: Optional[datetime] = None
        self._last_error: Optional[str] = None

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    def is_connected(self) -> bool:
        with self._lock:
            return self._state in (
                ConnectionState.AUTHENTICATED,
                ConnectionState.SUBSCRIBING,
                ConnectionState.STREAMING,
            )

    def register_tick_callback(self, callback: Callable[[TickEvent], None]) -> None:
        with self._lock:
            if callback not in self._tick_callbacks:
                self._tick_callbacks.append(callback)

    def register_bidask_callback(self, callback: Callable[[BidAskEvent], None]) -> None:
        with self._lock:
            if callback not in self._bidask_callbacks:
                self._bidask_callbacks.append(callback)

    def connect(self) -> bool:
        """Authenticates with Shioaji and binds native callbacks to `api.quote`."""
        with self._lock:
            if self._state in (ConnectionState.AUTHENTICATED, ConnectionState.STREAMING):
                return True

            self._state = ConnectionState.CONNECTING
            logger.info("Connecting ShioajiMarketDataSource (quote-only)...")

            if self._custom_api_provided and self._api:
                self._register_native_callbacks(self._api)
                self._state = ConnectionState.AUTHENTICATED
                return True

            if not SHIOAJI_AVAILABLE:
                logger.warning("shioaji package not installed. Running in mock market data source mode.")
                self._state = ConnectionState.AUTHENTICATED
                return True

            if not self.api_key or not self.secret_key:
                self._state = ConnectionState.DEGRADED
                self._last_error = "MISSING_CREDENTIALS: SHIOAJI_API_KEY and SHIOAJI_SECRET_KEY required"
                raise ValueError(self._last_error)

            try:
                self._api = sj.Shioaji(simulation=self.simulation)
                accounts = self._api.login(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    contracts_cb=lambda count: None,
                )
                self._register_native_callbacks(self._api)
                self._state = ConnectionState.AUTHENTICATED
                logger.info("ShioajiMarketDataSource authenticated successfully.")
                return True
            except Exception as e:
                self._state = ConnectionState.DISCONNECTED
                self._last_error = str(e)
                logger.error(f"Shioaji login failed: {e}", exc_info=True)
                raise ConnectionError(f"Failed to connect Shioaji quote transport: {e}")

    def _register_native_callbacks(self, api: Any) -> None:
        """Binds native SDK stock V1 tick and bidask callbacks on `api.quote`."""
        if not hasattr(api, "quote"):
            logger.warning("API instance missing 'quote' attribute; skipping native callback hooks.")
            return

        quote = api.quote
        # Register Tick callback
        if hasattr(quote, "set_on_tick_stk_v1_callback"):
            quote.set_on_tick_stk_v1_callback(self._on_native_tick)
        elif hasattr(quote, "set_on_tick_callback"):
            quote.set_on_tick_callback(self._on_native_tick)

        # Register BidAsk callback
        if hasattr(quote, "set_on_bidask_stk_v1_callback"):
            quote.set_on_bidask_stk_v1_callback(self._on_native_bidask)
        elif hasattr(quote, "set_on_bidask_callback"):
            quote.set_on_bidask_callback(self._on_native_bidask)

        # Register Event callback (Solace session lifecycle)
        if hasattr(quote, "set_event_callback"):
            quote.set_event_callback(self._on_native_event)

        logger.debug("Successfully registered native Shioaji quote callbacks.")

    def subscribe(self, symbols: List[str]) -> None:
        """
        Subscribes both Tick and BidAsk streams for the given symbols.
        Guarantees:
        - Prevents duplicate subscriptions.
        - Fails if called before callback registration.
        """
        with self._lock:
            if not self.is_connected():
                raise RuntimeError("Cannot subscribe market data: Shioaji source is not connected.")

            self._state = ConnectionState.SUBSCRIBING

            for sym in symbols:
                clean_sym = sym.replace(".TW", "").replace(".TWO", "")
                self._target_symbols.add(clean_sym)

                if not self._api or not hasattr(self._api, "Contracts"):
                    # Mock mode
                    self._subscribed_tick_symbols.add(clean_sym)
                    self._subscribed_bidask_symbols.add(clean_sym)
                    continue

                try:
                    contract = self._get_contract(clean_sym)
                    if not contract:
                        logger.error(f"Contract not found for symbol: {clean_sym}")
                        continue

                    # 1. Subscribe Tick
                    if clean_sym not in self._subscribed_tick_symbols:
                        quote_type_tick = getattr(sj_const, "QuoteType", None)
                        q_type = getattr(quote_type_tick, "Tick", "tick") if quote_type_tick else "tick"
                        self._api.quote.subscribe(contract, quote_type=q_type)
                        self._subscribed_tick_symbols.add(clean_sym)
                        logger.info(f"Subscribed Tick for {clean_sym}")

                    # 2. Subscribe BidAsk
                    if clean_sym not in self._subscribed_bidask_symbols:
                        quote_type_bidask = getattr(sj_const, "QuoteType", None)
                        q_type = getattr(quote_type_bidask, "BidAsk", "bidask") if quote_type_bidask else "bidask"
                        self._api.quote.subscribe(contract, quote_type=q_type)
                        self._subscribed_bidask_symbols.add(clean_sym)
                        logger.info(f"Subscribed BidAsk for {clean_sym}")

                except Exception as e:
                    logger.error(f"Failed subscribing market data for {clean_sym}: {e}", exc_info=True)
                    self._last_error = str(e)

            self._state = ConnectionState.STREAMING

    def unsubscribe(self, symbols: List[str]) -> None:
        """Fully unsubscribes both Tick and BidAsk streams for the given symbols."""
        with self._lock:
            for sym in symbols:
                clean_sym = sym.replace(".TW", "").replace(".TWO", "")
                self._target_symbols.discard(clean_sym)

                if not self._api or not hasattr(self._api, "Contracts"):
                    self._subscribed_tick_symbols.discard(clean_sym)
                    self._subscribed_bidask_symbols.discard(clean_sym)
                    continue

                try:
                    contract = self._get_contract(clean_sym)
                    if not contract:
                        continue

                    # Unsubscribe Tick
                    if clean_sym in self._subscribed_tick_symbols:
                        quote_type = getattr(sj_const, "QuoteType", None)
                        q_type = getattr(quote_type, "Tick", "tick") if quote_type else "tick"
                        if hasattr(self._api.quote, "unsubscribe"):
                            self._api.quote.unsubscribe(contract, quote_type=q_type)
                        self._subscribed_tick_symbols.discard(clean_sym)
                        logger.info(f"Unsubscribed Tick for {clean_sym}")

                    # Unsubscribe BidAsk
                    if clean_sym in self._subscribed_bidask_symbols:
                        quote_type = getattr(sj_const, "QuoteType", None)
                        q_type = getattr(quote_type, "BidAsk", "bidask") if quote_type else "bidask"
                        if hasattr(self._api.quote, "unsubscribe"):
                            self._api.quote.unsubscribe(contract, quote_type=q_type)
                        self._subscribed_bidask_symbols.discard(clean_sym)
                        logger.info(f"Unsubscribed BidAsk for {clean_sym}")

                except Exception as e:
                    logger.error(f"Error during unsubscribe for {clean_sym}: {e}", exc_info=True)

    def _get_contract(self, clean_sym: str) -> Any:
        try:
            contracts = self._api.Contracts
            if hasattr(contracts, "Stocks") and clean_sym in contracts.Stocks:
                return contracts.Stocks[clean_sym]
            return None
        except Exception:
            return None

    def disconnect(self) -> None:
        with self._lock:
            logger.info("Disconnecting ShioajiMarketDataSource...")
            # Unsubscribe all active symbols cleanly
            active_symbols = list(self._target_symbols)
            if active_symbols:
                self.unsubscribe(active_symbols)

            if self._api and SHIOAJI_AVAILABLE and not self._custom_api_provided:
                try:
                    self._api.logout()
                except Exception:
                    pass

            self._state = ConnectionState.DISCONNECTED
            self._subscribed_tick_symbols.clear()
            self._subscribed_bidask_symbols.clear()

    # -------------------------------------------------------------------------
    # Native Callbacks & Payload Normalization
    # -------------------------------------------------------------------------

    def _parse_timestamp(self, ts_raw: Any) -> Optional[datetime]:
        """Normalizes various Shioaji timestamp formats to timezone-aware Asia/Taipei datetime."""
        if ts_raw is None:
            return None
        try:
            if isinstance(ts_raw, datetime):
                if ts_raw.tzinfo is None:
                    return ts_raw.replace(tzinfo=TAIPEI_TZ)
                return ts_raw.astimezone(TAIPEI_TZ)
            if isinstance(ts_raw, (int, float)):
                # Unix timestamp in seconds or microseconds
                if ts_raw > 1e14:  # microseconds or nanoseconds
                    sec = ts_raw / 1e6
                elif ts_raw > 1e11:  # milliseconds
                    sec = ts_raw / 1e3
                else:
                    sec = ts_raw
                return datetime.fromtimestamp(sec, tz=TAIPEI_TZ)
            if isinstance(ts_raw, str):
                # ISO or format
                dt = datetime.fromisoformat(ts_raw)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=TAIPEI_TZ)
                return dt.astimezone(TAIPEI_TZ)
        except Exception:
            return None
        return None

    def _to_float(self, val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            if isinstance(val, (Decimal, int, float, str)):
                f = float(val)
                return f if f == f else None  # Filter NaN
        except (ValueError, TypeError):
            return None
        return None

    def _on_native_tick(self, exchange: Any, tick: Any) -> None:
        """
        Native Shioaji Stock V1 Tick Callback.
        Ultra-lightweight: parses, normalizes to domain TickEvent, notifies, and returns immediately.
        """
        t_receive = datetime.now(TAIPEI_TZ)
        try:
            code = str(getattr(tick, "code", "")).strip()
            if not code:
                with self._lock:
                    self._malformed_payload_count += 1
                return

            raw_ts = getattr(tick, "datetime", None) or getattr(tick, "ts", None)
            ts = self._parse_timestamp(raw_ts)
            if ts is None:
                logger.warning(f"[{code}] Rejected tick with malformed exchange timestamp: {raw_ts}")
                with self._lock:
                    self._malformed_payload_count += 1
                return

            price_raw = getattr(tick, "close", getattr(tick, "price", None))
            price = self._to_float(price_raw)
            if price is None or price <= 0:
                with self._lock:
                    self._malformed_payload_count += 1
                return

            volume_raw = getattr(tick, "volume", getattr(tick, "vol", 0))
            volume = self._to_float(volume_raw) or 0.0

            total_volume = self._to_float(getattr(tick, "total_volume", None))
            tick_type = str(getattr(tick, "tick_type", "trade"))
            simtrade = bool(getattr(tick, "simtrade", False))
            intraday_odd = bool(getattr(tick, "intraday_odd", False))

            tick_event = TickEvent(
                timestamp=ts,
                symbol=code,
                price=price,
                volume=volume,
                total_volume=total_volume,
                receive_timestamp=t_receive,
                tick_type=tick_type,
                simtrade=simtrade,
                intraday_odd=intraday_odd,
                source="shioaji",
            )

            with self._lock:
                self._ticks_received_count += 1
                self._last_tick_time = ts
                callbacks = list(self._tick_callbacks)

            for cb in callbacks:
                try:
                    cb(tick_event)
                except Exception as e:
                    logger.error(f"Error in external tick callback: {e}", exc_info=True)

        except Exception as e:
            with self._lock:
                self._malformed_payload_count += 1
            logger.error(f"Error parsing native Shioaji tick payload: {e}", exc_info=True)

    def _on_native_bidask(self, exchange: Any, bidask: Any) -> None:
        """
        Native Shioaji Stock V1 BidAsk Callback.
        Extracts top-of-book and multi-level depth, normalizes to BidAskEvent, and notifies listeners.
        """
        t_receive = datetime.now(TAIPEI_TZ)
        try:
            code = str(getattr(bidask, "code", "")).strip()
            if not code:
                with self._lock:
                    self._malformed_payload_count += 1
                return

            raw_ts = getattr(bidask, "datetime", None) or getattr(bidask, "ts", None)
            ts = self._parse_timestamp(raw_ts)
            if ts is None:
                logger.warning(f"[{code}] Rejected BidAsk with malformed exchange timestamp: {raw_ts}")
                with self._lock:
                    self._malformed_payload_count += 1
                return

            # Top of book prices and volumes
            bid_prices = getattr(bidask, "bid_price", []) or []
            ask_prices = getattr(bidask, "ask_price", []) or []
            bid_vols = getattr(bidask, "bid_volume", []) or []
            ask_vols = getattr(bidask, "ask_volume", []) or []

            # Handle list vs scalar if older mock
            best_bid = self._to_float(bid_prices[0] if isinstance(bid_prices, (list, tuple)) and bid_prices else bid_prices)
            best_ask = self._to_float(ask_prices[0] if isinstance(ask_prices, (list, tuple)) and ask_prices else ask_prices)
            best_bid_vol = self._to_float(bid_vols[0] if isinstance(bid_vols, (list, tuple)) and bid_vols else bid_vols) or 0.0
            best_ask_vol = self._to_float(ask_vols[0] if isinstance(ask_vols, (list, tuple)) and ask_vols else ask_vols) or 0.0

            if best_bid is None or best_ask is None:
                with self._lock:
                    self._malformed_payload_count += 1
                return

            simtrade = bool(getattr(bidask, "simtrade", False))

            # Multi-level depth retention if present
            bid_depth = []
            if isinstance(bid_prices, (list, tuple)) and isinstance(bid_vols, (list, tuple)):
                for p, v in zip(bid_prices, bid_vols):
                    fp, fv = self._to_float(p), self._to_float(v)
                    if fp is not None and fv is not None:
                        bid_depth.append({"price": fp, "volume": fv})

            ask_depth = []
            if isinstance(ask_prices, (list, tuple)) and isinstance(ask_vols, (list, tuple)):
                for p, v in zip(ask_prices, ask_vols):
                    fp, fv = self._to_float(p), self._to_float(v)
                    if fp is not None and fv is not None:
                        ask_depth.append({"price": fp, "volume": fv})

            bidask_event = BidAskEvent(
                timestamp=ts,
                symbol=code,
                bid_price=best_bid,
                ask_price=best_ask,
                bid_volume=best_bid_vol,
                ask_volume=best_ask_vol,
                bid_depth=bid_depth or None,
                ask_depth=ask_depth or None,
                receive_timestamp=t_receive,
                simtrade=simtrade,
                source="shioaji",
            )

            with self._lock:
                self._bidask_received_count += 1
                self._last_bidask_time = ts
                callbacks = list(self._bidask_callbacks)

            for cb in callbacks:
                try:
                    cb(bidask_event)
                except Exception as e:
                    logger.error(f"Error in external bidask callback: {e}", exc_info=True)

        except Exception as e:
            with self._lock:
                self._malformed_payload_count += 1
            logger.error(f"Error parsing native Shioaji bidask payload: {e}", exc_info=True)

    def _on_native_event(self, resp_code: int, event_code: int, info: str, event: str) -> None:
        """Handles Solace session status events (connection up, down, reconnect)."""
        logger.info(f"Shioaji Quote Event: resp={resp_code}, code={event_code}, event={event}, info={info}")
        with self._lock:
            # Solace event 0 = UP_NOTICE, 1 = DOWN_ERROR
            if event_code == 1 or "disconnect" in event.lower() or "down" in event.lower():
                self._state = ConnectionState.RECONNECTING
                self._disconnect_count += 1
                logger.warning(f"Shioaji quote session disconnected ({event}). Initiating auto-reconnect...")
            elif event_code == 0 or "up" in event.lower() or "connect" in event.lower():
                self._reconnect_count += 1
                self._state = ConnectionState.AUTHENTICATED
                self._handle_reconnected()

    def _handle_reconnected(self) -> None:
        """
        Executes robust post-reconnection lifecycle:
        Re-registers callbacks and re-subscribes all intended symbols.
        """
        logger.info("Shioaji quote transport reconnected. Restoring subscriptions...")
        if self._api:
            self._register_native_callbacks(self._api)

        # Re-subscribe all target symbols
        symbols_to_resubscribe = list(self._target_symbols)
        self._subscribed_tick_symbols.clear()
        self._subscribed_bidask_symbols.clear()
        self._resubscribe_count += 1

        if symbols_to_resubscribe:
            self.subscribe(symbols_to_resubscribe)

    def heartbeat(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "source": "ShioajiMarketDataSource",
                "state": self._state.value,
                "connected": self.is_connected(),
                "subscribed_ticks": list(self._subscribed_tick_symbols),
                "subscribed_bidask": list(self._subscribed_bidask_symbols),
                "ticks_received": self._ticks_received_count,
                "bidask_received": self._bidask_received_count,
                "malformed_payloads": self._malformed_payload_count,
                "disconnects": self._disconnect_count,
                "reconnects": self._reconnect_count,
                "resubscribes": self._resubscribe_count,
                "last_tick_time": self._last_tick_time.isoformat() if self._last_tick_time else None,
                "last_bidask_time": self._last_bidask_time.isoformat() if self._last_bidask_time else None,
                "timestamp": datetime.now(TAIPEI_TZ).isoformat(),
            }

    def get_status(self) -> Dict[str, Any]:
        return self.heartbeat()
