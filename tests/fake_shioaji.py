"""
Fake Shioaji API & Native Payload Generators for Offline Testing
Faithfully mimics Shioaji SDK's quote subsystem, Contracts repository,
and native TickSTKv1 / BidAskSTKv1 payloads without external network calls.
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

TAIPEI_TZ = ZoneInfo("Asia/Taipei")


class FakeQuoteType(str, Enum):
    Tick = "tick"
    BidAsk = "bidask"


class FakeExchange(str, Enum):
    TSE = "TSE"
    OTC = "OTC"


class FakeTickSTKv1:
    """Matches official SinoPac TickSTKv1 object."""
    def __init__(
        self,
        code: str,
        datetime_val: datetime,
        close: Decimal,
        volume: int = 1,
        total_volume: int = 100,
        tick_type: int = 1,
        simtrade: bool = False,
        intraday_odd: bool = False,
    ):
        self.code = code
        self.datetime = datetime_val
        self.close = close
        self.volume = volume
        self.total_volume = total_volume
        self.tick_type = tick_type
        self.simtrade = simtrade
        self.intraday_odd = intraday_odd


class FakeBidAskSTKv1:
    """Matches official SinoPac BidAskSTKv1 object with 5-level depth lists."""
    def __init__(
        self,
        code: str,
        datetime_val: datetime,
        bid_price: List[Decimal],
        bid_volume: List[int],
        ask_price: List[Decimal],
        ask_volume: List[int],
        simtrade: bool = False,
    ):
        self.code = code
        self.datetime = datetime_val
        self.bid_price = bid_price
        self.bid_volume = bid_volume
        self.ask_price = ask_price
        self.ask_volume = ask_volume
        self.simtrade = simtrade


class FakeContract:
    def __init__(self, code: str, symbol: Optional[str] = None, name: str = "FakeStock"):
        self.code = code
        self.symbol = symbol or code
        self.name = name
        self.exchange = FakeExchange.TSE


class FakeQuote:
    """Mock for `api.quote` providing official callback and subscription interfaces."""
    def __init__(self):
        self.on_tick_callback: Optional[Callable] = None
        self.on_bidask_callback: Optional[Callable] = None
        self.on_event_callback: Optional[Callable] = None

        self.subscriptions: Set[tuple] = set()  # {(contract.code, quote_type)}
        self.unsubscriptions: Set[tuple] = set()

    def set_on_tick_stk_v1_callback(self, cb: Callable):
        self.on_tick_callback = cb

    def set_on_bidask_stk_v1_callback(self, cb: Callable):
        self.on_bidask_callback = cb

    def set_event_callback(self, cb: Callable):
        self.on_event_callback = cb

    def subscribe(self, contract: FakeContract, quote_type: Any):
        q_type_str = str(getattr(quote_type, "value", quote_type)).lower()
        self.subscriptions.add((contract.code, q_type_str))

    def unsubscribe(self, contract: FakeContract, quote_type: Any):
        q_type_str = str(getattr(quote_type, "value", quote_type)).lower()
        self.subscriptions.discard((contract.code, q_type_str))
        self.unsubscriptions.add((contract.code, q_type_str))

    # Helper simulation methods
    def emit_tick(
        self,
        code: str,
        price: float,
        volume: int = 10,
        total_volume: int = 500,
        timestamp: Optional[datetime] = None,
        tick_type: int = 1,
        simtrade: bool = False,
        intraday_odd: bool = False,
    ):
        if not self.on_tick_callback:
            raise RuntimeError("Cannot emit tick: set_on_tick_stk_v1_callback was never called!")
        ts = timestamp or datetime.now(TAIPEI_TZ)
        tick_obj = FakeTickSTKv1(
            code=code,
            datetime_val=ts,
            close=Decimal(str(price)),
            volume=volume,
            total_volume=total_volume,
            tick_type=tick_type,
            simtrade=simtrade,
            intraday_odd=intraday_odd,
        )
        self.on_tick_callback(FakeExchange.TSE, tick_obj)

    def emit_bidask(
        self,
        code: str,
        bid_price: float,
        ask_price: float,
        bid_volume: int = 50,
        ask_volume: int = 50,
        timestamp: Optional[datetime] = None,
        simtrade: bool = False,
    ):
        if not self.on_bidask_callback:
            raise RuntimeError("Cannot emit bidask: set_on_bidask_stk_v1_callback was never called!")
        ts = timestamp or datetime.now(TAIPEI_TZ)
        bidask_obj = FakeBidAskSTKv1(
            code=code,
            datetime_val=ts,
            bid_price=[Decimal(str(bid_price)), Decimal(str(bid_price - 1.0))],
            bid_volume=[bid_volume, bid_volume * 2],
            ask_price=[Decimal(str(ask_price)), Decimal(str(ask_price + 1.0))],
            ask_volume=[ask_volume, ask_volume * 2],
            simtrade=simtrade,
        )
        self.on_bidask_callback(FakeExchange.TSE, bidask_obj)

    def emit_event(self, resp_code: int, event_code: int, info: str, event: str):
        if self.on_event_callback:
            self.on_event_callback(resp_code, event_code, info, event)


class FakeContracts:
    def __init__(self):
        self.Stocks = {
            "2330": FakeContract("2330", name="TSMC"),
            "2454": FakeContract("2454", name="MediaTek"),
            "2890": FakeContract("2890", name="SinoPac"),
        }


class FakeShioajiAPI:
    """High-fidelity mock of Shioaji API instance."""
    def __init__(self, simulation: bool = True):
        self.simulation = simulation
        self.quote = FakeQuote()
        self.Contracts = FakeContracts()
        self._logged_in = False
        self.place_order_call_count = 0

    def login(self, api_key: str, secret_key: str, contracts_cb: Optional[Callable] = None) -> List[Any]:
        if not api_key or not secret_key:
            raise ValueError("Invalid credentials")
        self._logged_in = True
        return [self]

    def logout(self) -> None:
        self._logged_in = False

    def place_order(self, contract: Any, order: Any) -> Any:
        self.place_order_call_count += 1
        raise RuntimeError("FATAL: REAL ORDER PATH TOUCHED! Shioaji.place_order must be unreachable in shadow mode.")
