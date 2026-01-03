from dataclasses import dataclass
from typing import Union, Literal, Optional, List, Tuple, Sequence

from .event import Event
from ..order import Side


@dataclass(frozen=True, slots=True)
class BookSnapshot(Event):
    market: str
    asset_id: str
    ts_ms: int
    book_hash: str
    bids: Sequence[tuple[str, str]]
    asks : Sequence[tuple[str, str]]
    last_trade_price: Optional[str] = None


@dataclass(frozen=True, slots=True)
class LastTrade(Event):
    market: str
    asset_id: str
    ts_ms: int
    price: str
    size: str
    side: Side

@dataclass(frozen=True, slots=True)
class PriceLevelUpdate:
    asset_id: str
    price: str
    size: str
    side: Side
    book_hash: str
    best_bid: Optional[str] = None
    best_ask : Optional[str] = None


@dataclass(frozen=True, slots=True)
class PriceChangeBatch(Event):
    market: str
    ts_ms: int
    changes: Sequence[PriceLevelUpdate]

@dataclass(frozen=True, slots=True)
class BestBidAsk(Event):
    market: str
    asset_id: str
    ts_ms: int
    best_bid : str
    best_ask: str
    spread: Optional[str] = None

@dataclass(frozen=True, slots=True)
class TickSizeChange(Event):
    market: str
    asset_id : str
    ts_ms: int
    old_tick_size: str
    new_tick_size: str

@dataclass(frozen=True, slots=True)
class UnknownMarketEvent:
    market: Optional[str]
    event_type: Optional[str]
    raw: dict

MarketEvent = Union[
    BookSnapshot, LastTrade, PriceChangeBatch, BestBidAsk, TickSizeChange, UnknownMarketEvent
]
    


