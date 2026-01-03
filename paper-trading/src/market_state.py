from dataclasses import dataclass, field
from typing import Dict, Union

from .market import Market
from .orderbook_state import OrderBookState
from .events.event import Event
from .events.market_events import BookSnapshot, LastTrade, PriceChangeBatch, BestBidAsk, TickSizeChange, UnknownMarketEvent


@dataclass
class MarketState:
    market: Market 
    books: Dict[str, OrderBookState]  = field(default_factory=dict)

    def __post_init__(self):
        for aid in self.market.get_token_ids():
            self.books.setdefault(aid, OrderBookState())

    def book(self, asset_id: str):
        return self.books.setdefault(asset_id, OrderBookState())

    def apply(self, e: Event):
        if isinstance(e, BookSnapshot):
            self.book(e.asset_id).apply_book_snapshot(e)
        elif isinstance(e, LastTrade):
            self.book(e.asset_id).apply_last_trade(e)
        elif isinstance(e, PriceChangeBatch):
            for u in e.changes:
                self.book(u.asset_id).apply_level_update(u, e.ts_ms)
        elif isinstance(e, BestBidAsk):
            self.book(e.asset_id).apply_best_bid_ask(e)
        elif isinstance(e, TickSizeChange):
                self.book(e.asset_id).apply_tick_size_change(e)
        elif isinstance(e, UnknownMarketEvent):
            # ignore others for now
            return


        else:
            raise TypeError(f"Unknown event {type(e)}")

    def get_market(self):
        return self.market
    
    def get_book(self):
        for key, book in self.books.items():
            return book








