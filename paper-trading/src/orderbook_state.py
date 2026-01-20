import heapq
from dataclasses import dataclass, field
from typing import Dict, Optional
from decimal import Decimal

from .events.market_events import BookSnapshot, PriceLevelUpdate, LastTrade, BestBidAsk, TickSizeChange
from .outcome import Outcome 

@dataclass
class OrderBookState:
    tick_size: Decimal = Decimal("0.001")
    bids: Dict[int, Decimal] = field(default_factory=dict) 
    asks:  Dict[int, Decimal] = field(default_factory=dict) 
    best_bid: Optional[int] = None
    best_ask : Optional[int] = None
    ts_ms:  Optional[int] = None
    book_hash: Optional[str] = None
    last_trade_price: Optional[Decimal] = None

    ## Debugging vars
    update_seq: int = 0
    last_event_type: Optional[str] = None

    ## debugging method
    def _mark_updated(self, event_type: str, ts_ms: Optional[int] = None):
        self.update_seq +=1
        self.last_event_type = event_type
        if ts_ms is not None:
            self.ts_ms = ts_ms


    def _to_tick(self, px: str):
        q = Decimal(px) / self.tick_size
        tick = int(q.to_integral_value())
        if q != Decimal(tick):
            raise ValueError(f"Price {px} not aligned to tick_size {self.tick_size}")
        return tick

    def _tick_to_price(self, tick: int) -> Decimal:
        return Decimal(tick) * self.tick_size


    def apply_book_snapshot(self, e: BookSnapshot):
        self.bids.clear()
        self.asks.clear()


        for p,s in e.bids:
            size = Decimal(s)
            if size > 0:
                self.bids[self._to_tick(p)] = size

        for p, s in e.asks:
            size = Decimal(s)
            if size > 0:
                self.asks[self._to_tick(p)] = size

        self.best_bid = max(self.bids) if self.bids else None
        self.best_ask = min(self.asks ) if self.asks else None
        self.ts_ms = e.ts_ms
        self.book_hash = e.book_hash
        if e.last_trade_price is not None:
            self.last_trade_price = Decimal(e.last_trade_price)
        # debug
        self._mark_updated("Book")



    def apply_level_update(self, u: PriceLevelUpdate, ts_ms):
        tick = self._to_tick(u.price)
        size = Decimal(u.size)

        side_map = self.bids if u.side == "BUY" else self.asks
        if size == 0:
            side_map.pop(tick, None)
        else:
            side_map[tick] = size

        self.best_bid = max(self.bids) if self.bids else None
        self.best_ask = min(self.asks ) if self.asks else None
        # debug
        if u.book_hash is not None:
            self.book_hash = u.book_hash
        self._mark_updated("Level", ts_ms)



    def apply_last_trade(self, t: LastTrade):
        self.last_trade_price = Decimal(t.price)
        # debug
        self._mark_updated("Trade" )



    def apply_best_bid_ask(self, e: BestBidAsk):
        self.best_bid = self._to_tick(e.best_bid) if e.best_bid is not None else None 
        self.best_ask = self._to_tick(e.best_ask) if e.best_ask is not None else None 
        self.ts_ms = e.ts_ms
        
        # debug
        self._mark_updated("bid/ask" )

    def apply_tick_size_change(self, e: TickSizeChange):
        self.tick_size = Decimal(e.new_tick_size)
        self.bids.clear()
        self.asks.clear()
        self.best_bid = self.bes_ask = None
        self.book_hash = None
        self._mark_updated("TickSize", e.ts_ms)

    # Vis Stuff

    def summary(self):
        bb = self._tick_to_price(self.best_bid) if self.best_bid is not None else None
        ba = self._tick_to_price(self.best_ask) if self.best_ask is not None else None
        return (
            f"OrderBook(seq={self.update_seq}, last={self.last_event_type}, "
            f"levels(bid/ask)={len(self.bids)}/{len(self.asks)}, "
            f"best_bid={bb}, best_ask={ba}, ts_ms={self.ts_ms}, hash={self.book_hash})"
        )


    def depth(self, side, n: int = 5):
        book = self.bids if side == "BUY" else self.asks
        ticks = sorted(book.keys(), reverse=(side == "BUY"))[:n]
        return [(self._tick_to_price(t), book[t]) for t in ticks]

    def pretty(self, n: int = 5) :
        view = {
            "summary": self.summary(),
            "top_bids": self.depth("BUY", n),
            "top_asks": self.depth("SELL", n),
            "last_trade_price": str(self.last_trade_price) if self.last_trade_price else None,
        }
        return view

    #------ Some Microstructure Stuff --------------

    def depth_sum(self, side: str, n: int = 5) -> Decimal:

        book = self.bids if side=="BUY" else self.asks
        if not book or n<=0:
            return Decimal("0")

        if side == "BUY":
            ticks = heapq.nlargest(n, book.keys())
        else:
            ticks = heapq.nsmallest(n, book.keys())

        s = Decimal("0")
        for t in ticks:
            s += book[t]
        return s
        






