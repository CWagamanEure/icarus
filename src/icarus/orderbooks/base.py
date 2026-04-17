from __future__ import annotations

from decimal import Decimal

from icarus.observations import OrderBookLevel, OrderBookObservation


class OrderBook:
    """Canonical in-memory order book using absolute per-level quantities."""

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}

    def load_snapshot(self, levels: tuple[OrderBookLevel, ...]) -> None:
        self._bids.clear()
        self._asks.clear()
        for level in levels:
            self.apply_level(level.side, level.price, level.size)

    def apply_delta(self, levels: tuple[OrderBookLevel, ...]) -> None:
        for level in levels:
            self.apply_level(level.side, level.price, level.size)

    def truncate(self, depth: int) -> None:
        if depth <= 0:
            self._bids.clear()
            self._asks.clear()
            return
        self._bids = dict(sorted(self._bids.items(), reverse=True)[:depth])
        self._asks = dict(sorted(self._asks.items())[:depth])

    def apply_level(self, side: str, price: Decimal, size: Decimal) -> None:
        book_side = self._bids if side == "buy" else self._asks
        if size <= 0:
            book_side.pop(price, None)
            return
        book_side[price] = size

    def best_bid_level(self) -> OrderBookLevel | None:
        if not self._bids:
            return None
        best_price = max(self._bids)
        return OrderBookLevel(side="buy", price=best_price, size=self._bids[best_price])

    def best_ask_level(self) -> OrderBookLevel | None:
        if not self._asks:
            return None
        best_price = min(self._asks)
        return OrderBookLevel(side="sell", price=best_price, size=self._asks[best_price])

    def levels(self) -> tuple[OrderBookLevel, ...]:
        bid_levels = tuple(
            OrderBookLevel(side="buy", price=price, size=size)
            for price, size in sorted(self._bids.items(), reverse=True)
        )
        ask_levels = tuple(
            OrderBookLevel(side="sell", price=price, size=size)
            for price, size in sorted(self._asks.items())
        )
        return bid_levels + ask_levels

    def to_observation(
        self,
        *,
        exchange: str,
        market: str,
        source_timestamp_ms: int | None,
        received_timestamp_ms: int | None,
        raw_message: dict[str, object],
    ) -> OrderBookObservation:
        return OrderBookObservation(
            exchange=exchange,
            market=market,
            source_timestamp_ms=source_timestamp_ms,
            received_timestamp_ms=received_timestamp_ms,
            raw_message=dict(raw_message),
            update_type="snapshot",
            levels=self.levels(),
        )
