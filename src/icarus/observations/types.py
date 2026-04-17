from __future__ import annotations

import abc
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

type Side = Literal["buy", "sell"]
type BookUpdateType = Literal["snapshot", "update"]


@dataclass(frozen=True, slots=True, repr=False)
class Observation(abc.ABC):
    exchange: str
    market: str
    source_timestamp_ms: int | None
    received_timestamp_ms: int | None
    raw_message: dict[str, Any]

    @abc.abstractmethod
    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        """Return subclass-specific fields for compact display output."""

    def __repr__(self) -> str:
        rendered_fields = ", ".join(
            f"{field_name}={field_value!r}" for field_name, field_value in self.display_fields()
        )
        return f"{self.__class__.__name__}({rendered_fields})"


@dataclass(frozen=True, slots=True, repr=False)
class BBOObservation(Observation):
    bid_price: Decimal
    bid_size: Decimal
    ask_price: Decimal
    ask_size: Decimal

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("bid", (self.bid_price, self.bid_size)),
            ("ask", (self.ask_price, self.ask_size)),
            ("source_timestamp_ms", self.source_timestamp_ms),
        )

    @property
    def midprice(self) -> Decimal:
        return (self.bid_price + self.ask_price) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> Decimal | None:
        if self.midprice == 0:
            return None
        return (self.spread / self.midprice) * Decimal("10000")

    @property
    def microprice(self) -> Decimal | None:
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return None
        return ((self.ask_price * self.bid_size) + (self.bid_price * self.ask_size)) / total_size


@dataclass(frozen=True, slots=True, repr=False)
class TradeObservation(Observation):
    trade_id: str | None
    side: Side
    price: Decimal
    size: Decimal

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("trade_id", self.trade_id),
            ("side", self.side),
            ("price", self.price),
            ("size", self.size),
            ("source_timestamp_ms", self.source_timestamp_ms),
        )


@dataclass(frozen=True, slots=True, repr=False)
class CandleObservation(Observation):
    interval: str | None
    open_timestamp_ms: int
    close_timestamp_ms: int
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    trade_count: int | None

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("interval", self.interval),
            ("open_timestamp_ms", self.open_timestamp_ms),
            ("close_timestamp_ms", self.close_timestamp_ms),
            ("ohlc", (self.open_price, self.high_price, self.low_price, self.close_price)),
            ("volume", self.volume),
            ("trade_count", self.trade_count),
        )


@dataclass(frozen=True, slots=True, repr=False)
class OrderBookLevel:
    side: Side
    price: Decimal
    size: Decimal

    def __repr__(self) -> str:
        return f"OrderBookLevel(side={self.side!r}, price={self.price!r}, size={self.size!r})"


@dataclass(frozen=True, slots=True, repr=False)
class OrderBookObservation(Observation):
    update_type: BookUpdateType
    levels: tuple[OrderBookLevel, ...]

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        preview_levels = self.levels[:4]
        remaining_levels = len(self.levels) - len(preview_levels)
        level_preview: list[Any] = [*preview_levels]
        if remaining_levels > 0:
            level_preview.append(f"...+{remaining_levels} more")

        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("update_type", self.update_type),
            ("levels", tuple(level_preview)),
            ("source_timestamp_ms", self.source_timestamp_ms),
        )

    @property
    def best_bid_level(self) -> OrderBookLevel | None:
        bid_levels = [level for level in self.levels if level.side == "buy"]
        return max(bid_levels, key=lambda level: level.price) if bid_levels else None

    @property
    def best_ask_level(self) -> OrderBookLevel | None:
        ask_levels = [level for level in self.levels if level.side == "sell"]
        return min(ask_levels, key=lambda level: level.price) if ask_levels else None

    @property
    def midprice(self) -> Decimal | None:
        best_bid = self.best_bid_level
        best_ask = self.best_ask_level
        if best_bid is None or best_ask is None:
            return None
        return (best_bid.price + best_ask.price) / Decimal("2")

    @property
    def spread(self) -> Decimal | None:
        best_bid = self.best_bid_level
        best_ask = self.best_ask_level
        if best_bid is None or best_ask is None:
            return None
        return best_ask.price - best_bid.price

    @property
    def spread_bps(self) -> Decimal | None:
        midprice = self.midprice
        spread = self.spread
        if midprice is None or spread is None or midprice == 0:
            return None
        return (spread / midprice) * Decimal("10000")

    @property
    def microprice(self) -> Decimal | None:
        best_bid = self.best_bid_level
        best_ask = self.best_ask_level
        if best_bid is None or best_ask is None:
            return None
        total_size = best_bid.size + best_ask.size
        if total_size == 0:
            return None
        return ((best_ask.price * best_bid.size) + (best_bid.price * best_ask.size)) / total_size


@dataclass(frozen=True, slots=True, repr=False)
class OrderBookDeltaObservation(Observation):
    """Incremental book patch; not a full current book state."""

    levels: tuple[OrderBookLevel, ...]

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        preview_levels = self.levels[:4]
        remaining_levels = len(self.levels) - len(preview_levels)
        level_preview: list[Any] = [*preview_levels]
        if remaining_levels > 0:
            level_preview.append(f"...+{remaining_levels} more")

        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("levels", tuple(level_preview)),
            ("source_timestamp_ms", self.source_timestamp_ms),
        )
