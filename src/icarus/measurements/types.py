from __future__ import annotations

import abc
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True, repr=False)
class Measurement(abc.ABC):
    exchange: str
    market: str
    timestamp_ms: int

    @abc.abstractmethod
    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        """Return subclass-specific fields for compact display output."""

    def __repr__(self) -> str:
        rendered_fields = ", ".join(
            f"{field_name}={field_value!r}" for field_name, field_value in self.display_fields()
        )
        return f"{self.__class__.__name__}({rendered_fields})"


@dataclass(frozen=True, slots=True, repr=False)
class MarketMeasurement(Measurement):
    midprice: Decimal | None
    microprice: Decimal | None
    spread_bps: Decimal | None
    top_bid_depth: Decimal | None
    top_ask_depth: Decimal | None
    depth_imbalance: Decimal | None
    quote_age_ms: int | None
    mid_volatility_bps: Decimal | None
    micro_volatility_bps: Decimal | None

    def display_fields(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("exchange", self.exchange),
            ("market", self.market),
            ("timestamp_ms", self.timestamp_ms),
            ("midprice", self.midprice),
            ("microprice", self.microprice),
            ("spread_bps", self.spread_bps),
            ("top_depth", (self.top_bid_depth, self.top_ask_depth)),
            ("depth_imbalance", self.depth_imbalance),
            ("quote_age_ms", self.quote_age_ms),
            ("mid_volatility_bps", self.mid_volatility_bps),
            ("micro_volatility_bps", self.micro_volatility_bps),
        )
