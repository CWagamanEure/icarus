from __future__ import annotations
# ruff: noqa: I001

from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True, slots=True)
class FairValueFeatures:
    midprice: Decimal | None
    microprice: Decimal | None
    spread_bps: Decimal | None
    depth_imbalance: Decimal | None
    quote_age_ms: int | None
    mid_volatility_bps: Decimal | None
    micro_volatility_bps: Decimal | None
    top_bid_depth: Decimal | None
    top_ask_depth: Decimal | None


@dataclass(frozen=True, slots=True)
class RawFairValueEstimate:
    timestamp_ms: int
    exchange: str
    market: str
    raw_fair_value: Decimal | None
    measurement_variance: Decimal | None
    micro_alpha: Decimal | None
    used_midprice: Decimal | None
    used_microprice: Decimal | None


@dataclass(frozen=True, slots=True)
class VenueFairValueState:
    exchange: str
    market: str
    timestamp_ms: int
    fair_value: Decimal
    variance: Decimal


@dataclass(frozen=True, slots=True)
class CombinedFairValueEstimate:
    market: str
    timestamp_ms: int
    fair_value: Decimal
    variance: Decimal
    contributing_exchanges: tuple[str, ...]

    
