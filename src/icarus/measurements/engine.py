from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from math import sqrt

from icarus.measurements.types import MarketMeasurement
from icarus.observations import (
    BBOObservation,
    Observation,
    OrderBookDeltaObservation,
    OrderBookObservation,
)


@dataclass(frozen=True, slots=True)
class MeasurementEngineConfig:
    volatility_window_ms: int = 60_000


@dataclass(slots=True)
class _MeasurementState:
    last_quote_observation: BBOObservation | OrderBookObservation | None = None
    last_quote_timestamp_ms: int | None = None
    recent_midpoints: deque[tuple[int, Decimal]] = field(default_factory=deque)
    recent_microprices: deque[tuple[int, Decimal]] = field(default_factory=deque)


class MarketMeasurementEngine:
    def __init__(
        self,
        *,
        exchange: str,
        market: str,
        config: MeasurementEngineConfig | None = None,
    ) -> None:
        self.exchange = exchange
        self.market = market
        self.config = config or MeasurementEngineConfig()
        self._state = _MeasurementState()

    def on_observation(self, observation: Observation) -> MarketMeasurement | None:
        if observation.exchange != self.exchange or observation.market != self.market:
            raise ValueError(
                f"Observation {observation.exchange}/{observation.market} does not match "
                f"engine {self.exchange}/{self.market}."
            )

        timestamp_ms = (
            observation.received_timestamp_ms
            if observation.received_timestamp_ms is not None
            else observation.source_timestamp_ms
        )
        if timestamp_ms is None:
            return None

        if isinstance(observation, BBOObservation | OrderBookObservation):
            self._state.last_quote_observation = observation
            self._state.last_quote_timestamp_ms = timestamp_ms
            self._record_prices(observation, timestamp_ms)

        if self._state.last_quote_observation is None:
            return None

        if isinstance(observation, OrderBookDeltaObservation):
            return None

        return self.current_measurement(timestamp_ms)

    def current_measurement(self, timestamp_ms: int) -> MarketMeasurement | None:
        if self._state.last_quote_observation is None:
            return None

        self._prune_old_points(self._state.recent_midpoints, timestamp_ms)
        self._prune_old_points(self._state.recent_microprices, timestamp_ms)

        current_quote = self._state.last_quote_observation
        midprice = current_quote.midprice
        microprice = current_quote.microprice
        spread_bps = current_quote.spread_bps
        bid_price, ask_price = self._extract_top_prices(current_quote)
        top_bid_depth, top_ask_depth = self._extract_top_depths(current_quote)
        depth_imbalance = self._compute_depth_imbalance(top_bid_depth, top_ask_depth)
        mid_volatility_bps = self._compute_rolling_volatility_bps(self._state.recent_midpoints)
        micro_volatility_bps = self._compute_rolling_volatility_bps(self._state.recent_microprices)
        quote_age_ms = (
            timestamp_ms - self._state.last_quote_timestamp_ms
            if self._state.last_quote_timestamp_ms is not None
            else None
        )

        return MarketMeasurement(
            exchange=self.exchange,
            market=self.market,
            timestamp_ms=timestamp_ms,
            midprice=midprice,
            microprice=microprice,
            spread_bps=spread_bps,
            top_bid_depth=top_bid_depth,
            top_ask_depth=top_ask_depth,
            depth_imbalance=depth_imbalance,
            quote_age_ms=quote_age_ms,
            mid_volatility_bps=mid_volatility_bps,
            micro_volatility_bps=micro_volatility_bps,
            bid_price=bid_price,
            ask_price=ask_price,
        )

    def _record_prices(
        self,
        observation: BBOObservation | OrderBookObservation,
        timestamp_ms: int,
    ) -> None:
        if observation.midprice is not None:
            self._state.recent_midpoints.append((timestamp_ms, observation.midprice))
        if observation.microprice is not None:
            self._state.recent_microprices.append((timestamp_ms, observation.microprice))
        self._prune_old_points(self._state.recent_midpoints, timestamp_ms)
        self._prune_old_points(self._state.recent_microprices, timestamp_ms)

    def _prune_old_points(self, values: deque[tuple[int, Decimal]], timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.config.volatility_window_ms
        while values and values[0][0] < cutoff:
            values.popleft()

    def _extract_top_depths(
        self,
        observation: BBOObservation | OrderBookObservation,
    ) -> tuple[Decimal | None, Decimal | None]:
        if isinstance(observation, BBOObservation):
            return observation.bid_size, observation.ask_size
        best_bid = observation.best_bid_level
        best_ask = observation.best_ask_level
        return (
            best_bid.size if best_bid is not None else None,
            best_ask.size if best_ask is not None else None,
        )

    def _extract_top_prices(
        self,
        observation: BBOObservation | OrderBookObservation,
    ) -> tuple[Decimal | None, Decimal | None]:
        if isinstance(observation, BBOObservation):
            return observation.bid_price, observation.ask_price
        best_bid = observation.best_bid_level
        best_ask = observation.best_ask_level
        return (
            best_bid.price if best_bid is not None else None,
            best_ask.price if best_ask is not None else None,
        )

    def _compute_depth_imbalance(
        self,
        top_bid_depth: Decimal | None,
        top_ask_depth: Decimal | None,
    ) -> Decimal | None:
        if top_bid_depth is None or top_ask_depth is None:
            return None
        total_depth = top_bid_depth + top_ask_depth
        if total_depth == 0:
            return None
        return (top_bid_depth - top_ask_depth) / total_depth

    def _compute_rolling_volatility_bps(
        self,
        values: Iterable[tuple[int, Decimal]],
    ) -> Decimal | None:
        price_values = [value for _, value in values]
        if len(price_values) < 2:
            return None

        returns_bps: list[float] = []
        for previous_price, current_price in zip(price_values, price_values[1:], strict=False):
            if previous_price == 0:
                continue
            returns_bps.append(
                float(((current_price - previous_price) / previous_price) * Decimal("10000"))
            )

        if not returns_bps:
            return None
        if len(returns_bps) == 1:
            return Decimal(str(abs(returns_bps[0])))

        mean_return = sum(returns_bps) / len(returns_bps)
        variance = sum((value - mean_return) ** 2 for value in returns_bps) / len(returns_bps)
        return Decimal(str(sqrt(variance)))
