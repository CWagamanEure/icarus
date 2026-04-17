from __future__ import annotations

from icarus.observations import Observation, OrderBookObservation
from icarus.orderbooks.base import OrderBook


class OkxOrderBookBuilder:
    """Reconstruct full OKX order book state from snapshot observations.

    OKX books5 always delivers complete top-of-book snapshots, so no delta
    merging is required. Each incoming OrderBookObservation seeds the local
    book and is re-emitted as a canonical full-book observation.
    """

    def __init__(self) -> None:
        self._book = OrderBook()
        self._is_initialized = False

    def on_observation(self, observation: Observation) -> OrderBookObservation | None:
        if observation.exchange != "okx":
            raise ValueError("OkxOrderBookBuilder only supports okx observations.")

        if isinstance(observation, OrderBookObservation):
            self._book.load_snapshot(observation.levels)
            self._is_initialized = True
            return self._book.to_observation(
                exchange=observation.exchange,
                market=observation.market,
                source_timestamp_ms=observation.source_timestamp_ms,
                received_timestamp_ms=observation.received_timestamp_ms,
                raw_message=observation.raw_message,
            )

        return None

    def reset(self) -> None:
        self._book = OrderBook()
        self._is_initialized = False
