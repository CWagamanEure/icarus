from __future__ import annotations

from typing import Any

from icarus.observations import Observation, OrderBookDeltaObservation, OrderBookObservation
from icarus.orderbooks.base import OrderBook


class CoinbaseOrderBookBuilder:
    """Reconstruct full Coinbase order book state from snapshot plus delta observations."""

    def __init__(self) -> None:
        self._book = OrderBook()
        self._is_initialized = False
        self._last_sequence_num: int | None = None

    def on_observation(self, observation: Observation) -> OrderBookObservation | None:
        if observation.exchange != "coinbase":
            raise ValueError("CoinbaseOrderBookBuilder only supports coinbase observations.")

        if isinstance(observation, OrderBookObservation):
            self._book.load_snapshot(observation.levels)
            self._is_initialized = True
            self._last_sequence_num = self._extract_sequence_num(observation.raw_message)
            return self._book.to_observation(
                exchange=observation.exchange,
                market=observation.market,
                source_timestamp_ms=observation.source_timestamp_ms,
                received_timestamp_ms=observation.received_timestamp_ms,
                raw_message=observation.raw_message,
            )

        if isinstance(observation, OrderBookDeltaObservation):
            if not self._is_initialized:
                # Deltas are unusable until a snapshot seeds local state.
                return None
            sequence_num = self._extract_sequence_num(observation.raw_message)
            if sequence_num is not None and self._last_sequence_num is None:
                # Without sequence metadata on the seeding snapshot, delta continuity
                # cannot be validated safely. Wait for a fresh snapshot with sequence.
                self.reset()
                return None
            if (
                sequence_num is not None
                and self._last_sequence_num is not None
                and sequence_num != self._last_sequence_num + 1
            ):
                # Missing, stale, or out-of-order deltas can corrupt local state. Drop
                # state and wait for the next full snapshot to safely resynchronize.
                self.reset()
                return None
            self._book.apply_delta(observation.levels)
            if sequence_num is not None:
                self._last_sequence_num = sequence_num
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
        self._last_sequence_num = None

    @staticmethod
    def _extract_sequence_num(raw_message: dict[str, Any]) -> int | None:
        sequence_num = raw_message.get("sequence_num")
        return sequence_num if isinstance(sequence_num, int) else None
