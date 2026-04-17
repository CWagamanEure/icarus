from __future__ import annotations

import zlib
from decimal import Decimal

from icarus.observations import Observation, OrderBookDeltaObservation, OrderBookObservation
from icarus.orderbooks.base import OrderBook


class KrakenOrderBookBuilder:
    """Reconstruct full Kraken order book state from snapshot plus delta observations."""

    def __init__(self, *, depth: int = 10) -> None:
        self._book = OrderBook()
        self._is_initialized = False
        self._depth = depth
        self._needs_resync = False

    def on_observation(self, observation: Observation) -> OrderBookObservation | None:
        if observation.exchange != "kraken":
            raise ValueError("KrakenOrderBookBuilder only supports kraken observations.")

        if isinstance(observation, OrderBookObservation):
            self._book.load_snapshot(observation.levels)
            self._book.truncate(self._depth)
            if not self._has_valid_checksum(observation):
                self.request_resync()
                return None
            self._is_initialized = True
            self._needs_resync = False
            return self._book.to_observation(
                exchange=observation.exchange,
                market=observation.market,
                source_timestamp_ms=observation.source_timestamp_ms,
                received_timestamp_ms=observation.received_timestamp_ms,
                raw_message=observation.raw_message,
            )

        if isinstance(observation, OrderBookDeltaObservation):
            if not self._is_initialized:
                # Kraken book updates are incremental patches. Wait for a snapshot
                # before applying them so top-of-book features remain semantically valid.
                return None
            self._book.apply_delta(observation.levels)
            self._book.truncate(self._depth)
            if not self._has_valid_checksum(observation):
                self.request_resync()
                return None
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
        self._needs_resync = False

    def request_resync(self) -> None:
        self._book = OrderBook()
        self._is_initialized = False
        self._needs_resync = True

    def consume_resync_request(self) -> bool:
        needs_resync = self._needs_resync
        self._needs_resync = False
        return needs_resync

    def _has_valid_checksum(self, observation: Observation) -> bool:
        expected = self._extract_checksum(observation.raw_message)
        if expected is None:
            return True
        return self._compute_checksum() == expected

    def _compute_checksum(self) -> int:
        ask_levels = [
            level for level in self._book.levels() if level.side == "sell"
        ][:10]
        bid_levels = [
            level for level in self._book.levels() if level.side == "buy"
        ][:10]
        checksum_input = "".join(
            self._format_checksum_part(level.price, level.size) for level in ask_levels
        )
        checksum_input += "".join(
            self._format_checksum_part(level.price, level.size) for level in bid_levels
        )
        return zlib.crc32(checksum_input.encode("utf-8")) & 0xFFFFFFFF

    @staticmethod
    def _format_checksum_part(price: Decimal, size: Decimal) -> str:
        return (
            KrakenOrderBookBuilder._normalize_checksum_value(price)
            + KrakenOrderBookBuilder._normalize_checksum_value(size)
        )

    @staticmethod
    def _normalize_checksum_value(value: Decimal) -> str:
        normalized = str(value).replace(".", "").lstrip("0")
        return normalized or "0"

    @staticmethod
    def _extract_checksum(raw_message: dict[str, object]) -> int | None:
        data = raw_message.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            return None
        checksum = data[0].get("checksum")
        return checksum if isinstance(checksum, int) else None
