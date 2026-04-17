from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from icarus.observations import KrakenObservationNormalizer, Observation
from icarus.orderbooks import KrakenOrderBookBuilder
from icarus.sockets.base import BaseSocket


class KrakenSocket(BaseSocket):
    """Stream raw market updates from Kraken spot WebSocket v2."""

    MAINNET_WS_URL = "wss://ws.kraken.com/v2"

    def __init__(
        self,
        symbols: str | list[str],
        *,
        ticker_event_trigger: str = "bbo",
        include_ticker: bool = True,
        include_book: bool = True,
        include_trades: bool = True,
        book_depth: int = 10,
    ) -> None:
        super().__init__(self.MAINNET_WS_URL)
        if isinstance(symbols, str):
            symbols = [symbols]

        self.symbols = [symbol.upper() for symbol in symbols]
        self.ticker_event_trigger = ticker_event_trigger
        self.include_ticker = include_ticker
        self.include_book = include_book
        self.include_trades = include_trades
        self.book_depth = book_depth
        self.observation_normalizer = KrakenObservationNormalizer()
        self._orderbook_builders: dict[str, KrakenOrderBookBuilder] = {}

    async def after_connect(self) -> None:
        self._orderbook_builders.clear()
        for payload in self.subscription_messages():
            await self.send_json(payload)

    def subscription_messages(self) -> list[dict[str, Any]]:
        subscriptions: list[dict[str, Any]] = []
        if self.include_ticker:
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "channel": "ticker",
                        "symbol": self.symbols,
                        "event_trigger": self.ticker_event_trigger,
                        "snapshot": True,
                    },
                }
            )
        if self.include_book:
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "channel": "book",
                        "symbol": self.symbols,
                        "depth": self.book_depth,
                        "snapshot": True,
                    },
                }
            )
        if self.include_trades:
            subscriptions.append(
                {
                    "method": "subscribe",
                    "params": {
                        "channel": "trade",
                        "symbol": self.symbols,
                        "snapshot": False,
                    },
                }
            )
        return subscriptions

    def convert_message_to_observations(
        self,
        raw_message: dict[str, Any],
        *,
        received_timestamp_ms: int | None = None,
    ) -> list[Observation]:
        return self.observation_normalizer.normalize_message(
            raw_message,
            received_timestamp_ms=received_timestamp_ms,
        )

    def parse_message(self, raw_message: str) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(raw_message, parse_float=str))

    def _pipeline_observation(self, observation: Observation) -> Observation | None:
        from icarus.observations import OrderBookDeltaObservation, OrderBookObservation

        if not isinstance(observation, OrderBookObservation | OrderBookDeltaObservation):
            return observation

        builder = self._orderbook_builders.get(observation.market)
        if builder is None:
            builder = KrakenOrderBookBuilder(depth=self.book_depth)
            self._orderbook_builders[observation.market] = builder
        pipeline_observation = builder.on_observation(observation)
        if pipeline_observation is None and builder.consume_resync_request():
            raise ConnectionError(
                f"Kraken order book checksum mismatch for {observation.market}; reconnecting."
            )
        return pipeline_observation

    async def stream_observations(self) -> AsyncIterator[Observation]:
        async for raw_message in self.stream_messages():
            for observation in self.convert_message_to_observations(
                raw_message,
                received_timestamp_ms=time.time_ns() // 1_000_000,
            ):
                pipeline_observation = self._pipeline_observation(observation)
                if pipeline_observation is not None:
                    yield pipeline_observation
