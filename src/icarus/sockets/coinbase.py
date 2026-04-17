from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from icarus.observations import CoinbaseObservationNormalizer, Observation
from icarus.orderbooks import CoinbaseOrderBookBuilder
from icarus.sockets.base import BaseSocket


class CoinbaseSocket(BaseSocket):
    """Stream raw market updates from Coinbase."""

    MAINNET_WS_URL = "wss://advanced-trade-ws.coinbase.com"
    SANDBOX_WS_URL = "wss://advanced-trade-ws-public.sandbox.exchange.coinbase.com"
    DEFAULT_MAX_MESSAGE_SIZE = 8 * 1024 * 1024

    def __init__(
        self,
        product_ids: str | list[str],
        *,
        channels: list[str] | None = None,
        sandbox: bool = False,
    ) -> None:
        super().__init__(
            self.SANDBOX_WS_URL if sandbox else self.MAINNET_WS_URL,
            max_message_size=self.DEFAULT_MAX_MESSAGE_SIZE,
        )
        if isinstance(product_ids, str):
            product_ids = [product_ids]

        self.product_ids = [product_id.upper() for product_id in product_ids]
        self.channels = channels or ["ticker", "heartbeats", "level2"]
        self.observation_normalizer = CoinbaseObservationNormalizer()
        self._orderbook_builders: dict[str, CoinbaseOrderBookBuilder] = {}

    async def after_connect(self) -> None:
        # A reconnect invalidates any locally reconstructed book state until a
        # fresh snapshot arrives on the new connection.
        self._orderbook_builders.clear()
        for payload in self.subscription_messages():
            await self.send_json(payload)

    def subscription_messages(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "subscribe",
                "product_ids": self.product_ids,
                "channel": channel,
            }
            for channel in self.channels
        ]

    def parse_message(self, raw_message: str) -> dict[str, Any]:
        message = super().parse_message(raw_message)

        events = message.get("events")
        if isinstance(events, list):
            message["events"] = [event for event in events if isinstance(event, dict)]

        return message

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

    def _pipeline_observation(self, observation: Observation) -> Observation | None:
        from icarus.observations import OrderBookDeltaObservation, OrderBookObservation

        if not isinstance(observation, OrderBookObservation | OrderBookDeltaObservation):
            return observation

        builder = self._orderbook_builders.get(observation.market)
        if builder is None:
            builder = CoinbaseOrderBookBuilder()
            self._orderbook_builders[observation.market] = builder
        return builder.on_observation(observation)

    async def stream_observations(self) -> AsyncIterator[Observation]:
        async for raw_message in self.stream_messages():
            for observation in self.convert_message_to_observations(
                raw_message,
                received_timestamp_ms=time.time_ns() // 1_000_000,
            ):
                pipeline_observation = self._pipeline_observation(observation)
                if pipeline_observation is not None:
                    yield pipeline_observation
