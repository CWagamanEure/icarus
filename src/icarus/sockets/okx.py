from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from icarus.observations import Observation, OkxObservationNormalizer, OrderBookObservation
from icarus.orderbooks import OkxOrderBookBuilder
from icarus.sockets.base import BaseSocket


class OkxSocket(BaseSocket):
    """Stream raw public market updates from OKX."""

    MAINNET_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(
        self,
        inst_ids: str | list[str],
        *,
        include_tickers: bool = True,
        include_books5: bool = True,
        include_trades: bool = True,
    ) -> None:
        super().__init__(self.MAINNET_WS_URL)
        if isinstance(inst_ids, str):
            inst_ids = [inst_ids]

        self.inst_ids = [inst_id.upper() for inst_id in inst_ids]
        self.include_tickers = include_tickers
        self.include_books5 = include_books5
        self.include_trades = include_trades
        self.observation_normalizer = OkxObservationNormalizer()
        self._orderbook_builders: dict[str, OkxOrderBookBuilder] = {}

    async def after_connect(self) -> None:
        # A reconnect invalidates any locally reconstructed book state until a
        # fresh snapshot arrives on the new connection.
        self._orderbook_builders.clear()
        for payload in self.subscription_messages():
            await self.send_json(payload)

    def subscription_messages(self) -> list[dict[str, Any]]:
        args: list[dict[str, str]] = []
        for inst_id in self.inst_ids:
            if self.include_tickers:
                args.append({"channel": "tickers", "instId": inst_id})
            if self.include_books5:
                args.append({"channel": "books5", "instId": inst_id})
            if self.include_trades:
                args.append({"channel": "trades", "instId": inst_id})

        return [{"op": "subscribe", "args": args}] if args else []

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
        if not isinstance(observation, OrderBookObservation):
            return observation

        builder = self._orderbook_builders.get(observation.market)
        if builder is None:
            builder = OkxOrderBookBuilder()
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
