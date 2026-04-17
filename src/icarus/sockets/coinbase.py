from __future__ import annotations

from typing import Any

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

    async def after_connect(self) -> None:
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
