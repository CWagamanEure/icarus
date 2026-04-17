from __future__ import annotations

from typing import Any

from icarus.sockets.base import BaseSocket


class HyperliquidSocket(BaseSocket):
    """Stream raw market updates from Hyperliquid."""

    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"

    def __init__(
        self,
        market: str,
        *,
        subscription_coin: str | None = None,
        testnet: bool = False,
        candle_interval: str | None = "1m",
        include_trades: bool = True,
        include_l2_book: bool = True,
        include_bbo: bool = True,
        include_active_asset_ctx: bool = True,
    ) -> None:
        super().__init__(self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL)
        self.market = market.upper()
        self.subscription_coin = subscription_coin or self.market
        self.candle_interval = candle_interval
        self.include_trades = include_trades
        self.include_l2_book = include_l2_book
        self.include_bbo = include_bbo
        self.include_active_asset_ctx = include_active_asset_ctx

    async def after_connect(self) -> None:
        for subscription in self.subscriptions():
            await self.send_json(
                {
                    "method": "subscribe",
                    "subscription": subscription,
                }
            )

    def subscriptions(self) -> list[dict[str, Any]]:
        subscriptions: list[dict[str, Any]] = []

        if self.include_trades:
            subscriptions.append({"type": "trades", "coin": self.subscription_coin})
        if self.include_l2_book:
            subscriptions.append({"type": "l2Book", "coin": self.subscription_coin})
        if self.include_bbo:
            subscriptions.append({"type": "bbo", "coin": self.subscription_coin})
        if self.include_active_asset_ctx:
            subscriptions.append({"type": "activeAssetCtx", "coin": self.subscription_coin})
        if self.candle_interval:
            subscriptions.append(
                {
                    "type": "candle",
                    "coin": self.subscription_coin,
                    "interval": self.candle_interval,
                }
            )

        return subscriptions
