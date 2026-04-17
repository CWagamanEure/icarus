from __future__ import annotations

from icarus.sockets.coinbase import CoinbaseSocket
from icarus.sockets.hyperliquid import HyperliquidSocket


def test_hyperliquid_builds_default_subscriptions() -> None:
    socket = HyperliquidSocket("btc")

    assert socket.subscriptions() == [
        {"type": "trades", "coin": "BTC"},
        {"type": "l2Book", "coin": "BTC"},
        {"type": "bbo", "coin": "BTC"},
        {"type": "activeAssetCtx", "coin": "BTC"},
        {"type": "candle", "coin": "BTC", "interval": "1m"},
    ]


def test_hyperliquid_respects_disabled_streams() -> None:
    socket = HyperliquidSocket(
        "eth",
        candle_interval=None,
        include_trades=False,
        include_bbo=False,
        include_active_asset_ctx=False,
    )

    assert socket.subscriptions() == [{"type": "l2Book", "coin": "ETH"}]


def test_coinbase_normalizes_products_and_channels() -> None:
    socket = CoinbaseSocket("btc-usd")

    assert socket.product_ids == ["BTC-USD"]
    assert socket.max_message_size == 8 * 1024 * 1024
    assert socket.subscription_messages() == [
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "ticker"},
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "heartbeats"},
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "level2"},
    ]


def test_coinbase_filters_non_object_events() -> None:
    socket = CoinbaseSocket(["BTC-USD"], channels=["ticker"])

    parsed = socket.parse_message(
        '{"channel":"ticker","events":[{"type":"snapshot"},"skip-me",1,{"type":"update"}]}'
    )

    assert parsed == {
        "channel": "ticker",
        "events": [{"type": "snapshot"}, {"type": "update"}],
    }
