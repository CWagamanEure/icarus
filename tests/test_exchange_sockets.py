from __future__ import annotations

from decimal import Decimal

import pytest

from icarus.observations import OrderBookDeltaObservation, OrderBookLevel, OrderBookObservation
from icarus.sockets.coinbase import CoinbaseSocket
from icarus.sockets.hyperliquid import HyperliquidSocket


class DummyWebSocket:
    async def close(self) -> None:
        return None

    async def send(self, payload: str) -> None:
        return None


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


def test_coinbase_pipeline_observation_rebuilds_snapshot_and_delta() -> None:
    socket = CoinbaseSocket(["BTC-USD"], channels=["level2"])
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot", "sequence_num": 1},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update", "sequence_num": 2},
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("1.5")),
        ),
    )

    rebuilt_snapshot = socket._pipeline_observation(snapshot)
    rebuilt_delta = socket._pipeline_observation(delta)

    assert isinstance(rebuilt_snapshot, OrderBookObservation)
    assert rebuilt_snapshot.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100"),
        size=Decimal("2"),
    )
    assert isinstance(rebuilt_delta, OrderBookObservation)
    assert rebuilt_delta.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100.5"),
        size=Decimal("5"),
    )
    assert rebuilt_delta.best_ask_level == OrderBookLevel(
        side="sell",
        price=Decimal("101"),
        size=Decimal("1.5"),
    )


@pytest.mark.asyncio
async def test_coinbase_after_connect_resets_reconstructed_books() -> None:
    socket = CoinbaseSocket(["BTC-USD"], channels=["level2"])
    socket._ws = DummyWebSocket()

    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot", "sequence_num": 1},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update", "sequence_num": 2},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )

    assert isinstance(socket._pipeline_observation(snapshot), OrderBookObservation)
    await socket.after_connect()

    assert socket._pipeline_observation(delta) is None
