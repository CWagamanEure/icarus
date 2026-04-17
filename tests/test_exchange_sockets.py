from __future__ import annotations

import zlib
from decimal import Decimal

import pytest

from icarus.observations import OrderBookDeltaObservation, OrderBookLevel, OrderBookObservation
from icarus.sockets.coinbase import CoinbaseSocket
from icarus.sockets.hyperliquid import HyperliquidSocket
from icarus.sockets.kraken import KrakenSocket
from icarus.sockets.okx import OkxSocket


def _kraken_checksum(levels: tuple[OrderBookLevel, ...]) -> int:
    asks = [level for level in levels if level.side == "sell"][:10]
    bids = [level for level in levels if level.side == "buy"][:10]

    def normalize(value: Decimal) -> str:
        formatted = str(value).replace(".", "").lstrip("0")
        return formatted or "0"

    checksum_input = "".join(normalize(level.price) + normalize(level.size) for level in asks)
    checksum_input += "".join(normalize(level.price) + normalize(level.size) for level in bids)
    return zlib.crc32(checksum_input.encode("utf-8")) & 0xFFFFFFFF


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


def test_kraken_normalizes_symbols_and_subscriptions() -> None:
    socket = KrakenSocket("btc/usd")

    assert socket.symbols == ["BTC/USD"]
    assert socket.subscription_messages() == [
        {
            "method": "subscribe",
            "params": {
                "channel": "ticker",
                "symbol": ["BTC/USD"],
                "event_trigger": "bbo",
                "snapshot": True,
            },
        },
        {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["BTC/USD"],
                "depth": 10,
                "snapshot": True,
            },
        },
        {
            "method": "subscribe",
            "params": {
                "channel": "trade",
                "symbol": ["BTC/USD"],
                "snapshot": False,
            },
        },
    ]


def test_kraken_respects_disabled_streams() -> None:
    socket = KrakenSocket(
        "ETH/USD",
        include_ticker=False,
        include_trades=False,
        book_depth=25,
    )

    assert socket.subscription_messages() == [
        {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": ["ETH/USD"],
                "depth": 25,
                "snapshot": True,
            },
        }
    ]


def test_kraken_parse_message_preserves_float_precision_as_strings() -> None:
    socket = KrakenSocket("BTC/USD")

    parsed = socket.parse_message(
        '{"channel":"book","type":"snapshot","data":[{"symbol":"BTC/USD","bids":[{"price":45296.1,"qty":0.35380000}],"asks":[],"checksum":3310070434,"timestamp":"2023-10-06T17:35:55.440295Z"}]}'
    )

    book = parsed["data"][0]
    assert book["bids"][0]["price"] == "45296.1"
    assert book["bids"][0]["qty"] == "0.35380000"


def test_kraken_pipeline_observation_rebuilds_snapshot_and_delta() -> None:
    socket = KrakenSocket(["BTC/USD"], include_ticker=False, include_trades=False)
    snapshot = OrderBookObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot"},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "update"},
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("1.5")),
        ),
    )

    rebuilt_snapshot = socket._pipeline_observation(snapshot)
    rebuilt_delta = socket._pipeline_observation(delta)

    assert isinstance(rebuilt_snapshot, OrderBookObservation)
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
async def test_kraken_after_connect_resets_reconstructed_books() -> None:
    socket = KrakenSocket(["BTC/USD"], include_ticker=False, include_trades=False)
    socket._ws = DummyWebSocket()

    snapshot = OrderBookObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot"},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "update"},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )

    assert isinstance(socket._pipeline_observation(snapshot), OrderBookObservation)
    await socket.after_connect()

    assert socket._pipeline_observation(delta) is None


def test_kraken_pipeline_observation_raises_on_checksum_mismatch() -> None:
    socket = KrakenSocket(["BTC/USD"], include_ticker=False, include_trades=False)
    snapshot_levels = (
        OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2.00000000")),
        OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3.00000000")),
    )
    snapshot = OrderBookObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={
            "type": "snapshot",
            "data": [{"checksum": _kraken_checksum(snapshot_levels)}],
        },
        update_type="snapshot",
        levels=snapshot_levels,
    )
    bad_delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "update", "data": [{"checksum": 1}]},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5.00000000")),),
    )

    assert isinstance(socket._pipeline_observation(snapshot), OrderBookObservation)
    with pytest.raises(ConnectionError, match="checksum mismatch"):
        socket._pipeline_observation(bad_delta)


def test_okx_normalizes_inst_ids_and_subscriptions() -> None:
    socket = OkxSocket("btc-usdt")

    assert socket.inst_ids == ["BTC-USDT"]
    assert socket.subscription_messages() == [
        {
            "op": "subscribe",
            "args": [
                {"channel": "tickers", "instId": "BTC-USDT"},
                {"channel": "books5", "instId": "BTC-USDT"},
                {"channel": "trades", "instId": "BTC-USDT"},
            ],
        }
    ]


def test_okx_respects_disabled_streams() -> None:
    socket = OkxSocket("ETH-USDT", include_tickers=False, include_trades=False)

    assert socket.subscription_messages() == [
        {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": "ETH-USDT"}],
        }
    ]
