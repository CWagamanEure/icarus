from __future__ import annotations

import zlib
from decimal import Decimal

from icarus.observations import OrderBookDeltaObservation, OrderBookLevel, OrderBookObservation
from icarus.orderbooks import CoinbaseOrderBookBuilder, KrakenOrderBookBuilder


def _kraken_checksum(levels: tuple[OrderBookLevel, ...]) -> int:
    asks = [level for level in levels if level.side == "sell"][:10]
    bids = [level for level in levels if level.side == "buy"][:10]

    def normalize(value: Decimal) -> str:
        formatted = str(value).replace(".", "").lstrip("0")
        return formatted or "0"

    checksum_input = "".join(normalize(level.price) + normalize(level.size) for level in asks)
    checksum_input += "".join(normalize(level.price) + normalize(level.size) for level in bids)
    return zlib.crc32(checksum_input.encode("utf-8")) & 0xFFFFFFFF


def test_coinbase_orderbook_builder_reconstructs_full_book_from_snapshot_and_delta() -> None:
    builder = CoinbaseOrderBookBuilder()
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot"},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="buy", price=Decimal("99"), size=Decimal("1")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
            OrderBookLevel(side="sell", price=Decimal("102"), size=Decimal("4")),
        ),
    )
    delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update"},
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("0")),
            OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("1.5")),
        ),
    )

    first = builder.on_observation(snapshot)
    second = builder.on_observation(delta)

    assert first is not None
    assert first.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100"),
        size=Decimal("2"),
    )
    assert first.best_ask_level == OrderBookLevel(
        side="sell",
        price=Decimal("101"),
        size=Decimal("3"),
    )

    assert second is not None
    assert second.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100.5"),
        size=Decimal("5"),
    )
    assert second.best_ask_level == OrderBookLevel(
        side="sell",
        price=Decimal("101"),
        size=Decimal("1.5"),
    )
    assert all(
        not (level.side == "buy" and level.price == Decimal("100"))
        for level in second.levels
    )


def test_coinbase_orderbook_builder_ignores_delta_before_snapshot() -> None:
    builder = CoinbaseOrderBookBuilder()
    delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update"},
        levels=(OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("0")),),
    )

    assert builder.on_observation(delta) is None


def test_coinbase_orderbook_builder_resets_on_stale_sequence() -> None:
    builder = CoinbaseOrderBookBuilder()
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot", "sequence_num": 10},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    stale_delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update", "sequence_num": 9},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )
    fresh_delta_after_reset = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1200,
        received_timestamp_ms=1300,
        raw_message={"type": "l2update", "sequence_num": 11},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )

    assert builder.on_observation(snapshot) is not None
    assert builder.on_observation(stale_delta) is None
    assert builder.on_observation(fresh_delta_after_reset) is None


def test_coinbase_orderbook_builder_resets_on_sequence_gap() -> None:
    builder = CoinbaseOrderBookBuilder()
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={"type": "snapshot", "sequence_num": 10},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("3")),
        ),
    )
    gap_delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update", "sequence_num": 12},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )
    post_reset_delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1200,
        received_timestamp_ms=1300,
        raw_message={"type": "l2update", "sequence_num": 13},
        levels=(OrderBookLevel(side="buy", price=Decimal("101"), size=Decimal("4")),),
    )

    assert builder.on_observation(snapshot) is not None
    assert builder.on_observation(gap_delta) is None
    assert builder.on_observation(post_reset_delta) is None


def test_coinbase_orderbook_builder_rejects_deltas_after_unsequenced_snapshot() -> None:
    builder = CoinbaseOrderBookBuilder()
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
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
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "l2update", "sequence_num": 11},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )
    sequenced_snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1300,
        received_timestamp_ms=1300,
        raw_message={"type": "snapshot", "sequence_num": 20},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("99"), size=Decimal("1")),
            OrderBookLevel(side="sell", price=Decimal("102"), size=Decimal("4")),
        ),
    )
    valid_delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1400,
        received_timestamp_ms=1500,
        raw_message={"type": "l2update", "sequence_num": 21},
        levels=(OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5")),),
    )

    assert builder.on_observation(snapshot) is not None
    assert builder.on_observation(delta) is None
    assert builder.on_observation(sequenced_snapshot) is not None
    rebuilt = builder.on_observation(valid_delta)
    assert rebuilt is not None
    assert rebuilt.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100.5"),
        size=Decimal("5"),
    )


def test_kraken_orderbook_builder_reconstructs_full_book_from_snapshot_and_delta() -> None:
    builder = KrakenOrderBookBuilder()
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
    updated_levels = (
        OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5.00000000")),
        OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2.00000000")),
        OrderBookLevel(side="sell", price=Decimal("101.5"), size=Decimal("1.50000000")),
    )
    delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={
            "type": "update",
            "data": [{"checksum": _kraken_checksum(updated_levels)}],
        },
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100.5"), size=Decimal("5.00000000")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("0")),
            OrderBookLevel(side="sell", price=Decimal("101.5"), size=Decimal("1.50000000")),
        ),
    )

    first = builder.on_observation(snapshot)
    second = builder.on_observation(delta)

    assert first is not None
    assert second is not None
    assert second.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100.5"),
        size=Decimal("5"),
    )
    assert second.best_ask_level == OrderBookLevel(
        side="sell",
        price=Decimal("101.5"),
        size=Decimal("1.5"),
    )


def test_kraken_orderbook_builder_ignores_delta_before_snapshot() -> None:
    builder = KrakenOrderBookBuilder()
    delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1100,
        received_timestamp_ms=1200,
        raw_message={"type": "update"},
        levels=(OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("1")),),
    )

    assert builder.on_observation(delta) is None


def test_kraken_orderbook_builder_resets_on_checksum_mismatch() -> None:
    builder = KrakenOrderBookBuilder()
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
    post_reset_delta = OrderBookDeltaObservation(
        exchange="kraken",
        market="BTC/USD",
        source_timestamp_ms=1200,
        received_timestamp_ms=1300,
        raw_message={"type": "update", "data": [{"checksum": 1}]},
        levels=(OrderBookLevel(side="buy", price=Decimal("101"), size=Decimal("4.00000000")),),
    )

    assert builder.on_observation(snapshot) is not None
    assert builder.on_observation(bad_delta) is None
    assert builder.on_observation(post_reset_delta) is None
