from __future__ import annotations

from decimal import Decimal

from icarus.observations import (
    BBOObservation,
    OrderBookLevel,
    OrderBookObservation,
    TradeObservation,
)


def test_trade_measurement_repr_omits_raw_message() -> None:
    observation = TradeObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1776384014957,
        received_timestamp_ms=1776384015000,
        raw_message={"huge": "payload"},
        trade_id="123",
        side="buy",
        price=Decimal("75165.93"),
        size=Decimal("0.01"),
    )

    rendered = repr(observation)

    assert "TradeObservation(" in rendered
    assert "raw_message" not in rendered
    assert "price=Decimal('75165.93')" in rendered


def test_order_book_measurement_repr_truncates_levels() -> None:
    observation = OrderBookObservation(
        exchange="hyperliquid",
        market="BTC",
        source_timestamp_ms=123,
        received_timestamp_ms=456,
        raw_message={"huge": "payload"},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("1"), size=Decimal("2")),
            OrderBookLevel(side="buy", price=Decimal("3"), size=Decimal("4")),
            OrderBookLevel(side="sell", price=Decimal("5"), size=Decimal("6")),
            OrderBookLevel(side="sell", price=Decimal("7"), size=Decimal("8")),
            OrderBookLevel(side="sell", price=Decimal("9"), size=Decimal("10")),
        ),
    )

    rendered = repr(observation)

    assert "OrderBookObservation(" in rendered
    assert "...+1 more" in rendered
    assert "raw_message" not in rendered


def test_bbo_measurement_repr_is_compact() -> None:
    observation = BBOObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=123,
        received_timestamp_ms=456,
        raw_message={"ignore": True},
        bid_price=Decimal("10"),
        bid_size=Decimal("1"),
        ask_price=Decimal("11"),
        ask_size=Decimal("2"),
    )

    rendered = repr(observation)

    assert "bid=(Decimal('10'), Decimal('1'))" in rendered
    assert "ask=(Decimal('11'), Decimal('2'))" in rendered


def test_bbo_observation_pricing_properties() -> None:
    observation = BBOObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=123,
        received_timestamp_ms=456,
        raw_message={},
        bid_price=Decimal("10"),
        bid_size=Decimal("1"),
        ask_price=Decimal("14"),
        ask_size=Decimal("3"),
    )

    assert observation.midprice == Decimal("12")
    assert observation.spread == Decimal("4")
    assert observation.spread_bps == Decimal("3333.333333333333333333333333")
    assert observation.microprice == Decimal("11")


def test_order_book_observation_pricing_properties() -> None:
    observation = OrderBookObservation(
        exchange="hyperliquid",
        market="BTC",
        source_timestamp_ms=123,
        received_timestamp_ms=456,
        raw_message={},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("99"), size=Decimal("1")),
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("2")),
            OrderBookLevel(side="sell", price=Decimal("102"), size=Decimal("5")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("4")),
        ),
    )

    assert observation.best_bid_level == OrderBookLevel(
        side="buy",
        price=Decimal("100"),
        size=Decimal("2"),
    )
    assert observation.best_ask_level == OrderBookLevel(
        side="sell",
        price=Decimal("101"),
        size=Decimal("4"),
    )
    assert observation.midprice == Decimal("100.5")
    assert observation.spread == Decimal("1")
    assert observation.spread_bps == Decimal("99.50248756218905472636815920")
    assert observation.microprice == Decimal("100.3333333333333333333333333")
