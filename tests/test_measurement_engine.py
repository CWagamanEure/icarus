from __future__ import annotations

from decimal import Decimal

from icarus.measurements import MarketMeasurement, MarketMeasurementEngine
from icarus.observations import (
    BBOObservation,
    OrderBookDeltaObservation,
    OrderBookLevel,
    OrderBookObservation,
    TradeObservation,
)


def test_measurement_engine_emits_from_bbo_observation() -> None:
    engine = MarketMeasurementEngine(exchange="coinbase", market="BTC-USD")
    observation = BBOObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1100,
        raw_message={},
        bid_price=Decimal("100"),
        bid_size=Decimal("2"),
        ask_price=Decimal("101"),
        ask_size=Decimal("3"),
    )

    measurement = engine.on_observation(observation)

    assert isinstance(measurement, MarketMeasurement)
    assert measurement.midprice == Decimal("100.5")
    assert measurement.microprice == Decimal("100.4")
    assert measurement.spread_bps == Decimal("99.50248756218905472636815920")
    assert measurement.top_bid_depth == Decimal("2")
    assert measurement.top_ask_depth == Decimal("3")
    assert measurement.depth_imbalance == Decimal("-0.2")
    assert measurement.quote_age_ms == 0


def test_measurement_engine_tracks_quote_age_and_volatility() -> None:
    engine = MarketMeasurementEngine(exchange="hyperliquid", market="BTC")
    quote_1 = BBOObservation(
        exchange="hyperliquid",
        market="BTC",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={},
        bid_price=Decimal("100"),
        bid_size=Decimal("1"),
        ask_price=Decimal("102"),
        ask_size=Decimal("1"),
    )
    quote_2 = OrderBookObservation(
        exchange="hyperliquid",
        market="BTC",
        source_timestamp_ms=2000,
        received_timestamp_ms=2000,
        raw_message={},
        update_type="snapshot",
        levels=(
            OrderBookLevel(side="buy", price=Decimal("101"), size=Decimal("4")),
            OrderBookLevel(side="sell", price=Decimal("103"), size=Decimal("5")),
        ),
    )
    trade = TradeObservation(
        exchange="hyperliquid",
        market="BTC",
        source_timestamp_ms=2500,
        received_timestamp_ms=2600,
        raw_message={},
        trade_id="abc",
        side="buy",
        price=Decimal("102"),
        size=Decimal("0.1"),
    )

    first_measurement = engine.on_observation(quote_1)
    second_measurement = engine.on_observation(quote_2)
    third_measurement = engine.on_observation(trade)

    assert first_measurement is not None
    assert second_measurement is not None
    assert second_measurement.mid_volatility_bps is not None
    assert third_measurement is not None
    assert third_measurement.quote_age_ms == 600
    assert third_measurement.midprice == Decimal("102")
    assert third_measurement.top_bid_depth == Decimal("4")
    assert third_measurement.top_ask_depth == Decimal("5")


def test_measurement_engine_ignores_delta_observation_without_state_update() -> None:
    engine = MarketMeasurementEngine(exchange="coinbase", market="BTC-USD")
    snapshot = OrderBookObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={},
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
        raw_message={},
        levels=(
            OrderBookLevel(side="buy", price=Decimal("99"), size=Decimal("0")),
            OrderBookLevel(side="sell", price=Decimal("105"), size=Decimal("1")),
        ),
    )

    first_measurement = engine.on_observation(snapshot)
    second_measurement = engine.on_observation(delta)

    assert first_measurement is not None
    assert first_measurement.midprice == Decimal("100.5")
    assert second_measurement is None


def test_measurement_engine_delta_only_event_emits_no_bogus_features() -> None:
    engine = MarketMeasurementEngine(exchange="coinbase", market="BTC-USD")
    delta = OrderBookDeltaObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1000,
        received_timestamp_ms=1000,
        raw_message={},
        levels=(
            OrderBookLevel(side="buy", price=Decimal("100"), size=Decimal("0")),
            OrderBookLevel(side="sell", price=Decimal("101"), size=Decimal("1")),
        ),
    )

    measurement = engine.on_observation(delta)

    assert measurement is None
