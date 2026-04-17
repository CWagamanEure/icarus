from __future__ import annotations

from decimal import Decimal

from icarus.observations import (
    BBOObservation,
    CandleObservation,
    CoinbaseObservationNormalizer,
    HyperliquidObservationNormalizer,
    OrderBookDeltaObservation,
    OrderBookObservation,
    TradeObservation,
)


def test_coinbase_normalizer_emits_bbo_and_l2_observations() -> None:
    normalizer = CoinbaseObservationNormalizer()

    ticker_message = {
        "channel": "ticker",
        "timestamp": "2026-04-17T00:00:14.95780388Z",
        "events": [
            {
                "type": "snapshot",
                "tickers": [
                    {
                        "product_id": "BTC-USD",
                        "best_bid": "75165.93",
                        "best_bid_quantity": "0.22851376",
                        "best_ask": "75165.94",
                        "best_ask_quantity": "0.20719345",
                    }
                ],
            }
        ],
    }
    level2_message = {
        "channel": "level2",
        "timestamp": "2026-04-17T00:00:15.00000000Z",
        "events": [
            {
                "type": "snapshot",
                "product_id": "BTC-USD",
                "updates": [
                    {"side": "bid", "price_level": "75165.93", "new_quantity": "0.22851376"},
                    {"side": "offer", "price_level": "75165.94", "new_quantity": "0.20719345"},
                ],
            }
        ],
    }

    bbo_measurements = normalizer.normalize_message(ticker_message, received_timestamp_ms=123)
    book_measurements = normalizer.normalize_message(level2_message, received_timestamp_ms=456)
    assert isinstance(bbo_measurements[0], BBOObservation)
    assert isinstance(book_measurements[0], OrderBookObservation)

    assert bbo_measurements[0] == BBOObservation(
        exchange="coinbase",
        market="BTC-USD",
        source_timestamp_ms=1776384014957,
        received_timestamp_ms=123,
        raw_message=ticker_message,
        bid_price=Decimal("75165.93"),
        bid_size=Decimal("0.22851376"),
        ask_price=Decimal("75165.94"),
        ask_size=Decimal("0.20719345"),
    )
    assert book_measurements[0].exchange == "coinbase"
    assert book_measurements[0].market == "BTC-USD"
    assert book_measurements[0].update_type == "snapshot"
    assert book_measurements[0].source_timestamp_ms == 1776384015000
    assert book_measurements[0].received_timestamp_ms == 456
    assert book_measurements[0].raw_message == level2_message
    assert book_measurements[0].levels[0].side == "buy"
    assert book_measurements[0].levels[1].side == "sell"


def test_coinbase_normalizer_emits_delta_observation_for_l2update() -> None:
    normalizer = CoinbaseObservationNormalizer()
    level2_update_message = {
        "channel": "level2",
        "timestamp": "2026-04-17T00:00:16.00000000Z",
        "events": [
            {
                "type": "l2update",
                "product_id": "BTC-USD",
                "updates": [
                    {"side": "bid", "price_level": "75165.92", "new_quantity": "0"},
                    {"side": "offer", "price_level": "75165.95", "new_quantity": "1.5"},
                ],
            }
        ],
    }

    observations = normalizer.normalize_message(level2_update_message, received_timestamp_ms=789)

    assert len(observations) == 1
    assert isinstance(observations[0], OrderBookDeltaObservation)
    assert observations[0].market == "BTC-USD"
    assert observations[0].levels[0].side == "buy"
    assert observations[0].levels[1].side == "sell"


def test_coinbase_normalizer_accepts_documented_l2_data_channel() -> None:
    normalizer = CoinbaseObservationNormalizer()
    level2_message = {
        "channel": "l2_data",
        "timestamp": "2026-04-17T00:00:15.00000000Z",
        "events": [
            {
                "type": "snapshot",
                "product_id": "BTC-USD",
                "updates": [
                    {"side": "bid", "price_level": "75165.93", "new_quantity": "0.22851376"},
                    {"side": "offer", "price_level": "75165.94", "new_quantity": "0.20719345"},
                ],
            }
        ],
    }

    observations = normalizer.normalize_message(level2_message, received_timestamp_ms=456)

    assert len(observations) == 1
    assert isinstance(observations[0], OrderBookObservation)
    assert observations[0].market == "BTC-USD"
    assert observations[0].levels[0].side == "buy"
    assert observations[0].levels[1].side == "sell"


def test_coinbase_normalizer_emits_trade_and_candle_measurements() -> None:
    normalizer = CoinbaseObservationNormalizer()

    trades_message = {
        "channel": "market_trades",
        "events": [
            {
                "type": "update",
                "trades": [
                    {
                        "trade_id": "123",
                        "product_id": "ETH-USD",
                        "price": "1260.01",
                        "size": "0.3",
                        "side": "BUY",
                        "time": "2019-08-14T20:42:27.265Z",
                    }
                ],
            }
        ],
    }
    candles_message = {
        "channel": "candles",
        "events": [
            {
                "type": "snapshot",
                "candles": [
                    {
                        "start": "1688998200",
                        "high": "1867.72",
                        "low": "1865.63",
                        "open": "1867.38",
                        "close": "1866.81",
                        "volume": "0.20269406",
                        "product_id": "ETH-USD",
                    }
                ],
            }
        ],
    }

    trade_measurements = normalizer.normalize_message(trades_message, received_timestamp_ms=1)
    candle_measurements = normalizer.normalize_message(candles_message, received_timestamp_ms=2)

    assert isinstance(trade_measurements[0], TradeObservation)
    assert trade_measurements[0].side == "buy"
    assert trade_measurements[0].price == Decimal("1260.01")
    assert isinstance(candle_measurements[0], CandleObservation)
    assert candle_measurements[0].open_timestamp_ms == 1688998200000
    assert candle_measurements[0].volume == Decimal("0.20269406")


def test_hyperliquid_normalizer_emits_measurements() -> None:
    normalizer = HyperliquidObservationNormalizer()

    bbo_message = {
        "channel": "bbo",
        "data": {
            "coin": "BTC",
            "time": 1744848015000,
            "levels": [
                [{"px": "75165.93", "sz": "0.22", "n": 1}],
                [{"px": "75165.94", "sz": "0.20", "n": 1}],
            ],
        },
    }
    trades_message = {
        "channel": "trades",
        "data": [
            {
                "coin": "BTC",
                "side": "B",
                "time": 1744848015000,
                "px": "75165.93",
                "sz": "0.01",
                "hash": "0xabc",
            }
        ],
    }
    candle_message = {
        "channel": "candle",
        "data": [
            {
                "t": 1744848000000,
                "T": 1744848059999,
                "s": "BTC",
                "i": "1m",
                "o": 75100.0,
                "c": 75165.93,
                "h": 75170.0,
                "l": 75090.0,
                "v": 1.23,
                "n": 42,
            }
        ],
    }
    book_message = {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": 1744848015000,
            "levels": [
                [{"px": "75165.93", "sz": "0.22", "n": 1}],
                [{"px": "75165.94", "sz": "0.20", "n": 1}],
            ],
        },
    }

    bbo_measurements = normalizer.normalize_message(bbo_message, received_timestamp_ms=1)
    trade_measurements = normalizer.normalize_message(trades_message, received_timestamp_ms=2)
    candle_measurements = normalizer.normalize_message(candle_message, received_timestamp_ms=3)
    book_measurements = normalizer.normalize_message(book_message, received_timestamp_ms=4)

    assert isinstance(bbo_measurements[0], BBOObservation)
    assert bbo_measurements[0].bid_price == Decimal("75165.93")
    assert isinstance(trade_measurements[0], TradeObservation)
    assert trade_measurements[0].side == "buy"
    assert isinstance(candle_measurements[0], CandleObservation)
    assert candle_measurements[0].interval == "1m"
    assert isinstance(book_measurements[0], OrderBookObservation)
    assert len(book_measurements[0].levels) == 2
