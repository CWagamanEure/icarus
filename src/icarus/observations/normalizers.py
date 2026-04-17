from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from icarus.observations.types import (
    BBOObservation,
    CandleObservation,
    Observation,
    OrderBookDeltaObservation,
    OrderBookLevel,
    OrderBookObservation,
    TradeObservation,
)


def parse_decimal(value: str | int | float | Decimal) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def parse_iso8601_to_ms(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    return int(datetime.fromisoformat(normalized).timestamp() * 1000)


class BaseObservationNormalizer(abc.ABC):
    exchange: str

    @abc.abstractmethod
    def normalize_message(
        self,
        raw_message: dict[str, Any],
        *,
        received_timestamp_ms: int | None = None,
    ) -> list[Observation]:
        """Convert a raw websocket payload into zero or more normalized observations."""

    async def normalize_stream(
        self,
        raw_messages: AsyncIterator[dict[str, Any]],
        *,
        received_timestamp_ms: int | None = None,
    ) -> AsyncIterator[Observation]:
        async for raw_message in raw_messages:
            for observation in self.normalize_message(
                raw_message,
                received_timestamp_ms=received_timestamp_ms,
            ):
                yield observation


class HyperliquidObservationNormalizer(BaseObservationNormalizer):
    exchange = "hyperliquid"

    def normalize_message(
        self,
        raw_message: dict[str, Any],
        *,
        received_timestamp_ms: int | None = None,
    ) -> list[Observation]:
        channel = raw_message.get("channel")
        data = raw_message.get("data")
        if not isinstance(data, dict | list):
            return []

        if channel == "bbo" and isinstance(data, dict):
            bbo_observation = self._normalize_bbo(raw_message, data, received_timestamp_ms)
            return [bbo_observation] if bbo_observation is not None else []
        if channel == "trades" and isinstance(data, list):
            return self._normalize_trades(raw_message, data, received_timestamp_ms)
        if channel == "candle" and isinstance(data, list):
            return self._normalize_candles(raw_message, data, received_timestamp_ms)
        if channel == "l2Book" and isinstance(data, dict):
            book_observation = self._normalize_l2_book(raw_message, data, received_timestamp_ms)
            return [book_observation] if book_observation is not None else []
        return []

    def _normalize_bbo(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> BBOObservation | None:
        levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            return None
        bids, asks = levels
        if not isinstance(bids, list) or not bids or not isinstance(asks, list) or not asks:
            return None
        top_bid = bids[0]
        top_ask = asks[0]
        if not isinstance(top_bid, dict) or not isinstance(top_ask, dict):
            return None
        return BBOObservation(
            exchange=self.exchange,
            market=str(data.get("coin", "")),
            source_timestamp_ms=self._extract_ms_timestamp(data.get("time")),
            received_timestamp_ms=received_timestamp_ms,
            raw_message=deepcopy(raw_message),
            bid_price=parse_decimal(top_bid["px"]),
            bid_size=parse_decimal(top_bid["sz"]),
            ask_price=parse_decimal(top_ask["px"]),
            ask_size=parse_decimal(top_ask["sz"]),
        )

    def _normalize_trades(
        self,
        raw_message: dict[str, Any],
        trades: list[Any],
        received_timestamp_ms: int | None,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            side = trade.get("side")
            if side not in {"A", "B"}:
                continue
            observations.append(
                TradeObservation(
                    exchange=self.exchange,
                    market=str(trade.get("coin", "")),
                    source_timestamp_ms=self._extract_ms_timestamp(trade.get("time")),
                    received_timestamp_ms=received_timestamp_ms,
                    raw_message=deepcopy(raw_message),
                    trade_id=self._coerce_optional_str(trade.get("hash")),
                    side="buy" if side == "B" else "sell",
                    price=parse_decimal(trade["px"]),
                    size=parse_decimal(trade["sz"]),
                )
            )
        return observations

    def _normalize_candles(
        self,
        raw_message: dict[str, Any],
        candles: list[Any],
        received_timestamp_ms: int | None,
    ) -> list[Observation]:
        observations: list[Observation] = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            observations.append(
                CandleObservation(
                    exchange=self.exchange,
                    market=str(candle.get("s", "")),
                    source_timestamp_ms=int(candle["t"]),
                    received_timestamp_ms=received_timestamp_ms,
                    raw_message=deepcopy(raw_message),
                    interval=self._coerce_optional_str(candle.get("i")),
                    open_timestamp_ms=int(candle["t"]),
                    close_timestamp_ms=int(candle["T"]),
                    open_price=parse_decimal(candle["o"]),
                    high_price=parse_decimal(candle["h"]),
                    low_price=parse_decimal(candle["l"]),
                    close_price=parse_decimal(candle["c"]),
                    volume=parse_decimal(candle["v"]),
                    trade_count=self._coerce_optional_int(candle.get("n")),
                )
            )
        return observations

    def _normalize_l2_book(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> OrderBookObservation | None:
        levels = data.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            return None

        normalized_levels: list[OrderBookLevel] = []
        for side_name, side_levels in zip(("buy", "sell"), levels, strict=True):
            if not isinstance(side_levels, list):
                return None
            for level in side_levels:
                if not isinstance(level, dict):
                    continue
                normalized_levels.append(
                    OrderBookLevel(
                        side=cast("Any", side_name),
                        price=parse_decimal(level["px"]),
                        size=parse_decimal(level["sz"]),
                    )
                )
        return OrderBookObservation(
            exchange=self.exchange,
            market=str(data.get("coin", "")),
            source_timestamp_ms=self._extract_ms_timestamp(data.get("time")),
            received_timestamp_ms=received_timestamp_ms,
            raw_message=deepcopy(raw_message),
            update_type="snapshot",
            levels=tuple(normalized_levels),
        )

    @staticmethod
    def _extract_ms_timestamp(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            if value.isdigit():
                return int(value)
            return parse_iso8601_to_ms(value)
        return None

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        return int(value) if isinstance(value, int | str) and str(value).isdigit() else None

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        return str(value) if value is not None else None


class CoinbaseObservationNormalizer(BaseObservationNormalizer):
    exchange = "coinbase"

    def normalize_message(
        self,
        raw_message: dict[str, Any],
        *,
        received_timestamp_ms: int | None = None,
    ) -> list[Observation]:
        channel = raw_message.get("channel")
        events = raw_message.get("events")
        if not isinstance(events, list):
            return []

        observations: list[Observation] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if channel in {"ticker", "ticker_batch"}:
                observations.extend(
                    self._normalize_tickers(raw_message, event, received_timestamp_ms)
                )
            elif channel == "market_trades":
                observations.extend(
                    self._normalize_market_trades(raw_message, event, received_timestamp_ms)
                )
            elif channel == "candles":
                observations.extend(
                    self._normalize_candles(raw_message, event, received_timestamp_ms)
                )
            elif channel in {"level2", "l2_data"}:
                book_observation = self._normalize_level2(raw_message, event, received_timestamp_ms)
                if book_observation is not None:
                    observations.append(book_observation)
        return observations

    def _normalize_tickers(
        self,
        raw_message: dict[str, Any],
        event: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> list[Observation]:
        tickers = event.get("tickers")
        if not isinstance(tickers, list):
            return []
        source_timestamp_ms = self._extract_source_timestamp(raw_message)
        observations: list[Observation] = []
        for ticker in tickers:
            if not isinstance(ticker, dict):
                continue
            if not {
                "best_bid",
                "best_bid_quantity",
                "best_ask",
                "best_ask_quantity",
            } <= ticker.keys():
                continue
            observations.append(
                BBOObservation(
                    exchange=self.exchange,
                    market=str(ticker.get("product_id", "")),
                    source_timestamp_ms=source_timestamp_ms,
                    received_timestamp_ms=received_timestamp_ms,
                    raw_message=deepcopy(raw_message),
                    bid_price=parse_decimal(ticker["best_bid"]),
                    bid_size=parse_decimal(ticker["best_bid_quantity"]),
                    ask_price=parse_decimal(ticker["best_ask"]),
                    ask_size=parse_decimal(ticker["best_ask_quantity"]),
                )
            )
        return observations

    def _normalize_market_trades(
        self,
        raw_message: dict[str, Any],
        event: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> list[Observation]:
        trades = event.get("trades")
        if not isinstance(trades, list):
            return []
        observations: list[Observation] = []
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            side = trade.get("side")
            if side not in {"BUY", "SELL"}:
                continue
            observations.append(
                TradeObservation(
                    exchange=self.exchange,
                    market=str(trade.get("product_id", "")),
                    source_timestamp_ms=self._extract_source_timestamp(trade),
                    received_timestamp_ms=received_timestamp_ms,
                    raw_message=deepcopy(raw_message),
                    trade_id=self._coerce_optional_str(trade.get("trade_id")),
                    side="buy" if side == "BUY" else "sell",
                    price=parse_decimal(trade["price"]),
                    size=parse_decimal(trade["size"]),
                )
            )
        return observations

    def _normalize_candles(
        self,
        raw_message: dict[str, Any],
        event: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> list[Observation]:
        candles = event.get("candles")
        if not isinstance(candles, list):
            return []
        observations: list[Observation] = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            open_timestamp_ms = int(candle["start"]) * 1000
            observations.append(
                CandleObservation(
                    exchange=self.exchange,
                    market=str(candle.get("product_id", "")),
                    source_timestamp_ms=open_timestamp_ms,
                    received_timestamp_ms=received_timestamp_ms,
                    raw_message=deepcopy(raw_message),
                    interval=None,
                    open_timestamp_ms=open_timestamp_ms,
                    close_timestamp_ms=open_timestamp_ms,
                    open_price=parse_decimal(candle["open"]),
                    high_price=parse_decimal(candle["high"]),
                    low_price=parse_decimal(candle["low"]),
                    close_price=parse_decimal(candle["close"]),
                    volume=parse_decimal(candle["volume"]),
                    trade_count=None,
                )
            )
        return observations

    def _normalize_level2(
        self,
        raw_message: dict[str, Any],
        event: dict[str, Any],
        received_timestamp_ms: int | None,
    ) -> Observation | None:
        updates = event.get("updates")
        if not isinstance(updates, list):
            return None
        levels: list[OrderBookLevel] = []
        for update in updates:
            if not isinstance(update, dict):
                continue
            side = update.get("side")
            if side not in {"bid", "offer"}:
                continue
            levels.append(
                OrderBookLevel(
                    side="buy" if side == "bid" else "sell",
                    price=parse_decimal(update["price_level"]),
                    size=parse_decimal(update["new_quantity"]),
                )
            )
        update_type = event.get("type")
        if update_type not in {"snapshot", "update", "l2update"}:
            return None

        market = str(event.get("product_id", ""))
        source_timestamp_ms = self._extract_source_timestamp(raw_message)
        raw_message_copy = deepcopy(raw_message)
        if update_type == "snapshot":
            return OrderBookObservation(
                exchange=self.exchange,
                market=market,
                source_timestamp_ms=source_timestamp_ms,
                received_timestamp_ms=received_timestamp_ms,
                raw_message=raw_message_copy,
                update_type="snapshot",
                levels=tuple(levels),
            )

        # Coinbase level2 update/l2update events contain only changed price levels.
        # They are deltas, not a full current book, so downstream consumers must not
        # treat them as reconstructable top-of-book state without a local book builder.
        return OrderBookDeltaObservation(
            exchange=self.exchange,
            market=market,
            source_timestamp_ms=source_timestamp_ms,
            received_timestamp_ms=received_timestamp_ms,
            raw_message=raw_message_copy,
            levels=tuple(levels),
        )

    @staticmethod
    def _extract_source_timestamp(payload: dict[str, Any]) -> int | None:
        timestamp = payload.get("timestamp") or payload.get("time")
        if isinstance(timestamp, str):
            return parse_iso8601_to_ms(timestamp)
        return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        return str(value) if value is not None else None
