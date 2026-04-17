#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from decimal import InvalidOperation
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from icarus.measurements import MarketMeasurementEngine  # noqa: E402
from icarus.observations import Observation  # noqa: E402
from icarus.strategy.fair_value.estimator import RawFairValueEstimator  # noqa: E402
from icarus.strategy.fair_value.filters.ema import EMAFairValueFilter  # noqa: E402
from icarus.strategy.fair_value.filters.kalman_1d import (  # noqa: E402
    Kalman1DFairValueFilter,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from icarus.sockets.base import BaseSocket


@dataclass(frozen=True, slots=True)
class FilteredFairValueEstimate:
    timestamp_ms: int
    exchange: str
    market: str
    filter_name: str
    raw_fair_value: Decimal
    raw_measurement_variance: Decimal
    filtered_fair_value: Decimal
    filter_output_variance: Decimal
    micro_alpha: Decimal | None
    used_midprice: Decimal | None
    used_microprice: Decimal | None


def ensure_project_src_on_path() -> None:
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))


def parse_decimal_arg(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {value!r}") from exc
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to a websocket feed and print live fair value estimates.",
    )
    parser.add_argument(
        "exchange",
        nargs="?",
        choices=["hyperliquid", "coinbase"],
        default="coinbase",
        help="Which websocket feed to connect to.",
    )
    parser.add_argument(
        "market",
        nargs="*",
        default=None,
        help=(
            "Market identifier. Use a coin symbol like BTC for Hyperliquid or one or more "
            "product ids like BTC-USD ETH-USD for Coinbase."
        ),
    )
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        help="Coinbase channel to subscribe to. Repeat to provide multiple channels.",
    )
    parser.add_argument(
        "--candle-interval",
        default="1m",
        help="Hyperliquid candle interval. Pass an empty string to disable candles.",
    )
    parser.add_argument(
        "--disable-trades",
        action="store_true",
        help="Disable Hyperliquid trades subscription.",
    )
    parser.add_argument(
        "--disable-l2-book",
        action="store_true",
        help="Disable Hyperliquid level-2 book subscription.",
    )
    parser.add_argument(
        "--disable-bbo",
        action="store_true",
        help="Disable Hyperliquid best bid/offer subscription.",
    )
    parser.add_argument(
        "--disable-active-asset-ctx",
        action="store_true",
        help="Disable Hyperliquid active asset context subscription.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Coinbase sandbox websocket.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Hyperliquid testnet websocket.",
    )
    parser.add_argument(
        "--filter",
        choices=["kalman", "ema"],
        default="kalman",
        help="Fair value filter to apply to the raw estimate.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=parse_decimal_arg,
        default=Decimal("0.2"),
        help="EMA alpha when using --filter ema.",
    )
    parser.add_argument(
        "--kalman-process-variance-per-second",
        type=float,
        default=1e-6,
        help="Kalman process variance per second when using --filter kalman.",
    )
    parser.add_argument(
        "--kalman-initial-variance",
        type=float,
        default=1e-4,
        help="Initial Kalman variance when using --filter kalman.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Stop after printing this many filtered fair value estimates. "
            "Zero means stream forever."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for reconnect messages.",
    )
    return parser


def build_socket(args: argparse.Namespace) -> BaseSocket:
    ensure_project_src_on_path()

    from icarus.sockets.coinbase import CoinbaseSocket
    from icarus.sockets.hyperliquid import HyperliquidSocket

    markets = args.market or (["BTC"] if args.exchange == "hyperliquid" else ["BTC-USD"])

    if args.exchange == "hyperliquid":
        if len(markets) != 1:
            raise ValueError("Hyperliquid accepts exactly one market symbol.")

        candle_interval = args.candle_interval or None
        return HyperliquidSocket(
            markets[0],
            testnet=args.testnet,
            candle_interval=candle_interval,
            include_trades=not args.disable_trades,
            include_l2_book=not args.disable_l2_book,
            include_bbo=not args.disable_bbo,
            include_active_asset_ctx=not args.disable_active_asset_ctx,
        )

    if args.testnet:
        raise ValueError("--testnet only applies to Hyperliquid.")
    if any(
        [
            args.disable_trades,
            args.disable_l2_book,
            args.disable_bbo,
            args.disable_active_asset_ctx,
            args.candle_interval != "1m",
        ]
    ):
        raise ValueError("Hyperliquid-only flags cannot be used with Coinbase.")

    return CoinbaseSocket(
        markets,
        channels=args.channels,
        sandbox=args.sandbox,
    )


def observation_stream(socket: BaseSocket) -> AsyncIterator[Observation]:
    ensure_project_src_on_path()

    async def stream() -> AsyncIterator[Observation]:
        async for observation in socket.stream_observations():  # type: ignore[attr-defined]
            yield observation

    return stream()


def build_filter(args: argparse.Namespace) -> EMAFairValueFilter | Kalman1DFairValueFilter:
    if args.filter == "ema":
        return EMAFairValueFilter(alpha=args.ema_alpha)
    return Kalman1DFairValueFilter(
        process_variance_per_second=args.kalman_process_variance_per_second,
        initial_variance=args.kalman_initial_variance,
    )


async def stream_fair_values(socket: BaseSocket, args: argparse.Namespace) -> None:
    count = 0
    measurement_engines: dict[tuple[str, str], MarketMeasurementEngine] = {}
    estimators: dict[tuple[str, str], RawFairValueEstimator] = {}
    filters: dict[tuple[str, str], EMAFairValueFilter | Kalman1DFairValueFilter] = {}

    try:
        async for observation in observation_stream(socket):
            engine_key = (observation.exchange, observation.market)

            measurement_engine = measurement_engines.get(engine_key)
            if measurement_engine is None:
                measurement_engine = MarketMeasurementEngine(
                    exchange=observation.exchange,
                    market=observation.market,
                )
                measurement_engines[engine_key] = measurement_engine

            measurement = measurement_engine.on_observation(observation)
            if measurement is None:
                continue

            estimator = estimators.get(engine_key)
            if estimator is None:
                estimator = RawFairValueEstimator()
                estimators[engine_key] = estimator

            raw_estimate = estimator.estimate(measurement)
            if raw_estimate.raw_fair_value is None or raw_estimate.measurement_variance is None:
                continue

            fair_value_filter = filters.get(engine_key)
            if fair_value_filter is None:
                fair_value_filter = build_filter(args)
                filters[engine_key] = fair_value_filter

            filtered_value, filtered_variance = fair_value_filter.update(
                measurement=raw_estimate.raw_fair_value,
                measurement_variance=raw_estimate.measurement_variance,
                timestamp_ms=raw_estimate.timestamp_ms,
            )

            print(
                FilteredFairValueEstimate(
                    timestamp_ms=raw_estimate.timestamp_ms,
                    exchange=raw_estimate.exchange,
                    market=raw_estimate.market,
                    filter_name=args.filter,
                    raw_fair_value=raw_estimate.raw_fair_value,
                    raw_measurement_variance=raw_estimate.measurement_variance,
                    filtered_fair_value=filtered_value,
                    filter_output_variance=filtered_variance,
                    micro_alpha=raw_estimate.micro_alpha,
                    used_midprice=raw_estimate.used_midprice,
                    used_microprice=raw_estimate.used_microprice,
                )
            )
            count += 1
            if limit := args.limit:
                if count >= limit:
                    break
    finally:
        await socket.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        socket = build_socket(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        asyncio.run(stream_fair_values(socket, args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
