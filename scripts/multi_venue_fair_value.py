#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from icarus.measurements import MarketMeasurementEngine  # noqa: E402
from icarus.observations import Observation  # noqa: E402
from icarus.sockets.coinbase import CoinbaseSocket  # noqa: E402
from icarus.sockets.hyperliquid import HyperliquidSocket  # noqa: E402
from icarus.strategy.fair_value.combiner import (  # noqa: E402
    CrossVenueCombinerConfig,
    CrossVenueFairValueCombiner,
)
from icarus.strategy.fair_value.estimator import RawFairValueEstimator  # noqa: E402
from icarus.strategy.fair_value.filters.ema import EMAFairValueFilter  # noqa: E402
from icarus.strategy.fair_value.filters.kalman_1d import (  # noqa: E402
    Kalman1DFairValueFilter,
)
from icarus.strategy.fair_value.types import VenueFairValueState  # noqa: E402
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin  # noqa: E402

if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


@dataclass(frozen=True, slots=True)
class MultiVenueEfficientPrice:
    asset: str
    timestamp_ms: int
    combined_fair_value: Decimal
    combined_variance: Decimal
    coinbase_fair_value: Decimal | None
    hyperliquid_fair_value: Decimal | None
    contributing_exchanges: tuple[str, ...]


def parse_decimal_arg(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a combined BTC efficient price from Coinbase and Hyperliquid.",
    )
    parser.add_argument(
        "--asset",
        default="BTC",
        help="Canonical asset symbol for the combined output.",
    )
    parser.add_argument(
        "--coinbase-market",
        default="BTC-USD",
        help="Coinbase product id.",
    )
    parser.add_argument(
        "--hyperliquid-market",
        default="BTC/USDC",
        help="Hyperliquid spot market to resolve and subscribe to.",
    )
    parser.add_argument(
        "--hyperliquid-subscription-coin",
        default=None,
        help="Optional explicit Hyperliquid subscription coin. If omitted, resolve from spotMeta.",
    )
    parser.add_argument(
        "--coinbase-channel",
        action="append",
        dest="coinbase_channels",
        help="Coinbase channel to subscribe to. Repeat to provide multiple channels.",
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
        choices=["none", "kalman", "ema"],
        default="none",
        help="Venue-local fair value filter to apply before combining.",
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
        "--stale-after-ms",
        type=int,
        default=500,
        help="Drop venue states older than this many milliseconds.",
    )
    parser.add_argument(
        "--age-penalty-per-second",
        type=parse_decimal_arg,
        default=Decimal("1"),
        help="Inflate venue variance by this age penalty per second before combining.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after printing this many combined updates. Zero means stream forever.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for reconnect messages.",
    )
    return parser


def build_filter(
    args: argparse.Namespace,
) -> EMAFairValueFilter | Kalman1DFairValueFilter | None:
    if args.filter == "none":
        return None

    if args.filter == "ema":
        return EMAFairValueFilter(alpha=args.ema_alpha)

    if args.kalman_process_variance_per_second < 0:
        raise ValueError("--kalman-process-variance-per-second must be non-negative.")
    if args.kalman_initial_variance <= 0:
        raise ValueError("--kalman-initial-variance must be positive.")

    return Kalman1DFairValueFilter(
        process_variance_per_second=args.kalman_process_variance_per_second,
        initial_variance=args.kalman_initial_variance,
    )


async def stream_socket_observations(
    socket: BaseSocket,
    output_queue: asyncio.Queue[Observation],
) -> None:
    async for observation in socket.stream_observations():  # type: ignore[attr-defined]
        await output_queue.put(observation)


async def run_multi_venue(args: argparse.Namespace) -> None:
    observation_queue: asyncio.Queue[Observation] = asyncio.Queue()
    hyperliquid_subscription_coin = (
        args.hyperliquid_subscription_coin
        or resolve_hyperliquid_spot_subscription_coin(
            args.hyperliquid_market,
            testnet=args.testnet,
        )
    )
    sockets: list[BaseSocket] = [
        CoinbaseSocket(
            args.coinbase_market,
            channels=args.coinbase_channels or ["ticker", "heartbeats", "level2"],
            sandbox=args.sandbox,
        ),
        HyperliquidSocket(
            args.hyperliquid_market.split("/", 1)[0],
            subscription_coin=hyperliquid_subscription_coin,
            testnet=args.testnet,
        ),
    ]

    tasks = [
        asyncio.create_task(stream_socket_observations(socket, observation_queue))
        for socket in sockets
    ]

    measurement_engines: dict[tuple[str, str], MarketMeasurementEngine] = {}
    estimators: dict[tuple[str, str], RawFairValueEstimator] = {}
    filters: dict[tuple[str, str], EMAFairValueFilter | Kalman1DFairValueFilter | None] = {}
    latest_venue_states: dict[str, VenueFairValueState] = {}
    combiner = CrossVenueFairValueCombiner(
        args.asset,
        config=CrossVenueCombinerConfig(
            stale_after_ms=args.stale_after_ms,
            age_penalty_per_second=args.age_penalty_per_second,
        ),
    )

    count = 0
    try:
        while True:
            observation = await observation_queue.get()
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
            if engine_key not in filters:
                filters[engine_key] = build_filter(args)
            fair_value_filter = filters[engine_key]

            if fair_value_filter is None:
                filtered_value = raw_estimate.raw_fair_value
                filtered_variance = raw_estimate.measurement_variance
            else:
                filtered_value, filtered_variance = fair_value_filter.update(
                    measurement=raw_estimate.raw_fair_value,
                    measurement_variance=raw_estimate.measurement_variance,
                    timestamp_ms=raw_estimate.timestamp_ms,
                )

            venue_state = VenueFairValueState(
                exchange=observation.exchange,
                market=args.asset,
                timestamp_ms=raw_estimate.timestamp_ms,
                fair_value=filtered_value,
                variance=filtered_variance,
            )
            latest_venue_states[observation.exchange] = venue_state
            combined = combiner.update(venue_state, now_ms=raw_estimate.timestamp_ms)
            if combined is None:
                continue

            coinbase_state = latest_venue_states.get("coinbase")
            hyperliquid_state = latest_venue_states.get("hyperliquid")

            print(
                MultiVenueEfficientPrice(
                    asset=args.asset,
                    timestamp_ms=combined.timestamp_ms,
                    combined_fair_value=combined.fair_value,
                    combined_variance=combined.variance,
                    coinbase_fair_value=(
                        coinbase_state.fair_value if coinbase_state is not None else None
                    ),
                    hyperliquid_fair_value=(
                        hyperliquid_state.fair_value if hyperliquid_state is not None else None
                    ),
                    contributing_exchanges=combined.contributing_exchanges,
                )
            )
            count += 1
            if args.limit and count >= args.limit:
                break
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket in sockets:
            await socket.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_multi_venue(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
