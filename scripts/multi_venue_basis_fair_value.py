#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, replace
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
from icarus.sockets.kraken import KrakenSocket  # noqa: E402
from icarus.sockets.okx import OkxSocket  # noqa: E402
from icarus.strategy.fair_value.combiner import (  # noqa: E402
    CrossVenueCombinerConfig,
    CrossVenueFairValueCombiner,
)
from icarus.strategy.fair_value.estimator import RawFairValueEstimator  # noqa: E402
from icarus.strategy.fair_value.filters.ema import EMAFairValueFilter  # noqa: E402
from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (  # noqa: E402
    VenueBasisKalmanConfig,
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)
from icarus.strategy.fair_value.types import VenueFairValueState  # noqa: E402
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin  # noqa: E402

if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


@dataclass(frozen=True, slots=True)
class MultiVenueBasisEfficientPrice:
    asset: str
    timestamp_ms: int
    composite_efficient_price: Decimal
    basis_filtered_common_price: Decimal | None
    basis_estimate_is_live: bool
    composite_variance: Decimal
    basis_estimates: dict[str, Decimal]
    basis_stddevs: dict[str, Decimal]
    observation_variances: dict[str, Decimal]
    active_venues: tuple[str, ...]
    contributing_exchanges: tuple[str, ...]


def parse_decimal_arg(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the experimental venue-basis fair-value filter live.",
    )
    parser.add_argument("--asset", default="BTC", help="Canonical asset symbol for output.")
    parser.add_argument("--coinbase-market", default="BTC-USD", help="Coinbase product id.")
    parser.add_argument(
        "--hyperliquid-market",
        default="BTC/USDC",
        help="Hyperliquid spot market to resolve and subscribe to.",
    )
    parser.add_argument("--okx-market", default="BTC-USDT", help="OKX instrument id.")
    parser.add_argument("--kraken-market", default="BTC/USDT", help="Kraken v2 pair symbol.")
    parser.add_argument("--disable-kraken", action="store_true", help="Skip the Kraken venue.")
    parser.add_argument(
        "--disable-hyperliquid",
        action="store_true",
        help="Skip the Hyperliquid venue.",
    )
    parser.add_argument("--disable-okx", action="store_true", help="Skip the OKX venue.")
    parser.add_argument(
        "--hyperliquid-subscription-coin",
        default=None,
        help="Optional explicit Hyperliquid subscription coin.",
    )
    parser.add_argument(
        "--enable-hyperliquid-perp",
        action="store_true",
        help="Add a separate Hyperliquid perp ingestion path to the experimental basis filter.",
    )
    parser.add_argument(
        "--hyperliquid-perp-market",
        default="BTC",
        help="Hyperliquid perp market symbol.",
    )
    parser.add_argument(
        "--hyperliquid-perp-subscription-coin",
        default=None,
        help="Optional explicit Hyperliquid perp subscription coin.",
    )
    parser.add_argument(
        "--coinbase-channel",
        action="append",
        dest="coinbase_channels",
        help="Coinbase channel to subscribe to. Repeat for multiple.",
    )
    parser.add_argument("--sandbox", action="store_true", help="Use Coinbase sandbox websocket.")
    parser.add_argument("--testnet", action="store_true", help="Use Hyperliquid testnet websocket.")
    parser.add_argument(
        "--filter",
        choices=["none", "ema"],
        default="none",
        help="Venue-local fair value filter to apply before cross-venue filtering.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=parse_decimal_arg,
        default=Decimal("0.2"),
        help="EMA alpha when using --filter ema.",
    )
    parser.add_argument(
        "--stale-after-ms",
        type=int,
        default=1500,
        help="Drop venue states older than this many milliseconds in the combiner.",
    )
    parser.add_argument(
        "--age-penalty-per-second",
        type=parse_decimal_arg,
        default=Decimal("0.1"),
        help="Combiner-only age penalty used for the printed composite estimate.",
    )
    _add_basis_filter_args(parser)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after printing this many updates. Zero means stream forever.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for reconnect messages.",
    )
    return parser


def build_filter(args: argparse.Namespace) -> EMAFairValueFilter | None:
    if args.filter == "none":
        return None
    return EMAFairValueFilter(alpha=args.ema_alpha)


def _add_basis_filter_args(parser: argparse.ArgumentParser) -> None:
    defaults = VenueBasisKalmanConfig(anchor_exchange="coinbase")
    group = parser.add_argument_group("Experimental venue-basis filter")
    group.add_argument(
        "--basis-anchor-exchange",
        default=defaults.anchor_exchange,
        help="Exchange whose basis is fixed to zero for identifiability.",
    )
    group.add_argument(
        "--basis-common-price-process-var-per-sec",
        type=float,
        default=defaults.common_price_process_var_per_sec,
        help="Random-walk process variance per second for the common price state.",
    )
    group.add_argument(
        "--basis-process-var-per-sec",
        type=float,
        default=defaults.default_basis_process_var_per_sec,
        help="Default process variance per second for non-anchor basis states.",
    )
    group.add_argument(
        "--basis-rho-per-second",
        type=float,
        default=defaults.default_basis_rho_per_second,
        help="Default one-second persistence for basis AR(1) states.",
    )
    group.add_argument(
        "--basis-perp-process-var-per-sec",
        type=float,
        default=defaults.default_perp_basis_process_var_per_sec,
        help="Default process variance per second for perp-basis states.",
    )
    group.add_argument(
        "--basis-perp-rho-per-second",
        type=float,
        default=defaults.default_perp_basis_rho_per_second,
        help="Default one-second persistence for perp-basis AR(1) states.",
    )
    group.add_argument(
        "--basis-initial-common-price-variance",
        type=float,
        default=defaults.initial_common_price_variance,
        help="Initial covariance on the common price state.",
    )
    group.add_argument(
        "--basis-initial-basis-variance",
        type=float,
        default=defaults.initial_basis_variance,
        help="Initial covariance on each non-anchor basis state.",
    )
    group.add_argument(
        "--basis-stale-cutoff-ms",
        type=float,
        default=defaults.stale_cutoff_ms,
        help="Drop venue observations older than this many milliseconds.",
    )
    group.add_argument(
        "--basis-local-var-floor",
        type=float,
        default=defaults.local_var_floor,
        help="Lower bound on per-venue observation variance.",
    )


def build_basis_filter_config(args: argparse.Namespace) -> VenueBasisKalmanConfig:
    ordered_venues = ["coinbase"]
    if not args.disable_hyperliquid:
        ordered_venues.append("hyperliquid")
    if not args.disable_okx:
        ordered_venues.append("okx")
    if not args.disable_kraken:
        ordered_venues.append("kraken")
    perp_exchange_order: list[str] = []
    if args.enable_hyperliquid_perp:
        perp_exchange_order.append("hyperliquid_perp")
    return VenueBasisKalmanConfig(
        anchor_exchange=args.basis_anchor_exchange,
        venue_order=tuple(ordered_venues),
        perp_exchange_order=tuple(perp_exchange_order),
        common_price_process_var_per_sec=args.basis_common_price_process_var_per_sec,
        default_basis_process_var_per_sec=args.basis_process_var_per_sec,
        default_basis_rho_per_second=args.basis_rho_per_second,
        default_perp_basis_process_var_per_sec=args.basis_perp_process_var_per_sec,
        default_perp_basis_rho_per_second=args.basis_perp_rho_per_second,
        initial_common_price_variance=args.basis_initial_common_price_variance,
        initial_basis_variance=args.basis_initial_basis_variance,
        stale_cutoff_ms=args.basis_stale_cutoff_ms,
        local_var_floor=args.basis_local_var_floor,
    )


async def stream_socket_observations(
    socket: BaseSocket,
    output_queue: asyncio.Queue[Observation],
    *,
    exchange_override: str | None = None,
    market_override: str | None = None,
) -> None:
    async for observation in socket.stream_observations():  # type: ignore[attr-defined]
        if exchange_override is not None or market_override is not None:
            observation = replace(
                observation,
                exchange=exchange_override or observation.exchange,
                market=market_override or observation.market,
            )
        await output_queue.put(observation)


def is_perp_exchange(exchange: str) -> bool:
    return exchange.endswith("_perp")


def build_venue_state(
    *,
    engine: MarketMeasurementEngine,
    estimator: RawFairValueEstimator,
    fair_value_filter: EMAFairValueFilter | None,
    market: str,
    now_ms: int,
) -> VenueFairValueState | None:
    measurement = engine.current_measurement(now_ms)
    if measurement is None:
        return None

    raw_estimate = estimator.estimate(measurement)
    if raw_estimate.raw_fair_value is None or raw_estimate.measurement_variance is None:
        return None

    if fair_value_filter is None:
        fair_value = raw_estimate.raw_fair_value
        variance = raw_estimate.measurement_variance
    else:
        fair_value, variance = fair_value_filter.update(
            measurement=raw_estimate.raw_fair_value,
            measurement_variance=raw_estimate.measurement_variance,
            timestamp_ms=raw_estimate.timestamp_ms,
        )

    return VenueFairValueState(
        exchange=engine.exchange,
        market=market,
        timestamp_ms=(
            raw_estimate.timestamp_ms - measurement.quote_age_ms
            if measurement.quote_age_ms is not None
            else raw_estimate.timestamp_ms
        ),
        fair_value=fair_value,
        variance=variance,
    )


async def run_multi_venue_basis(args: argparse.Namespace) -> None:
    observation_queue: asyncio.Queue[Observation] = asyncio.Queue()
    sockets: list[tuple[BaseSocket, str | None, str | None]] = [
        (
            CoinbaseSocket(
                args.coinbase_market,
                channels=args.coinbase_channels or ["ticker", "heartbeats", "level2"],
                sandbox=args.sandbox,
            ),
            None,
            None,
        ),
    ]
    if not args.disable_hyperliquid:
        hyperliquid_subscription_coin = (
            args.hyperliquid_subscription_coin
            or resolve_hyperliquid_spot_subscription_coin(
                args.hyperliquid_market,
                testnet=args.testnet,
            )
        )
        sockets.append(
            (
                HyperliquidSocket(
                    args.hyperliquid_market.split("/", 1)[0],
                    subscription_coin=hyperliquid_subscription_coin,
                    testnet=args.testnet,
                ),
                None,
                None,
            )
        )
    if args.enable_hyperliquid_perp:
        sockets.append(
            (
                HyperliquidSocket(
                    args.hyperliquid_perp_market,
                    subscription_coin=(
                        args.hyperliquid_perp_subscription_coin or args.hyperliquid_perp_market
                    ),
                    testnet=args.testnet,
                ),
                "hyperliquid_perp",
                f"{args.hyperliquid_perp_market}-PERP",
            )
        )
    if not args.disable_okx:
        sockets.append((OkxSocket(args.okx_market), None, None))
    if not args.disable_kraken:
        sockets.append((KrakenSocket(args.kraken_market), None, None))

    tasks = [
        asyncio.create_task(
            stream_socket_observations(
                socket,
                observation_queue,
                exchange_override=exchange_override,
                market_override=market_override,
            )
        )
        for socket, exchange_override, market_override in sockets
    ]

    measurement_engines: dict[tuple[str, str], MarketMeasurementEngine] = {}
    estimators: dict[tuple[str, str], RawFairValueEstimator] = {}
    filters: dict[tuple[str, str], EMAFairValueFilter | None] = {}
    latest_venue_states: dict[str, VenueFairValueState] = {}
    combiner = CrossVenueFairValueCombiner(
        args.asset,
        config=CrossVenueCombinerConfig(
            stale_after_ms=args.stale_after_ms,
            age_penalty_per_second=args.age_penalty_per_second,
        ),
    )
    basis_filter = VenueBasisKalmanFilter(config=build_basis_filter_config(args))
    last_basis_result = None

    count = 0
    try:
        while True:
            observation = await observation_queue.get()
            now_ms = (
                observation.received_timestamp_ms
                if observation.received_timestamp_ms is not None
                else observation.source_timestamp_ms
            )
            if now_ms is None:
                continue

            engine_key = (observation.exchange, observation.market)
            measurement_engine = measurement_engines.get(engine_key)
            if measurement_engine is None:
                measurement_engine = MarketMeasurementEngine(
                    exchange=observation.exchange,
                    market=observation.market,
                )
                measurement_engines[engine_key] = measurement_engine

            measurement_engine.on_observation(observation)

            latest_venue_states.clear()
            for current_engine_key, current_engine in measurement_engines.items():
                estimator = estimators.get(current_engine_key)
                if estimator is None:
                    estimator = RawFairValueEstimator()
                    estimators[current_engine_key] = estimator

                if current_engine_key not in filters:
                    filters[current_engine_key] = build_filter(args)
                fair_value_filter = filters[current_engine_key]

                venue_state = build_venue_state(
                    engine=current_engine,
                    estimator=estimator,
                    fair_value_filter=fair_value_filter,
                    market=args.asset,
                    now_ms=now_ms,
                )
                if venue_state is None:
                    continue
                latest_venue_states[current_engine.exchange] = venue_state

            if not latest_venue_states:
                continue

            spot_venue_states = [
                state
                for state in latest_venue_states.values()
                if not is_perp_exchange(state.exchange)
            ]
            if not spot_venue_states:
                continue
            for venue_state in spot_venue_states:
                combiner.update(venue_state, now_ms=now_ms)
            combined = combiner.combine(now_ms=now_ms)
            if combined is None:
                continue

            basis_result = basis_filter.update(
                timestamp_s=now_ms / 1000.0,
                observations=[
                    VenueBasisObservation(
                        name=state.exchange,
                        fair_value=state.fair_value,
                        local_variance=state.variance,
                        age_ms=float(max(now_ms - state.timestamp_ms, 0)),
                        venue_kind="perp" if is_perp_exchange(state.exchange) else "spot",
                    )
                    for state in latest_venue_states.values()
                ],
            )
            if basis_result is not None:
                last_basis_result = basis_result
            if last_basis_result is None:
                continue

            print(
                MultiVenueBasisEfficientPrice(
                    asset=args.asset,
                    timestamp_ms=combined.timestamp_ms,
                    composite_efficient_price=combined.fair_value,
                    basis_filtered_common_price=Decimal(str(last_basis_result.common_price)),
                    basis_estimate_is_live=basis_result is not None,
                    composite_variance=combined.variance,
                    basis_estimates={
                        exchange: Decimal(str(value))
                        for exchange, value in last_basis_result.basis_estimates.items()
                    },
                    basis_stddevs={
                        exchange: Decimal(str(value))
                        for exchange, value in last_basis_result.basis_stddevs.items()
                    },
                    observation_variances={
                        exchange: Decimal(str(value))
                        for exchange, value in last_basis_result.observation_variances.items()
                    },
                    active_venues=tuple(last_basis_result.active_venues),
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
        for socket, _, _ in sockets:
            await socket.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_multi_venue_basis(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
