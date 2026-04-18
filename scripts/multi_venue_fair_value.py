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
from icarus.sockets.kraken import KrakenSocket  # noqa: E402
from icarus.sockets.okx import OkxSocket  # noqa: E402
from icarus.strategy.fair_value.combiner import (  # noqa: E402
    CrossVenueCombinerConfig,
    CrossVenueFairValueCombiner,
)
from icarus.strategy.fair_value.estimator import RawFairValueEstimator  # noqa: E402
from icarus.strategy.fair_value.filters.ema import EMAFairValueFilter  # noqa: E402
from icarus.strategy.fair_value.filters.kalman_1d import (  # noqa: E402
    AdaptiveEfficientPriceKalman,
    KalmanFilterConfig,
    VenueObservation,
)
from icarus.strategy.fair_value.types import VenueFairValueState  # noqa: E402
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin  # noqa: E402

if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


@dataclass(frozen=True, slots=True)
class MultiVenueEfficientPrice:
    asset: str
    timestamp_ms: int
    composite_efficient_price: Decimal
    filtered_efficient_price: Decimal | None
    composite_variance: Decimal
    coinbase_fair_value: Decimal | None
    hyperliquid_fair_value: Decimal | None
    okx_fair_value: Decimal | None
    kraken_fair_value: Decimal | None
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
        "--okx-market",
        default="BTC-USDT",
        help="OKX instrument id.",
    )
    parser.add_argument(
        "--kraken-market",
        default="BTC/USD",
        help="Kraken v2 pair symbol.",
    )
    parser.add_argument(
        "--disable-kraken",
        action="store_true",
        help="Skip the Kraken venue entirely.",
    )
    parser.add_argument(
        "--disable-hyperliquid",
        action="store_true",
        help="Skip the Hyperliquid venue entirely.",
    )
    parser.add_argument(
        "--disable-okx",
        action="store_true",
        help="Skip the OKX venue entirely.",
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
        choices=["none", "ema"],
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
        "--stale-after-ms",
        type=int,
        default=1500,
        help="Drop venue states older than this many milliseconds.",
    )
    parser.add_argument(
        "--age-penalty-per-second",
        type=parse_decimal_arg,
        default=Decimal("0.1"),
        help=(
            "Inflate venue variance by this age penalty per second before combining. "
            "Note: variance.py already applies an age-factor to measurement variance, "
            "so this combiner-level penalty is additive/compounding. Keep small."
        ),
    )
    _add_kalman_cli_args(parser)
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
) -> EMAFairValueFilter | None:
    if args.filter == "none":
        return None
    return EMAFairValueFilter(alpha=args.ema_alpha)


def _add_kalman_cli_args(parser: argparse.ArgumentParser) -> None:
    defaults = KalmanFilterConfig()
    group = parser.add_argument_group("Kalman filter")
    group.add_argument(
        "--kalman-initial-variance",
        type=float,
        default=defaults.initial_variance,
        help="Initial posterior variance P_0 (dollars^2).",
    )
    group.add_argument(
        "--kalman-q-base-per-sec",
        type=float,
        default=defaults.q_base_per_sec,
        help="Baseline process variance per second. LOWER => smoother.",
    )
    group.add_argument(
        "--kalman-q-vol-scale",
        type=float,
        default=defaults.q_vol_scale,
        help="Multiplier on EWMA observed-move variance. LOWER => smoother.",
    )
    group.add_argument(
        "--kalman-r-floor",
        type=float,
        default=defaults.r_floor,
        help="Minimum observation variance R_t. HIGHER => smoother.",
    )
    group.add_argument(
        "--kalman-local-var-floor",
        type=float,
        default=defaults.local_var_floor,
        help="Minimum per-venue local variance.",
    )
    group.add_argument(
        "--kalman-disagreement-scale",
        type=float,
        default=defaults.disagreement_scale,
        help="Inflate R_t when venues disagree. HIGHER => smoother on disagreement.",
    )
    group.add_argument(
        "--kalman-stale-cutoff-ms",
        type=float,
        default=defaults.stale_cutoff_ms,
        help="Drop venue observations older than this many ms.",
    )
    group.add_argument(
        "--kalman-age-variance-scale",
        type=float,
        default=defaults.age_variance_scale,
        help="Quadratic penalty applied to older venue observations.",
    )
    group.add_argument(
        "--kalman-obs-var-ewma-alpha",
        type=float,
        default=defaults.obs_var_ewma_alpha,
        help="EWMA alpha for process-noise proxy. LOWER => slower vol adaptation.",
    )


def build_kalman_config(args: argparse.Namespace) -> KalmanFilterConfig:
    return KalmanFilterConfig(
        initial_variance=args.kalman_initial_variance,
        q_base_per_sec=args.kalman_q_base_per_sec,
        q_vol_scale=args.kalman_q_vol_scale,
        r_floor=args.kalman_r_floor,
        local_var_floor=args.kalman_local_var_floor,
        disagreement_scale=args.kalman_disagreement_scale,
        stale_cutoff_ms=args.kalman_stale_cutoff_ms,
        age_variance_scale=args.kalman_age_variance_scale,
        obs_var_ewma_alpha=args.kalman_obs_var_ewma_alpha,
    )


async def stream_socket_observations(
    socket: BaseSocket,
    output_queue: asyncio.Queue[Observation],
) -> None:
    async for observation in socket.stream_observations():  # type: ignore[attr-defined]
        await output_queue.put(observation)


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


async def run_multi_venue(args: argparse.Namespace) -> None:
    observation_queue: asyncio.Queue[Observation] = asyncio.Queue()
    sockets: list[BaseSocket] = [
        CoinbaseSocket(
            args.coinbase_market,
            channels=args.coinbase_channels or ["ticker", "heartbeats", "level2"],
            sandbox=args.sandbox,
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
            HyperliquidSocket(
                args.hyperliquid_market.split("/", 1)[0],
                subscription_coin=hyperliquid_subscription_coin,
                testnet=args.testnet,
            )
        )
    if not args.disable_okx:
        sockets.append(OkxSocket(args.okx_market))
    if not args.disable_kraken:
        sockets.append(KrakenSocket(args.kraken_market))

    tasks = [
        asyncio.create_task(stream_socket_observations(socket, observation_queue))
        for socket in sockets
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
    kalman = AdaptiveEfficientPriceKalman(config=build_kalman_config(args))

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

            for venue_state in latest_venue_states.values():
                combiner.update(venue_state, now_ms=now_ms)
            combined = combiner.combine(now_ms=now_ms)
            if combined is None:
                continue

            kalman_result = kalman.update(
                timestamp_s=now_ms / 1000.0,
                observations=[
                    VenueObservation(
                        name=state.exchange,
                        fair_value=float(state.fair_value),
                        local_variance=float(state.variance),
                        age_ms=float(max(now_ms - state.timestamp_ms, 0)),
                    )
                    for state in latest_venue_states.values()
                ],
            )
            filtered_price = (
                Decimal(str(kalman_result.filtered_price))
                if kalman_result is not None
                else None
            )

            coinbase_state = latest_venue_states.get("coinbase")
            hyperliquid_state = latest_venue_states.get("hyperliquid")
            okx_state = latest_venue_states.get("okx")
            kraken_state = latest_venue_states.get("kraken")

            print(
                MultiVenueEfficientPrice(
                    asset=args.asset,
                    timestamp_ms=combined.timestamp_ms,
                    composite_efficient_price=combined.fair_value,
                    filtered_efficient_price=filtered_price,
                    composite_variance=combined.variance,
                    coinbase_fair_value=(
                        coinbase_state.fair_value if coinbase_state is not None else None
                    ),
                    hyperliquid_fair_value=(
                        hyperliquid_state.fair_value if hyperliquid_state is not None else None
                    ),
                    okx_fair_value=(
                        okx_state.fair_value if okx_state is not None else None
                    ),
                    kraken_fair_value=(
                        kraken_state.fair_value if kraken_state is not None else None
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
