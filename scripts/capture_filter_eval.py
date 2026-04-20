#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multi_venue_basis_fair_value import (  # noqa: E402
    apply_asset_defaults,
    build_basis_filter_config,
    build_filter,
    build_parser as build_basis_parser,
    build_venue_state,
    is_perp_exchange,
    stream_socket_observations,
)
from multi_venue_fair_value import (  # noqa: E402
    _add_kalman_cli_args,
    build_kalman_config,
)
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
from icarus.strategy.fair_value.filters.kalman_1d import (  # noqa: E402
    AdaptiveEfficientPriceKalman,
    VenueObservation,
)
from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (  # noqa: E402
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)
from icarus.strategy.fair_value.types import VenueFairValueState  # noqa: E402
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin  # noqa: E402

if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


class EvalCaptureDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._pending = 0
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER NOT NULL,
                event_exchange TEXT NOT NULL,
                event_market TEXT NOT NULL,
                anchor_exchange TEXT NOT NULL,
                anchor_present INTEGER NOT NULL,
                basis_is_live INTEGER NOT NULL,
                composite_price REAL,
                composite_variance REAL,
                kalman_filtered_price REAL,
                kalman_raw_fused_price REAL,
                basis_common_price REAL,
                basis_common_stddev REAL,
                contributing_exchanges_json TEXT NOT NULL,
                kalman_used_venues_json TEXT NOT NULL,
                basis_active_venues_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_updates_timestamp ON updates(timestamp_ms, id);

            CREATE TABLE IF NOT EXISTS venue_states (
                update_id INTEGER NOT NULL,
                exchange TEXT NOT NULL,
                venue_kind TEXT NOT NULL,
                fair_value REAL NOT NULL,
                variance REAL NOT NULL,
                age_ms REAL NOT NULL,
                PRIMARY KEY (update_id, exchange)
            );

            CREATE TABLE IF NOT EXISTS basis_states (
                update_id INTEGER NOT NULL,
                exchange TEXT NOT NULL,
                basis_estimate REAL NOT NULL,
                basis_stddev REAL NOT NULL,
                PRIMARY KEY (update_id, exchange)
            );
            """
        )
        self.conn.commit()

    def insert_snapshot(
        self,
        *,
        now_ms: int,
        event_exchange: str,
        event_market: str,
        anchor_exchange: str,
        latest_venue_states: dict[str, VenueFairValueState],
        composite_price: float | None,
        composite_variance: float | None,
        contributing_exchanges: tuple[str, ...],
        kalman_filtered_price: float | None,
        kalman_raw_fused_price: float | None,
        kalman_used_venues: list[str],
        basis_common_price: float | None,
        basis_common_stddev: float | None,
        basis_is_live: bool,
        basis_active_venues: list[str],
        basis_estimates: dict[str, float],
        basis_stddevs: dict[str, float],
    ) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO updates (
                timestamp_ms,
                event_exchange,
                event_market,
                anchor_exchange,
                anchor_present,
                basis_is_live,
                composite_price,
                composite_variance,
                kalman_filtered_price,
                kalman_raw_fused_price,
                basis_common_price,
                basis_common_stddev,
                contributing_exchanges_json,
                kalman_used_venues_json,
                basis_active_venues_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_ms,
                event_exchange,
                event_market,
                anchor_exchange,
                int(anchor_exchange in latest_venue_states),
                int(basis_is_live),
                composite_price,
                composite_variance,
                kalman_filtered_price,
                kalman_raw_fused_price,
                basis_common_price,
                basis_common_stddev,
                json.dumps(list(contributing_exchanges)),
                json.dumps(kalman_used_venues),
                json.dumps(basis_active_venues),
            ),
        )
        update_id = int(cursor.lastrowid)

        venue_rows = [
            (
                update_id,
                state.exchange,
                "perp" if is_perp_exchange(state.exchange) else "spot",
                float(state.fair_value),
                float(state.variance),
                float(max(now_ms - state.timestamp_ms, 0)),
            )
            for state in latest_venue_states.values()
        ]
        self.conn.executemany(
            """
            INSERT INTO venue_states (
                update_id,
                exchange,
                venue_kind,
                fair_value,
                variance,
                age_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            venue_rows,
        )

        basis_rows = [
            (
                update_id,
                exchange,
                float(basis_estimates[exchange]),
                float(basis_stddevs.get(exchange, 0.0)),
            )
            for exchange in sorted(basis_estimates)
        ]
        self.conn.executemany(
            """
            INSERT INTO basis_states (
                update_id,
                exchange,
                basis_estimate,
                basis_stddev
            ) VALUES (?, ?, ?, ?)
            """,
            basis_rows,
        )
        self._pending += 1

    def maybe_commit(self, commit_every: int) -> None:
        if self._pending >= commit_every:
            self.conn.commit()
            self._pending = 0

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = build_basis_parser()
    parser.description = "Capture live fair-value snapshots and filter outputs into SQLite."
    _add_kalman_cli_args(parser)
    parser.add_argument(
        "--db-path",
        default="data/filter_eval.sqlite3",
        help="SQLite file to append captured snapshots to.",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit SQLite writes every N updates.",
    )
    return parser


def build_socket_specs(args: argparse.Namespace) -> list[tuple[BaseSocket, str | None, str | None]]:
    specs: list[tuple[BaseSocket, str | None, str | None]] = [
        (
            CoinbaseSocket(
                args.coinbase_market,
                channels=args.coinbase_channels or ["ticker", "heartbeats", "level2"],
                sandbox=args.sandbox,
            ),
            None,
            None,
        )
    ]
    if not args.disable_hyperliquid:
        hyperliquid_subscription_coin = (
            args.hyperliquid_subscription_coin
            or resolve_hyperliquid_spot_subscription_coin(
                args.hyperliquid_market,
                testnet=args.testnet,
            )
        )
        specs.append(
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
        specs.append(
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
        specs.append((OkxSocket(args.okx_market), None, None))
    if not args.disable_kraken:
        specs.append((KrakenSocket(args.kraken_market), None, None))
    return specs


async def run_capture(args: argparse.Namespace) -> None:
    db = EvalCaptureDB(Path(args.db_path))
    observation_queue: asyncio.Queue[Observation] = asyncio.Queue()
    socket_specs = build_socket_specs(args)
    tasks = [
        asyncio.create_task(
            stream_socket_observations(
                socket,
                observation_queue,
                exchange_override=exchange_override,
                market_override=market_override,
            )
        )
        for socket, exchange_override, market_override in socket_specs
    ]

    measurement_engines: dict[tuple[str, str], MarketMeasurementEngine] = {}
    estimators: dict[tuple[str, str], RawFairValueEstimator] = {}
    filters: dict[tuple[str, str], object | None] = {}
    latest_venue_states: dict[str, VenueFairValueState] = {}
    combiner = CrossVenueFairValueCombiner(
        args.asset,
        config=CrossVenueCombinerConfig(
            stale_after_ms=args.stale_after_ms,
            age_penalty_per_second=args.age_penalty_per_second,
        ),
    )
    kalman = AdaptiveEfficientPriceKalman(config=build_kalman_config(args))
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
                venue_state = build_venue_state(
                    engine=current_engine,
                    estimator=estimator,
                    fair_value_filter=filters[current_engine_key],
                    market=args.asset,
                    now_ms=now_ms,
                )
                if venue_state is None:
                    continue
                latest_venue_states[current_engine.exchange] = venue_state

            if not latest_venue_states:
                continue

            spot_venue_states = [
                state for state in latest_venue_states.values() if not is_perp_exchange(state.exchange)
            ]
            if not spot_venue_states:
                continue

            for venue_state in spot_venue_states:
                combiner.update(venue_state, now_ms=now_ms)
            combined = combiner.combine(now_ms=now_ms)

            kalman_result = kalman.update(
                timestamp_s=now_ms / 1000.0,
                observations=[
                    VenueObservation(
                        name=state.exchange,
                        fair_value=float(state.fair_value),
                        local_variance=float(state.variance),
                        age_ms=float(max(now_ms - state.timestamp_ms, 0)),
                    )
                    for state in spot_venue_states
                ],
            )

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
            basis_snapshot = basis_result if basis_result is not None else last_basis_result

            db.insert_snapshot(
                now_ms=now_ms,
                event_exchange=observation.exchange,
                event_market=observation.market,
                anchor_exchange=args.basis_anchor_exchange,
                latest_venue_states=latest_venue_states,
                composite_price=float(combined.fair_value) if combined is not None else None,
                composite_variance=float(combined.variance) if combined is not None else None,
                contributing_exchanges=combined.contributing_exchanges if combined is not None else (),
                kalman_filtered_price=(
                    float(kalman_result.filtered_price) if kalman_result is not None else None
                ),
                kalman_raw_fused_price=(
                    float(kalman_result.raw_fused_price) if kalman_result is not None else None
                ),
                kalman_used_venues=kalman_result.used_venues if kalman_result is not None else [],
                basis_common_price=(
                    float(basis_snapshot.common_price)
                    if basis_snapshot is not None
                    else None
                ),
                basis_common_stddev=(
                    float(basis_snapshot.common_price_stddev)
                    if basis_snapshot is not None
                    else None
                ),
                basis_is_live=basis_result is not None,
                basis_active_venues=(
                    basis_snapshot.active_venues if basis_snapshot is not None else []
                ),
                basis_estimates=(
                    basis_snapshot.basis_estimates if basis_snapshot is not None else {}
                ),
                basis_stddevs=(
                    basis_snapshot.basis_stddevs if basis_snapshot is not None else {}
                ),
            )
            db.maybe_commit(args.commit_every)

            count += 1
            if count % 1000 == 0:
                logging.info("captured %s updates into %s", count, args.db_path)
            if args.limit and count >= args.limit:
                break
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket, _, _ in socket_specs:
            await socket.close()
        db.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_asset_defaults(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_capture(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
