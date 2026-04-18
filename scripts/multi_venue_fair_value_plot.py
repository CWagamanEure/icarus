#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import sys
import threading
import tkinter as tk
from collections import deque
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
class PlotPoint:
    timestamp_ms: int
    composite_efficient_price: float
    filtered_efficient_price: float | None
    coinbase_fair_value: float | None
    hyperliquid_fair_value: float | None
    okx_fair_value: float | None
    kraken_fair_value: float | None
    diagnostics_lines: tuple[str, ...]


def parse_decimal_arg(value: str) -> Decimal:
    parsed = Decimal(value)
    if not parsed.is_finite():
        raise argparse.ArgumentTypeError(f"decimal value must be finite: {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot combined BTC efficient price from Coinbase and Hyperliquid live.",
    )
    parser.add_argument("--asset", default="BTC", help="Canonical asset symbol for output.")
    parser.add_argument("--coinbase-market", default="BTC-USD", help="Coinbase product id.")
    parser.add_argument(
        "--hyperliquid-market",
        default="BTC/USDC",
        help="Hyperliquid spot market to resolve and subscribe to.",
    )
    parser.add_argument("--okx-market", default="BTC-USDT", help="OKX instrument id.")
    parser.add_argument("--kraken-market", default="BTC/USD", help="Kraken v2 pair symbol.")
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
    parser.add_argument("--sandbox", action="store_true", help="Use Coinbase sandbox websocket.")
    parser.add_argument("--testnet", action="store_true", help="Use Hyperliquid testnet websocket.")
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
    parser.add_argument("--max-points", type=int, default=300, help="Maximum points to keep.")
    parser.add_argument("--width", type=int, default=1100, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=700, help="Window height in pixels.")
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


def build_diagnostics_text(
    *,
    venue_states: dict[str, VenueFairValueState],
    combiner: CrossVenueFairValueCombiner,
    now_ms: int,
) -> tuple[str, ...]:
    rows: list[str] = []
    weighted: list[tuple[str, Decimal, int, Decimal]] = []

    for exchange in ("coinbase", "hyperliquid", "okx", "kraken"):
        state = venue_states.get(exchange)
        if state is None:
            rows.append(f"{exchange:<11} px=-- age=-- var=-- eff=-- w=--")
            continue
        age_ms = max(now_ms - state.timestamp_ms, 0)
        if age_ms > combiner.config.stale_after_ms:
            rows.append(
                f"{exchange:<11} px={float(state.fair_value):>9.2f} age={age_ms:>4}ms stale"
            )
            continue

        effective_variance = combiner._effective_variance(state.variance, age_ms)
        if effective_variance <= 0:
            rows.append(
                f"{exchange:<11} px={float(state.fair_value):>9.2f} "
                f"age={age_ms:>4}ms var={float(state.variance):>8.2e} dropped"
            )
            continue

        weight = Decimal("1") / effective_variance
        weighted.append((exchange, weight, age_ms, effective_variance))

    weight_sum = sum((weight for _, weight, _, _ in weighted), start=Decimal("0"))
    weight_by_exchange = {
        exchange: (weight / weight_sum if weight_sum > 0 else Decimal("0"))
        for exchange, weight, _, _ in weighted
    }

    for exchange in ("coinbase", "hyperliquid", "okx", "kraken"):
        state = venue_states.get(exchange)
        if state is None:
            continue
        age_ms = max(now_ms - state.timestamp_ms, 0)
        effective_variance = combiner._effective_variance(state.variance, age_ms)
        norm_weight = weight_by_exchange.get(exchange)
        if norm_weight is None:
            continue
        rows.append(
            f"{exchange:<11} px={float(state.fair_value):>9.2f} age={age_ms:>4}ms "
            f"var={float(state.variance):>8.2e} eff={float(effective_variance):>8.2e} "
            f"w={float(norm_weight):>6.1%}"
        )

    return tuple(rows)


async def stream_plot_points(
    args: argparse.Namespace,
    output_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> None:
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

    try:
        while not stop_event.is_set():
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
                kalman_result.filtered_price if kalman_result is not None else None
            )

            coinbase_state = latest_venue_states.get("coinbase")
            hyperliquid_state = latest_venue_states.get("hyperliquid")
            okx_state = latest_venue_states.get("okx")
            kraken_state = latest_venue_states.get("kraken")
            output_queue.put(
                PlotPoint(
                    timestamp_ms=combined.timestamp_ms,
                    composite_efficient_price=float(combined.fair_value),
                    filtered_efficient_price=filtered_price,
                    coinbase_fair_value=(
                        float(coinbase_state.fair_value) if coinbase_state is not None else None
                    ),
                    hyperliquid_fair_value=(
                        float(hyperliquid_state.fair_value)
                        if hyperliquid_state is not None
                        else None
                    ),
                    okx_fair_value=(
                        float(okx_state.fair_value) if okx_state is not None else None
                    ),
                    kraken_fair_value=(
                        float(kraken_state.fair_value) if kraken_state is not None else None
                    ),
                    diagnostics_lines=build_diagnostics_text(
                        venue_states=latest_venue_states,
                        combiner=combiner,
                        now_ms=now_ms,
                    ),
                )
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket in sockets:
            await socket.close()


class LiveMultiVenuePlot:
    COMBINED_COLOR = "#ff8c42"
    FILTERED_COLOR = "#ffd700"
    COINBASE_COLOR = "#4ea1ff"
    HYPERLIQUID_COLOR = "#7bd389"
    OKX_COLOR = "#e06bff"
    KRAKEN_COLOR = "#f7b733"

    def __init__(
        self,
        *,
        root: tk.Tk,
        title: str,
        width: int,
        height: int,
        max_points: int,
        data_queue: queue.Queue[PlotPoint],
        stop_event: threading.Event,
    ) -> None:
        self.root = root
        self.root.title(title)
        self.root.configure(bg="#111111")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.width = width
        self.height = height
        self.max_points = max_points
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.points: deque[PlotPoint] = deque(maxlen=max_points)

        self.status_var = tk.StringVar(value="Waiting for multi-venue data...")
        self.status_label = tk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
            bg="#111111",
            fg="#dddddd",
            font=("Helvetica", 12),
            padx=12,
            pady=8,
        )
        self.status_label.pack(fill="x")

        self.canvas = tk.Canvas(
            root,
            width=width,
            height=height,
            bg="#181818",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.after(100, self.refresh)

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def refresh(self) -> None:
        while True:
            try:
                self.points.append(self.data_queue.get_nowait())
            except queue.Empty:
                break

        self._draw()
        if not self.stop_event.is_set():
            self.root.after(100, self.refresh)

    def _draw(self) -> None:
        self.canvas.delete("all")
        if len(self.points) < 2:
            self.canvas.create_text(
                self.width / 2,
                self.height / 2,
                text="Waiting for enough points to draw...",
                fill="#999999",
                font=("Helvetica", 16),
            )
            return

        all_values: list[float] = []
        for point in self.points:
            all_values.append(point.composite_efficient_price)
            if point.filtered_efficient_price is not None:
                all_values.append(point.filtered_efficient_price)
            if point.coinbase_fair_value is not None:
                all_values.append(point.coinbase_fair_value)
            if point.hyperliquid_fair_value is not None:
                all_values.append(point.hyperliquid_fair_value)
            if point.okx_fair_value is not None:
                all_values.append(point.okx_fair_value)
            if point.kraken_fair_value is not None:
                all_values.append(point.kraken_fair_value)

        min_y = min(all_values)
        max_y = max(all_values)
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0

        pad_left = 70
        pad_right = 20
        pad_top = 126
        pad_bottom = 50
        plot_width = self.width - pad_left - pad_right
        plot_height = self.height - pad_top - pad_bottom

        def x_at(index: int) -> float:
            return pad_left + (plot_width * index / max(len(self.points) - 1, 1))

        def y_at(value: float) -> float:
            scale = (value - min_y) / (max_y - min_y)
            return pad_top + plot_height - (scale * plot_height)

        for i in range(5):
            y = pad_top + plot_height * i / 4
            self.canvas.create_line(pad_left, y, pad_left + plot_width, y, fill="#2a2a2a")
            label_value = max_y - (max_y - min_y) * i / 4
            self.canvas.create_text(
                pad_left - 10,
                y,
                text=f"{label_value:.2f}",
                fill="#bbbbbb",
                anchor="e",
                font=("Helvetica", 10),
            )

        self.canvas.create_line(
            pad_left,
            pad_top + plot_height,
            pad_left + plot_width,
            pad_top + plot_height,
            fill="#444444",
        )
        self.canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_height, fill="#444444")

        combined_coords: list[float] = []
        filtered_coords: list[float] = []
        coinbase_coords: list[float] = []
        hyperliquid_coords: list[float] = []
        okx_coords: list[float] = []
        kraken_coords: list[float] = []
        for index, point in enumerate(self.points):
            x = x_at(index)
            combined_coords.extend((x, y_at(point.composite_efficient_price)))
            if point.filtered_efficient_price is not None:
                filtered_coords.extend((x, y_at(point.filtered_efficient_price)))
            if point.coinbase_fair_value is not None:
                coinbase_coords.extend((x, y_at(point.coinbase_fair_value)))
            if point.hyperliquid_fair_value is not None:
                hyperliquid_coords.extend((x, y_at(point.hyperliquid_fair_value)))
            if point.okx_fair_value is not None:
                okx_coords.extend((x, y_at(point.okx_fair_value)))
            if point.kraken_fair_value is not None:
                kraken_coords.extend((x, y_at(point.kraken_fair_value)))

        self.canvas.create_line(
            *combined_coords,
            fill=self.COMBINED_COLOR,
            width=2,
            smooth=False,
            dash=(6, 3),
        )
        if len(filtered_coords) >= 4:
            self.canvas.create_line(
                *filtered_coords,
                fill=self.FILTERED_COLOR,
                width=3,
                smooth=False,
            )
        if len(coinbase_coords) >= 4:
            self.canvas.create_line(
                *coinbase_coords,
                fill=self.COINBASE_COLOR,
                width=2,
                smooth=False,
            )
        if len(hyperliquid_coords) >= 4:
            self.canvas.create_line(
                *hyperliquid_coords,
                fill=self.HYPERLIQUID_COLOR,
                width=2,
                smooth=False,
            )
        if len(okx_coords) >= 4:
            self.canvas.create_line(
                *okx_coords,
                fill=self.OKX_COLOR,
                width=2,
                smooth=False,
            )
        if len(kraken_coords) >= 4:
            self.canvas.create_line(
                *kraken_coords,
                fill=self.KRAKEN_COLOR,
                width=2,
                smooth=False,
            )

        latest = self.points[-1]
        status_parts = [f"Composite: {latest.composite_efficient_price:.2f}"]
        if latest.filtered_efficient_price is not None:
            status_parts.append(f"Filtered: {latest.filtered_efficient_price:.2f}")
        if latest.coinbase_fair_value is not None:
            status_parts.append(f"Coinbase: {latest.coinbase_fair_value:.2f}")
        if latest.hyperliquid_fair_value is not None:
            status_parts.append(f"Hyperliquid: {latest.hyperliquid_fair_value:.2f}")
        if latest.okx_fair_value is not None:
            status_parts.append(f"OKX: {latest.okx_fair_value:.2f}")
        if latest.kraken_fair_value is not None:
            status_parts.append(f"Kraken: {latest.kraken_fair_value:.2f}")
        self.status_var.set("    ".join(status_parts))

        legend_y = pad_top + 16
        legend_items: list[tuple[str, str]] = [
            (self.FILTERED_COLOR, "Filtered"),
            (self.COMBINED_COLOR, "Raw Composite"),
            (self.COINBASE_COLOR, "Coinbase"),
            (self.HYPERLIQUID_COLOR, "Hyperliquid"),
            (self.OKX_COLOR, "OKX"),
            (self.KRAKEN_COLOR, "Kraken"),
        ]
        legend_x = pad_left
        for color, label in legend_items:
            self.canvas.create_rectangle(
                legend_x, legend_y - 6, legend_x + 18, legend_y + 6,
                fill=color, outline="",
            )
            self.canvas.create_text(
                legend_x + 26, legend_y, text=label,
                fill="#dddddd", anchor="w", font=("Helvetica", 11, "bold"),
            )
            legend_x += 26 + len(label) * 9 + 20
        panel_x0 = pad_left
        panel_y0 = 42
        panel_x1 = self.width - pad_right
        panel_y1 = 114
        self.canvas.create_rectangle(
            panel_x0,
            panel_y0,
            panel_x1,
            panel_y1,
            fill="#141414",
            outline="#2a2a2a",
        )
        self.canvas.create_text(
            panel_x0 + 10,
            panel_y0 + 10,
            text="Diagnostics",
            fill="#dddddd",
            anchor="nw",
            font=("Helvetica", 11, "bold"),
        )
        for index, line in enumerate(latest.diagnostics_lines):
            self.canvas.create_text(
                panel_x0 + 10,
                panel_y0 + 28 + index * 16,
                text=line,
                fill="#bbbbbb",
                anchor="nw",
                font=("Courier", 11),
            )
        self.canvas.create_text(
            pad_left,
            self.height - 20,
            text="Live plot of composite multi-venue efficient price and venue-local fair values",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11),
        )


def start_stream_thread(
    *,
    args: argparse.Namespace,
    data_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> threading.Thread:
    def runner() -> None:
        asyncio.run(stream_plot_points(args, data_queue, stop_event))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_queue: queue.Queue[PlotPoint] = queue.Queue()
    stop_event = threading.Event()

    try:
        start_stream_thread(args=args, data_queue=data_queue, stop_event=stop_event)
    except ValueError as exc:
        parser.error(str(exc))

    root = tk.Tk()
    root.geometry(f"{args.width}x{args.height}")
    LiveMultiVenuePlot(
        root=root,
        title=f"Multi-Venue Efficient Price: {args.asset}",
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        data_queue=data_queue,
        stop_event=stop_event,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
