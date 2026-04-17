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
class PlotPoint:
    timestamp_ms: int
    combined_fair_value: float
    coinbase_fair_value: float | None
    hyperliquid_fair_value: float | None


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


async def stream_plot_points(
    args: argparse.Namespace,
    output_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> None:
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

    try:
        while not stop_event.is_set():
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
            output_queue.put(
                PlotPoint(
                    timestamp_ms=combined.timestamp_ms,
                    combined_fair_value=float(combined.fair_value),
                    coinbase_fair_value=(
                        float(coinbase_state.fair_value) if coinbase_state is not None else None
                    ),
                    hyperliquid_fair_value=(
                        float(hyperliquid_state.fair_value)
                        if hyperliquid_state is not None
                        else None
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
    COINBASE_COLOR = "#4ea1ff"
    HYPERLIQUID_COLOR = "#7bd389"

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
            all_values.append(point.combined_fair_value)
            if point.coinbase_fair_value is not None:
                all_values.append(point.coinbase_fair_value)
            if point.hyperliquid_fair_value is not None:
                all_values.append(point.hyperliquid_fair_value)

        min_y = min(all_values)
        max_y = max(all_values)
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0

        pad_left = 70
        pad_right = 20
        pad_top = 20
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
        coinbase_coords: list[float] = []
        hyperliquid_coords: list[float] = []
        for index, point in enumerate(self.points):
            x = x_at(index)
            combined_coords.extend((x, y_at(point.combined_fair_value)))
            if point.coinbase_fair_value is not None:
                coinbase_coords.extend((x, y_at(point.coinbase_fair_value)))
            if point.hyperliquid_fair_value is not None:
                hyperliquid_coords.extend((x, y_at(point.hyperliquid_fair_value)))

        self.canvas.create_line(
            *combined_coords,
            fill=self.COMBINED_COLOR,
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

        latest = self.points[-1]
        self.status_var.set(
            "Combined: "
            f"{latest.combined_fair_value:.2f}    "
            "Coinbase: "
            f"{latest.coinbase_fair_value:.2f}    "
            if latest.coinbase_fair_value is not None
            else "Combined only    "
            + (
                "Hyperliquid: "
                f"{latest.hyperliquid_fair_value:.2f}"
                if latest.hyperliquid_fair_value is not None
                else ""
            )
        )

        legend_y = pad_top + 16
        self.canvas.create_rectangle(
            pad_left,
            legend_y - 6,
            pad_left + 18,
            legend_y + 6,
            fill=self.COMBINED_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 26,
            legend_y,
            text="Combined",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_rectangle(
            pad_left + 140,
            legend_y - 6,
            pad_left + 158,
            legend_y + 6,
            fill=self.COINBASE_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 166,
            legend_y,
            text="Coinbase",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_rectangle(
            pad_left + 280,
            legend_y - 6,
            pad_left + 298,
            legend_y + 6,
            fill=self.HYPERLIQUID_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 306,
            legend_y,
            text="Hyperliquid",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_text(
            pad_left,
            self.height - 20,
            text="Live plot of combined multi-venue efficient price and venue-local fair values",
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
