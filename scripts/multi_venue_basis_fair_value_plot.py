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
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multi_venue_basis_fair_value import (  # noqa: E402
    build_basis_filter_config,
    build_filter,
    build_parser,
    build_venue_state,
    stream_socket_observations,
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
from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (  # noqa: E402
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)
from icarus.strategy.fair_value.types import VenueFairValueState  # noqa: E402
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin  # noqa: E402

if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


class PlotPoint:
    def __init__(
        self,
        *,
        timestamp_ms: int,
        composite_efficient_price: float,
        basis_filtered_common_price: float | None,
        basis_estimate_is_live: bool,
        coinbase_fair_value: float | None,
        hyperliquid_fair_value: float | None,
        okx_fair_value: float | None,
        kraken_fair_value: float | None,
        diagnostics_lines: tuple[str, ...],
    ) -> None:
        self.timestamp_ms = timestamp_ms
        self.composite_efficient_price = composite_efficient_price
        self.basis_filtered_common_price = basis_filtered_common_price
        self.basis_estimate_is_live = basis_estimate_is_live
        self.coinbase_fair_value = coinbase_fair_value
        self.hyperliquid_fair_value = hyperliquid_fair_value
        self.okx_fair_value = okx_fair_value
        self.kraken_fair_value = kraken_fair_value
        self.diagnostics_lines = diagnostics_lines


def build_diagnostics_text(
    *,
    venue_states: dict[str, VenueFairValueState],
    basis_common_price: float | None,
    basis_estimates: dict[str, float],
    basis_stddevs: dict[str, float],
    anchor_exchange: str,
    basis_estimate_is_live: bool,
) -> tuple[str, ...]:
    rows: list[str] = []
    if basis_common_price is not None:
        status = "live" if basis_estimate_is_live else "held"
        rows.append(
            f"common={basis_common_price:>10.2f} anchor={anchor_exchange} status={status}"
        )
    for exchange in ("coinbase", "hyperliquid", "okx", "kraken"):
        state = venue_states.get(exchange)
        basis = basis_estimates.get(exchange)
        basis_std = basis_stddevs.get(exchange)
        if state is None:
            rows.append(f"{exchange:<11} px=-- basis=-- std=--")
            continue
        basis_text = "--" if basis is None else f"{basis:+7.2f}"
        std_text = "--" if basis_std is None else f"{basis_std:>6.2f}"
        rows.append(
            f"{exchange:<11} px={float(state.fair_value):>9.2f} basis={basis_text} std={std_text}"
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
    filters: dict[tuple[str, str], object | None] = {}
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

            basis_result = basis_filter.update(
                timestamp_s=now_ms / 1000.0,
                observations=[
                    VenueBasisObservation(
                        name=state.exchange,
                        fair_value=state.fair_value,
                        local_variance=state.variance,
                        age_ms=float(max(now_ms - state.timestamp_ms, 0)),
                    )
                    for state in latest_venue_states.values()
                ],
            )
            if basis_result is not None:
                last_basis_result = basis_result

            basis_common_price = (
                last_basis_result.common_price if last_basis_result is not None else None
            )
            basis_estimate_is_live = basis_result is not None
            basis_estimates = (
                last_basis_result.basis_estimates if last_basis_result is not None else {}
            )
            basis_stddevs = (
                last_basis_result.basis_stddevs if last_basis_result is not None else {}
            )

            coinbase_state = latest_venue_states.get("coinbase")
            hyperliquid_state = latest_venue_states.get("hyperliquid")
            okx_state = latest_venue_states.get("okx")
            kraken_state = latest_venue_states.get("kraken")
            output_queue.put(
                PlotPoint(
                    timestamp_ms=combined.timestamp_ms,
                    composite_efficient_price=float(combined.fair_value),
                    basis_filtered_common_price=basis_common_price,
                    basis_estimate_is_live=basis_estimate_is_live,
                    coinbase_fair_value=(
                        float(coinbase_state.fair_value) if coinbase_state is not None else None
                    ),
                    hyperliquid_fair_value=(
                        float(hyperliquid_state.fair_value) if hyperliquid_state is not None else None
                    ),
                    okx_fair_value=float(okx_state.fair_value) if okx_state is not None else None,
                    kraken_fair_value=(
                        float(kraken_state.fair_value) if kraken_state is not None else None
                    ),
                    diagnostics_lines=build_diagnostics_text(
                        venue_states=latest_venue_states,
                        basis_common_price=basis_common_price,
                        basis_estimates=basis_estimates,
                        basis_stddevs=basis_stddevs,
                        anchor_exchange=args.basis_anchor_exchange,
                        basis_estimate_is_live=basis_estimate_is_live,
                    ),
                )
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket in sockets:
            await socket.close()


class LiveBasisPlot:
    COMBINED_COLOR = "#ff8c42"
    BASIS_COLOR = "#ffd700"
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

        self.status_var = tk.StringVar(value="Waiting for basis-filter data...")
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
            if point.basis_filtered_common_price is not None:
                all_values.append(point.basis_filtered_common_price)
            for value in (
                point.coinbase_fair_value,
                point.hyperliquid_fair_value,
                point.okx_fair_value,
                point.kraken_fair_value,
            ):
                if value is not None:
                    all_values.append(value)

        min_y = min(all_values)
        max_y = max(all_values)
        if min_y == max_y:
            min_y -= 1.0
            max_y += 1.0

        latest = self.points[-1]

        pad_left = 70
        pad_right = 20
        pad_bottom = 50

        panel_x0 = pad_left
        panel_y0 = 42
        panel_title_height = 24
        panel_line_height = 14
        panel_inner_pad = 10
        panel_y1 = (
            panel_y0
            + panel_inner_pad
            + panel_title_height
            + len(latest.diagnostics_lines) * panel_line_height
            + panel_inner_pad
        )
        legend_y = panel_y1 + 22
        pad_top = legend_y + 26

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

        series = {
            "combined": [],
            "basis": [],
            "coinbase": [],
            "hyperliquid": [],
            "okx": [],
            "kraken": [],
        }

        for index, point in enumerate(self.points):
            x = x_at(index)
            series["combined"].extend((x, y_at(point.composite_efficient_price)))
            if point.basis_filtered_common_price is not None:
                series["basis"].extend((x, y_at(point.basis_filtered_common_price)))
            if point.coinbase_fair_value is not None:
                series["coinbase"].extend((x, y_at(point.coinbase_fair_value)))
            if point.hyperliquid_fair_value is not None:
                series["hyperliquid"].extend((x, y_at(point.hyperliquid_fair_value)))
            if point.okx_fair_value is not None:
                series["okx"].extend((x, y_at(point.okx_fair_value)))
            if point.kraken_fair_value is not None:
                series["kraken"].extend((x, y_at(point.kraken_fair_value)))

        self.canvas.create_line(*series["combined"], fill=self.COMBINED_COLOR, width=2, dash=(6, 3))
        if len(series["basis"]) >= 4:
            self.canvas.create_line(*series["basis"], fill=self.BASIS_COLOR, width=3)
        for key, color in (
            ("coinbase", self.COINBASE_COLOR),
            ("hyperliquid", self.HYPERLIQUID_COLOR),
            ("okx", self.OKX_COLOR),
            ("kraken", self.KRAKEN_COLOR),
        ):
            if len(series[key]) >= 4:
                self.canvas.create_line(*series[key], fill=color, width=2)

        status_parts = [f"Composite: {latest.composite_efficient_price:.2f}"]
        if latest.basis_filtered_common_price is not None:
            suffix = "" if latest.basis_estimate_is_live else " (held)"
            status_parts.append(f"Basis Common: {latest.basis_filtered_common_price:.2f}{suffix}")
        if latest.coinbase_fair_value is not None:
            status_parts.append(f"Coinbase: {latest.coinbase_fair_value:.2f}")
        if latest.kraken_fair_value is not None:
            status_parts.append(f"Kraken: {latest.kraken_fair_value:.2f}")
        self.status_var.set("    ".join(status_parts))

        legend_items: list[tuple[str, str]] = [
            (self.BASIS_COLOR, "Basis Common"),
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

        panel_x1 = self.width - pad_right
        self.canvas.create_rectangle(
            panel_x0, panel_y0, panel_x1, panel_y1,
            fill="#141414", outline="#2a2a2a",
        )
        self.canvas.create_text(
            panel_x0 + 10, panel_y0 + 10,
            text="Basis Diagnostics",
            fill="#dddddd", anchor="nw", font=("Helvetica", 11, "bold"),
        )
        for index, line in enumerate(latest.diagnostics_lines):
            self.canvas.create_text(
                panel_x0 + 10,
                panel_y0 + panel_inner_pad + panel_title_height + index * panel_line_height,
                text=line,
                fill="#bbbbbb",
                anchor="nw",
                font=("Courier", 11),
            )
        self.canvas.create_text(
            pad_left,
            self.height - 20,
            text="Live plot of raw composite, basis-filter common price, and venue fair values",
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
    parser.description = "Plot the experimental venue-basis fair-value filter live."
    parser.add_argument("--max-points", type=int, default=300, help="Maximum points to keep.")
    parser.add_argument("--width", type=int, default=1100, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=700, help="Window height in pixels.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_queue: queue.Queue[PlotPoint] = queue.Queue()
    stop_event = threading.Event()
    start_stream_thread(args=args, data_queue=data_queue, stop_event=stop_event)

    root = tk.Tk()
    root.geometry(f"{args.width}x{args.height}")
    LiveBasisPlot(
        root=root,
        title=f"Venue-Basis Fair Value: {args.asset}",
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        data_queue=data_queue,
        stop_event=stop_event,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
