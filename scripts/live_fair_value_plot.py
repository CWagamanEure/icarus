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

from icarus.measurements import MarketMeasurementEngine  # noqa: E402
from icarus.observations import Observation  # noqa: E402
from icarus.sockets.coinbase import CoinbaseSocket  # noqa: E402
from icarus.strategy.fair_value.estimator import RawFairValueEstimator  # noqa: E402
from icarus.strategy.fair_value.filters.ema import EMAFairValueFilter  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass(frozen=True, slots=True)
class PlotPoint:
    timestamp_ms: int
    quoted_midprice: float
    efficient_price: float
    baseline_ema_price: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Coinbase quoted midprice versus filtered efficient price live.",
    )
    parser.add_argument(
        "market",
        nargs="?",
        default="BTC-USD",
        help="Coinbase product id, e.g. BTC-USD.",
    )
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        help="Coinbase channel to subscribe to. Repeat to provide multiple channels.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Coinbase sandbox websocket.",
    )
    parser.add_argument(
        "--filter",
        choices=["none", "ema"],
        default="ema",
        help="Fair value filter to apply to the raw estimate.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA alpha when using --filter ema.",
    )
    parser.add_argument(
        "--baseline-ema-alpha",
        type=float,
        default=0.2,
        help="EMA alpha for the baseline reference line.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=300,
        help="Maximum number of plotted points to keep on screen.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1100,
        help="Window width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Window height in pixels.",
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
    alpha = Decimal(str(args.ema_alpha))
    return EMAFairValueFilter(alpha=alpha)


def observation_stream(socket: CoinbaseSocket) -> AsyncIterator[Observation]:
    async def stream() -> AsyncIterator[Observation]:
        async for observation in socket.stream_observations():
            yield observation

    return stream()


async def stream_plot_points(
    *,
    socket: CoinbaseSocket,
    fair_value_filter: EMAFairValueFilter | None,
    baseline_ema_filter: EMAFairValueFilter,
    output_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> None:
    measurement_engine = MarketMeasurementEngine(exchange="coinbase", market=socket.product_ids[0])
    estimator = RawFairValueEstimator()

    try:
        async for observation in observation_stream(socket):
            if stop_event.is_set():
                break

            measurement = measurement_engine.on_observation(observation)
            if measurement is None or measurement.midprice is None:
                continue

            raw_estimate = estimator.estimate(measurement)
            if raw_estimate.raw_fair_value is None or raw_estimate.measurement_variance is None:
                continue

            if fair_value_filter is not None:
                filtered_value, _ = fair_value_filter.update(
                    measurement=raw_estimate.raw_fair_value,
                    measurement_variance=raw_estimate.measurement_variance,
                    timestamp_ms=raw_estimate.timestamp_ms,
                )
            else:
                filtered_value = raw_estimate.raw_fair_value
            baseline_ema_value, _ = baseline_ema_filter.update(
                measurement=raw_estimate.raw_fair_value,
                measurement_variance=raw_estimate.measurement_variance,
                timestamp_ms=raw_estimate.timestamp_ms,
            )
            output_queue.put(
                PlotPoint(
                    timestamp_ms=raw_estimate.timestamp_ms,
                    quoted_midprice=float(measurement.midprice),
                    efficient_price=float(filtered_value),
                    baseline_ema_price=float(baseline_ema_value),
                )
            )
    finally:
        await socket.close()


class LiveFairValuePlot:
    QUOTED_MID_COLOR = "#4ea1ff"
    EFFICIENT_PRICE_COLOR = "#ff8c42"
    BASELINE_EMA_COLOR = "#7bd389"

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

        self.status_var = tk.StringVar(value="Waiting for Coinbase data...")
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

        quoted_values = [point.quoted_midprice for point in self.points]
        efficient_values = [point.efficient_price for point in self.points]
        baseline_ema_values = [point.baseline_ema_price for point in self.points]
        all_values = quoted_values + efficient_values + baseline_ema_values
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

        quoted_coords: list[float] = []
        efficient_coords: list[float] = []
        baseline_ema_coords: list[float] = []
        for index, point in enumerate(self.points):
            x = x_at(index)
            quoted_coords.extend((x, y_at(point.quoted_midprice)))
            efficient_coords.extend((x, y_at(point.efficient_price)))
            baseline_ema_coords.extend((x, y_at(point.baseline_ema_price)))

        self.canvas.create_line(
            *quoted_coords,
            fill=self.QUOTED_MID_COLOR,
            width=2,
            smooth=False,
        )
        self.canvas.create_line(
            *efficient_coords,
            fill=self.EFFICIENT_PRICE_COLOR,
            width=2,
            smooth=False,
        )
        self.canvas.create_line(
            *baseline_ema_coords,
            fill=self.BASELINE_EMA_COLOR,
            width=2,
            dash=(6, 4),
            smooth=False,
        )

        latest = self.points[-1]
        self.status_var.set(
            "Quoted mid: "
            f"{latest.quoted_midprice:.2f}    "
            "Efficient price: "
            f"{latest.efficient_price:.2f}    "
            "Baseline EMA: "
            f"{latest.baseline_ema_price:.2f}    "
            "Spread: "
            f"{(latest.efficient_price - latest.quoted_midprice):+.4f}"
        )
        legend_y = pad_top + 16
        self.canvas.create_rectangle(
            pad_left,
            legend_y - 6,
            pad_left + 18,
            legend_y + 6,
            fill=self.QUOTED_MID_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 26,
            legend_y,
            text="Quoted mid",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_rectangle(
            pad_left + 140,
            legend_y - 6,
            pad_left + 158,
            legend_y + 6,
            fill=self.EFFICIENT_PRICE_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 166,
            legend_y,
            text="Efficient price",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_rectangle(
            pad_left + 300,
            legend_y - 6,
            pad_left + 318,
            legend_y + 6,
            fill=self.BASELINE_EMA_COLOR,
            outline="",
        )
        self.canvas.create_text(
            pad_left + 326,
            legend_y,
            text="Baseline EMA",
            fill="#dddddd",
            anchor="w",
            font=("Helvetica", 11, "bold"),
        )
        self.canvas.create_text(
            pad_left,
            self.height - 20,
            text=(
                "Live plot of Coinbase mid, filtered efficient price, and EMA baseline"
            ),
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
    fair_value_filter = build_filter(args)
    baseline_ema_filter = EMAFairValueFilter(alpha=Decimal(str(args.baseline_ema_alpha)))
    channels = args.channels or ["ticker", "heartbeats", "level2"]
    socket = CoinbaseSocket(args.market, channels=channels, sandbox=args.sandbox)

    def runner() -> None:
        asyncio.run(
            stream_plot_points(
                socket=socket,
                fair_value_filter=fair_value_filter,
                baseline_ema_filter=baseline_ema_filter,
                output_queue=data_queue,
                stop_event=stop_event,
            )
        )

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
    LiveFairValuePlot(
        root=root,
        title=f"Efficient Price vs Coinbase Mid: {args.market}",
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        data_queue=data_queue,
        stop_event=stop_event,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
