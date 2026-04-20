#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Plot filter-vs-anchor deltas from a captured filter-eval SQLite DB.

Reads snapshots produced by scripts/capture_filter_eval.py and plots
    basis_common_price - anchor_fair_value
alongside the same delta for the raw composite and the 1D Kalman filter, so
you can eyeball whether the basis filter's common price is biased/lagged vs
the anchor venue across the captured run.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


@dataclass(frozen=True, slots=True)
class DeltaRow:
    timestamp_ms: int
    basis_delta: float | None
    kalman_delta: float | None
    composite_delta: float | None
    anchor_price: float | None


@dataclass(frozen=True, slots=True)
class SeriesStats:
    name: str
    mean: float
    median: float
    rmse: float
    stddev: float
    abs_max: float
    count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default="data/filter_eval.sqlite3",
        help="SQLite file produced by capture_filter_eval.py.",
    )
    parser.add_argument(
        "--anchor-exchange",
        default="coinbase",
        help="Venue whose fair value is subtracted from each filter output.",
    )
    parser.add_argument(
        "--max-venue-age-ms",
        type=float,
        default=750.0,
        help="Only use anchor rows whose captured age is <= this threshold.",
    )
    parser.add_argument(
        "--skip-warmup-rows",
        type=int,
        default=0,
        help="Drop the first N rows to exclude filter bootstrap transients.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Keep at most this many rows (evenly downsampled) for plotting.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Window (in plotted points) for the rolling mean overlay.",
    )
    parser.add_argument("--width", type=int, default=1300, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=780, help="Window height in pixels.")
    return parser


def load_rows(
    db_path: Path,
    *,
    anchor_exchange: str,
    max_venue_age_ms: float,
    skip_warmup_rows: int,
) -> list[DeltaRow]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            """
            SELECT
                u.id AS id,
                u.timestamp_ms AS timestamp_ms,
                u.composite_price AS composite_price,
                u.kalman_filtered_price AS kalman_filtered_price,
                u.basis_common_price AS basis_common_price,
                v.fair_value AS anchor_fair_value,
                v.age_ms AS anchor_age_ms
            FROM updates u
            LEFT JOIN venue_states v
                ON v.update_id = u.id AND v.exchange = ?
            ORDER BY u.timestamp_ms, u.id
            """,
            (anchor_exchange,),
        )
        out: list[DeltaRow] = []
        for row in cursor:
            anchor_price = row["anchor_fair_value"]
            anchor_age_ms = row["anchor_age_ms"]
            anchor_valid = (
                anchor_price is not None
                and anchor_age_ms is not None
                and float(anchor_age_ms) <= max_venue_age_ms
            )
            anchor_price_f = float(anchor_price) if anchor_valid else None

            def delta(column: str) -> float | None:
                value = row[column]
                if value is None or anchor_price_f is None:
                    return None
                return float(value) - anchor_price_f

            out.append(
                DeltaRow(
                    timestamp_ms=int(row["timestamp_ms"]),
                    basis_delta=delta("basis_common_price"),
                    kalman_delta=delta("kalman_filtered_price"),
                    composite_delta=delta("composite_price"),
                    anchor_price=anchor_price_f,
                )
            )
    finally:
        conn.close()

    if skip_warmup_rows > 0:
        out = out[skip_warmup_rows:]
    return out


def downsample(rows: list[DeltaRow], max_points: int) -> list[DeltaRow]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    step = len(rows) / max_points
    return [rows[min(int(i * step), len(rows) - 1)] for i in range(max_points)]


def series_stats(name: str, values: list[float]) -> SeriesStats:
    count = len(values)
    if count == 0:
        return SeriesStats(name=name, mean=0.0, median=0.0, rmse=0.0, stddev=0.0, abs_max=0.0, count=0)
    mean = sum(values) / count
    sorted_vals = sorted(values)
    median = (
        sorted_vals[count // 2]
        if count % 2 == 1
        else 0.5 * (sorted_vals[count // 2 - 1] + sorted_vals[count // 2])
    )
    rmse = (sum(v * v for v in values) / count) ** 0.5
    variance = sum((v - mean) ** 2 for v in values) / count
    stddev = variance**0.5
    abs_max = max(abs(v) for v in values)
    return SeriesStats(
        name=name,
        mean=mean,
        median=median,
        rmse=rmse,
        stddev=stddev,
        abs_max=abs_max,
        count=count,
    )


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return list(values)
    buf: deque[float] = deque(maxlen=window)
    out: list[float | None] = []
    for value in values:
        if value is not None:
            buf.append(value)
        if buf:
            out.append(sum(buf) / len(buf))
        else:
            out.append(None)
    return out


class DeltaPlot:
    BG = "#111111"
    PANEL_BG = "#181818"
    GRID = "#2a2a2a"
    AXIS = "#444444"
    ZERO = "#666666"
    TEXT = "#dddddd"
    SUBTEXT = "#bbbbbb"
    BASIS_COLOR = "#ffd700"
    KALMAN_COLOR = "#4ea1ff"
    COMPOSITE_COLOR = "#ff8c42"
    BASIS_ROLL_COLOR = "#ffae00"
    KALMAN_ROLL_COLOR = "#76baff"
    COMPOSITE_ROLL_COLOR = "#ff6a00"

    def __init__(
        self,
        *,
        root: tk.Tk,
        title: str,
        width: int,
        height: int,
        rows: list[DeltaRow],
        rolling_window: int,
        anchor_exchange: str,
    ) -> None:
        self.root = root
        self.root.title(title)
        self.root.configure(bg=self.BG)
        self.width = width
        self.height = height
        self.rows = rows
        self.rolling_window = rolling_window
        self.anchor_exchange = anchor_exchange

        self.status_var = tk.StringVar(value=self._status_text())
        tk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
            bg=self.BG,
            fg=self.TEXT,
            font=("Helvetica", 12),
            padx=12,
            pady=8,
        ).pack(fill="x")

        self.canvas = tk.Canvas(
            root,
            width=width,
            height=height,
            bg=self.PANEL_BG,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<Configure>", self._on_resize)
        self._draw()

    def _status_text(self) -> str:
        total = len(self.rows)
        first_ts = self.rows[0].timestamp_ms if total else 0
        last_ts = self.rows[-1].timestamp_ms if total else 0
        span_sec = (last_ts - first_ts) / 1000.0 if total else 0.0
        return (
            f"anchor={self.anchor_exchange}  rows={total}  span={span_sec:.1f}s  "
            f"rolling_window={self.rolling_window}"
        )

    def _on_resize(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.widget is self.root:
            self.width = max(event.width, 400)
            self.height = max(event.height - 40, 300)
            self.canvas.configure(width=self.width, height=self.height)
            self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        if not self.rows:
            self.canvas.create_text(
                self.width / 2,
                self.height / 2,
                text="No rows loaded from SQLite.",
                fill=self.SUBTEXT,
                font=("Helvetica", 16),
            )
            return

        basis_values = [row.basis_delta for row in self.rows]
        kalman_values = [row.kalman_delta for row in self.rows]
        composite_values = [row.composite_delta for row in self.rows]
        basis_roll = rolling_mean(basis_values, self.rolling_window)
        kalman_roll = rolling_mean(kalman_values, self.rolling_window)
        composite_roll = rolling_mean(composite_values, self.rolling_window)

        present = [v for v in basis_values + kalman_values + composite_values if v is not None]
        if not present:
            self.canvas.create_text(
                self.width / 2,
                self.height / 2,
                text="No anchor-aligned deltas available.",
                fill=self.SUBTEXT,
                font=("Helvetica", 16),
            )
            return
        min_y = min(present)
        max_y = max(present)
        # Symmetric-ish padding so zero sits inside the frame.
        span = max(max_y - min_y, 1e-6)
        min_y -= 0.05 * span
        max_y += 0.05 * span
        if min_y > 0:
            min_y = min(min_y, -0.05 * span)
        if max_y < 0:
            max_y = max(max_y, 0.05 * span)

        pad_left = 80
        pad_right = 340  # room for stats panel on right
        pad_top = 30
        pad_bottom = 48

        plot_width = max(self.width - pad_left - pad_right, 50)
        plot_height = max(self.height - pad_top - pad_bottom, 50)
        n = len(self.rows)

        def x_at(index: int) -> float:
            return pad_left + (plot_width * index / max(n - 1, 1))

        def y_at(value: float) -> float:
            scale = (value - min_y) / (max_y - min_y)
            return pad_top + plot_height - (scale * plot_height)

        # Grid + labels.
        for i in range(6):
            y = pad_top + plot_height * i / 5
            self.canvas.create_line(pad_left, y, pad_left + plot_width, y, fill=self.GRID)
            label_value = max_y - (max_y - min_y) * i / 5
            self.canvas.create_text(
                pad_left - 10,
                y,
                text=f"{label_value:+.2f}",
                fill=self.SUBTEXT,
                anchor="e",
                font=("Helvetica", 10),
            )
        # Axes + zero line.
        self.canvas.create_line(
            pad_left,
            pad_top + plot_height,
            pad_left + plot_width,
            pad_top + plot_height,
            fill=self.AXIS,
        )
        self.canvas.create_line(
            pad_left, pad_top, pad_left, pad_top + plot_height, fill=self.AXIS
        )
        zero_y = y_at(0.0)
        if pad_top <= zero_y <= pad_top + plot_height:
            self.canvas.create_line(
                pad_left, zero_y, pad_left + plot_width, zero_y, fill=self.ZERO, dash=(4, 4)
            )

        # Raw per-row series (thin).
        self._draw_series(basis_values, x_at, y_at, self.BASIS_COLOR, width=1)
        self._draw_series(kalman_values, x_at, y_at, self.KALMAN_COLOR, width=1)
        self._draw_series(composite_values, x_at, y_at, self.COMPOSITE_COLOR, width=1)
        # Rolling means (thicker, brighter).
        self._draw_series(basis_roll, x_at, y_at, self.BASIS_ROLL_COLOR, width=2)
        self._draw_series(kalman_roll, x_at, y_at, self.KALMAN_ROLL_COLOR, width=2)
        self._draw_series(composite_roll, x_at, y_at, self.COMPOSITE_ROLL_COLOR, width=2)

        # X-axis time tick labels (start / mid / end).
        for fraction in (0.0, 0.5, 1.0):
            index = int((n - 1) * fraction)
            ts_ms = self.rows[index].timestamp_ms
            start_ts = self.rows[0].timestamp_ms
            self.canvas.create_text(
                x_at(index),
                pad_top + plot_height + 16,
                text=f"t+{(ts_ms - start_ts) / 1000.0:.1f}s",
                fill=self.SUBTEXT,
                anchor="n",
                font=("Helvetica", 10),
            )

        # Legend under the plot.
        legend_y = pad_top + plot_height + 32
        legend_items = [
            (self.BASIS_ROLL_COLOR, "basis - anchor (roll)"),
            (self.BASIS_COLOR, "basis - anchor"),
            (self.KALMAN_ROLL_COLOR, "kalman_1d - anchor (roll)"),
            (self.KALMAN_COLOR, "kalman_1d - anchor"),
            (self.COMPOSITE_ROLL_COLOR, "composite - anchor (roll)"),
            (self.COMPOSITE_COLOR, "composite - anchor"),
        ]
        legend_x = pad_left
        for color, label in legend_items:
            self.canvas.create_rectangle(
                legend_x, legend_y - 5, legend_x + 14, legend_y + 5, fill=color, outline=""
            )
            self.canvas.create_text(
                legend_x + 20, legend_y, text=label, fill=self.TEXT, anchor="w",
                font=("Helvetica", 10),
            )
            legend_x += 20 + len(label) * 7 + 16

        # Stats panel.
        stats = [
            series_stats("basis", [v for v in basis_values if v is not None]),
            series_stats("kalman_1d", [v for v in kalman_values if v is not None]),
            series_stats("composite", [v for v in composite_values if v is not None]),
        ]
        self._draw_stats(stats, pad_left + plot_width + 20, pad_top)

    def _draw_series(
        self,
        values: list[float | None],
        x_at,  # type: ignore[no-untyped-def]
        y_at,  # type: ignore[no-untyped-def]
        color: str,
        *,
        width: int,
    ) -> None:
        segment: list[float] = []
        for index, value in enumerate(values):
            if value is None:
                if len(segment) >= 4:
                    self.canvas.create_line(*segment, fill=color, width=width)
                segment = []
                continue
            segment.extend((x_at(index), y_at(value)))
        if len(segment) >= 4:
            self.canvas.create_line(*segment, fill=color, width=width)

    def _draw_stats(self, stats: list[SeriesStats], x: float, y: float) -> None:
        panel_width = 310
        line_height = 16
        title_height = 22
        inner_pad = 10
        panel_height = inner_pad + title_height + (len(stats) * 6) * line_height + inner_pad
        self.canvas.create_rectangle(
            x, y, x + panel_width, y + panel_height, fill="#141414", outline=self.GRID
        )
        self.canvas.create_text(
            x + inner_pad,
            y + inner_pad,
            text=f"Delta vs {self.anchor_exchange} (all deltas in price units)",
            fill=self.TEXT,
            anchor="nw",
            font=("Helvetica", 11, "bold"),
        )
        cursor_y = y + inner_pad + title_height
        for stat in stats:
            self.canvas.create_text(
                x + inner_pad,
                cursor_y,
                text=f"{stat.name}  n={stat.count}",
                fill=self.TEXT,
                anchor="nw",
                font=("Helvetica", 11, "bold"),
            )
            cursor_y += line_height
            for label, value in (
                ("mean", stat.mean),
                ("median", stat.median),
                ("rmse", stat.rmse),
                ("stddev", stat.stddev),
                ("|max|", stat.abs_max),
            ):
                self.canvas.create_text(
                    x + inner_pad + 14,
                    cursor_y,
                    text=f"{label:<7}{value:>+12.4f}",
                    fill=self.SUBTEXT,
                    anchor="nw",
                    font=("Courier", 11),
                )
                cursor_y += line_height


def main() -> None:
    args = build_parser().parse_args()
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    rows = load_rows(
        db_path,
        anchor_exchange=args.anchor_exchange,
        max_venue_age_ms=args.max_venue_age_ms,
        skip_warmup_rows=args.skip_warmup_rows,
    )
    rows = downsample(rows, args.max_points)

    root = tk.Tk()
    root.geometry(f"{args.width}x{args.height}")
    DeltaPlot(
        root=root,
        title=f"Basis vs {args.anchor_exchange}: {db_path.name}",
        width=args.width,
        height=args.height,
        rows=rows,
        rolling_window=max(args.rolling_window, 1),
        anchor_exchange=args.anchor_exchange,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
