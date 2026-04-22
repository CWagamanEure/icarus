#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Live plot: per-venue price, my basis-reconstructed fair value, and drift prediction.

The drift Kalman (state = regression coefficient vector β) must be calibrated before
going live. We do that by loading a historical SQLite capture and calling the same
calibrate_predictor() used in simulate_basis_maker.py. After calibration the script
connects to live sockets, emits drift_hat per venue on every tick, and feeds the
realized 500ms-later drift back into the filter so β keeps adapting in real time.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import sqlite3
import sys
import threading
import tkinter as tk
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multi_venue_basis_fair_value import (
    apply_asset_defaults,
    build_basis_filter_config,
    build_filter,
    build_parser as build_basis_parser,
    build_venue_state,
    stream_socket_observations,
)
from icarus.measurements import MarketMeasurementEngine
from icarus.observations import Observation, TradeObservation
from icarus.sockets.base import BaseSocket
from icarus.sockets.coinbase import CoinbaseSocket
from icarus.sockets.hyperliquid import HyperliquidSocket
from icarus.sockets.kraken import KrakenSocket
from icarus.sockets.okx import OkxSocket
from icarus.strategy.fair_value.combiner import (
    CrossVenueCombinerConfig,
    CrossVenueFairValueCombiner,
)
from icarus.strategy.fair_value.estimator import RawFairValueEstimator
from _drift_predictor import DriftPredictor, FEATURE_NAMES  # noqa: F401
from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin


VENUES = ("coinbase", "hyperliquid", "okx", "kraken")


@dataclass
class TradeFlowTracker:
    window_ms: int = 1000

    def __post_init__(self) -> None:
        self._events: deque[tuple[int, float, float]] = deque()

    def add(self, ts_ms: int, side: str, size: float) -> None:
        buy = size if side == "buy" else 0.0
        sell = size if side == "sell" else 0.0
        self._events.append((ts_ms, buy, sell))

    def snapshot(self, now_ms: int) -> tuple[float, float]:
        cutoff = now_ms - self.window_ms
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        buy = sum(e[1] for e in self._events)
        sell = sum(e[2] for e in self._events)
        return (buy - sell, buy + sell)


def _estimate_q_windowed(
    xi: np.ndarray,
    y: np.ndarray,
    ts_ms: list[int],
    window_sec: int,
    floor: float,
    min_window_obs: int = 100,
) -> np.ndarray | None:
    """Fit OLS β inside non-overlapping time windows, then Q = var(Δβ) / updates_per_window.

    Self-consistent: no dependence on a seed Q, unlike a Kalman-walk-based estimate.
    """
    window_betas: list[np.ndarray] = []
    t0 = ts_ms[0]
    w_xi: list[np.ndarray] = []
    w_y: list[float] = []
    for t, xi_row, y_val in zip(ts_ms, xi, y):
        if t - t0 >= window_sec * 1000 and len(w_y) >= min_window_obs:
            b, *_ = np.linalg.lstsq(np.asarray(w_xi), np.asarray(w_y), rcond=None)
            window_betas.append(b)
            w_xi, w_y = [], []
            t0 = t
        w_xi.append(xi_row)
        w_y.append(float(y_val))
    if len(window_betas) < 3:
        return None
    deltas = np.diff(np.asarray(window_betas), axis=0)
    updates_per_window = len(y) / len(window_betas)
    return np.maximum(np.var(deltas, axis=0) / updates_per_window, floor)


def calibrate_predictors_from_db(
    db_path: Path,
    horizon_ms: int,
    q_floor: float,
    q_window_sec: int,
    min_oos_r2: float,
    min_oos_sign_accuracy: float,
    train_fraction: float = 0.8,
) -> dict[str, DriftPredictor]:
    """Fit a drift predictor per venue with OOS validation and windowed-OLS Q estimation.

    Train/test split is temporal (first train_fraction of samples are train). Standardization
    and β are fit on train only so OOS metrics describe exactly the predictor we ship.
    Venues that fail the OOS gate are dropped.
    """
    conn = sqlite3.connect(str(db_path))
    query = """
        SELECT u.timestamp_ms, v.exchange, v.fair_value, v.microprice,
               v.depth_imbalance, v.mid_volatility_bps,
               COALESCE(v.trade_net_flow, 0.0),
               COALESCE(v.trade_buy_size, 0.0) + COALESCE(v.trade_sell_size, 0.0)
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        WHERE v.microprice IS NOT NULL
          AND v.depth_imbalance IS NOT NULL
          AND v.mid_volatility_bps IS NOT NULL
        ORDER BY v.exchange, u.timestamp_ms, u.id
    """
    per_venue: dict[str, list[tuple[int, float, tuple[float, ...]]]] = defaultdict(list)
    for ts, exchange, mid, microprice, imb, vol, net_flow, total_size in conn.execute(query):
        per_venue[exchange].append(
            (
                ts,
                float(mid),
                (
                    float(microprice) - float(mid),
                    float(imb),
                    float(vol),
                    float(net_flow),
                    float(total_size),
                ),
            )
        )
    conn.close()

    predictors: dict[str, DriftPredictor] = {}
    for venue, rows in per_venue.items():
        if len(rows) < 500:
            print(f"  {venue}: SKIP  only {len(rows)} raw rows")
            continue
        ts_list = [r[0] for r in rows]
        mids = [r[1] for r in rows]
        feats = [r[2] for r in rows]
        xs: list[tuple[float, ...]] = []
        ys: list[float] = []
        ts_of_sample: list[int] = []
        for i in range(len(rows)):
            target_ts = ts_list[i] + horizon_ms
            j = bisect_left(ts_list, target_ts, lo=i + 1)
            if j >= len(rows):
                break
            if ts_list[j] - target_ts > horizon_ms:
                continue
            xs.append(feats[i])
            ys.append(mids[j] - mids[i])
            ts_of_sample.append(ts_list[i])
        if len(ys) < 500:
            print(f"  {venue}: SKIP  only {len(ys)} valid (feature, target) pairs")
            continue

        x = np.asarray(xs)
        y = np.asarray(ys)
        split = int(len(y) * train_fraction)

        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        ts_train = ts_of_sample[:split]

        mean = x_train.mean(axis=0)
        std = np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0)
        xi_train = np.hstack([np.ones((len(y_train), 1)), (x_train - mean) / std])
        xi_test = np.hstack([np.ones((len(y_test), 1)), (x_test - mean) / std])

        beta0, *_ = np.linalg.lstsq(xi_train, y_train, rcond=None)
        r = float(np.var(y_train - xi_train @ beta0)) or 1.0

        y_hat_test = xi_test @ beta0
        ss_res = float(((y_test - y_hat_test) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mag_cut = np.quantile(np.abs(y_hat_test), 0.5)
        sign_mask = np.abs(y_hat_test) >= mag_cut
        sign_acc = (
            float((np.sign(y_hat_test[sign_mask]) == np.sign(y_test[sign_mask])).mean())
            if sign_mask.sum() > 0
            else float("nan")
        )

        if not (r2_oos >= min_oos_r2 and sign_acc >= min_oos_sign_accuracy):
            print(
                f"  {venue}: FAIL  r2_oos={r2_oos:+.4f}  sign_acc={sign_acc:.3f}"
                f"  n_train={len(y_train)}  n_test={len(y_test)}  — venue dropped"
            )
            continue

        q_vec = _estimate_q_windowed(
            xi_train, y_train, ts_train,
            window_sec=q_window_sec, floor=q_floor,
        )
        if q_vec is None:
            print(
                f"  {venue}: SKIP  not enough windows for Q"
                f" (need ≥3 of ≥100 obs within {q_window_sec}s each)"
            )
            continue

        print(
            f"  {venue}: PASS  r2_oos={r2_oos:+.4f}  sign_acc={sign_acc:.3f}"
            f"  n_train={len(y_train)}  n_test={len(y_test)}  Q={q_vec.round(6)}"
        )

        predictors[venue] = DriftPredictor(
            beta=beta0.copy(),
            P=np.eye(xi_train.shape[1]),
            Q=np.diag(q_vec),
            r=r,
            mean=mean,
            std=std,
        )
    return predictors


@dataclass
class PlotPoint:
    timestamp_ms: int
    basis_common_price: float | None
    basis_is_live: bool
    # Per venue: venue mid, basis-reconstructed fair value, drift_hat.
    venue_mid: dict[str, float]
    venue_fair: dict[str, float]
    venue_drift: dict[str, float]
    diagnostics_lines: tuple[str, ...]


def build_diagnostics(
    venue_mid: dict[str, float],
    venue_fair: dict[str, float],
    venue_drift: dict[str, float],
    basis_common_price: float | None,
    is_live: bool,
) -> tuple[str, ...]:
    lines: list[str] = []
    if basis_common_price is not None:
        lines.append(
            f"common={basis_common_price:>10.2f}  "
            f"status={'live' if is_live else 'held'}"
        )
    lines.append(f"{'venue':<14} {'mid':>10} {'fair':>10} {'drift_hat':>10}")
    for venue in VENUES:
        mid = venue_mid.get(venue)
        fair = venue_fair.get(venue)
        drift = venue_drift.get(venue)
        mid_s = f"{mid:>10.2f}" if mid is not None else f"{'--':>10}"
        fair_s = f"{fair:>10.2f}" if fair is not None else f"{'--':>10}"
        drift_s = f"{drift:+10.3f}" if drift is not None else f"{'--':>10}"
        lines.append(f"{venue:<14} {mid_s} {fair_s} {drift_s}")
    return tuple(lines)


async def stream_plot_points(
    args: argparse.Namespace,
    predictors: dict[str, DriftPredictor],
    output_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> None:
    observation_queue: asyncio.Queue[Observation] = asyncio.Queue()
    socket_specs: list[tuple[BaseSocket, str | None, str | None]] = [
        (
            CoinbaseSocket(
                args.coinbase_market,
                channels=args.coinbase_channels
                or ["ticker", "heartbeats", "level2", "market_trades"],
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
        socket_specs.append(
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
    if not args.disable_okx:
        socket_specs.append((OkxSocket(args.okx_market), None, None))
    if not args.disable_kraken:
        socket_specs.append((KrakenSocket(args.kraken_market), None, None))

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
    basis_filter = VenueBasisKalmanFilter(config=build_basis_filter_config(args))
    combiner = CrossVenueFairValueCombiner(
        args.asset,
        config=CrossVenueCombinerConfig(
            stale_after_ms=args.stale_after_ms,
            age_penalty_per_second=args.age_penalty_per_second,
        ),
    )
    last_basis_result = None
    trade_flow_trackers: dict[str, TradeFlowTracker] = {}
    # Per venue: queue of (target_ts, xi_standardized, mid_at_t)
    pending_updates: dict[str, deque[tuple[int, np.ndarray, float]]] = {
        v: deque() for v in VENUES
    }

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

            if isinstance(observation, TradeObservation):
                tracker = trade_flow_trackers.get(observation.exchange)
                if tracker is None:
                    tracker = TradeFlowTracker()
                    trade_flow_trackers[observation.exchange] = tracker
                tracker.add(now_ms, observation.side, float(observation.size))
                continue

            engine_key = (observation.exchange, observation.market)
            engine = measurement_engines.get(engine_key)
            if engine is None:
                engine = MarketMeasurementEngine(
                    exchange=observation.exchange,
                    market=observation.market,
                )
                measurement_engines[engine_key] = engine
            engine.on_observation(observation)

            # Build venue states (for basis filter).
            latest_states = {}
            venue_features: dict[str, tuple[float, ...]] = {}
            venue_mids: dict[str, float] = {}
            for key, eng in measurement_engines.items():
                estimator = estimators.get(key)
                if estimator is None:
                    estimator = RawFairValueEstimator()
                    estimators[key] = estimator
                if key not in filters:
                    filters[key] = build_filter(args)
                vs = build_venue_state(
                    engine=eng,
                    estimator=estimator,
                    fair_value_filter=filters[key],
                    market=args.asset,
                    now_ms=now_ms,
                )
                if vs is None:
                    continue
                latest_states[eng.exchange] = vs
                m = eng.current_measurement(now_ms)
                if m is None or m.microprice is None or m.depth_imbalance is None or m.mid_volatility_bps is None:
                    continue
                mid = float(vs.fair_value)
                venue_mids[eng.exchange] = mid
                tracker = trade_flow_trackers.get(eng.exchange)
                net_flow, total_size = tracker.snapshot(now_ms) if tracker else (0.0, 0.0)
                venue_features[eng.exchange] = (
                    float(m.microprice) - mid,
                    float(m.depth_imbalance),
                    float(m.mid_volatility_bps),
                    net_flow,
                    total_size,
                )

            if not latest_states:
                continue

            for vs in latest_states.values():
                combiner.update(vs, now_ms=now_ms)
            combiner.combine(now_ms=now_ms)

            basis_result = basis_filter.update(
                timestamp_s=now_ms / 1000.0,
                observations=[
                    VenueBasisObservation(
                        name=s.exchange,
                        fair_value=s.fair_value,
                        local_variance=s.variance,
                        age_ms=float(max(now_ms - s.timestamp_ms, 0)),
                        venue_kind="perp" if s.exchange.endswith("_perp") else "spot",
                    )
                    for s in latest_states.values()
                ],
            )
            if basis_result is not None:
                last_basis_result = basis_result
            basis_common = (
                last_basis_result.common_price if last_basis_result is not None else None
            )
            is_live = basis_result is not None
            basis_estimates = (
                last_basis_result.basis_estimates if last_basis_result is not None else {}
            )

            # Drift predictions + updates.
            venue_drift: dict[str, float] = {}
            for venue, predictor in predictors.items():
                pend = pending_updates[venue]
                mid_now = venue_mids.get(venue)
                if mid_now is not None:
                    while pend and pend[0][0] <= now_ms:
                        _, xi_past, mid_past = pend.popleft()
                        predictor.update(xi_past, mid_now - mid_past)
                features = venue_features.get(venue)
                if features is None or mid_now is None:
                    continue
                xi = predictor.standardize(features)
                drift_hat = predictor.predict(xi)
                venue_drift[venue] = drift_hat
                pend.append((now_ms + args.drift_horizon_ms, xi, mid_now))

            # Basis-reconstructed fair value per venue = common + basis estimate.
            venue_fair: dict[str, float] = {}
            if basis_common is not None:
                for venue in VENUES:
                    est = basis_estimates.get(venue)
                    if est is not None:
                        venue_fair[venue] = basis_common + est

            output_queue.put(
                PlotPoint(
                    timestamp_ms=now_ms,
                    basis_common_price=basis_common,
                    basis_is_live=is_live,
                    venue_mid=venue_mids,
                    venue_fair=venue_fair,
                    venue_drift=venue_drift,
                    diagnostics_lines=build_diagnostics(
                        venue_mids, venue_fair, venue_drift, basis_common, is_live
                    ),
                )
            )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket, _, _ in socket_specs:
            await socket.close()


class LiveDriftPlot:
    VENUE_COLORS = {
        "coinbase": "#4ea1ff",
        "hyperliquid": "#7bd389",
        "okx": "#e06bff",
        "kraken": "#f7b733",
    }
    BASIS_COLOR = "#ffd700"

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

        self.status_var = tk.StringVar(value="Waiting for live data...")
        tk.Label(
            root,
            textvariable=self.status_var,
            anchor="w",
            bg="#111111",
            fg="#dddddd",
            font=("Helvetica", 12),
            padx=12,
            pady=8,
        ).pack(fill="x")

        self.canvas = tk.Canvas(root, width=width, height=height, bg="#181818", highlightthickness=0)
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
                self.width / 2, self.height / 2,
                text="Waiting for enough points to draw...",
                fill="#999999", font=("Helvetica", 16),
            )
            return

        values: list[float] = []
        for p in self.points:
            values.extend(p.venue_mid.values())
            values.extend(p.venue_fair.values())
        if not values:
            return
        min_y, max_y = min(values), max(values)
        if max_y - min_y < 0.5:
            mid = (max_y + min_y) / 2.0
            min_y, max_y = mid - 0.5, mid + 0.5

        latest = self.points[-1]
        pad_left, pad_right, pad_bottom = 70, 20, 50
        diag_pad = 10
        diag_title_h = 24
        diag_line_h = 14
        diag_y0 = 42
        diag_y1 = diag_y0 + diag_pad + diag_title_h + len(latest.diagnostics_lines) * diag_line_h + diag_pad
        legend_y = diag_y1 + 22
        pad_top = legend_y + 26

        plot_w = self.width - pad_left - pad_right
        plot_h = self.height - pad_top - pad_bottom

        def x_at(i: int) -> float:
            return pad_left + plot_w * i / max(len(self.points) - 1, 1)

        def y_at(v: float) -> float:
            scale = (v - min_y) / (max_y - min_y)
            return pad_top + plot_h - scale * plot_h

        for i in range(5):
            y = pad_top + plot_h * i / 4
            self.canvas.create_line(pad_left, y, pad_left + plot_w, y, fill="#2a2a2a")
            self.canvas.create_text(
                pad_left - 10, y, text=f"{max_y - (max_y - min_y) * i / 4:.2f}",
                fill="#bbbbbb", anchor="e", font=("Helvetica", 10),
            )
        self.canvas.create_line(pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, fill="#444444")
        self.canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_h, fill="#444444")

        # Basis common price (single series, gold).
        basis_pts: list[float] = []
        for i, p in enumerate(self.points):
            if p.basis_common_price is not None:
                basis_pts.extend((x_at(i), y_at(p.basis_common_price)))
        if len(basis_pts) >= 4:
            self.canvas.create_line(*basis_pts, fill=self.BASIS_COLOR, width=3)

        # Per venue: solid venue_mid, dashed venue_fair (basis-reconstructed).
        for venue, color in self.VENUE_COLORS.items():
            mid_pts: list[float] = []
            fair_pts: list[float] = []
            for i, p in enumerate(self.points):
                x = x_at(i)
                m = p.venue_mid.get(venue)
                f = p.venue_fair.get(venue)
                if m is not None:
                    mid_pts.extend((x, y_at(m)))
                if f is not None:
                    fair_pts.extend((x, y_at(f)))
            if len(mid_pts) >= 4:
                self.canvas.create_line(*mid_pts, fill=color, width=2)
            if len(fair_pts) >= 4:
                self.canvas.create_line(*fair_pts, fill=color, width=1, dash=(4, 3))

            # Drift arrow on the most recent point: from current mid to mid + drift_hat.
            last_mid = latest.venue_mid.get(venue)
            drift = latest.venue_drift.get(venue)
            if last_mid is not None and drift is not None and drift != 0.0:
                x_end = x_at(len(self.points) - 1)
                y_from = y_at(last_mid)
                y_to = y_at(last_mid + drift)
                self.canvas.create_line(
                    x_end, y_from, x_end + 18, y_to,
                    fill=color, width=2, arrow=tk.LAST,
                )

        # Status bar
        parts: list[str] = []
        if latest.basis_common_price is not None:
            parts.append(f"Common: {latest.basis_common_price:.2f}")
        for venue in VENUES:
            m = latest.venue_mid.get(venue)
            d = latest.venue_drift.get(venue)
            if m is not None:
                part = f"{venue[:2]}={m:.2f}"
                if d is not None:
                    part += f" (drift={d:+.3f})"
                parts.append(part)
        self.status_var.set("  |  ".join(parts))

        # Diagnostics panel
        diag_x1 = self.width - pad_right
        self.canvas.create_rectangle(
            pad_left, diag_y0, diag_x1, diag_y1, fill="#141414", outline="#2a2a2a",
        )
        self.canvas.create_text(
            pad_left + 10, diag_y0 + 10,
            text="Live Drift Predictions",
            fill="#dddddd", anchor="nw", font=("Helvetica", 11, "bold"),
        )
        for i, line in enumerate(latest.diagnostics_lines):
            self.canvas.create_text(
                pad_left + 10,
                diag_y0 + diag_pad + diag_title_h + i * diag_line_h,
                text=line, fill="#bbbbbb", anchor="nw", font=("Courier", 11),
            )

        # Legend
        lx = pad_left
        self.canvas.create_rectangle(lx, legend_y - 6, lx + 18, legend_y + 6, fill=self.BASIS_COLOR, outline="")
        self.canvas.create_text(
            lx + 26, legend_y, text="Basis Common",
            fill="#dddddd", anchor="w", font=("Helvetica", 11, "bold"),
        )
        lx += 26 + len("Basis Common") * 9 + 20
        for venue, color in self.VENUE_COLORS.items():
            self.canvas.create_rectangle(lx, legend_y - 6, lx + 18, legend_y + 6, fill=color, outline="")
            self.canvas.create_text(
                lx + 26, legend_y, text=f"{venue} (solid=mid, dash=fair)",
                fill="#dddddd", anchor="w", font=("Helvetica", 10),
            )
            lx += 26 + len(f"{venue} (solid=mid, dash=fair)") * 7 + 12

        self.canvas.create_text(
            pad_left, self.height - 20,
            text="Solid = venue mid. Dashed = basis-reconstructed fair value. Arrow = predicted 500ms drift.",
            fill="#dddddd", anchor="w", font=("Helvetica", 11),
        )


def start_stream_thread(
    *,
    args: argparse.Namespace,
    predictors: dict[str, DriftPredictor],
    data_queue: queue.Queue[PlotPoint],
    stop_event: threading.Event,
) -> threading.Thread:
    def runner() -> None:
        asyncio.run(stream_plot_points(args, predictors, data_queue, stop_event))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = build_basis_parser()
    parser.description = "Live plot: per-venue price, basis fair value, and drift prediction."
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=750)
    parser.add_argument(
        "--calibration-db",
        type=Path,
        default=Path("data/filter_eval_maker.sqlite3"),
        help="SQLite capture used to calibrate the drift Kalman before going live",
    )
    parser.add_argument("--drift-horizon-ms", type=int, default=500)
    parser.add_argument("--drift-q-floor", type=float, default=1e-9)
    parser.add_argument(
        "--q-window-seconds",
        type=int,
        default=600,
        help="Window length (seconds) for the windowed-OLS Q estimator",
    )
    parser.add_argument(
        "--min-oos-r2",
        type=float,
        default=0.0,
        help="Refuse a venue's predictor if out-of-sample R² falls below this",
    )
    parser.add_argument(
        "--min-oos-sign-accuracy",
        type=float,
        default=0.5,
        help="Refuse a venue's predictor if top-50% |ŷ| sign accuracy falls below this",
    )
    args = parser.parse_args()
    apply_asset_defaults(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"Calibrating drift predictors from {args.calibration_db}...")
    predictors = calibrate_predictors_from_db(
        args.calibration_db,
        args.drift_horizon_ms,
        args.drift_q_floor,
        args.q_window_seconds,
        args.min_oos_r2,
        args.min_oos_sign_accuracy,
    )
    if not predictors:
        raise SystemExit(
            f"No predictors calibrated from {args.calibration_db}. "
            "All venues failed OOS gates or have insufficient data."
        )

    data_queue: queue.Queue[PlotPoint] = queue.Queue()
    stop_event = threading.Event()
    start_stream_thread(
        args=args, predictors=predictors, data_queue=data_queue, stop_event=stop_event
    )

    root = tk.Tk()
    root.geometry(f"{args.width}x{args.height}")
    LiveDriftPlot(
        root=root,
        title=f"Live Drift Prediction: {args.asset}",
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        data_queue=data_queue,
        stop_event=stop_event,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
