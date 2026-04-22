#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Live paper-trade visualization of the Ridge+lagged drift edge on coinbase.

At startup: load a calibration SQLite capture, build the same 40-feature
(cross-venue raw + lagged diffs) training set used by walk_forward_lagged.py,
fit a single Ridge model, and freeze its coefficients.

Live: connect sockets for all 4 venues, compute coinbase's 40-feature vector
every coinbase tick (uses latest other-venue snapshots and self-lagged diffs
from 2s/5s buffers), predict drift_hat via the frozen Ridge. Run a paper
maker simulator that quotes around venue_mid + skew * drift_hat with the
upgraded fill rule from simulate_basis_maker.py.

Plot: coinbase mid, live bid/ask quotes, fill markers, running P&L + inventory.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import logging
import queue
import sqlite3
import sys
import threading
import tkinter as tk
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

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
from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)
from _hyperliquid_spot import resolve_hyperliquid_spot_subscription_coin
from walk_forward_lagged import (
    OWN_FEATURE_NAMES,
    LAGGED_FEATURE_INDICES,
    LAGGED_FEATURE_NAMES,
    load_series,
    build_dataset,
    build_feature_names,
)

VENUES = ("coinbase", "hyperliquid", "okx", "kraken")
PREDICT_VENUE = "coinbase"


# -----------------------------------------------------------------------------
# Ridge training (one-shot at startup, from calibration DB)
# -----------------------------------------------------------------------------


@dataclass
class RidgeModel:
    coef: np.ndarray  # (n_features,)
    intercept: float
    feature_names: list[str]
    other_venues: list[str]  # sorted, excluding predict venue
    lag_windows_ms: list[int]
    horizon_ms: int
    n_train: int

    def predict(self, feats: np.ndarray) -> float:
        return float(self.coef @ feats + self.intercept)


def train_ridge(
    db_path: Path,
    horizon_ms: int,
    tolerance_ms: int,
    lag_windows_ms: list[int],
    alpha: float,
    enabled_venues: set[str] | None = None,
) -> RidgeModel:
    print(f"loading calibration DB: {db_path}")
    series = load_series([db_path])
    if enabled_venues is not None:
        series = {v: s for v, s in series.items() if v in enabled_venues}
    if PREDICT_VENUE not in series:
        raise SystemExit(f"calibration DB has no {PREDICT_VENUE} data")
    other_venues = sorted(v for v in series if v != PREDICT_VENUE)
    print(f"series: {[(v, len(series[v].ts)) for v in series]}")
    built = build_dataset(
        predict_venue=PREDICT_VENUE,
        series=series,
        horizon_ms=horizon_ms,
        tolerance_ms=tolerance_ms,
        lag_windows_ms=lag_windows_ms,
    )
    if built is None:
        raise SystemExit("build_dataset returned None (insufficient aligned rows)")
    x, y, feature_names = built
    print(f"training Ridge: n={len(y)}  n_features={x.shape[1]}  alpha={alpha}")
    model = Ridge(alpha=alpha)
    model.fit(x, y)
    return RidgeModel(
        coef=np.asarray(model.coef_),
        intercept=float(model.intercept_),
        feature_names=feature_names,
        other_venues=other_venues,
        lag_windows_ms=lag_windows_ms,
        horizon_ms=horizon_ms,
        n_train=len(y),
    )


# -----------------------------------------------------------------------------
# Live feature assembly
# -----------------------------------------------------------------------------


@dataclass
class TradeFlowTracker:
    window_ms: int = 1000
    _events: deque[tuple[int, float, float]] = field(default_factory=deque)

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


@dataclass
class VenueBuffer:
    """Per-venue rolling buffer of (ts_ms, 6-feature vector). Lists + bisect for O(log n) lookup."""
    ts: list[int] = field(default_factory=list)
    feats: list[np.ndarray] = field(default_factory=list)
    retain_ms: int = 20_000
    _cleanup_every: int = 256
    _since_cleanup: int = 0

    def push(self, ts_ms: int, feat: np.ndarray) -> None:
        self.ts.append(ts_ms)
        self.feats.append(feat)
        self._since_cleanup += 1
        if self._since_cleanup >= self._cleanup_every:
            self._since_cleanup = 0
            cutoff = ts_ms - self.retain_ms
            i = bisect.bisect_left(self.ts, cutoff)
            if i > 0:
                del self.ts[:i]
                del self.feats[:i]

    def lookup(self, target_ts: int, max_gap_ms: int) -> np.ndarray | None:
        if not self.ts:
            return None
        i = bisect.bisect_right(self.ts, target_ts) - 1
        if i < 0:
            return None
        if target_ts - self.ts[i] > max_gap_ms:
            return None
        return self.feats[i]

    def latest(self) -> tuple[int, np.ndarray] | None:
        if not self.ts:
            return None
        return self.ts[-1], self.feats[-1]


def assemble_feature_vector(
    model: RidgeModel,
    now_ms: int,
    own_feat: np.ndarray,
    buffers: dict[str, VenueBuffer],
) -> tuple[np.ndarray | None, str | None]:
    """Build the 40-dim feature vector; return (features, fail_reason)."""
    parts: list[np.ndarray] = [own_feat]
    for ov in model.other_venues:
        latest = buffers[ov].latest()
        if latest is None:
            return None, f"no-latest:{ov}"
        parts.append(latest[1])

    lagged: list[float] = []
    max_gap = {w: 2 * w for w in model.lag_windows_ms}
    venue_order = [PREDICT_VENUE] + model.other_venues
    for v in venue_order:
        if v == PREDICT_VENUE:
            cur = own_feat
        else:
            latest = buffers[v].latest()
            if latest is None:
                return None, f"no-latest:{v}"
            cur = latest[1]
        for fi in LAGGED_FEATURE_INDICES:
            for w in model.lag_windows_ms:
                past = buffers[v].lookup(now_ms - w, max_gap[w])
                if past is None:
                    oldest = buffers[v].ts[0] if buffers[v].ts else None
                    newest = buffers[v].ts[-1] if buffers[v].ts else None
                    return None, (
                        f"lag-miss:{v}@-{w}ms (target={now_ms - w}, "
                        f"buf=[{oldest},{newest}], n={len(buffers[v].ts)})"
                    )
                lagged.append(float(cur[fi]) - float(past[fi]))
    parts.append(np.asarray(lagged))
    return np.concatenate(parts), None


# -----------------------------------------------------------------------------
# Paper-trade simulator (distilled from simulate_basis_maker.py, live version)
# -----------------------------------------------------------------------------


@dataclass
class PaperTradeState:
    half_spread: float
    skew_coef: float
    fee_bps: float
    max_inventory: float
    latency_ms: int
    queue_scale: float
    requote_threshold: float
    order_size: float

    inventory: float = 0.0
    cash: float = 0.0

    live_bid_price: float | None = None
    live_bid_queue: float = 0.0
    live_bid_go_live_ts: int = 0
    live_ask_price: float | None = None
    live_ask_queue: float = 0.0
    live_ask_go_live_ts: int = 0

    fills: list["FillEvent"] = field(default_factory=list)
    total_quotes_posted: int = 0


@dataclass
class FillEvent:
    ts_ms: int
    side: str
    price: float
    fee: float
    drift_hat: float


def step_paper_trade(
    state: PaperTradeState,
    now_ms: int,
    venue_mid: float,
    venue_bid: float | None,
    venue_ask: float | None,
    top_bid_depth: float,
    top_ask_depth: float,
    trade_buy_size: float,
    trade_sell_size: float,
    drift_hat: float,
) -> None:
    center = venue_mid + state.skew_coef * drift_hat
    desired_bid = center - state.half_spread
    desired_ask = center + state.half_spread

    # Manage live bid
    if state.inventory < state.max_inventory:
        if (
            state.live_bid_price is None
            or abs(state.live_bid_price - desired_bid) > state.requote_threshold
        ):
            state.live_bid_price = desired_bid
            if venue_bid is not None and desired_bid > venue_bid:
                state.live_bid_queue = 0.0
            else:
                state.live_bid_queue = top_bid_depth * state.queue_scale
            state.live_bid_go_live_ts = now_ms + state.latency_ms
            state.total_quotes_posted += 1
    else:
        state.live_bid_price = None

    # Manage live ask
    if state.inventory > -state.max_inventory:
        if (
            state.live_ask_price is None
            or abs(state.live_ask_price - desired_ask) > state.requote_threshold
        ):
            state.live_ask_price = desired_ask
            if venue_ask is not None and desired_ask < venue_ask:
                state.live_ask_queue = 0.0
            else:
                state.live_ask_queue = top_ask_depth * state.queue_scale
            state.live_ask_go_live_ts = now_ms + state.latency_ms
            state.total_quotes_posted += 1
    else:
        state.live_ask_price = None

    fee_rate = state.fee_bps / 10_000.0

    # Bid fill
    if (
        state.live_bid_price is not None
        and now_ms >= state.live_bid_go_live_ts
        and state.inventory < state.max_inventory
    ):
        crossed = venue_ask is not None and venue_ask <= state.live_bid_price
        at_or_inside = venue_bid is None or state.live_bid_price >= venue_bid
        fill = False
        if crossed:
            fill = True
        elif at_or_inside and trade_sell_size > 0:
            consumed = min(trade_sell_size, state.live_bid_queue)
            state.live_bid_queue -= consumed
            remaining = trade_sell_size - consumed
            if remaining > 0 and state.live_bid_queue <= 0.0:
                fill = True
        if fill:
            p = state.live_bid_price
            fee = fee_rate * p * state.order_size
            state.cash -= p * state.order_size + fee
            state.inventory += state.order_size
            state.fills.append(FillEvent(now_ms, "buy", p, fee, drift_hat))
            state.live_bid_price = None

    # Ask fill
    if (
        state.live_ask_price is not None
        and now_ms >= state.live_ask_go_live_ts
        and state.inventory > -state.max_inventory
    ):
        crossed = venue_bid is not None and venue_bid >= state.live_ask_price
        at_or_inside = venue_ask is None or state.live_ask_price <= venue_ask
        fill = False
        if crossed:
            fill = True
        elif at_or_inside and trade_buy_size > 0:
            consumed = min(trade_buy_size, state.live_ask_queue)
            state.live_ask_queue -= consumed
            remaining = trade_buy_size - consumed
            if remaining > 0 and state.live_ask_queue <= 0.0:
                fill = True
        if fill:
            p = state.live_ask_price
            fee = fee_rate * p * state.order_size
            state.cash += p * state.order_size - fee
            state.inventory -= state.order_size
            state.fills.append(FillEvent(now_ms, "sell", p, fee, drift_hat))
            state.live_ask_price = None


def mtm(state: PaperTradeState, last_mid: float) -> float:
    unwind = abs(state.inventory) * state.half_spread
    return state.cash + state.inventory * last_mid - unwind


# -----------------------------------------------------------------------------
# Plot data model
# -----------------------------------------------------------------------------


@dataclass
class PlotPoint:
    ts_ms: int
    mid: float
    bid_quote: float | None
    ask_quote: float | None
    drift_hat: float | None
    fills_since_last: list[FillEvent]
    mtm: float
    inventory: float
    total_fills: int
    quotes_posted: int


# -----------------------------------------------------------------------------
# Live streaming
# -----------------------------------------------------------------------------


async def stream_plot_points(
    args: argparse.Namespace,
    model: RidgeModel,
    paper: PaperTradeState,
    output_queue: "queue.Queue[PlotPoint]",
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
        hl_coin = (
            args.hyperliquid_subscription_coin
            or resolve_hyperliquid_spot_subscription_coin(
                args.hyperliquid_market, testnet=args.testnet,
            )
        )
        socket_specs.append(
            (
                HyperliquidSocket(
                    args.hyperliquid_market.split("/", 1)[0],
                    subscription_coin=hl_coin,
                    testnet=args.testnet,
                ),
                None, None,
            )
        )
    if not args.disable_okx:
        socket_specs.append((OkxSocket(args.okx_market), None, None))
    if not args.disable_kraken:
        socket_specs.append((KrakenSocket(args.kraken_market), None, None))

    tasks = [
        asyncio.create_task(
            stream_socket_observations(
                socket, observation_queue,
                exchange_override=ex, market_override=mk,
            )
        )
        for socket, ex, mk in socket_specs
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
    trade_flow_trackers: dict[str, TradeFlowTracker] = {
        v: TradeFlowTracker() for v in VENUES
    }
    buffers: dict[str, VenueBuffer] = {v: VenueBuffer() for v in VENUES}
    last_basis_result = None
    fills_since_last_push: list[FillEvent] = []
    # Diagnostics: log once per stage.
    seen_venues: set[str] = set()
    logged_basis_live = False
    logged_first_prediction = False
    skip_reasons: defaultdict[str, int] = defaultdict(int)
    last_diag_ms = 0
    predictions_since_last = 0
    last_heartbeat_ms = 0

    try:
        while not stop_event.is_set():
            try:
                observation = await asyncio.wait_for(
                    observation_queue.get(), timeout=5.0,
                )
            except asyncio.TimeoutError:
                print(
                    f"[live] no observations for 5s. socket task states: "
                    f"{[('done' if t.done() else 'running') for t in tasks]}"
                )
                for i, t in enumerate(tasks):
                    if t.done() and not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            print(f"[live] socket task {i} died: {exc!r}")
                continue
            # Backpressure: if we're falling behind, drain and keep only the latest per venue.
            qsize = observation_queue.qsize()
            if qsize > 200:
                print(f"[live] backpressure: draining queue (depth={qsize})")
                latest_per_venue: dict[tuple[str, type], Observation] = {}
                latest_per_venue[(observation.exchange, type(observation))] = observation
                while True:
                    try:
                        obs2 = observation_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    latest_per_venue[(obs2.exchange, type(obs2))] = obs2
                # Re-enqueue the latest we kept; process oldest first.
                for obs2 in latest_per_venue.values():
                    observation_queue.put_nowait(obs2)
                observation = await observation_queue.get()
            now_ms = (
                observation.received_timestamp_ms
                if observation.received_timestamp_ms is not None
                else observation.source_timestamp_ms
            )
            if now_ms is None:
                continue

            if observation.exchange not in seen_venues:
                seen_venues.add(observation.exchange)
                print(f"[live] first message from {observation.exchange}  "
                      f"(have: {sorted(seen_venues)})")

            if isinstance(observation, TradeObservation):
                tr = trade_flow_trackers.get(observation.exchange)
                if tr is not None:
                    tr.add(now_ms, observation.side, float(observation.size))
                continue

            key = (observation.exchange, observation.market)
            eng = measurement_engines.get(key)
            if eng is None:
                eng = MarketMeasurementEngine(
                    exchange=observation.exchange, market=observation.market,
                )
                measurement_engines[key] = eng
            eng.on_observation(observation)

            # Refresh all venue states so basis filter + combiner stay current.
            latest_states = {}
            for k2, eng2 in measurement_engines.items():
                est = estimators.get(k2)
                if est is None:
                    est = RawFairValueEstimator()
                    estimators[k2] = est
                if k2 not in filters:
                    filters[k2] = build_filter(args)
                vs = build_venue_state(
                    engine=eng2, estimator=est,
                    fair_value_filter=filters[k2],
                    market=args.asset, now_ms=now_ms,
                )
                if vs is None:
                    continue
                latest_states[eng2.exchange] = vs

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
                if not logged_basis_live:
                    logged_basis_live = True
                    print(f"[live] basis filter LIVE at {now_ms}  "
                          f"common={basis_result.common_price:.2f}  "
                          f"estimates={ {k: round(v, 2) for k, v in basis_result.basis_estimates.items()} }")
            if last_basis_result is None:
                skip_reasons["basis_not_live"] += 1
                if now_ms - last_diag_ms > 5000:
                    last_diag_ms = now_ms
                    print(f"[live] still warming up: basis filter not live yet  "
                          f"(venues seen: {sorted(seen_venues)})")
                continue
            basis_common = last_basis_result.common_price
            basis_estimates = last_basis_result.basis_estimates

            # Push each venue's current state into its buffer (keeps lookups dense).
            for venue, vs in latest_states.items():
                eng_for_venue = None
                for (exch, _), e in measurement_engines.items():
                    if exch == venue:
                        eng_for_venue = e
                        break
                if eng_for_venue is None:
                    continue
                m = eng_for_venue.current_measurement(now_ms)
                if (
                    m is None or m.microprice is None
                    or m.depth_imbalance is None or m.mid_volatility_bps is None
                ):
                    continue
                est = basis_estimates.get(venue)
                if est is None:
                    continue
                reconstructed = basis_common + est
                net_flow, total_size = trade_flow_trackers[venue].snapshot(now_ms)
                mid = float(vs.fair_value)
                feat = np.asarray(
                    (
                        float(m.microprice) - mid,
                        float(m.depth_imbalance),
                        float(m.mid_volatility_bps),
                        net_flow,
                        total_size,
                        mid - reconstructed,
                    )
                )
                buffers[venue].push(now_ms, feat)

            # Only run paper-trade on coinbase ticks.
            if observation.exchange != PREDICT_VENUE:
                continue
            own_vs = latest_states.get(PREDICT_VENUE)
            if own_vs is None:
                continue
            own_latest = buffers[PREDICT_VENUE].latest()
            if own_latest is None:
                continue
            own_feat = own_latest[1]
            features, fail_reason = assemble_feature_vector(model, now_ms, own_feat, buffers)
            if features is None:
                skip_reasons["features_missing"] += 1
                if now_ms - last_diag_ms > 5000:
                    last_diag_ms = now_ms
                    print(f"[live] assembly failed: {fail_reason}")
                continue
            drift_hat_raw = model.predict(features)
            drift_hat = max(-paper.half_spread, min(paper.half_spread, drift_hat_raw))
            predictions_since_last += 1
            if not logged_first_prediction:
                logged_first_prediction = True
                print(f"[live] FIRST PREDICTION at {now_ms}  drift_hat={drift_hat:+.4f} "
                      f"(raw={drift_hat_raw:+.4f})  (paper-trade active)")
            if last_heartbeat_ms == 0:
                last_heartbeat_ms = now_ms
            elif now_ms - last_heartbeat_ms >= 5000:
                elapsed_s = (now_ms - last_heartbeat_ms) / 1000.0
                rate = predictions_since_last / elapsed_s
                print(f"[live] heartbeat: {rate:.1f} pred/s  qsize={observation_queue.qsize()}  "
                      f"last drift={drift_hat:+.3f}  skips={dict(skip_reasons)}")
                last_heartbeat_ms = now_ms
                predictions_since_last = 0
                skip_reasons.clear()

            m_cb = measurement_engines.get((PREDICT_VENUE, args.coinbase_market))
            if m_cb is None:
                continue
            meas = m_cb.current_measurement(now_ms)
            if meas is None:
                continue
            venue_bid = float(meas.bid_price) if meas.bid_price is not None else None
            venue_ask = float(meas.ask_price) if meas.ask_price is not None else None
            top_bid_depth = float(meas.top_bid_depth or 0.0)
            top_ask_depth = float(meas.top_ask_depth or 0.0)
            tracker = trade_flow_trackers[PREDICT_VENUE]
            net_flow, total_flow = tracker.snapshot(now_ms)
            # snapshot returns (buy-sell, buy+sell) → recover each side.
            buy_flow = max(0.0, (total_flow + net_flow) / 2.0)
            sell_flow = max(0.0, (total_flow - net_flow) / 2.0)
            mid_cb = float(own_vs.fair_value)

            prev_fill_count = len(paper.fills)
            step_paper_trade(
                paper,
                now_ms=now_ms,
                venue_mid=mid_cb,
                venue_bid=venue_bid,
                venue_ask=venue_ask,
                top_bid_depth=top_bid_depth,
                top_ask_depth=top_ask_depth,
                trade_buy_size=buy_flow,
                trade_sell_size=sell_flow,
                drift_hat=drift_hat,
            )
            new_fills = paper.fills[prev_fill_count:]
            fills_since_last_push.extend(new_fills)

            output_queue.put(
                PlotPoint(
                    ts_ms=now_ms,
                    mid=mid_cb,
                    bid_quote=paper.live_bid_price,
                    ask_quote=paper.live_ask_price,
                    drift_hat=drift_hat,
                    fills_since_last=list(fills_since_last_push),
                    mtm=mtm(paper, mid_cb),
                    inventory=paper.inventory,
                    total_fills=len(paper.fills),
                    quotes_posted=paper.total_quotes_posted,
                )
            )
            fills_since_last_push.clear()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for socket, _, _ in socket_specs:
            await socket.close()


# -----------------------------------------------------------------------------
# Tk plot
# -----------------------------------------------------------------------------


class LivePaperTradePlot:
    BG = "#181818"
    GRID = "#2a2a2a"
    AXIS = "#444444"
    TEXT_DIM = "#bbbbbb"
    MID_COLOR = "#4ea1ff"
    BID_COLOR = "#7bd389"
    ASK_COLOR = "#e06b6b"
    PNL_POS = "#7bd389"
    PNL_NEG = "#e06b6b"
    BUY_FILL = "#7bd389"
    SELL_FILL = "#e06b6b"

    def __init__(
        self,
        *,
        root: tk.Tk,
        title: str,
        width: int,
        height: int,
        max_points: int,
        data_queue: "queue.Queue[PlotPoint]",
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
        self.all_fills: list[FillEvent] = []
        self.start_ts_ms: int | None = None

        self.status_var = tk.StringVar(value="Waiting for live data...")
        tk.Label(
            root, textvariable=self.status_var, anchor="w",
            bg="#111111", fg="#dddddd",
            font=("Helvetica", 14, "bold"),
            padx=12, pady=8,
        ).pack(fill="x")

        self.canvas = tk.Canvas(
            root, width=width, height=height,
            bg=self.BG, highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.after(100, self.refresh)

    def close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    def refresh(self) -> None:
        try:
            drained = 0
            while drained < 500:
                try:
                    pt = self.data_queue.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                if self.start_ts_ms is None:
                    self.start_ts_ms = pt.ts_ms
                self.points.append(pt)
                for f in pt.fills_since_last:
                    self.all_fills.append(f)
            if self.points:
                ts0 = self.points[0].ts_ms
                self.all_fills = [f for f in self.all_fills if f.ts_ms >= ts0]
            self._draw()
        except Exception:
            import traceback
            print("[plot] refresh error:", flush=True)
            traceback.print_exc()
        if not self.stop_event.is_set():
            self.root.after(100, self.refresh)

    def _draw(self) -> None:
        self.canvas.delete("all")
        if len(self.points) < 2:
            self.canvas.create_text(
                self.width / 2, self.height / 2,
                text="Waiting for live data...",
                fill="#999999", font=("Helvetica", 16),
            )
            return

        latest = self.points[-1]
        pnl = latest.mtm
        color = self.PNL_POS if pnl >= 0 else self.PNL_NEG
        sign = "+" if pnl >= 0 else ""
        buys = sum(1 for f in self.all_fills if f.side == "buy")
        sells = len(self.all_fills) - buys
        elapsed = (latest.ts_ms - (self.start_ts_ms or latest.ts_ms)) / 1000.0
        per_hour = pnl / (elapsed / 3600.0) if elapsed > 30 else 0.0
        self.status_var.set(
            f"P&L {sign}${pnl:,.2f}   •   {per_hour:+,.0f}/hour   •   "
            f"inv {latest.inventory:+.1f}   •   fills {buys}B/{sells}S   •   "
            f"mid {latest.mid:,.2f}"
        )

        pad_left, pad_right, pad_top, pad_bottom = 70, 20, 20, 30
        plot_w = self.width - pad_left - pad_right
        plot_h = self.height - pad_top - pad_bottom

        ts0 = self.points[0].ts_ms
        ts1 = latest.ts_ms
        span = max(ts1 - ts0, 1)

        vals: list[float] = []
        for p in self.points:
            vals.append(p.mid)
            if p.bid_quote is not None:
                vals.append(p.bid_quote)
            if p.ask_quote is not None:
                vals.append(p.ask_quote)
        pmin, pmax = min(vals), max(vals)
        if pmax - pmin < 1.0:
            c = (pmax + pmin) / 2
            pmin, pmax = c - 1.0, c + 1.0
        else:
            pad = (pmax - pmin) * 0.08
            pmin -= pad; pmax += pad

        def x_at(ts: int) -> float:
            return pad_left + plot_w * (ts - ts0) / span

        def y_at(v: float) -> float:
            return pad_top + plot_h - plot_h * (v - pmin) / (pmax - pmin)

        for i in range(5):
            y = pad_top + plot_h * i / 4
            self.canvas.create_line(pad_left, y, pad_left + plot_w, y, fill=self.GRID)
            v = pmax - (pmax - pmin) * i / 4
            self.canvas.create_text(
                pad_left - 10, y, text=f"{v:,.2f}",
                fill=self.TEXT_DIM, anchor="e", font=("Helvetica", 10),
            )
        self.canvas.create_line(pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, fill=self.AXIS)
        self.canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_h, fill=self.AXIS)

        def draw_line(extract, color, width):
            seg: list[float] = []
            for p in self.points:
                v = extract(p)
                if v is None:
                    if len(seg) >= 4:
                        self.canvas.create_line(*seg, fill=color, width=width)
                    seg = []
                    continue
                seg.extend((x_at(p.ts_ms), y_at(v)))
            if len(seg) >= 4:
                self.canvas.create_line(*seg, fill=color, width=width)

        draw_line(lambda p: p.bid_quote, self.BID_COLOR, 1)
        draw_line(lambda p: p.ask_quote, self.ASK_COLOR, 1)
        draw_line(lambda p: p.mid, self.MID_COLOR, 2)

        for f in self.all_fills:
            if f.ts_ms < ts0:
                continue
            c = self.BUY_FILL if f.side == "buy" else self.SELL_FILL
            cx, cy = x_at(f.ts_ms), y_at(f.price)
            self.canvas.create_oval(
                cx - 4, cy - 4, cx + 4, cy + 4, fill=c, outline="",
            )


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def start_stream_thread(
    *, args, model, paper, data_queue, stop_event,
) -> threading.Thread:
    def runner() -> None:
        try:
            asyncio.run(stream_plot_points(args, model, paper, data_queue, stop_event))
        except Exception:
            import traceback
            print("[live] STREAM THREAD CRASHED:", flush=True)
            traceback.print_exc()
            stop_event.set()
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = build_basis_parser()
    parser.description = (
        "Live paper-trade visualization of the Ridge+lagged edge on coinbase."
    )
    parser.add_argument("--max-points", type=int, default=400)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--calibration-db", type=Path,
        default=Path("data/capture/2026-04-21.sqlite3"),
        help="SQLite capture used to train the Ridge model",
    )
    parser.add_argument("--horizon-ms", type=int, default=5000)
    parser.add_argument("--tolerance-ms", type=int, default=10_000)
    parser.add_argument(
        "--lag-windows-ms", default="2000,5000",
        help="comma-separated lag windows for lagged-diff features",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--half-spread", type=float, default=2.0)
    parser.add_argument("--skew-coef", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=-1.0)
    parser.add_argument("--max-inventory", type=float, default=10.0)
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument("--queue-scale", type=float, default=1.0)
    parser.add_argument("--requote-threshold", type=float, default=1.0)
    parser.add_argument("--order-size", type=float, default=1.0)
    args = parser.parse_args()
    apply_asset_defaults(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    lag_windows = [int(x) for x in args.lag_windows_ms.split(",") if x]
    enabled_venues = {PREDICT_VENUE}
    if not args.disable_hyperliquid:
        enabled_venues.add("hyperliquid")
    if not args.disable_okx:
        enabled_venues.add("okx")
    if not args.disable_kraken:
        enabled_venues.add("kraken")
    print(f"enabled venues: {sorted(enabled_venues)}")
    model = train_ridge(
        args.calibration_db,
        horizon_ms=args.horizon_ms,
        tolerance_ms=args.tolerance_ms,
        lag_windows_ms=lag_windows,
        alpha=args.ridge_alpha,
        enabled_venues=enabled_venues,
    )

    paper = PaperTradeState(
        half_spread=args.half_spread,
        skew_coef=args.skew_coef,
        fee_bps=args.fee_bps,
        max_inventory=args.max_inventory,
        latency_ms=args.latency_ms,
        queue_scale=args.queue_scale,
        requote_threshold=args.requote_threshold,
        order_size=args.order_size,
    )
    print(
        f"paper-trade: hs=${args.half_spread}  skew={args.skew_coef}  "
        f"fee_bps={args.fee_bps}  max_inv={args.max_inventory}"
    )

    data_queue: "queue.Queue[PlotPoint]" = queue.Queue()
    stop_event = threading.Event()
    start_stream_thread(
        args=args, model=model, paper=paper,
        data_queue=data_queue, stop_event=stop_event,
    )

    root = tk.Tk()
    root.geometry(f"{args.width}x{args.height}")
    LivePaperTradePlot(
        root=root,
        title=f"Live Paper Trade — {PREDICT_VENUE} {args.asset}",
        width=args.width,
        height=args.height,
        max_points=args.max_points,
        data_queue=data_queue,
        stop_event=stop_event,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
