#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Simulate maker quoting on one venue, centered on that venue's own fair value.

Quotes are posted continuously around venue_mid (the per-venue Kalman fair
value):
    bid = quote_center - half_spread
    ask = quote_center + half_spread

`quote_center` defaults to venue_mid. When `--drift-skew-coefs` includes a
nonzero value, a drift predictor (internal Kalman or external CSV) emits
drift_hat per tick and:
    quote_center = venue_mid + drift_skew_coef * drift_hat
                    - inventory * inventory_skew

Note: earlier versions centered on basis_reconstructed (= basis_common +
basis_estimate). That is the venue's price under the cross-venue common-factor
model, which differs from the venue's own book by the real per-venue premium
(e.g., ~$9 on hyperliquid, ~+$9 on okx). Centering on venue_mid avoids
quoting systematically off-market on venues with persistent premia.

Fill rule: my bid is filled when the venue's best ask drops to or below my bid
price (a taker-sell is willing to trade there). Symmetric for the ask.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _drift_predictor import DriftPredictor, FEATURE_NAMES  # noqa: F401


@dataclass(frozen=True, slots=True)
class Tick:
    timestamp_ms: int
    venue_mid: float
    venue_bid: float | None
    venue_ask: float | None
    reconstructed: float
    basis_stddev: float
    age_ms: float
    features: tuple[float, ...] | None  # None if any microstructure field was NULL
    top_bid_depth: float  # base units resting at top of bid (0 if missing)
    top_ask_depth: float
    trade_buy_size: float  # buy-side contra flow since last tick
    trade_sell_size: float


@dataclass
class Fill:
    timestamp_ms: int
    side: str
    price: float
    reference: float
    fee: float = 0.0  # signed: positive = paid, negative = rebate received
    drift_hat: float = 0.0


# Rough 2026 maker-side fee/rebate rates (bps). Positive = pay, negative = rebate.
DEFAULT_FEE_BPS = {
    "coinbase": -1.0,
    "hyperliquid": 1.5,
    "kraken": 16.0,
    "okx": 2.0,
}


def load_ticks(db_path: Path, venue: str, max_age_ms: float) -> list[Tick]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    has_bbo = _has_bid_ask_columns(conn)
    has_flow = _has_trade_flow_columns(conn)
    has_depth = _has_depth_columns(conn)
    bbo_select = (
        "v.bid_price AS bid_price, v.ask_price AS ask_price"
        if has_bbo
        else "NULL AS bid_price, NULL AS ask_price"
    )
    flow_select = (
        "COALESCE(v.trade_net_flow, 0.0) AS trade_net_flow, "
        "COALESCE(v.trade_buy_size, 0.0) + COALESCE(v.trade_sell_size, 0.0) "
        "AS trade_total_size, "
        "COALESCE(v.trade_buy_size, 0.0) AS trade_buy_size, "
        "COALESCE(v.trade_sell_size, 0.0) AS trade_sell_size"
        if has_flow
        else "0.0 AS trade_net_flow, 0.0 AS trade_total_size, "
             "0.0 AS trade_buy_size, 0.0 AS trade_sell_size"
    )
    depth_select = (
        "COALESCE(v.top_bid_depth, 0.0) AS top_bid_depth, "
        "COALESCE(v.top_ask_depth, 0.0) AS top_ask_depth"
        if has_depth
        else "0.0 AS top_bid_depth, 0.0 AS top_ask_depth"
    )
    query = f"""
        SELECT
            u.timestamp_ms,
            u.basis_common_price,
            v.fair_value,
            v.age_ms,
            v.microprice,
            v.depth_imbalance,
            v.mid_volatility_bps,
            {bbo_select},
            {flow_select},
            {depth_select},
            b.basis_estimate,
            b.basis_stddev
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        LEFT JOIN basis_states b
          ON b.update_id = u.id AND b.exchange = v.exchange
        WHERE u.basis_is_live = 1
          AND u.basis_common_price IS NOT NULL
          AND v.exchange = ?
          AND b.basis_estimate IS NOT NULL
        ORDER BY u.timestamp_ms, u.id
    """
    ticks: list[Tick] = []
    for row in conn.execute(query, (venue,)):
        if row["age_ms"] is None or row["age_ms"] > max_age_ms:
            continue
        mid = row["fair_value"]
        microprice = row["microprice"]
        imb = row["depth_imbalance"]
        vol = row["mid_volatility_bps"]
        if microprice is None or imb is None or vol is None:
            features: tuple[float, ...] | None = None
        else:
            features = (
                float(microprice) - float(mid),
                float(imb),
                float(vol),
                float(row["trade_net_flow"]),
                float(row["trade_total_size"]),
            )
        bid = row["bid_price"] if has_bbo else None
        ask = row["ask_price"] if has_bbo else None
        ticks.append(
            Tick(
                timestamp_ms=row["timestamp_ms"],
                venue_mid=float(mid),
                venue_bid=float(bid) if bid is not None else None,
                venue_ask=float(ask) if ask is not None else None,
                reconstructed=float(row["basis_common_price"]) + float(row["basis_estimate"]),
                basis_stddev=float(row["basis_stddev"] or 0.0),
                age_ms=float(row["age_ms"]),
                features=features,
                top_bid_depth=float(row["top_bid_depth"]) if has_depth else 0.0,
                top_ask_depth=float(row["top_ask_depth"]) if has_depth else 0.0,
                trade_buy_size=float(row["trade_buy_size"]) if has_flow else 0.0,
                trade_sell_size=float(row["trade_sell_size"]) if has_flow else 0.0,
            )
        )
    conn.close()
    return ticks


def _has_bid_ask_columns(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(venue_states)")}
    return "bid_price" in cols and "ask_price" in cols


def _has_trade_flow_columns(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(venue_states)")}
    return "trade_net_flow" in cols


def _has_depth_columns(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(venue_states)")}
    return "top_bid_depth" in cols and "top_ask_depth" in cols


def calibrate_predictor(
    ticks: list[Tick],
    calibration_frac: float,
    horizon_ms: int,
    q_floor: float,
    q_ceiling: float,
    q_init: float,
) -> tuple[DriftPredictor, int]:
    """Compute mean/std, Q per coefficient, and an initial β from the calibration slice.

    Returns (predictor, calibration_end_index). No drift skew should be applied
    to ticks before calibration_end_index — the predictor hasn't seen enough data.
    """
    if not any(t.features is not None for t in ticks):
        raise ValueError("no ticks have microstructure features — cannot calibrate")
    cal_end = max(100, int(len(ticks) * calibration_frac))
    cal_end = min(cal_end, len(ticks))

    # Build (features, target) pairs over calibration window.
    timestamps = [t.timestamp_ms for t in ticks]
    x_list: list[tuple[float, ...]] = []
    y_list: list[float] = []
    for i in range(cal_end):
        tick = ticks[i]
        if tick.features is None:
            continue
        target_ts = tick.timestamp_ms + horizon_ms
        j = bisect_left(timestamps, target_ts, lo=i + 1)
        if j >= len(ticks):
            break
        future = ticks[j]
        if future.timestamp_ms - target_ts > horizon_ms:
            continue
        x_list.append(tick.features)
        y_list.append(future.venue_mid - tick.venue_mid)
    if len(y_list) < 50:
        raise ValueError(
            f"only {len(y_list)} (x, y) pairs in calibration window — need >=50"
        )
    x = np.asarray(x_list)
    y = np.asarray(y_list)

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    x_std = (x - mean) / std

    # Initial β via static OLS on standardized features (gives Kalman a sane prior).
    xi = np.hstack([np.ones((len(y), 1)), x_std])
    beta0, *_ = np.linalg.lstsq(xi, y, rcond=None)
    resid = y - xi @ beta0
    r = float(np.var(resid)) or 1.0

    # Per-coefficient Q: run a loose Kalman pass over calibration and measure Δβ variance.
    betas = np.zeros((len(y), xi.shape[1]))
    beta = beta0.copy()
    P = np.eye(xi.shape[1])
    Q_init_mat = np.eye(xi.shape[1]) * q_init
    for t in range(len(y)):
        P_pred = P + Q_init_mat
        h = xi[t]
        innovation = y[t] - h @ beta
        S = float(h @ P_pred @ h + r)
        K = (P_pred @ h) / S
        beta = beta + K * innovation
        P = P_pred - np.outer(K, h) @ P_pred
        betas[t] = beta
    burn = min(200, len(y) // 4)
    q_vec = np.clip(np.var(np.diff(betas[burn:], axis=0), axis=0), q_floor, q_ceiling)

    predictor = DriftPredictor(
        beta=beta0.copy(),
        P=np.eye(xi.shape[1]),
        Q=np.diag(q_vec),
        r=r,
        mean=mean,
        std=std,
    )
    return predictor, cal_end


def simulate(
    ticks: list[Tick],
    half_spread: float,
    fee_bps: float,
    inventory_skew: float,
    max_inventory: float,
    max_basis_stddev: float | None,
    single_sided: bool,
    drift_skew_coef: float,
    predictor: DriftPredictor | None,
    calibration_end: int,
    horizon_ms: int,
    latency_ms: int,
    queue_scale: float,
    order_size: float,
    requote_threshold: float,
    unwind_half_spread_mult: float,
    external_predictions: dict[int, float] | None = None,
) -> tuple[list[Fill], float, dict[str, float]]:
    fills: list[Fill] = []
    inventory = 0.0
    cash = 0.0
    fee_rate = fee_bps / 10_000.0
    drift_hats: list[float] = []

    # Pending updates: (target_ts, xi_standardized, mid_at_t). Processed FIFO
    # when current tick ts >= target_ts so the Kalman update only uses realized data.
    pending: deque[tuple[int, np.ndarray, float]] = deque()

    # Live quote state — persisted across ticks so the queue model can reason
    # about "did we sit on this price long enough to be at the front?".
    # When the desired quote moves by more than requote_threshold we cancel +
    # repost, eating the latency penalty and resetting the queue.
    live_bid_price: float | None = None
    live_bid_queue: float = 0.0  # base units ahead of our order at the bid
    live_bid_go_live_ts: int = 0  # ts when our bid post becomes live (post + latency)
    live_ask_price: float | None = None
    live_ask_queue: float = 0.0
    live_ask_go_live_ts: int = 0

    # Diagnostics
    total_quotes_posted = 0  # bid or ask postings
    total_fills = 0
    total_quote_ms_bid = 0
    total_quote_ms_ask = 0
    last_tick_ts = ticks[0].timestamp_ms if ticks else 0

    for idx, tick in enumerate(ticks):
        dt_ms = tick.timestamp_ms - last_tick_ts
        last_tick_ts = tick.timestamp_ms

        # Apply any pending Kalman updates whose horizon has elapsed.
        if predictor is not None:
            while pending and pending[0][0] <= tick.timestamp_ms:
                _, xi_past, mid_past = pending.popleft()
                predictor.update(xi_past, tick.venue_mid - mid_past)

        drift_hat = 0.0
        if drift_skew_coef != 0.0:
            if external_predictions is not None:
                ext = external_predictions.get(tick.timestamp_ms)
                if ext is not None:
                    drift_hat = max(-half_spread, min(half_spread, float(ext)))
                    drift_hats.append(drift_hat)
            elif (
                predictor is not None
                and idx >= calibration_end
                and tick.features is not None
            ):
                xi = predictor.standardize(tick.features)
                drift_hat = predictor.predict(xi)
                drift_hat = max(-half_spread, min(half_spread, drift_hat))
                drift_hats.append(drift_hat)
                pending.append((tick.timestamp_ms + horizon_ms, xi, tick.venue_mid))

        if max_basis_stddev is not None and tick.basis_stddev > max_basis_stddev:
            # Cancel any live quotes when basis uncertainty spikes.
            live_bid_price = None
            live_ask_price = None
            continue

        skew = -inventory * inventory_skew
        center = tick.venue_mid + drift_skew_coef * drift_hat + skew
        desired_bid = center - half_spread
        desired_ask = center + half_spread

        quote_bid = inventory < max_inventory
        quote_ask = inventory > -max_inventory
        if single_sided:
            dislocation = tick.venue_mid - tick.reconstructed
            quote_bid = quote_bid and dislocation < 0
            quote_ask = quote_ask and dislocation > 0

        # --- Bid side: manage live quote ---
        if quote_bid:
            if live_bid_price is None or abs(live_bid_price - desired_bid) > requote_threshold:
                # Repost — pay latency. Queue: 0 if we're strictly inside best,
                # else top-of-book depth scaled (we're behind existing resters).
                live_bid_price = desired_bid
                if tick.venue_bid is not None and live_bid_price > tick.venue_bid:
                    live_bid_queue = 0.0
                else:
                    live_bid_queue = tick.top_bid_depth * queue_scale
                live_bid_go_live_ts = tick.timestamp_ms + latency_ms
                total_quotes_posted += 1
            else:
                total_quote_ms_bid += dt_ms
        else:
            live_bid_price = None

        # --- Ask side: manage live quote ---
        if quote_ask:
            if live_ask_price is None or abs(live_ask_price - desired_ask) > requote_threshold:
                live_ask_price = desired_ask
                if tick.venue_ask is not None and live_ask_price < tick.venue_ask:
                    live_ask_queue = 0.0
                else:
                    live_ask_queue = tick.top_ask_depth * queue_scale
                live_ask_go_live_ts = tick.timestamp_ms + latency_ms
                total_quotes_posted += 1
            else:
                total_quote_ms_ask += dt_ms
        else:
            live_ask_price = None

        # --- Fill check ---
        # A maker bid fills when (a) the book crosses us (venue_ask <= our bid),
        # or (b) we're at-or-inside the best bid and enough sell flow arrived to
        # drain the queue ahead of us. Symmetric for ask.
        bid_fills = False
        if (
            live_bid_price is not None
            and tick.timestamp_ms >= live_bid_go_live_ts
            and inventory < max_inventory
        ):
            crossed = tick.venue_ask is not None and tick.venue_ask <= live_bid_price
            at_or_inside = tick.venue_bid is None or live_bid_price >= tick.venue_bid
            if crossed:
                bid_fills = True
            elif at_or_inside and tick.trade_sell_size > 0:
                consumed = min(tick.trade_sell_size, live_bid_queue)
                live_bid_queue -= consumed
                remaining = tick.trade_sell_size - consumed
                if remaining > 0 and live_bid_queue <= 0.0:
                    bid_fills = True

        ask_fills = False
        if (
            live_ask_price is not None
            and tick.timestamp_ms >= live_ask_go_live_ts
            and inventory > -max_inventory
        ):
            crossed = tick.venue_bid is not None and tick.venue_bid >= live_ask_price
            at_or_inside = tick.venue_ask is None or live_ask_price <= tick.venue_ask
            if crossed:
                ask_fills = True
            elif at_or_inside and tick.trade_buy_size > 0:
                consumed = min(tick.trade_buy_size, live_ask_queue)
                live_ask_queue -= consumed
                remaining = tick.trade_buy_size - consumed
                if remaining > 0 and live_ask_queue <= 0.0:
                    ask_fills = True

        if bid_fills:
            fill_price = live_bid_price
            fee = fee_rate * fill_price * order_size
            cash -= fill_price * order_size + fee
            inventory += order_size
            fills.append(
                Fill(tick.timestamp_ms, "buy", fill_price, tick.venue_mid, fee, drift_hat)
            )
            total_fills += 1
            live_bid_price = None
        if ask_fills:
            fill_price = live_ask_price
            fee = fee_rate * fill_price * order_size
            cash += fill_price * order_size - fee
            inventory -= order_size
            fills.append(
                Fill(tick.timestamp_ms, "sell", fill_price, tick.venue_mid, fee, drift_hat)
            )
            total_fills += 1
            live_ask_price = None

    last_mid = ticks[-1].venue_mid if ticks else 0.0
    unwind_cost = abs(inventory) * half_spread * unwind_half_spread_mult
    mtm = cash + inventory * last_mid - unwind_cost

    diagnostics = {
        "quotes_posted": float(total_quotes_posted),
        "fill_rate": (total_fills / total_quotes_posted) if total_quotes_posted else 0.0,
        "avg_quote_ms_bid": (total_quote_ms_bid / max(total_quotes_posted, 1)),
        "avg_quote_ms_ask": (total_quote_ms_ask / max(total_quotes_posted, 1)),
        "unwind_cost": unwind_cost,
        "final_inventory": inventory,
    }
    if drift_hats:
        arr = np.asarray(drift_hats)
        print(
            f"    drift_hat stats: n={len(arr)}  "
            f"mean={arr.mean():+.4f}  std={arr.std():.4f}  "
            f"p5={np.percentile(arr, 5):+.3f}  p95={np.percentile(arr, 95):+.3f}  "
            f"max|={np.max(np.abs(arr)):.2f}"
        )
    return fills, mtm, diagnostics


def summarize(
    fills: list[Fill],
    mtm: float,
    hours: float,
    label: str,
    ticks: list[Tick],
    horizon_ms: int,
    diagnostics: dict[str, float],
) -> None:
    if not fills:
        print(
            f"  {label}  no fills  mtm={mtm:+.4f}  "
            f"quotes_posted={int(diagnostics.get('quotes_posted', 0))}"
        )
        return
    n = len(fills)
    buys = sum(1 for f in fills if f.side == "buy")
    sells = n - buys
    turnover = sum(f.price for f in fills)
    total_fees = sum(f.fee for f in fills)
    per_hour = mtm / hours if hours > 0 else 0.0
    print(
        f"  {label}  fills={n} (b={buys} s={sells})  "
        f"turnover=${turnover:,.0f}  "
        f"fees=${total_fees:+.2f}  "
        f"unwind=${diagnostics.get('unwind_cost', 0.0):.2f}  "
        f"mtm=${mtm:+.2f}  per_hour=${per_hour:+.2f}"
    )
    fill_rate = diagnostics.get("fill_rate", 0.0)
    print(
        f"    quotes={int(diagnostics.get('quotes_posted', 0))}  "
        f"fill/quote={fill_rate:.4f}  "
        f"final_inv={diagnostics.get('final_inventory', 0.0):+.2f}"
    )

    # Fill-conditional drift accuracy: does drift_hat sign at fill time
    # match the realized mid move over the next horizon_ms?
    non_zero = [f for f in fills if f.drift_hat != 0.0]
    if not non_zero:
        return
    tick_ts = [t.timestamp_ms for t in ticks]
    agree = 0
    disagree = 0
    realized_buy: list[float] = []
    realized_sell: list[float] = []
    for f in non_zero:
        target_ts = f.timestamp_ms + horizon_ms
        j = bisect_left(tick_ts, target_ts)
        if j >= len(ticks):
            continue
        future_mid = ticks[j].venue_mid
        realized = future_mid - f.reference
        if f.side == "buy":
            realized_buy.append(realized)
        else:
            realized_sell.append(realized)
        if (f.drift_hat >= 0 and realized >= 0) or (f.drift_hat < 0 and realized < 0):
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    acc = agree / total if total > 0 else 0.0
    rb = np.mean(realized_buy) if realized_buy else 0.0
    rs = np.mean(realized_sell) if realized_sell else 0.0
    print(
        f"    fill-drift agree={acc:.3f} (n={total})  "
        f"avg realized post-fill: buys={rb:+.3f}  sells={rs:+.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite capture. Defaults to newest file under data/capture/.",
    )
    parser.add_argument("--venue", default="hyperliquid")
    parser.add_argument("--half-spreads", default="0.5,1,2,5,10")
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=None,
        help="override maker fee (bps); negative = rebate. Defaults to DEFAULT_FEE_BPS[venue].",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=50,
        help="one-way latency: new quotes go live after this delay (0 disables)",
    )
    parser.add_argument(
        "--queue-scale",
        type=float,
        default=1.0,
        help="multiply top-of-book depth by this to get queue-ahead size. "
             "0 = you're always front-of-queue (optimistic).",
    )
    parser.add_argument(
        "--order-size",
        type=float,
        default=1.0,
        help="order size in base units (BTC). All fills use this size.",
    )
    parser.add_argument(
        "--requote-threshold",
        type=float,
        default=0.0,
        help="re-post only when desired quote moves more than this ($). 0 = always re-quote.",
    )
    parser.add_argument(
        "--unwind-half-spread-mult",
        type=float,
        default=1.0,
        help="final-inventory unwind cost = |inv| * half_spread * this",
    )
    parser.add_argument("--inventory-skew", type=float, default=0.0)
    parser.add_argument("--max-inventory", type=float, default=5.0)
    parser.add_argument("--max-venue-age-ms", type=float, default=500.0)
    parser.add_argument("--max-basis-stddev", type=float, default=None)
    parser.add_argument("--single-sided", action="store_true")
    parser.add_argument(
        "--drift-skew-coefs",
        default="0",
        help="comma-separated drift skew coefficients to A/B (0 disables predictor)",
    )
    parser.add_argument("--drift-horizon-ms", type=int, default=500)
    parser.add_argument("--drift-calibration-frac", type=float, default=0.3)
    parser.add_argument("--drift-q-init", type=float, default=1e-4)
    parser.add_argument("--drift-q-floor", type=float, default=1e-9)
    parser.add_argument("--drift-q-ceiling", type=float, default=1e-3)
    parser.add_argument(
        "--predictions-path", type=Path, default=None,
        help="CSV (timestamp_ms,drift_hat) from walk_forward_lagged.py. "
        "When set, bypasses the internal Kalman predictor and uses these "
        "pre-computed cross-venue+lagged Ridge predictions.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    half_spreads = [float(x) for x in args.half_spreads.split(",") if x]
    skew_coefs = [float(x) for x in args.drift_skew_coefs.split(",") if x]
    if args.db_path is None:
        captures = sorted(Path("data/capture").glob("*.sqlite3"))
        if not captures:
            raise SystemExit("No captures found under data/capture/ — pass --db-path.")
        args.db_path = captures[-1]
        print(f"using latest capture: {args.db_path}")
    ticks = load_ticks(args.db_path, args.venue, args.max_venue_age_ms)
    if not ticks:
        print(f"no ticks for venue={args.venue}")
        return
    duration_ms = ticks[-1].timestamp_ms - ticks[0].timestamp_ms
    hours = duration_ms / 3_600_000.0

    predictor_template: DriftPredictor | None = None
    cal_end = 0
    needs_predictor = any(c != 0.0 for c in skew_coefs)
    external_predictions: dict[int, float] | None = None
    if needs_predictor and args.predictions_path is not None:
        external_predictions = {}
        with args.predictions_path.open() as f:
            header = f.readline().strip().split(",")
            ts_i = header.index("timestamp_ms")
            dh_i = header.index("drift_hat")
            for line in f:
                parts = line.strip().split(",")
                external_predictions[int(parts[ts_i])] = float(parts[dh_i])
        print(
            f"loaded {len(external_predictions)} external predictions "
            f"from {args.predictions_path}"
        )
    elif needs_predictor:
        predictor_template, cal_end = calibrate_predictor(
            ticks,
            args.drift_calibration_frac,
            args.drift_horizon_ms,
            args.drift_q_floor,
            args.drift_q_ceiling,
            args.drift_q_init,
        )
        print(
            f"calibrated drift predictor: cal_end={cal_end}  "
            f"Q={np.diag(predictor_template.Q).round(6)}"
        )

    fee_bps = (
        args.fee_bps if args.fee_bps is not None else DEFAULT_FEE_BPS.get(args.venue, 0.0)
    )
    print(
        f"venue={args.venue}  ticks={len(ticks)}  duration={hours*60:.1f} min  "
        f"fee_bps={fee_bps:+.2f}  latency={args.latency_ms}ms  "
        f"queue_scale={args.queue_scale}  order_size={args.order_size}  "
        f"max_inv={args.max_inventory}  skew_coefs={skew_coefs}"
    )
    for hs in half_spreads:
        print(f"half_spread=${hs:.2f}")
        for coef in skew_coefs:
            predictor = None
            if coef != 0.0 and predictor_template is not None:
                predictor = DriftPredictor(
                    beta=predictor_template.beta.copy(),
                    P=predictor_template.P.copy(),
                    Q=predictor_template.Q.copy(),
                    r=predictor_template.r,
                    mean=predictor_template.mean.copy(),
                    std=predictor_template.std.copy(),
                )
            fills, mtm, diagnostics = simulate(
                ticks,
                hs,
                fee_bps,
                args.inventory_skew,
                args.max_inventory,
                args.max_basis_stddev,
                args.single_sided,
                coef,
                predictor,
                cal_end,
                args.drift_horizon_ms,
                args.latency_ms,
                args.queue_scale,
                args.order_size,
                args.requote_threshold,
                args.unwind_half_spread_mult,
                external_predictions,
            )
            summarize(
                fills,
                mtm,
                hours,
                f"skew_coef={coef:+.2f}",
                ticks,
                args.drift_horizon_ms,
                diagnostics,
            )


if __name__ == "__main__":
    main()
