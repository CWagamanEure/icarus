#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Basis mean-reversion backtest.

Hypothesis: each venue's basis_estimate (vs the cross-venue common price) has
a persistent mean but mean-reverting short-term deviations. When a venue's
basis deviates N sigma from its rolling mean, trade the convergence: short
the rich leg, long the cheap reference leg. Exit when z returns to ~0, or
stop-out on timeout / extreme move.

Two-leg execution: for each entry we cross the spread on BOTH legs (taker).
For each exit we cross the spread on BOTH legs (taker). 4 fills per trade.

P&L per trade:
  entry: sell venue at venue_bid  ->  cash += venue_bid
         buy  ref   at ref_ask    ->  cash -= ref_ask
  exit:  buy  venue at venue_ask  ->  cash -= venue_ask (close short)
         sell ref   at ref_bid    ->  cash += ref_bid  (close long)
  + accumulated fees (sum of notional × fee_bps across all 4 legs)

Positive z (venue rich) → short venue, long ref.
Negative z (venue cheap) → long venue, short ref.
"""

from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAKER_FEE_BPS = {
    "coinbase": 6.0,
    "hyperliquid": 3.5,
    "okx": 10.0,
    "kraken": 26.0,
}


@dataclass(frozen=True, slots=True)
class VenueTicks:
    venue: str
    ts: np.ndarray              # int64, milliseconds
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    basis_estimate: np.ndarray  # reconstructed - basis_common


def load_venue_ticks(db_path: Path) -> dict[str, VenueTicks]:
    conn = sqlite3.connect(str(db_path))
    q = """
        SELECT u.timestamp_ms, v.exchange, v.fair_value, v.bid_price, v.ask_price,
               b.basis_estimate
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        JOIN basis_states b ON b.update_id = u.id AND b.exchange = v.exchange
        WHERE v.bid_price IS NOT NULL AND v.ask_price IS NOT NULL
          AND b.basis_estimate IS NOT NULL
        ORDER BY v.exchange, u.timestamp_ms, u.id
    """
    rows: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for ts, ex, mid, bid, ask, be in conn.execute(q):
        rows[ex].append((int(ts), float(mid), float(bid), float(ask), float(be)))
    out: dict[str, VenueTicks] = {}
    for ex, lst in rows.items():
        lst.sort(key=lambda r: r[0])
        out[ex] = VenueTicks(
            venue=ex,
            ts=np.asarray([r[0] for r in lst], dtype=np.int64),
            mid=np.asarray([r[1] for r in lst]),
            bid=np.asarray([r[2] for r in lst]),
            ask=np.asarray([r[3] for r in lst]),
            basis_estimate=np.asarray([r[4] for r in lst]),
        )
    conn.close()
    return out


def rolling_mean_std(
    x: np.ndarray, window_ms: int, ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean and std of x over a trailing window_ms. Expanding until full."""
    n = len(x)
    mean = np.zeros(n)
    std = np.zeros(n)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    j_start = 0
    for i in range(n):
        while ts[i] - ts[j_start] > window_ms and j_start < i:
            j_start += 1
        count = i - j_start + 1
        s1 = csum[i] - (csum[j_start - 1] if j_start > 0 else 0.0)
        s2 = csum2[i] - (csum2[j_start - 1] if j_start > 0 else 0.0)
        m = s1 / count
        var = max(s2 / count - m * m, 0.0)
        mean[i] = m
        std[i] = var ** 0.5
    return mean, std


def describe_deviations(venues: dict[str, VenueTicks], window_ms: int) -> None:
    print(f"\nrolling basis stats (window={window_ms/60000:.0f} min):")
    print(f"{'venue':<14} {'n':>8} {'mean':>10} {'std_of_basis':>14} "
          f"{'abs_z>2':>10} {'abs_z>3':>10}")
    print("-" * 72)
    for venue, vt in sorted(venues.items()):
        mean, std = rolling_mean_std(vt.basis_estimate, window_ms, vt.ts)
        good = std > 0.01
        z = np.where(good, (vt.basis_estimate - mean) / np.maximum(std, 1e-9), 0.0)
        pct2 = (np.abs(z[good]) > 2).mean() if good.any() else 0.0
        pct3 = (np.abs(z[good]) > 3).mean() if good.any() else 0.0
        print(f"{venue:<14} {len(vt.ts):>8} {vt.basis_estimate.mean():>+10.3f} "
              f"{vt.basis_estimate.std():>14.3f} {pct2:>10.2%} {pct3:>10.2%}")


@dataclass
class Trade:
    venue: str
    side: str  # "short_venue" or "long_venue"
    entry_ts: int
    entry_z: float
    entry_venue_px: float
    entry_ref_px: float
    exit_ts: int = 0
    exit_z: float = 0.0
    exit_venue_px: float = 0.0
    exit_ref_px: float = 0.0
    fees: float = 0.0
    pnl: float = 0.0
    size: float = 0.0


def backtest_venue(
    venues: dict[str, VenueTicks],
    venue: str,
    ref_venue: str,
    window_ms: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    max_hold_ms: int,
    size_notional: float,
    use_raw_basis: bool = False,
) -> list[Trade]:
    vt = venues[venue]
    rt = venues[ref_venue]

    if use_raw_basis:
        # Raw mid-price basis: venue_mid(t) - ref_mid(t_<=t), step-aligned.
        ref_mid_arr = (rt.bid + rt.ask) / 2.0
        ref_ts_np = rt.ts
        # For each venue tick, find most-recent ref mid (right-align stale ref to the past).
        idx = np.searchsorted(ref_ts_np, vt.ts, side="right") - 1
        valid = idx >= 0
        ref_mid_aligned = np.where(valid, ref_mid_arr[np.clip(idx, 0, None)], np.nan)
        venue_mid_arr = (vt.bid + vt.ask) / 2.0
        basis_series = venue_mid_arr - ref_mid_aligned
        # Skip NaNs at the head.
        basis_series = np.where(np.isnan(basis_series), 0.0, basis_series)
    else:
        basis_series = vt.basis_estimate

    mean, std = rolling_mean_std(basis_series, window_ms, vt.ts)
    z = np.where(std > 0.01, (basis_series - mean) / np.maximum(std, 1e-9), 0.0)

    ref_ts = rt.ts.tolist()

    def ref_bid_ask_at(ts_target: int) -> tuple[float, float] | None:
        j = bisect_left(ref_ts, ts_target)
        # Pick nearest prior snapshot; bail if the ref venue has no recent state.
        if j == 0 and ref_ts[0] > ts_target:
            return None
        if j >= len(ref_ts):
            j = len(ref_ts) - 1
        elif ref_ts[j] > ts_target and j > 0:
            j -= 1
        if ts_target - ref_ts[j] > 2000:
            return None
        return float(rt.bid[j]), float(rt.ask[j])

    venue_fee = TAKER_FEE_BPS[venue] / 10_000.0
    ref_fee = TAKER_FEE_BPS[ref_venue] / 10_000.0

    trades: list[Trade] = []
    pos: Trade | None = None
    i = 0
    # Skip warmup until rolling window is filled.
    warmup_until = vt.ts[0] + window_ms
    while i < len(vt.ts) and vt.ts[i] < warmup_until:
        i += 1

    while i < len(vt.ts):
        ts = int(vt.ts[i])
        zi = float(z[i])

        if pos is None:
            if abs(zi) >= entry_z:
                ref = ref_bid_ask_at(ts)
                if ref is None:
                    i += 1
                    continue
                r_bid, r_ask = ref
                if zi > 0:
                    # venue rich: short venue (sell at its bid), long ref (buy its ask)
                    size = size_notional / float(vt.bid[i])
                    pos = Trade(
                        venue=venue, side="short_venue",
                        entry_ts=ts, entry_z=zi,
                        entry_venue_px=float(vt.bid[i]),
                        entry_ref_px=r_ask,
                        size=size,
                    )
                else:
                    size = size_notional / float(vt.ask[i])
                    pos = Trade(
                        venue=venue, side="long_venue",
                        entry_ts=ts, entry_z=zi,
                        entry_venue_px=float(vt.ask[i]),
                        entry_ref_px=r_bid,
                        size=size,
                    )
                pos.fees += pos.size * (pos.entry_venue_px * venue_fee + pos.entry_ref_px * ref_fee)
            i += 1
            continue

        # Position open: check for exit conditions.
        hold_ms = ts - pos.entry_ts
        exit_now = (
            abs(zi) <= exit_z
            or abs(zi) >= stop_z
            or hold_ms >= max_hold_ms
        )
        if exit_now:
            ref = ref_bid_ask_at(ts)
            if ref is None:
                i += 1
                continue
            r_bid, r_ask = ref
            if pos.side == "short_venue":
                # close: buy venue at ask, sell ref at bid
                pos.exit_venue_px = float(vt.ask[i])
                pos.exit_ref_px = r_bid
                pos.exit_ts = ts
                pos.exit_z = zi
                venue_leg = pos.size * (pos.entry_venue_px - pos.exit_venue_px)  # short gain
                ref_leg = pos.size * (pos.exit_ref_px - pos.entry_ref_px)        # long gain
            else:
                pos.exit_venue_px = float(vt.bid[i])
                pos.exit_ref_px = r_ask
                pos.exit_ts = ts
                pos.exit_z = zi
                venue_leg = pos.size * (pos.exit_venue_px - pos.entry_venue_px)  # long gain
                ref_leg = pos.size * (pos.entry_ref_px - pos.exit_ref_px)        # short gain
            pos.fees += pos.size * (pos.exit_venue_px * venue_fee + pos.exit_ref_px * ref_fee)
            pos.pnl = venue_leg + ref_leg - pos.fees
            trades.append(pos)
            pos = None
        i += 1

    return trades


def report(trades: list[Trade], label: str) -> None:
    if not trades:
        print(f"{label}: no trades")
        return
    pnls = np.asarray([t.pnl for t in trades])
    fees = np.asarray([t.fees for t in trades])
    gross = pnls + fees
    holds = np.asarray([(t.exit_ts - t.entry_ts) / 1000.0 for t in trades])
    wins = (pnls > 0).sum()
    gross_wins = (gross > 0).sum()
    print(f"{label}:")
    print(f"  n={len(trades)}  net_pnl=${pnls.sum():+,.2f}  "
          f"gross_pnl=${gross.sum():+,.2f}  fees=${fees.sum():,.0f}")
    print(f"  avg_net=${pnls.mean():+.2f}  avg_gross=${gross.mean():+.2f}  "
          f"net_win%={wins/len(trades):.1%}  gross_win%={gross_wins/len(trades):.1%}")
    print(f"  avg_hold={holds.mean():.1f}s  med_hold={np.median(holds):.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", type=Path, default=Path("data/capture/2026-04-21.sqlite3"))
    p.add_argument("--ref-venue", default="coinbase")
    p.add_argument("--window-ms", type=int, default=300_000, help="rolling window for mean/std")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--stop-z", type=float, default=5.0)
    p.add_argument("--max-hold-ms", type=int, default=600_000)
    p.add_argument("--size-notional", type=float, default=10_000.0,
                   help="dollar notional per trade leg")
    p.add_argument("--skip-explore", action="store_true")
    p.add_argument("--raw-basis", action="store_true",
                   help="use raw venue_mid - ref_mid instead of Kalman basis_estimate")
    args = p.parse_args()

    print(f"loading {args.db_path}")
    venues = load_venue_ticks(args.db_path)
    print(f"loaded venues: {sorted(venues.keys())}  sizes={[(v, len(vt.ts)) for v, vt in venues.items()]}")

    if not args.skip_explore:
        describe_deviations(venues, args.window_ms)

    if args.ref_venue not in venues:
        raise SystemExit(f"ref venue {args.ref_venue} missing")

    duration_hours = (max(vt.ts[-1] for vt in venues.values())
                      - min(vt.ts[0] for vt in venues.values())) / 3_600_000.0
    print(f"\ntest duration ≈ {duration_hours:.1f} hours  "
          f"(entry_z={args.entry_z}, exit_z={args.exit_z}, stop_z={args.stop_z}, "
          f"hold<={args.max_hold_ms/60000:.0f}min, notional=${args.size_notional:,.0f})\n")

    for v in sorted(venues.keys()):
        if v == args.ref_venue:
            continue
        trades = backtest_venue(
            venues, v, args.ref_venue, args.window_ms,
            args.entry_z, args.exit_z, args.stop_z, args.max_hold_ms,
            args.size_notional, use_raw_basis=args.raw_basis,
        )
        report(trades, f"{v} vs {args.ref_venue}")
        if trades:
            per_hour = sum(t.pnl for t in trades) / duration_hours
            print(f"  ${per_hour:+,.2f}/hour")


if __name__ == "__main__":
    main()
