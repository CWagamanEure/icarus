#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Bucket simulated maker fills by market-regime (forward mid return) and
report per-fill P&L per bucket. Tests whether losses concentrate in
downtrends vs uptrends.
"""

from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simulate_basis_maker import load_ticks, simulate, DEFAULT_FEE_BPS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", type=Path, default=Path("data/capture/2026-04-21.sqlite3"))
    p.add_argument("--venue", default="coinbase")
    p.add_argument("--predictions-path", type=Path,
                   default=Path("data/predictions/coinbase_5000ms.csv"))
    p.add_argument("--half-spread", type=float, default=2.0)
    p.add_argument("--skew-coef", type=float, default=2.0)
    p.add_argument("--fee-bps", type=float, default=None)
    p.add_argument("--order-size", type=float, default=1.0)
    p.add_argument("--max-inventory", type=float, default=10.0)
    p.add_argument("--latency-ms", type=int, default=50)
    p.add_argument("--queue-scale", type=float, default=1.0)
    p.add_argument("--requote-threshold", type=float, default=1.0)
    p.add_argument("--max-venue-age-ms", type=float, default=500.0)
    p.add_argument("--regime-window-ms", type=int, default=30_000,
                   help="forward window used to measure regime at each fill")
    p.add_argument("--pnl-window-ms", type=int, default=30_000,
                   help="mark-to-market window per fill for per-fill P&L")
    args = p.parse_args()

    fee_bps = args.fee_bps if args.fee_bps is not None else DEFAULT_FEE_BPS.get(args.venue, 0.0)
    print(f"venue={args.venue}  fee_bps={fee_bps:+.2f}  hs={args.half_spread}  skew={args.skew_coef}")

    ticks = load_ticks(args.db_path, args.venue, args.max_venue_age_ms)
    if not ticks:
        raise SystemExit(f"no ticks for {args.venue}")

    ext = {}
    with args.predictions_path.open() as f:
        header = f.readline().strip().split(",")
        ts_i, dh_i = header.index("timestamp_ms"), header.index("drift_hat")
        for line in f:
            parts = line.strip().split(",")
            ext[int(parts[ts_i])] = float(parts[dh_i])
    print(f"loaded {len(ext)} predictions")

    fills, mtm, diag = simulate(
        ticks=ticks,
        half_spread=args.half_spread,
        fee_bps=fee_bps,
        inventory_skew=0.0,
        max_inventory=args.max_inventory,
        max_basis_stddev=None,
        single_sided=False,
        drift_skew_coef=args.skew_coef,
        predictor=None,
        calibration_end=0,
        horizon_ms=5000,
        latency_ms=args.latency_ms,
        queue_scale=args.queue_scale,
        order_size=args.order_size,
        requote_threshold=args.requote_threshold,
        unwind_half_spread_mult=1.0,
        external_predictions=ext,
    )
    hours = (ticks[-1].timestamp_ms - ticks[0].timestamp_ms) / 3_600_000.0
    print(f"fills={len(fills)}  total_mtm=${mtm:+,.2f}  per_hour=${mtm/hours:+,.2f}")

    tick_ts = [t.timestamp_ms for t in ticks]
    tick_mid = [t.venue_mid for t in ticks]

    def mid_at_forward(ts: int, dt_ms: int) -> float | None:
        j = bisect_left(tick_ts, ts + dt_ms)
        if j >= len(ticks):
            return None
        return tick_mid[j]

    # Build per-fill records.
    records: list[tuple[float, float, str, float]] = []  # (regime_ret, pnl, side, size)
    for f in fills:
        pre_mid = f.reference
        post_pnl_mid = mid_at_forward(f.timestamp_ms, args.pnl_window_ms)
        post_regime_mid = mid_at_forward(f.timestamp_ms, args.regime_window_ms)
        if post_pnl_mid is None or post_regime_mid is None:
            continue
        regime_ret = (post_regime_mid - pre_mid) / pre_mid * 10_000.0  # bps
        size = args.order_size
        if f.side == "buy":
            pnl = (post_pnl_mid - f.price) * size - f.fee
        else:
            pnl = (f.price - post_pnl_mid) * size - f.fee
        records.append((regime_ret, pnl, f.side, size))

    if not records:
        print("no records to analyze")
        return

    regime_rets = np.asarray([r[0] for r in records])
    pnls = np.asarray([r[1] for r in records])

    # Quartile buckets by regime forward return.
    q = np.quantile(regime_rets, [0.0, 0.25, 0.50, 0.75, 1.0])
    labels = ["Q1 DOWN", "Q2 soft-down", "Q3 soft-up", "Q4 UP"]
    print(f"\nregime buckets (bps, {args.regime_window_ms}ms forward return)")
    print(f"  Q1: [{q[0]:+.1f}, {q[1]:+.1f}]  Q2: [{q[1]:+.1f}, {q[2]:+.1f}]  "
          f"Q3: [{q[2]:+.1f}, {q[3]:+.1f}]  Q4: [{q[3]:+.1f}, {q[4]:+.1f}]")

    print(f"\n{'regime':<14} {'n':>6} {'buys':>6} {'sells':>6} "
          f"{'avg_pnl':>10} {'total':>12} {'avg_regime':>12}")
    print("-" * 72)
    for i, label in enumerate(labels):
        lo, hi = q[i], q[i + 1]
        if i == 0:
            mask = regime_rets <= hi
        elif i == 3:
            mask = regime_rets > lo
        else:
            mask = (regime_rets > lo) & (regime_rets <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bucket_pnls = pnls[mask]
        bucket_sides = np.asarray([r[2] for r, m in zip(records, mask) if m])
        buys = int((bucket_sides == "buy").sum())
        sells = int((bucket_sides == "sell").sum())
        avg = bucket_pnls.mean()
        total = bucket_pnls.sum()
        avg_regime = regime_rets[mask].mean()
        print(f"{label:<14} {n:>6} {buys:>6} {sells:>6} "
              f"${avg:>+9.2f} ${total:>+11.2f} {avg_regime:>+10.1f} bps")

    print(f"\ntotal per-fill P&L: ${pnls.sum():+,.2f}  avg/fill: ${pnls.mean():+.2f}  n={len(records)}")


if __name__ == "__main__":
    main()
