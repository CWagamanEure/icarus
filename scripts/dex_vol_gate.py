#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Test vol-gating hypothesis for LP.

For each hour: compute fees earned, LVR realized (dv_ext for in-range),
and a realized-vol proxy from pool mid moves. Then measure:
  (1) corr(prev_vol, net_pnl) — if strongly negative, gating can work
  (2) backtest: stay out if prev_k_hours vol > threshold

Works on existing attrib CSV from dex_lp_backtest.py.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", type=Path, required=True)
    ap.add_argument("--bucket-s", type=int, default=3600,
                    help="seconds per window (default 3600 = 1h)")
    ap.add_argument("--vol-window", type=int, default=3,
                    help="number of prior windows to measure vol over")
    args = ap.parse_args()

    print(f"loading {args.attrib}")
    buckets: dict[int, dict] = defaultdict(lambda: {
        "fee": 0.0, "dv": 0.0, "in_range_n": 0, "n": 0,
        "pool_first": None, "pool_last": None, "ret2": 0.0,
        "prev_post": None,
    })
    with args.attrib.open() as fh:
        r = csv.DictReader(fh)
        for row in r:
            ts = int(row["ts"])
            b = ts // args.bucket_s
            d = buckets[b]
            fee = float(row["fee"])
            dv = float(row["dv_ext"])
            in_range = int(row["in_range"])
            pool_pre = float(row["pool_pre"])
            pool_post = float(row["pool_post"])
            d["fee"] += fee
            if in_range:
                d["dv"] += dv
                d["in_range_n"] += 1
            d["n"] += 1
            if d["pool_first"] is None:
                d["pool_first"] = pool_pre
            d["pool_last"] = pool_post
            # per-swap squared log return, summed = realized variance
            if d["prev_post"] is not None and pool_pre > 0 and d["prev_post"] > 0:
                lr = math.log(pool_pre / d["prev_post"])
                d["ret2"] += lr * lr
            if pool_pre > 0 and pool_post > 0:
                lr2 = math.log(pool_post / pool_pre)
                d["ret2"] += lr2 * lr2
            d["prev_post"] = pool_post

    keys = sorted(buckets.keys())
    print(f"  {len(keys):,} buckets, {len(keys)*args.bucket_s/86400:.1f}d")

    fees = np.array([buckets[k]["fee"] for k in keys])
    dvs = np.array([buckets[k]["dv"] for k in keys])
    net = fees + dvs  # fees earned, dv is already negative for LVR
    rv = np.sqrt(np.array([buckets[k]["ret2"] for k in keys]))
    pool_ret = np.array([
        math.log(buckets[k]["pool_last"] / buckets[k]["pool_first"])
        if buckets[k]["pool_first"] else 0.0
        for k in keys
    ])

    print(f"\ntotals: fees=${fees.sum():.2f}  LVR=${-dvs.sum():.2f}  "
          f"net=${net.sum():.2f}")
    print(f"hours with net>0: {(net>0).sum()}/{len(net)} "
          f"({100*(net>0).mean():.1f}%)")

    # Lagged vol correlations
    print(f"\n(prev_k={args.vol_window}h vol) correlations:")
    w = args.vol_window
    if w < 1:
        w = 1
    # prev rolling sum of variance
    ret2 = np.array([buckets[k]["ret2"] for k in keys])
    prev_var = np.zeros_like(ret2)
    csum = np.cumsum(ret2)
    for i in range(len(keys)):
        lo = max(0, i - w)
        prev_var[i] = csum[i - 1] - (csum[lo - 1] if lo > 0 else 0.0) if i > 0 else 0.0
    prev_vol = np.sqrt(np.maximum(prev_var, 0.0))

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"  corr(prev_vol, fees)     = {corr(prev_vol, fees):+.3f}")
    print(f"  corr(prev_vol, -dv(LVR)) = {corr(prev_vol, -dvs):+.3f}")
    print(f"  corr(prev_vol, net)      = {corr(prev_vol, net):+.3f}")
    print(f"  corr(prev_vol, |ret|)    = {corr(prev_vol, np.abs(pool_ret)):+.3f}")

    # Simple gating backtest: skip window if prev_vol > threshold
    print("\n=== gating backtest: skip hour if prev_{w}h vol > q ===")
    print(f"{'quantile':>9} {'thresh':>10} {'kept':>6} "
          f"{'fees_k':>9} {'LVR_k':>9} {'net_k':>9} {'baseline_diff':>14}")
    base = net.sum()
    for q in [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]:
        th = np.quantile(prev_vol, q) if q < 1.0 else np.inf
        keep = prev_vol <= th
        f_k = fees[keep].sum()
        d_k = -dvs[keep].sum()
        n_k = net[keep].sum()
        print(f"{q:>9.2f} {th:>10.5f} {keep.sum():>6} "
              f"{f_k:>9.2f} {d_k:>9.2f} {n_k:>9.2f} {n_k - base:>+14.2f}")

    # Fee/LVR concentration by vol quartile
    print("\n=== concentration by prev-vol quartile ===")
    q1, q2, q3 = np.quantile(prev_vol, [0.25, 0.5, 0.75])
    for lo, hi, lbl in [(0, q1, "Q1 low"), (q1, q2, "Q2"),
                        (q2, q3, "Q3"), (q3, np.inf, "Q4 high")]:
        mask = (prev_vol > lo) & (prev_vol <= hi)
        if not mask.any():
            continue
        print(f"  {lbl:<7} [{lo:.5f},{hi:.5f}]  n={mask.sum():>4}  "
              f"fees=${fees[mask].sum():>8.2f}  "
              f"LVR=${-dvs[mask].sum():>8.2f}  "
              f"net=${net[mask].sum():>+8.2f}")


if __name__ == "__main__":
    main()
