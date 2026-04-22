#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Re-analyze a saved ticker_poll_*.jsonl file without re-polling.

Same pairwise taker-arb metric as thin_alt_live_ticker_poll.py, but lets you
replay stored data under different fee assumptions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="path to ticker_poll_*.jsonl")
    p.add_argument("--cb-taker-bps", type=float, default=6.0)
    p.add_argument("--kr-taker-bps", type=float, default=25.0)
    p.add_argument("--okx-taker-bps", type=float, default=10.0)
    p.add_argument("--align-tolerance-sec", type=float, default=8.0)
    args = p.parse_args()

    # Load rows
    raw: list[dict] = []
    with args.input.open() as f:
        for line in f:
            raw.append(json.loads(line))

    # Group by (symbol, venue)
    ba_by: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    usdt: list[tuple[float, float]] = []
    for r in raw:
        if r["symbol"] == "USDT" and r["venue"] == "coinbase":
            usdt.append((float(r["ts"]), 0.5 * (float(r["bid"]) + float(r["ask"]))))
        else:
            ba_by[(r["symbol"], r["venue"])].append(
                (float(r["ts"]), float(r["bid"]), float(r["ask"]))
            )

    if not usdt:
        print("WARN: no USDT samples, using 1.0 for OKX correction")
        usdt_ts = np.asarray([0.0])
        usdt_mid = np.asarray([1.0])
    else:
        usdt.sort()
        usdt_ts = np.asarray([t for t, _ in usdt])
        usdt_mid = np.asarray([m for _, m in usdt])

    def usdt_at(ts: float) -> float:
        i = int(np.argmin(np.abs(usdt_ts - ts)))
        return float(usdt_mid[i])

    print(f"loaded {len(raw)} rows")
    print(f"usdt/usd samples: {len(usdt)}, median {float(np.median(usdt_mid)):.4f}")

    # Per-venue bid/ask spread table
    print("\n=== per-venue bid/ask (taker spread, bps) ===")
    print(f"{'symbol':<10} {'venue':<10} {'n':>5} {'median':>8} {'p95':>8}")
    for (sym, venue), rows in sorted(ba_by.items()):
        arr = np.asarray([(a - b) / (0.5 * (a + b)) * 1e4
                          for (_, b, a) in rows if a > 0 and b > 0])
        if arr.size == 0:
            continue
        print(f"{sym:<10} {venue:<10} {arr.size:>5} "
              f"{float(np.median(arr)):>8.1f} {float(np.percentile(arr, 95)):>8.1f}")

    # Pairwise taker arb
    fee_pair = {
        ("coinbase", "kraken"): args.cb_taker_bps + args.kr_taker_bps,
        ("coinbase", "okx"): args.cb_taker_bps + args.okx_taker_bps,
        ("kraken", "okx"): args.kr_taker_bps + args.okx_taker_bps,
    }

    symbols = sorted({s for (s, _) in ba_by.keys()})

    def series(sym: str, v: str):
        rows = ba_by.get((sym, v), [])
        rows.sort()
        ts = np.asarray([r[0] for r in rows])
        bid = np.asarray([r[1] for r in rows])
        ask = np.asarray([r[2] for r in rows])
        return ts, bid, ask

    print("\n=== pairwise cross-venue taker arb edge (bps, pre-fee) ===")
    print(f"{'symbol':<10} {'pair':<8} {'n':>5} "
          f"{'median':>8} {'p95':>8} {'max':>8} "
          f"{'frac>0':>8} {'frac>fee':>9} {'gross(>fee)':>12}")
    for sym in symbols:
        for v1, v2 in [("coinbase", "kraken"), ("coinbase", "okx"), ("kraken", "okx")]:
            t1, b1, a1 = series(sym, v1)
            t2, b2, a2 = series(sym, v2)
            if t1.size == 0 or t2.size == 0:
                continue
            edges: list[float] = []
            for i, ts in enumerate(t1):
                j = int(np.argmin(np.abs(t2 - ts)))
                if abs(t2[j] - ts) > args.align_tolerance_sec:
                    continue
                if v1 == "okx":
                    u = usdt_at(ts)
                    bid1, ask1 = b1[i] * u, a1[i] * u
                else:
                    bid1, ask1 = b1[i], a1[i]
                if v2 == "okx":
                    u = usdt_at(ts)
                    bid2, ask2 = b2[j] * u, a2[j] * u
                else:
                    bid2, ask2 = b2[j], a2[j]
                avg = 0.25 * (bid1 + ask1 + bid2 + ask2)
                edge = max(bid1 - ask2, bid2 - ask1) / avg * 1e4
                edges.append(edge)
            if not edges:
                continue
            arr = np.asarray(edges)
            fee = fee_pair[(v1, v2)]
            pair_label = v1[:2] + "-" + v2[:2]
            frac_pos = float((arr > 0).mean()) * 100
            frac_over = float((arr > fee).mean()) * 100
            # gross edge *given* we only trade when over fee (mean of arr[arr>fee] - fee)
            profitable = arr[arr > fee]
            gross = float(profitable.mean() - fee) if profitable.size else 0.0
            print(f"{sym:<10} {pair_label:<8} {arr.size:>5} "
                  f"{float(np.median(arr)):>8.1f} "
                  f"{float(np.percentile(arr, 95)):>8.1f} "
                  f"{float(arr.max()):>8.1f} "
                  f"{frac_pos:>7.1f}% {frac_over:>8.1f}% {gross:>12.1f}")

    print("\nfees (bps/leg): "
          f"cb={args.cb_taker_bps}, kr={args.kr_taker_bps}, okx={args.okx_taker_bps}")


if __name__ == "__main__":
    main()
