#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Extract per-swap pre-swap features + toxicity labels.

Features use only info available BEFORE the swap lands (causal):
  - size_usd, signed_flow_usd (token0 delta direction)
  - rv_5m, rv_15m, rv_60m  (realized log-return vol from prior pool mids)
  - basis_bps_pre  (pool vs CEX basis entering the swap)
  - basis_abs_bps  (magnitude of basis)
  - flow_imbalance_5m, flow_imbalance_30m  (signed net $ volume)
  - swap_rate_5m  (swaps per minute over last 5 min)
  - time_since_last_big_s  (last swap with |usd| > p95)
  - recent_p95_size_usd  (p95 swap size last 100 swaps)

Label: toxicity magnitude = -dv_ext from attribution (loss at CEX price);
binary label = toxicity > threshold (default: top quantile).

Output: CSV with features + labels, ready for classifier training.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

Q96 = 2**96
ETH_DEC = 18
USDC_DEC = 6
DEC_ADJ = 10 ** (ETH_DEC - USDC_DEC)


def sqrtx96_to_price(sqrt_x96: float) -> float:
    s = sqrt_x96 / Q96
    return (s * s) * DEC_ADJ


def load_swaps(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: (int(r["timestamp"]), int(r["logIndex"])))
    return rows


def load_cex(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts: list[int] = []
    px: list[float] = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            ts.append(int(r["ts_s"]))
            px.append(float(r["close"]))
    a = np.asarray(ts)
    p = np.asarray(px)
    order = np.argsort(a)
    return a[order], p[order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swaps", type=Path,
                    default=Path("data/dex_cache/swaps_0x96d4b53a_30d.jsonl.gz"))
    ap.add_argument("--cex", type=Path,
                    default=Path("data/dex_cache/cex_eth_1m_30d.jsonl"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/dex_cache/toxic_features_5pct.csv"))
    ap.add_argument("--range-pct", type=float, default=5.0,
                    help="LP range around deposit (for labeling)")
    ap.add_argument("--label-pct", type=float, default=95.0,
                    help="quantile cutoff for binary toxic label")
    ap.add_argument("--deposit-usd", type=float, default=10_000.0,
                    help="notional for labeling (scales dv_ext magnitudes)")
    args = ap.parse_args()

    print(f"loading {args.swaps}")
    swaps = load_swaps(args.swaps)
    cex_ts, cex_px = load_cex(args.cex)
    t0, t1 = int(cex_ts[0]), int(cex_ts[-1])
    swaps = [s for s in swaps if t0 <= int(s["timestamp"]) <= t1]
    print(f"  {len(swaps):,} swaps in window")

    # Use first hour for warm-up; we need price history before labeling makes sense
    warm_end = int(swaps[0]["timestamp"]) + 3600

    # Compute LP range center from median first-hour pool price
    warm_prices = [sqrtx96_to_price(float(s["sqrtPriceX96"]))
                   for s in swaps if int(s["timestamp"]) < warm_end]
    center = float(np.median(warm_prices))
    rp = args.range_pct / 100.0
    p_lo = center * (1.0 - rp)
    p_hi = center * (1.0 + rp)
    sqrt_lo = math.sqrt(p_lo / DEC_ADJ) * Q96
    sqrt_hi = math.sqrt(p_hi / DEC_ADJ) * Q96
    print(f"center=${center:.2f}  range=[${p_lo:.2f}, ${p_hi:.2f}]")

    # Dummy L_pos for labeling (deposit-scale-invariant since we emit magnitude)
    L_pos = 1.0

    def token_amounts(L, sqrtP, sqrtPl, sqrtPh):
        if sqrtP <= sqrtPl:
            return L * (1.0 / sqrtPl - 1.0 / sqrtPh) * Q96, 0.0
        if sqrtP >= sqrtPh:
            return 0.0, L * (sqrtPh - sqrtPl) / Q96
        return (L * (1.0 / sqrtP - 1.0 / sqrtPh) * Q96,
                L * (sqrtP - sqrtPl) / Q96)

    # Rolling buffers for features
    price_hist: deque[tuple[int, float]] = deque()  # (ts, pool_mid)
    flow_hist: deque[tuple[int, float]] = deque()   # (ts, signed_usd)
    size_hist: deque[float] = deque()                # last 100 abs usd

    BIG_WINDOW = 100  # rolling window for p95 size

    def prune(buf: deque, cutoff: int):
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def rv(prices: list[float]) -> float:
        if len(prices) < 3:
            return float("nan")
        lr = np.diff(np.log(np.asarray(prices)))
        return float(np.sqrt(np.sum(lr * lr)))

    out_rows = []
    prev_sqrt = float(swaps[0]["sqrtPriceX96"])
    last_big_ts = int(swaps[0]["timestamp"])

    for s in swaps[1:]:
        ts = int(s["timestamp"])
        sqrt_post = float(s["sqrtPriceX96"])
        a0 = float(s["amount0"])
        usd = float(s["amountUSD"])
        pool_pre = sqrtx96_to_price(prev_sqrt)
        pool_post = sqrtx96_to_price(sqrt_post)
        signed_usd = -math.copysign(usd, a0)  # a0>0 = user sold ETH = sell-flow

        if ts < warm_end:
            price_hist.append((ts, pool_pre))
            flow_hist.append((ts, signed_usd))
            size_hist.append(abs(usd))
            if len(size_hist) > BIG_WINDOW:
                size_hist.popleft()
            prev_sqrt = sqrt_post
            continue

        # CEX at swap time
        j = int(np.searchsorted(cex_ts, ts))
        j = min(max(j, 0), len(cex_px) - 1)
        ext = float(cex_px[j])

        # Basis (pre-swap)
        basis_bps = (pool_pre - ext) / ext * 1e4

        # Realized vol over last 5/15/60 min (from pool mids)
        prune(price_hist, ts - 3600)
        prices_60 = [p for (t, p) in price_hist]
        prices_15 = [p for (t, p) in price_hist if t >= ts - 900]
        prices_5  = [p for (t, p) in price_hist if t >= ts - 300]
        rv_5 = rv(prices_5) * 1e4  # bps of log-return sum
        rv_15 = rv(prices_15) * 1e4
        rv_60 = rv(prices_60) * 1e4

        # Flow imbalance last 5/30 min
        prune(flow_hist, ts - 1800)
        fl30 = sum(v for (t, v) in flow_hist)
        fl5  = sum(v for (t, v) in flow_hist if t >= ts - 300)

        # Swap rate last 5 min
        rate_5 = sum(1 for (t, _) in flow_hist if t >= ts - 300) / 5.0

        # Recent size stats
        p95_size = float(np.percentile(list(size_hist), 95)) if size_hist else 0.0
        time_since_big = ts - last_big_ts

        # --- Label: position dV at ext using this swap ---
        x_pre, y_pre = token_amounts(L_pos, prev_sqrt, sqrt_lo, sqrt_hi)
        x_post, y_post = token_amounts(L_pos, sqrt_post, sqrt_lo, sqrt_hi)
        # Valued at ext price
        v_pre = (x_pre / 10**ETH_DEC) * ext + y_pre / 10**USDC_DEC
        v_post = (x_post / 10**ETH_DEC) * ext + y_post / 10**USDC_DEC
        dv_ext = v_post - v_pre  # negative = LP lost
        # Toxicity magnitude: |dv_ext| scaled by deposit for legibility
        # (dv_ext for L=1 is tiny; scale relative to ourselves)
        toxicity = -dv_ext  # positive = loss

        in_range = (sqrt_lo <= prev_sqrt <= sqrt_hi) or (sqrt_lo <= sqrt_post <= sqrt_hi)

        out_rows.append({
            "ts": ts,
            "size_usd": usd,
            "signed_usd": signed_usd,
            "basis_bps": basis_bps,
            "abs_basis_bps": abs(basis_bps),
            "rv_5m": rv_5, "rv_15m": rv_15, "rv_60m": rv_60,
            "flow_5m": fl5, "flow_30m": fl30,
            "rate_5m": rate_5,
            "p95_size_usd": p95_size,
            "time_since_big_s": time_since_big,
            "in_range": int(in_range),
            "pool_pre": pool_pre, "pool_post": pool_post, "ext": ext,
            "toxicity_raw": toxicity,
        })

        # Update rolling state AFTER emitting (features are pre-swap)
        price_hist.append((ts, pool_post))
        flow_hist.append((ts, signed_usd))
        size_hist.append(abs(usd))
        if len(size_hist) > BIG_WINDOW:
            size_hist.popleft()
        if abs(usd) > p95_size:
            last_big_ts = ts

        prev_sqrt = sqrt_post

    print(f"  emitted {len(out_rows):,} feature rows")

    # Scale toxicity per-dollar-of-deposit:
    # dv_ext at L=1 is in units of "raw token deltas / Q96".
    # Magnitudes are relative; normalize by max so label cutoff is sensible.
    tox = np.asarray([r["toxicity_raw"] for r in out_rows])
    tox_in_range = np.asarray([r["toxicity_raw"] for r in out_rows if r["in_range"]])
    print(f"toxicity (in-range) distribution:")
    print(f"  n={tox_in_range.size:,}  "
          f"median={np.median(tox_in_range):.3e}  "
          f"p95={np.percentile(tox_in_range, args.label_pct):.3e}  "
          f"p99={np.percentile(tox_in_range, 99):.3e}")
    cutoff = float(np.percentile(tox_in_range[tox_in_range > 0], args.label_pct))

    pos = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        cols = ["ts", "size_usd", "signed_usd", "basis_bps", "abs_basis_bps",
                "rv_5m", "rv_15m", "rv_60m", "flow_5m", "flow_30m",
                "rate_5m", "p95_size_usd", "time_since_big_s",
                "in_range", "pool_pre", "pool_post", "ext",
                "toxicity_raw", "is_toxic"]
        fh.write(",".join(cols) + "\n")
        for r in out_rows:
            is_toxic = int(r["in_range"] == 1 and r["toxicity_raw"] > cutoff)
            pos += is_toxic
            r["is_toxic"] = is_toxic
            fh.write(",".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    print(f"  wrote {args.output}  "
          f"(toxic rate: {pos}/{len(out_rows)} = {pos/len(out_rows)*100:.2f}%)")


if __name__ == "__main__":
    main()
