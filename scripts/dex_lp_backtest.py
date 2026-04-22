#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Backtest a concentrated liquidity LP position on a V4 ETH/USDC pool.

Loads swap history + CEX reference, simulates a passive LP with a fixed
price range and deposit size, accumulates fees, tracks LVR vs HODL and
vs CEX reference.

Core math (Uniswap V3 concentrated liquidity):
    x = L * (1/sqrtP - 1/sqrtPh)   if sqrtPl <= sqrtP <= sqrtPh
    y = L * (sqrtP - sqrtPl)
    x fully (all token0) if sqrtP <= sqrtPl
    y fully (all token1) if sqrtP >= sqrtPh

Pool active L is estimated per-swap from (amount, sqrtPre, sqrtPost).
Our fee share = L_position / L_pool_active at that swap.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

Q96 = 2**96
ETH_DEC = 18
USDC_DEC = 6
DEC_ADJ = 10 ** (ETH_DEC - USDC_DEC)  # 10^12


def sqrtx96_to_price(sqrt_x96: float) -> float:
    """Convert sqrtPriceX96 to USDC/ETH price (human units)."""
    s = sqrt_x96 / Q96
    return (s * s) * DEC_ADJ


def price_to_sqrtx96(price_usdc_per_eth: float) -> float:
    raw = price_usdc_per_eth / DEC_ADJ
    return math.sqrt(raw) * Q96


def token_amounts(L: float, sqrtP: float, sqrtPl: float, sqrtPh: float):
    """Return (x_eth_raw, y_usdc_raw) for a position (raw integer-scaled amounts)."""
    if sqrtP <= sqrtPl:
        x = L * (1.0 / sqrtPl - 1.0 / sqrtPh) * Q96
        y = 0.0
    elif sqrtP >= sqrtPh:
        x = 0.0
        y = L * (sqrtPh - sqrtPl) / Q96
    else:
        x = L * (1.0 / sqrtP - 1.0 / sqrtPh) * Q96
        y = L * (sqrtP - sqrtPl) / Q96
    return x, y


def liquidity_for_deposit(
    deposit_usd: float,
    sqrtP: float,
    sqrtPl: float,
    sqrtPh: float,
    price: float,
) -> float:
    """Find L such that position value = deposit_usd at current price.

    Value(L=1) = x1 * price_eth_in_usdc + y1 (both in raw units, so
    convert x from 1e18-scaled to ETH and y from 1e6-scaled to USDC).
    """
    x1_raw, y1_raw = token_amounts(1.0, sqrtP, sqrtPl, sqrtPh)
    x1_eth = x1_raw / 10**ETH_DEC
    y1_usdc = y1_raw / 10**USDC_DEC
    v1 = x1_eth * price + y1_usdc
    if v1 <= 0:
        raise ValueError("zero-value unit position; check range relative to price")
    return deposit_usd / v1


def estimate_pool_L(amount0: float, amount1: float, sqrt_pre: float, sqrt_post: float) -> float:
    """Estimate active pool liquidity from a swap.

    amount1 delta and sqrt delta give: L ≈ |amount1_raw| * Q96 / |sqrt_post - sqrt_pre|
    We prefer token1 (USDC) because its 6-decimal units amplify precision.
    Falls back to token0 form if needed.
    """
    d_sqrt = abs(sqrt_post - sqrt_pre)
    if d_sqrt <= 0:
        return float("nan")
    # y = L * dSqrtP / Q96  →  L = |dy| * Q96 / dSqrtP
    y_raw = abs(amount1) * 10**USDC_DEC
    if y_raw > 0:
        return y_raw * Q96 / d_sqrt
    # fallback: dx = L * d(1/sqrtP) * Q96  →  dSqrt / (sqrt*sqrt) = 1/pre - 1/post
    x_raw = abs(amount0) * 10**ETH_DEC
    if x_raw > 0 and sqrt_pre > 0 and sqrt_post > 0:
        d_inv = abs(1.0 / sqrt_pre - 1.0 / sqrt_post)
        if d_inv > 0:
            return x_raw / (d_inv * Q96)
    return float("nan")


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
    ap.add_argument("--fee-bps", type=float, default=5.0,
                    help="pool fee tier in bps (0.05% = 5)")
    ap.add_argument("--range-pct", type=float, default=5.0,
                    help="+/- pct around deposit price for the LP range")
    ap.add_argument("--deposit-usd", type=float, default=10_000.0)
    ap.add_argument("--skip-hours", type=float, default=1.0,
                    help="skip first N hours to warm up pool_L estimate")
    ap.add_argument("--attrib-out", type=Path, default=None,
                    help="if set, write per-swap attribution CSV here")
    args = ap.parse_args()

    print(f"loading swaps from {args.swaps}")
    swaps = load_swaps(args.swaps)
    print(f"  {len(swaps):,} swaps")

    print(f"loading cex from {args.cex}")
    cex_ts, cex_px = load_cex(args.cex)
    print(f"  {len(cex_ts):,} cex bars, range ${cex_px.min():.1f}–${cex_px.max():.1f}")

    # Filter swaps to window covered by CEX reference
    t0 = int(cex_ts[0])
    t1 = int(cex_ts[-1])
    swaps = [s for s in swaps if t0 <= int(s["timestamp"]) <= t1]
    print(f"  {len(swaps):,} swaps in cex window  "
          f"[{t0} .. {t1}]  ({(t1-t0)/86400:.1f}d)")

    if len(swaps) < 100:
        raise SystemExit("too few swaps in window")

    # Warm-up: establish pool L estimate and deposit price
    warm_end = int(swaps[0]["timestamp"]) + int(args.skip_hours * 3600)
    deposit_idx = next(i for i, s in enumerate(swaps) if int(s["timestamp"]) >= warm_end)
    dep_swap = swaps[deposit_idx]
    dep_sqrt = float(dep_swap["sqrtPriceX96"])
    dep_price = sqrtx96_to_price(dep_sqrt)
    print(f"\ndeposit at swap #{deposit_idx}  ts={dep_swap['timestamp']}  "
          f"price=${dep_price:.2f}")

    rp = args.range_pct / 100.0
    p_lo = dep_price * (1.0 - rp)
    p_hi = dep_price * (1.0 + rp)
    sqrt_lo = price_to_sqrtx96(p_lo)
    sqrt_hi = price_to_sqrtx96(p_hi)
    print(f"range: ${p_lo:.2f} .. ${p_hi:.2f} (+/- {args.range_pct}%)")

    L_pos = liquidity_for_deposit(args.deposit_usd, dep_sqrt, sqrt_lo, sqrt_hi, dep_price)
    x0_raw, y0_raw = token_amounts(L_pos, dep_sqrt, sqrt_lo, sqrt_hi)
    x0_eth = x0_raw / 10**ETH_DEC
    y0_usdc = y0_raw / 10**USDC_DEC
    v0 = x0_eth * dep_price + y0_usdc
    print(f"L_position = {L_pos:.3e}")
    print(f"initial: {x0_eth:.4f} ETH + {y0_usdc:.2f} USDC  = ${v0:.2f}")

    fee_rate = args.fee_bps / 10_000.0

    # Iterate swaps, compute per-swap pool L, accumulate fees, attribute LVR
    fees_usd = 0.0
    lvr_usd = 0.0
    toxic_lvr = 0.0
    noise_lvr = 0.0
    in_range_seconds = 0
    out_range_seconds = 0
    prev_ts = int(dep_swap["timestamp"])
    prev_sqrt = dep_sqrt
    share_samples: list[float] = []
    pool_L_samples: list[float] = []
    attrib_rows: list[tuple] = []

    for s in swaps[deposit_idx + 1:]:
        ts = int(s["timestamp"])
        sqrt_post = float(s["sqrtPriceX96"])
        a0 = float(s["amount0"])
        a1 = float(s["amount1"])
        usd = float(s["amountUSD"])

        dt = ts - prev_ts
        in_range_pre = sqrt_lo <= prev_sqrt <= sqrt_hi
        in_range_post = sqrt_lo <= sqrt_post <= sqrt_hi
        in_range = in_range_pre or in_range_post
        if in_range_pre:
            in_range_seconds += dt
        else:
            out_range_seconds += dt

        pool_price_pre = sqrtx96_to_price(prev_sqrt)
        pool_price_post = sqrtx96_to_price(sqrt_post)
        j = int(np.searchsorted(cex_ts, ts))
        j = min(max(j, 0), len(cex_px) - 1)
        ext_price = float(cex_px[j])

        # Per-swap dV_LP valued at external oracle price. Convention:
        # dV is change in position USD value using ext_price to mark ETH.
        # Toward-oracle swaps (arb): pool_after closer to ext -> LP loses.
        x_pre, y_pre = token_amounts(L_pos, prev_sqrt, sqrt_lo, sqrt_hi)
        x_post, y_post = token_amounts(L_pos, sqrt_post, sqrt_lo, sqrt_hi)
        v_pre = (x_pre / 10**ETH_DEC) * ext_price + y_pre / 10**USDC_DEC
        v_post = (x_post / 10**ETH_DEC) * ext_price + y_post / 10**USDC_DEC
        dv_ext = v_post - v_pre  # negative = LP lost value vs oracle

        dist_pre = abs(pool_price_pre - ext_price)
        dist_post = abs(pool_price_post - ext_price)
        toward = dist_post < dist_pre  # swap moved pool toward oracle (toxic)

        L_pool = estimate_pool_L(a0, a1, prev_sqrt, sqrt_post)
        share = 0.0
        our_fee = 0.0
        if L_pool and L_pool == L_pool and L_pool > 0:
            pool_L_samples.append(L_pool)
            mid_sqrt = 0.5 * (prev_sqrt + sqrt_post)
            if sqrt_lo <= mid_sqrt <= sqrt_hi:
                share = L_pos / L_pool
                share_samples.append(share)
                our_fee = usd * fee_rate * share
                fees_usd += our_fee

        if in_range:
            lvr_usd += -dv_ext  # LVR = sum of losses
            if toward:
                toxic_lvr += -dv_ext
            else:
                noise_lvr += -dv_ext

        if args.attrib_out is not None:
            attrib_rows.append((
                ts, usd, pool_price_pre, pool_price_post, ext_price,
                dist_pre, dist_post, int(toward), int(in_range),
                share, our_fee, dv_ext,
            ))

        prev_ts = ts
        prev_sqrt = sqrt_post

    # Final position value at last swap price
    final_sqrt = prev_sqrt
    final_price = sqrtx96_to_price(final_sqrt)
    xf_raw, yf_raw = token_amounts(L_pos, final_sqrt, sqrt_lo, sqrt_hi)
    xf_eth = xf_raw / 10**ETH_DEC
    yf_usdc = yf_raw / 10**USDC_DEC
    v_lp = xf_eth * final_price + yf_usdc

    # HODL at final price: started with x0 ETH + y0 USDC
    v_hodl = x0_eth * final_price + y0_usdc

    # CEX-ref LVR: evaluate at external oracle price (final CEX bar)
    # idx of nearest cex bar to prev_ts
    j = int(np.argmin(np.abs(cex_ts - prev_ts)))
    ext_price = float(cex_px[j])

    # LVR vs CEX: continuously rebalance against external price.
    # Approx: same as vs pool HODL but using ext_price at end.
    v_hodl_ext = x0_eth * ext_price + y0_usdc
    # What LP would be worth if priced at CEX
    xf_e_raw, yf_e_raw = token_amounts(L_pos, price_to_sqrtx96(ext_price), sqrt_lo, sqrt_hi)
    v_lp_ext = (xf_e_raw / 10**ETH_DEC) * ext_price + (yf_e_raw / 10**USDC_DEC)

    span_s = prev_ts - int(dep_swap["timestamp"])
    span_d = span_s / 86400.0

    print("\n=== results ===")
    print(f"window: {span_d:.2f}d ({span_s} s)")
    print(f"in-range fraction: {in_range_seconds/(in_range_seconds+out_range_seconds+1e-9):.1%}")
    print(f"final pool price: ${final_price:.2f}  (deposit ${dep_price:.2f}, "
          f"Δ {100*(final_price/dep_price-1):+.2f}%)")
    print(f"final CEX ETH-USD: ${ext_price:.2f}")

    print(f"\nposition value now (pool price): ${v_lp:.2f}")
    print(f"HODL value now (pool price):     ${v_hodl:.2f}")
    print(f"   LVR vs HODL (pool):   ${v_hodl - v_lp:+.2f}")

    print(f"\nposition value (CEX price):  ${v_lp_ext:.2f}")
    print(f"HODL value (CEX price):      ${v_hodl_ext:.2f}")
    print(f"   LVR vs HODL (CEX):  ${v_hodl_ext - v_lp_ext:+.2f}")

    print(f"\nfees accrued (est):  ${fees_usd:.2f}")
    print(f"net P&L vs HODL (pool):  ${v_lp - v_hodl + fees_usd:+.2f}  "
          f"({(v_lp - v_hodl + fees_usd)/args.deposit_usd*100:+.2f}% on ${args.deposit_usd:.0f})")
    print(f"annualized:  {(v_lp - v_hodl + fees_usd)/args.deposit_usd * 365/span_d * 100:+.1f}%")

    if share_samples:
        sh = np.asarray(share_samples)
        print(f"\nfee-share samples: n={sh.size:,}  "
              f"median={np.median(sh)*1e4:.2f} bps  "
              f"p95={np.percentile(sh,95)*1e4:.2f} bps")
    if pool_L_samples:
        pl = np.asarray(pool_L_samples)
        print(f"pool active-L: median={np.median(pl):.3e}  p95={np.percentile(pl,95):.3e}")

    print(f"\nper-swap LVR attribution (vs CEX, in-range only):")
    print(f"  total LVR:  ${lvr_usd:+.2f}")
    print(f"  toxic (toward-oracle):  ${toxic_lvr:+.2f}")
    print(f"  noise (away-from-oracle): ${noise_lvr:+.2f}")
    print(f"  fees - toxic:  ${fees_usd - toxic_lvr:+.2f}  "
          f"(what we'd earn if we skipped toxic swaps)")

    if args.attrib_out is not None and attrib_rows:
        print(f"\nwriting attribution CSV to {args.attrib_out}")
        args.attrib_out.parent.mkdir(parents=True, exist_ok=True)
        with args.attrib_out.open("w") as fh:
            fh.write("ts,amount_usd,pool_pre,pool_post,ext,dist_pre,dist_post,"
                     "toward,in_range,share,fee,dv_ext\n")
            for r in attrib_rows:
                fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                                  for v in r) + "\n")

        # quick summary of how concentrated the LVR is
        arr = np.asarray([(r[7], r[10], r[11]) for r in attrib_rows
                          if r[8] == 1 and r[7] == 1])  # toward & in_range
        if arr.size:
            losses = -arr[:, 2]
            losses = losses[losses > 0]
            losses.sort()
            cum = np.cumsum(losses[::-1])
            total = losses.sum()
            top1 = 100 * cum[max(0, int(0.01*len(losses))-1)] / total if total else 0
            top5 = 100 * cum[max(0, int(0.05*len(losses))-1)] / total if total else 0
            top10 = 100 * cum[max(0, int(0.10*len(losses))-1)] / total if total else 0
            print(f"\nLVR concentration (toxic losses only, n={len(losses):,}):")
            print(f"  top  1% of swaps:  {top1:.1f}% of toxic LVR")
            print(f"  top  5% of swaps:  {top5:.1f}% of toxic LVR")
            print(f"  top 10% of swaps:  {top10:.1f}% of toxic LVR")


if __name__ == "__main__":
    main()
