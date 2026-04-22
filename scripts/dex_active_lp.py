#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Active-rebalancing LP simulator for a V4/V3 ETH/USDC pool.

Maintains a concentrated position of width `--range-pct`. When the pool
price drifts outside `--trigger-pct` of the current range center, closes
the position at pool price, pays gas, and re-opens a new position
centered on the current price. Tracks fees, LVR, rebalance costs.

Compares net P&L against passive wide-range baseline.
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
DEC_ADJ = 10 ** (ETH_DEC - USDC_DEC)


def sqrtx96_to_price(s):
    r = s / Q96
    return (r * r) * DEC_ADJ


def price_to_sqrtx96(p):
    return math.sqrt(p / DEC_ADJ) * Q96


def token_amounts(L, sp, sl, sh):
    if sp <= sl:
        return L * (1.0 / sl - 1.0 / sh) * Q96, 0.0
    if sp >= sh:
        return 0.0, L * (sh - sl) / Q96
    return (L * (1.0 / sp - 1.0 / sh) * Q96,
            L * (sp - sl) / Q96)


def position_value_usd(L, sp, sl, sh, mark_price):
    x, y = token_amounts(L, sp, sl, sh)
    return (x / 10**ETH_DEC) * mark_price + y / 10**USDC_DEC


def liquidity_for_deposit(deposit_usd, sp, sl, sh, price):
    v1 = position_value_usd(1.0, sp, sl, sh, price)
    if v1 <= 0:
        raise ValueError("zero-value unit position")
    return deposit_usd / v1


def estimate_pool_L(a0, a1, sp, sq):
    d = abs(sq - sp)
    if d <= 0:
        return float("nan")
    y = abs(a1) * 10**USDC_DEC
    if y > 0:
        return y * Q96 / d
    x = abs(a0) * 10**ETH_DEC
    if x > 0 and sp > 0 and sq > 0:
        di = abs(1.0 / sp - 1.0 / sq)
        if di > 0:
            return x / (di * Q96)
    return float("nan")


def load_swaps(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: (int(r["timestamp"]), int(r["logIndex"])))
    return rows


def load_cex(path):
    ts, px = [], []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            ts.append(int(r["ts_s"]))
            px.append(float(r["close"]))
    a = np.asarray(ts)
    p = np.asarray(px)
    o = np.argsort(a)
    return a[o], p[o]


def simulate(swaps, cex_ts, cex_px, *, deposit_usd, range_pct,
             trigger_pct, fee_bps, gas_per_rebalance_usd, skip_hours,
             oor_grace_s):
    """Run one scenario. Return dict of metrics."""
    fee_rate = fee_bps / 10_000.0
    rp = range_pct / 100.0
    tp = trigger_pct / 100.0

    # warm-up then deposit
    warm_end = int(swaps[0]["timestamp"]) + int(skip_hours * 3600)
    dep_idx = next(i for i, s in enumerate(swaps) if int(s["timestamp"]) >= warm_end)
    dep_swap = swaps[dep_idx]
    dep_sqrt = float(dep_swap["sqrtPriceX96"])
    center_price = sqrtx96_to_price(dep_sqrt)
    p_lo = center_price * (1.0 - rp)
    p_hi = center_price * (1.0 + rp)
    sqrt_lo = price_to_sqrtx96(p_lo)
    sqrt_hi = price_to_sqrtx96(p_hi)
    L_pos = liquidity_for_deposit(deposit_usd, dep_sqrt, sqrt_lo, sqrt_hi, center_price)
    x0_raw, y0_raw = token_amounts(L_pos, dep_sqrt, sqrt_lo, sqrt_hi)
    x0_eth = x0_raw / 10**ETH_DEC
    y0_usdc = y0_raw / 10**USDC_DEC

    fees_usd = 0.0
    gas_usd = 0.0
    rebalance_slippage = 0.0  # sum of (v_pool - v_ext) at each rebalance
    n_rebalances = 0
    in_range_s = 0
    out_range_s = 0
    out_range_run_s = 0
    prev_ts = int(dep_swap["timestamp"])
    prev_sqrt = dep_sqrt

    for s in swaps[dep_idx + 1:]:
        ts = int(s["timestamp"])
        sqrt_post = float(s["sqrtPriceX96"])
        a0 = float(s["amount0"])
        a1 = float(s["amount1"])
        usd = float(s["amountUSD"])
        dt = ts - prev_ts

        pp_pre = sqrtx96_to_price(prev_sqrt)
        in_range_pre = sqrt_lo <= prev_sqrt <= sqrt_hi

        if in_range_pre:
            in_range_s += dt
            out_range_run_s = 0
        else:
            out_range_s += dt
            out_range_run_s += dt

        # fees: only when the SWAP midpoint is in range
        L_pool = estimate_pool_L(a0, a1, prev_sqrt, sqrt_post)
        if L_pool and L_pool == L_pool and L_pool > 0:
            mid = 0.5 * (prev_sqrt + sqrt_post)
            if sqrt_lo <= mid <= sqrt_hi:
                share = L_pos / L_pool
                fees_usd += usd * fee_rate * share

        # Rebalance trigger:
        #  (a) price drift from center > trigger_pct, OR
        #  (b) out-of-range for > grace period
        pp_post = sqrtx96_to_price(sqrt_post)
        drift = abs(math.log(pp_post / center_price))
        should_rebalance = False
        if drift > tp:
            should_rebalance = True
        if not in_range_pre and out_range_run_s >= oor_grace_s:
            should_rebalance = True

        if should_rebalance:
            # Close at pool price
            x_old_raw, y_old_raw = token_amounts(L_pos, sqrt_post, sqrt_lo, sqrt_hi)
            x_old_eth = x_old_raw / 10**ETH_DEC
            y_old_usdc = y_old_raw / 10**USDC_DEC
            v_pool = x_old_eth * pp_post + y_old_usdc

            # Tentatively re-open centered on pool_post with post-gas cash
            v_after_gas = v_pool - gas_per_rebalance_usd
            new_center = pp_post
            new_lo = new_center * (1.0 - rp)
            new_hi = new_center * (1.0 + rp)
            sqrt_lo_n = price_to_sqrtx96(new_lo)
            sqrt_hi_n = price_to_sqrtx96(new_hi)
            if v_after_gas <= 0:
                L_new = 0.0
                x_new_eth = 0.0
                y_new_usdc = 0.0
            else:
                L_new = liquidity_for_deposit(v_after_gas, sqrt_post, sqrt_lo_n, sqrt_hi_n, new_center)
                x_new_raw, y_new_raw = token_amounts(L_new, sqrt_post, sqrt_lo_n, sqrt_hi_n)
                x_new_eth = x_new_raw / 10**ETH_DEC
                y_new_usdc = y_new_raw / 10**USDC_DEC

            # Rebalance swap: delta inventory must trade in the pool at fee_rate
            # Swap size in USD ≈ |Δx_eth| * price (== |Δy_usdc| by construction)
            dx_eth = x_new_eth - x_old_eth
            swap_size_usd = abs(dx_eth) * pp_post
            swap_fee = swap_size_usd * fee_rate
            # Pay the swap fee out of the new position cash
            v_final = v_after_gas - swap_fee
            if v_final <= 0:
                L_pos = 0.0
            else:
                L_pos = liquidity_for_deposit(v_final, sqrt_post, sqrt_lo_n, sqrt_hi_n, new_center)

            gas_usd += gas_per_rebalance_usd
            rebalance_slippage += swap_fee
            n_rebalances += 1
            center_price = new_center
            sqrt_lo = sqrt_lo_n
            sqrt_hi = sqrt_hi_n
            out_range_run_s = 0

        prev_ts = ts
        prev_sqrt = sqrt_post

    # final MTM
    final_price = sqrtx96_to_price(prev_sqrt)
    v_lp = position_value_usd(L_pos, prev_sqrt, sqrt_lo, sqrt_hi, final_price)
    # HODL baseline: starting inventory at final pool price
    v_hodl = x0_eth * final_price + y0_usdc

    span_s = prev_ts - int(dep_swap["timestamp"])
    span_d = span_s / 86400.0
    # total LP economics: current LP value + accumulated fees - gas spent
    # (gas already subtracted from L_pos at each rebalance, but we also
    # want separate reporting)
    net_pnl = v_lp + fees_usd - v_hodl  # vs HODL at pool price
    in_frac = in_range_s / (in_range_s + out_range_s + 1e-9)

    return {
        "range_pct": range_pct,
        "trigger_pct": trigger_pct,
        "n_rebalances": n_rebalances,
        "gas_usd": gas_usd,
        "fees_usd": fees_usd,
        "in_range_frac": in_frac,
        "v_lp": v_lp,
        "v_hodl": v_hodl,
        "final_price": final_price,
        "center_price": center_price,
        "rebalance_slippage": rebalance_slippage,
        "net_pnl": net_pnl,
        "net_pnl_pct": net_pnl / deposit_usd * 100,
        "annualized_pct": net_pnl / deposit_usd * 365.0 / span_d * 100,
        "span_d": span_d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swaps", type=Path, required=True)
    ap.add_argument("--cex", type=Path, required=True)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--deposit-usd", type=float, default=10_000.0)
    ap.add_argument("--gas-usd", type=float, default=0.30,
                    help="gas cost per rebalance (close+open)")
    ap.add_argument("--skip-hours", type=float, default=1.0)
    ap.add_argument("--oor-grace-hours", type=float, default=0.0,
                    help="wait this long while OOR before forcing rebalance")
    ap.add_argument("--range-pcts", type=str,
                    default="1,2,5,10,20",
                    help="comma-sep list of half-widths to sweep")
    ap.add_argument("--trigger-pcts", type=str,
                    default="0.5,1,2,5,10,20",
                    help="comma-sep list of drift triggers (in %)")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    print(f"loading swaps: {args.swaps}")
    swaps = load_swaps(args.swaps)
    print(f"  {len(swaps):,} swaps")
    print(f"loading cex: {args.cex}")
    cex_ts, cex_px = load_cex(args.cex)
    # clip swaps to cex window
    t0, t1 = int(cex_ts[0]), int(cex_ts[-1])
    swaps = [s for s in swaps if t0 <= int(s["timestamp"]) <= t1]
    print(f"  {len(swaps):,} swaps in cex window ({(t1-t0)/86400:.1f}d)")

    ranges = [float(x) for x in args.range_pcts.split(",")]
    triggers = [float(x) for x in args.trigger_pcts.split(",")]
    grace_s = int(args.oor_grace_hours * 3600)

    print(f"\ngrid: range_pcts={ranges}  trigger_pcts={triggers}  "
          f"gas=${args.gas_usd}  grace={args.oor_grace_hours}h")
    print()
    header = (f"{'range%':>6} {'trig%':>6} {'nrb':>5} "
              f"{'in_rng':>7} {'fees$':>9} {'gas$':>7} "
              f"{'swFee$':>8} {'netP&L%':>8} {'ann%':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for rp in ranges:
        for tp in triggers:
            if tp < rp * 0.25:
                continue  # silly to trigger at <1/4 of width
            r = simulate(swaps, cex_ts, cex_px,
                         deposit_usd=args.deposit_usd,
                         range_pct=rp, trigger_pct=tp,
                         fee_bps=args.fee_bps,
                         gas_per_rebalance_usd=args.gas_usd,
                         skip_hours=args.skip_hours,
                         oor_grace_s=grace_s)
            results.append(r)
            print(f"{r['range_pct']:>6.1f} {r['trigger_pct']:>6.1f} "
                  f"{r['n_rebalances']:>5} {r['in_range_frac']*100:>6.1f}% "
                  f"{r['fees_usd']:>9.2f} {r['gas_usd']:>7.2f} "
                  f"{r['rebalance_slippage']:>8.2f} "
                  f"{r['net_pnl_pct']:>7.2f}% {r['annualized_pct']:>6.1f}%")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(results[0].keys())
        with args.out_csv.open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in results:
                fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                                  for v in (r[k] for k in keys)) + "\n")
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
