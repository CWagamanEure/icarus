#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Reality check for Hyperliquid perp funding arb (retail-scale).

Strategy: long BTC spot on Coinbase + short BTC perp on Hyperliquid; held
across one or more hourly funding settlements; exit.

P&L per held window:
    pnl = funding_collected - basis_change - fees
where basis = perp_price - spot_price, so basis widening (perp premium rises)
costs the short-perp-long-spot position.

Data sources (public, no auth):
    - Hyperliquid: POST /info  fundingHistory and candleSnapshot for BTC perp.
    - Coinbase:    GET /products/BTC-USD/candles  hourly OHLC.

Assumptions:
    - Retail fees: Coinbase spot 6 bps taker (public advanced-trade top tier is
      40 bps but volume tiers bring it to 6 bps quickly), Hyperliquid perp
      3.5 bps taker. Round-trip = (6 + 6) + (3.5 + 3.5) = 19 bps on notional.
    - Enter at hour open, exit at the end of the last held hour.
    - Funding paid at each hourly settlement while position is open; funding
      rate comes from the API and is applied to mark notional.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HL_URL = "https://api.hyperliquid.xyz/info"
CB_URL_TEMPLATE = "https://api.exchange.coinbase.com/products/{product}/candles"

# Retail taker fees (bps).
CB_SPOT_TAKER_BPS = 6.0
HL_PERP_TAKER_BPS = 3.5


def post_json(url: str, body: dict) -> list | dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_json(url: str, params: dict) -> list | dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{qs}",
        headers={"User-Agent": "icarus-research/0.1 (github.com/icarus)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_funding_history(coin: str, start_ms: int, end_ms: int) -> list[dict]:
    """Hyperliquid returns list of {coin, fundingRate, premium, time}."""
    out: list[dict] = []
    cursor = start_ms
    # API returns at most 500 per request; page forward by last.time+1
    while cursor < end_ms:
        batch = post_json(HL_URL, {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": cursor,
            "endTime": end_ms,
        })
        if not batch:
            break
        assert isinstance(batch, list)
        out.extend(batch)
        if len(batch) < 500:
            break
        cursor = int(batch[-1]["time"]) + 1
    return out


def fetch_perp_candles(coin: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    """Hyperliquid candles: list of {t, T, c, h, l, o, n, s, v}."""
    out: list[dict] = []
    cursor = start_ms
    step_ms = 500 * 3_600_000  # ~500 hourly bars per page
    while cursor < end_ms:
        page_end = min(cursor + step_ms, end_ms)
        batch = post_json(HL_URL, {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": cursor, "endTime": page_end},
        })
        assert isinstance(batch, list)
        if not batch:
            cursor = page_end
            continue
        out.extend(batch)
        cursor = int(batch[-1]["t"]) + 3_600_000
    # dedupe by t
    seen: dict[int, dict] = {}
    for c in out:
        seen[int(c["t"])] = c
    return [seen[t] for t in sorted(seen)]


def fetch_coinbase_candles(product: str, start_ms: int, end_ms: int) -> list[list]:
    """Coinbase candles: [[time_s, low, high, open, close, volume], ...].
    Coinbase caps returns to 300 rows per request.
    """
    out: list[list] = []
    step_s = 300 * 3600  # 300 hourly bars per page
    cursor = start_ms // 1000
    end_s = end_ms // 1000
    url = CB_URL_TEMPLATE.format(product=product)
    while cursor < end_s:
        page_end = min(cursor + step_s, end_s)
        params = {
            "granularity": 3600,
            "start": datetime.fromtimestamp(cursor, UTC).isoformat(),
            "end": datetime.fromtimestamp(page_end, UTC).isoformat(),
        }
        try:
            batch = get_json(url, params)
        except Exception as e:
            print(f"coinbase request failed ({cursor}): {e}", file=sys.stderr)
            cursor = page_end
            time.sleep(1.0)
            continue
        assert isinstance(batch, list)
        out.extend(batch)
        cursor = page_end
        time.sleep(0.25)  # be polite to public API
    # dedupe + sort ascending by time
    seen: dict[int, list] = {}
    for c in out:
        seen[int(c[0])] = c
    return [seen[t] for t in sorted(seen)]


def simulate(
    funding: list[dict],
    perp_by_hr: dict[int, dict],
    spot_by_hr: dict[int, list],
    hold_hours: int,
    notional: float,
) -> list[dict]:
    """Simulate entering long spot + short perp, holding `hold_hours`, exit.
    Enter at hour T_enter open; exit at hour T_enter+hold_hours open.
    Collect funding at every funding settlement T where T_enter < T <= T_enter+hold_hours.
    """
    funding_by_hr: dict[int, float] = {
        int(f["time"]) // 3_600_000 * 3_600_000: float(f["fundingRate"]) for f in funding
    }
    hour_ms = 3_600_000
    all_hours = sorted(perp_by_hr.keys() & spot_by_hr.keys())
    if not all_hours:
        return []

    results: list[dict] = []
    cb_fee = CB_SPOT_TAKER_BPS / 10_000.0
    hl_fee = HL_PERP_TAKER_BPS / 10_000.0

    for enter_hr in all_hours:
        exit_hr = enter_hr + hold_hours * hour_ms
        if exit_hr not in perp_by_hr or exit_hr not in spot_by_hr:
            continue

        # Use opens as entry/exit marks. Coinbase candle order [t, low, high, open, close, vol].
        spot_in = float(spot_by_hr[enter_hr][3])
        spot_out = float(spot_by_hr[exit_hr][3])
        perp_in = float(perp_by_hr[enter_hr]["o"])
        perp_out = float(perp_by_hr[exit_hr]["o"])

        # 1 unit sizing: spot_size BTC long, perp_size BTC short, each $notional.
        spot_size = notional / spot_in
        perp_size = notional / perp_in

        spot_pnl = spot_size * (spot_out - spot_in)
        perp_pnl = -perp_size * (perp_out - perp_in)

        fees = notional * (2 * cb_fee + 2 * hl_fee)

        # Funding settlements in (enter_hr, exit_hr]. Each paid on short perp position.
        funding_cashflow = 0.0
        funding_sum_rate = 0.0
        n_periods = 0
        for t in range(enter_hr + hour_ms, exit_hr + hour_ms, hour_ms):
            rate = funding_by_hr.get(t)
            if rate is None:
                continue
            mark = float(perp_by_hr.get(t, {"o": perp_in})["o"])
            funding_cashflow += rate * perp_size * mark
            funding_sum_rate += rate
            n_periods += 1

        total_pnl = spot_pnl + perp_pnl + funding_cashflow - fees
        basis_in = perp_in - spot_in
        basis_out = perp_out - spot_out
        results.append({
            "enter_hr": enter_hr,
            "exit_hr": exit_hr,
            "spot_in": spot_in,
            "spot_out": spot_out,
            "perp_in": perp_in,
            "perp_out": perp_out,
            "basis_in": basis_in,
            "basis_out": basis_out,
            "basis_change": basis_out - basis_in,
            "spot_pnl": spot_pnl,
            "perp_pnl": perp_pnl,
            "funding_rate_sum": funding_sum_rate,
            "funding_cashflow": funding_cashflow,
            "fees": fees,
            "total_pnl": total_pnl,
            "n_funding_periods": n_periods,
        })
    return results


def summarize(results: list[dict], hold_hours: int, notional: float, label: str) -> None:
    if not results:
        print(f"{label}: no results")
        return
    pnl = np.array([r["total_pnl"] for r in results])
    fund_rates = np.array([r["funding_rate_sum"] for r in results]) * 10_000.0  # bps
    basis_chg = np.array([r["basis_change"] for r in results])
    bps_pnl = pnl / notional * 10_000.0

    print(f"\n=== {label} (hold={hold_hours}h, notional=${notional:,.0f}) ===")
    print(f"n periods: {len(results)}")
    print(f"funding (sum per hold), bps: mean={fund_rates.mean():+.2f}  "
          f"p25={np.percentile(fund_rates, 25):+.2f}  "
          f"med={np.percentile(fund_rates, 50):+.2f}  "
          f"p75={np.percentile(fund_rates, 75):+.2f}  "
          f"p95={np.percentile(fund_rates, 95):+.2f}")
    print(f"basis_change ($): mean={basis_chg.mean():+.2f}  std={basis_chg.std():.2f}  "
          f"abs.median={np.median(np.abs(basis_chg)):.2f}")
    print(f"net P&L per hold ($): mean=${pnl.mean():+.2f}  std=${pnl.std():.2f}  "
          f"min=${pnl.min():.2f}  max=${pnl.max():.2f}")
    print(f"net P&L per hold (bps): mean={bps_pnl.mean():+.2f}  "
          f"med={np.percentile(bps_pnl, 50):+.2f}  "
          f"std={bps_pnl.std():.2f}  win%={(pnl > 0).mean():.1%}")
    sharpe = pnl.mean() / (pnl.std() + 1e-12)
    holds_per_year = 24 * 365 / hold_hours
    ann_sharpe = sharpe * np.sqrt(holds_per_year)
    print(f"per-hold Sharpe={sharpe:.3f}  annualized Sharpe≈{ann_sharpe:.2f}  "
          f"(assumes independent holds)")

    # Conditional stats: what if we only take the top quartile of expected funding?
    # Proxy "expected" using the previous-period funding rate.
    # (Simple EMA-0 signal: prior-period funding.)
    rates = np.array([r["funding_rate_sum"] for r in results])
    # Reindex: for period i, use rate[i-hold] as naive forecast.
    filt_pnls = []
    for i in range(hold_hours, len(results)):
        prior_rate = rates[i - hold_hours]
        if prior_rate > 0:  # only enter when recent funding was positive
            filt_pnls.append(results[i]["total_pnl"])
    if filt_pnls:
        filt = np.array(filt_pnls)
        bps_filt = filt / notional * 10_000.0
        print(f"conditional (prior funding > 0): n={len(filt)}  "
              f"mean=${filt.mean():+.2f}  std=${filt.std():.2f}  "
              f"mean_bps={bps_filt.mean():+.2f}  "
              f"win%={(filt > 0).mean():.1%}  "
              f"Sharpe={filt.mean() / (filt.std() + 1e-12):.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coin", default="BTC", help="Hyperliquid perp coin")
    p.add_argument("--cb-product", default=None,
                   help="Coinbase product id (default: {COIN}-USD)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--notional", type=float, default=10_000.0)
    p.add_argument("--cache-dir", type=Path, default=Path("data/funding_cache"))
    args = p.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cb_product = args.cb_product or f"{args.coin}-USD"

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400 * 1000
    print(f"fetching {args.days}d  {datetime.fromtimestamp(start_ms/1000, UTC):%Y-%m-%d} → "
          f"{datetime.fromtimestamp(end_ms/1000, UTC):%Y-%m-%d}")

    fn_cache = args.cache_dir / f"funding_{args.coin}_{start_ms}_{end_ms}.json"
    hl_cache = args.cache_dir / f"perp_candles_{args.coin}_{start_ms}_{end_ms}.json"
    cb_cache = args.cache_dir / f"cb_candles_{cb_product}_{start_ms}_{end_ms}.json"

    if fn_cache.exists():
        funding = json.loads(fn_cache.read_text())
        print(f"funding (cached): {len(funding)} records")
    else:
        funding = fetch_funding_history(args.coin, start_ms, end_ms)
        fn_cache.write_text(json.dumps(funding))
        print(f"funding fetched: {len(funding)} records")

    if hl_cache.exists():
        hl_candles = json.loads(hl_cache.read_text())
        print(f"HL candles (cached): {len(hl_candles)} bars")
    else:
        hl_candles = fetch_perp_candles(args.coin, "1h", start_ms, end_ms)
        hl_cache.write_text(json.dumps(hl_candles))
        print(f"HL candles fetched: {len(hl_candles)} bars")

    if cb_cache.exists():
        cb_candles = json.loads(cb_cache.read_text())
        print(f"CB candles (cached): {len(cb_candles)} bars")
    else:
        cb_candles = fetch_coinbase_candles(cb_product, start_ms, end_ms)
        cb_cache.write_text(json.dumps(cb_candles))
        print(f"CB candles fetched: {len(cb_candles)} bars")

    # Index by hour.
    perp_by_hr = {int(c["t"]): c for c in hl_candles}
    spot_by_hr = {int(c[0]) * 1000: c for c in cb_candles}

    print(f"\noverlap: {len(perp_by_hr.keys() & spot_by_hr.keys())} common hours")

    # Funding distribution overall.
    rates_bps = np.array([float(f["fundingRate"]) for f in funding]) * 10_000
    print(f"\nfunding rate distribution (bps per 1h period):")
    print(f"  mean={rates_bps.mean():+.3f}  std={rates_bps.std():.3f}  "
          f"median={np.median(rates_bps):+.3f}")
    print(f"  p05={np.percentile(rates_bps, 5):+.3f}  "
          f"p95={np.percentile(rates_bps, 95):+.3f}  "
          f"|rate|>1bps: {(np.abs(rates_bps) > 1).mean():.1%}")

    # Round-trip fee in bps of notional:
    rt_fee_bps = 2 * CB_SPOT_TAKER_BPS + 2 * HL_PERP_TAKER_BPS
    print(f"\nround-trip fee: {rt_fee_bps:.1f} bps "
          f"(Coinbase {CB_SPOT_TAKER_BPS}bps × 2 + Hyperliquid {HL_PERP_TAKER_BPS}bps × 2)")

    for hold in [1, 4, 8, 24, 72, 168]:
        label = f"hold {hold}h"
        results = simulate(funding, perp_by_hr, spot_by_hr, hold_hours=hold,
                           notional=args.notional)
        summarize(results, hold, args.notional, label)


if __name__ == "__main__":
    main()
