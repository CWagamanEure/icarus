#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Scan for retail-tradeable cross-venue spreads in non-major spot tokens.

Hypothesis: BTC/ETH/SOL are tightly arbed (prior tests: <10 bps gross edge).
Small/mid-cap tokens that trade on CB + Kraken + OKX may have persistent
spreads because HFTs don't bother with them and natural market-maker inventory
is thinner.

Pipeline:
  Stage 1: fetch public product lists from Coinbase, Kraken, OKX (spot, USD
           or USDT quote). Compute base-currency overlap, excluding majors.
  Stage 2: for overlapping bases, fetch N days of hourly candles from each
           venue. Align timestamps (UTC hour). Report per-base-currency:
             - n aligned hours with data from all 3 venues
             - median / p95 cross-venue spread (bps)
             - half-life of the spread (AR(1) decay)
             - 24h volume per venue (from candle volume * close)
  Stage 3: rank candidates: persistent >20 bps median spread AND
           >$100k/day volume on every venue.

Caches both stages to data/altscan_cache/ to be nice to public APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

CB_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
CB_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
KR_ASSETPAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
KR_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
OKX_INSTR_URL = "https://www.okx.com/api/v5/public/instruments"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"

# Exclude from the "thin-alt" universe: these are known tightly-arbed.
MAJORS = {"BTC", "ETH", "SOL", "USDT", "USDC", "DAI", "BUSD", "TUSD", "USD", "EUR", "GBP"}

CACHE_DIR = Path("data/altscan_cache")


def _user_agent_headers() -> dict:
    return {"User-Agent": "icarus-altscan/0.1"}


def http_get(url: str, params: dict | None = None, timeout: int = 30) -> dict | list:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers=_user_agent_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _cached_json(name: str, fetch_fn, ttl_sec: int = 3600) -> dict | list:
    path = _cache_path(name)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_sec:
        return json.loads(path.read_text())
    data = fetch_fn()
    path.write_text(json.dumps(data))
    return data


# ----- Stage 1: product lists -----

def fetch_coinbase_products() -> list[dict]:
    """Returns list of {base, quote, id, online, volume_24h_usd_est}."""
    def _fetch():
        raw = http_get(CB_PRODUCTS_URL)
        return raw
    raw = _cached_json("cb_products.json", _fetch)
    out = []
    for p in raw:
        if p.get("trading_disabled") or p.get("status") != "online":
            continue
        quote = p.get("quote_currency", "")
        if quote not in ("USD", "USDT", "USDC"):
            continue
        base = p.get("base_currency", "")
        # product id = BASE-QUOTE
        out.append({
            "venue": "coinbase",
            "base": base,
            "quote": quote,
            "id": p["id"],
        })
    return out


def fetch_kraken_pairs() -> list[dict]:
    def _fetch():
        return http_get(KR_ASSETPAIRS_URL)
    raw = _cached_json("kr_assetpairs.json", _fetch)
    result = raw.get("result", {})
    out = []
    # Kraken uses legacy altnames like XBT for BTC, XDG for DOGE, etc.
    # Use 'wsname' (e.g., "BTC/USD", "DOGE/USD") as the canonical BASE/QUOTE.
    for pair_key, info in result.items():
        status = info.get("status", "online")
        if status != "online":
            continue
        wsname = info.get("wsname", "")
        if "/" not in wsname:
            continue
        base, quote = wsname.split("/", 1)
        # normalize Kraken weirdness
        if base == "XBT":
            base = "BTC"
        if base == "XDG":
            base = "DOGE"
        if quote not in ("USD", "USDT", "USDC"):
            continue
        out.append({
            "venue": "kraken",
            "base": base,
            "quote": quote,
            "id": pair_key,  # use pair_key (altname) for OHLC requests
            "wsname": wsname,
        })
    return out


def fetch_okx_spot_instruments() -> list[dict]:
    def _fetch():
        return http_get(OKX_INSTR_URL, {"instType": "SPOT"})
    raw = _cached_json("okx_instruments.json", _fetch)
    data = raw.get("data", [])
    out = []
    for i in data:
        if i.get("state") != "live":
            continue
        quote = i.get("quoteCcy", "")
        if quote not in ("USDT", "USDC", "USD"):
            continue
        base = i.get("baseCcy", "")
        out.append({
            "venue": "okx",
            "base": base,
            "quote": quote,
            "id": i["instId"],
        })
    return out


def compute_overlap(
    cb: list[dict], kr: list[dict], okx: list[dict]
) -> dict[str, dict[str, dict]]:
    """Return {base: {venue: {id, quote}}} for bases present on all three venues.

    Quote preference is venue-specific because each venue's primary quote market
    differs: CB and Kraken are USD-native; OKX's real liquidity is in USDT.
    """
    def index_by_base(items: list[dict], priority: dict[str, int]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for it in items:
            b = it["base"]
            if b in MAJORS:
                continue
            cur = out.get(b)
            if cur is None or priority[it["quote"]] < priority[cur["quote"]]:
                out[b] = it
        return out

    usd_first = {"USD": 0, "USDC": 1, "USDT": 2}
    usdt_first = {"USDT": 0, "USDC": 1, "USD": 2}
    cb_idx = index_by_base(cb, usd_first)
    kr_idx = index_by_base(kr, usd_first)
    okx_idx = index_by_base(okx, usdt_first)

    bases = set(cb_idx) & set(kr_idx) & set(okx_idx)
    return {
        b: {"coinbase": cb_idx[b], "kraken": kr_idx[b], "okx": okx_idx[b]}
        for b in sorted(bases)
    }


# ----- Stage 2: hourly candles -----

def fetch_coinbase_hourly(product: str, start_ms: int, end_ms: int) -> list[list]:
    """Returns list of [time_s, low, high, open, close, volume] ascending."""
    out: list[list] = []
    cursor = start_ms // 1000
    end_s = end_ms // 1000
    step_s = 300 * 3600
    url = CB_CANDLES_URL.format(product=product)
    while cursor < end_s:
        page_end = min(cursor + step_s, end_s)
        params = {
            "granularity": 3600,
            "start": datetime.fromtimestamp(cursor, UTC).isoformat(),
            "end": datetime.fromtimestamp(page_end, UTC).isoformat(),
        }
        try:
            batch = http_get(url, params)
        except Exception as e:
            print(f"  cb {product} fail: {e}", file=sys.stderr)
            cursor = page_end
            time.sleep(1.0)
            continue
        if isinstance(batch, list):
            out.extend(batch)
        cursor = page_end
        time.sleep(0.2)
    seen: dict[int, list] = {int(c[0]): c for c in out}
    return [seen[t] for t in sorted(seen)]


def fetch_kraken_hourly(pair_id: str, start_ms: int, end_ms: int) -> list[list]:
    """Kraken OHLC API returns at most ~720 bars (most recent). We accept that
    and just take whatever fits in window. Returns [[time_s, o, h, l, c, vwap, volume, count]]."""
    since_s = start_ms // 1000
    try:
        raw = http_get(KR_OHLC_URL, {"pair": pair_id, "interval": 60, "since": since_s})
    except Exception as e:
        print(f"  kr {pair_id} fail: {e}", file=sys.stderr)
        return []
    time.sleep(0.3)
    err = raw.get("error", []) if isinstance(raw, dict) else []
    if err:
        print(f"  kr {pair_id} err: {err}", file=sys.stderr)
        return []
    result = raw.get("result", {})
    # result has key pair_id (or similar) + "last"
    rows: list[list] = []
    for k, v in result.items():
        if k == "last":
            continue
        rows = v
        break
    end_s = end_ms // 1000
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
             float(r[4]), float(r[5]), float(r[6]), int(r[7])]
            for r in rows if int(r[0]) <= end_s]


def fetch_okx_hourly(inst_id: str, start_ms: int, end_ms: int) -> list[list]:
    """OKX history-candles: returns list of [ts_ms, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    in DESCENDING order. Paginate via `after` (exclusive upper bound ts_ms).
    Limit 100 per request for history-candles endpoint.
    """
    out: list[list] = []
    cursor = end_ms
    while cursor > start_ms:
        try:
            raw = http_get(OKX_CANDLES_URL, {
                "instId": inst_id, "bar": "1H",
                "after": str(cursor), "limit": "100",
            })
        except Exception as e:
            print(f"  okx {inst_id} fail: {e}", file=sys.stderr)
            break
        time.sleep(0.15)
        batch = raw.get("data", []) if isinstance(raw, dict) else []
        if not batch:
            break
        for r in batch:
            ts = int(r[0])
            if ts < start_ms:
                continue
            out.append([ts // 1000, float(r[1]), float(r[2]), float(r[3]),
                        float(r[4]), float(r[5])])
        # advance: the oldest ts in batch becomes the next `after`
        oldest = min(int(r[0]) for r in batch)
        if oldest >= cursor:
            break
        cursor = oldest
    # dedupe + ascending
    seen: dict[int, list] = {int(c[0]): c for c in out}
    return [seen[t] for t in sorted(seen)]


# ----- Stage 3: metrics -----

def _mids_from_candles(candles: list[list], venue: str) -> dict[int, float]:
    """Return {hour_ts_s: close_price}. Candle close is our mid proxy."""
    out: dict[int, float] = {}
    for c in candles:
        ts = int(c[0])
        if venue == "coinbase":
            close = float(c[4])
        elif venue == "kraken":
            close = float(c[4])
        elif venue == "okx":
            close = float(c[4])
        else:
            continue
        out[ts] = close
    return out


def _volumes_from_candles(candles: list[list], venue: str) -> dict[int, float]:
    """Return {hour_ts_s: usd_volume_that_hour}."""
    out: dict[int, float] = {}
    for c in candles:
        ts = int(c[0])
        if venue == "coinbase":
            close = float(c[4])
            vol_base = float(c[5])
        elif venue == "kraken":
            close = float(c[4])
            vol_base = float(c[6])
        elif venue == "okx":
            close = float(c[4])
            vol_base = float(c[5])
        else:
            continue
        out[ts] = close * vol_base
    return out


def compute_spread_series(mids: dict[str, dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Align hourly timestamps present in all venues; return (ts_s, spread_bps)."""
    venues = list(mids.keys())
    common = set(mids[venues[0]].keys())
    for v in venues[1:]:
        common &= set(mids[v].keys())
    ts = np.asarray(sorted(common), dtype=np.int64)
    if ts.size == 0:
        return ts, np.asarray([])
    M = np.asarray([[mids[v][t] for v in venues] for t in ts])  # n x 3
    mean_p = M.mean(axis=1)
    spread = (M.max(axis=1) - M.min(axis=1)) / mean_p * 10_000.0
    return ts, spread


def ar1_half_life(x: np.ndarray) -> float:
    """Fit x_{t+1} = rho * x_t + e; return half-life = ln(0.5)/ln(rho)."""
    if x.size < 10:
        return float("nan")
    x = x - x.mean()
    num = float(np.dot(x[:-1], x[1:]))
    den = float(np.dot(x[:-1], x[:-1]))
    if den <= 0:
        return float("nan")
    rho = num / den
    if not (0 < rho < 1):
        return float("nan")
    return float(np.log(0.5) / np.log(rho))


def venue_daily_volume_usd(vol_hr: dict[int, float]) -> float:
    """Median of 24h rolling sums. Approximate but robust."""
    if not vol_hr:
        return 0.0
    ts = np.asarray(sorted(vol_hr.keys()), dtype=np.int64)
    v = np.asarray([vol_hr[t] for t in ts])
    if v.size < 24:
        # Scale up partial window to daily
        return float(v.sum() * 24.0 / max(v.size, 1))
    # Non-overlapping 24h sums
    daily = [float(v[i:i + 24].sum()) for i in range(0, v.size - 23, 24)]
    return float(np.median(daily)) if daily else 0.0


# ----- main -----

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14,
                   help="lookback days for hourly candles")
    p.add_argument("--stage", choices=["overlap", "spreads", "all"], default="all")
    p.add_argument("--min-venue-volume-usd", type=float, default=100_000.0,
                   help="tradeable cutoff: daily USD volume on every venue")
    p.add_argument("--min-median-spread-bps", type=float, default=20.0)
    p.add_argument("--max-bases", type=int, default=0,
                   help="cap number of overlapping bases to fetch candles for (0 = all)")
    p.add_argument("--only", type=str, default="",
                   help="comma-separated base tickers to restrict to (debug)")
    args = p.parse_args()

    print("fetching venue product lists...")
    cb = fetch_coinbase_products()
    kr = fetch_kraken_pairs()
    okx = fetch_okx_spot_instruments()
    print(f"  coinbase: {len(cb)} usd-quoted products")
    print(f"  kraken:   {len(kr)} usd-quoted pairs")
    print(f"  okx:      {len(okx)} usd-quoted instruments")

    overlap = compute_overlap(cb, kr, okx)
    print(f"\noverlap (excluding majors): {len(overlap)} bases")
    print("  " + ", ".join(sorted(overlap.keys())[:40])
          + (" ..." if len(overlap) > 40 else ""))

    if args.stage == "overlap":
        return

    if args.only:
        wanted = {s.strip().upper() for s in args.only.split(",") if s.strip()}
        overlap = {b: v for b, v in overlap.items() if b in wanted}
        print(f"\nfiltered to {len(overlap)} bases via --only")

    if args.max_bases and len(overlap) > args.max_bases:
        keep = sorted(overlap.keys())[:args.max_bases]
        overlap = {b: overlap[b] for b in keep}
        print(f"\ncapped to first {args.max_bases} bases (alphabetical)")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400_000

    rows: list[dict] = []
    for i, (base, venues) in enumerate(overlap.items(), 1):
        print(f"\n[{i}/{len(overlap)}] {base}")
        cache_key = f"candles_{base}_{args.days}d.json"
        cache_path = _cache_path(cache_key)
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < 3600:
            candles_per_venue = json.loads(cache_path.read_text())
            print("  (cached)")
        else:
            candles_per_venue = {}
            print(f"  coinbase {venues['coinbase']['id']}...")
            candles_per_venue["coinbase"] = fetch_coinbase_hourly(
                venues["coinbase"]["id"], start_ms, end_ms)
            print(f"  kraken   {venues['kraken']['id']}...")
            candles_per_venue["kraken"] = fetch_kraken_hourly(
                venues["kraken"]["id"], start_ms, end_ms)
            print(f"  okx      {venues['okx']['id']}...")
            candles_per_venue["okx"] = fetch_okx_hourly(
                venues["okx"]["id"], start_ms, end_ms)
            cache_path.write_text(json.dumps(candles_per_venue))

        mids = {v: _mids_from_candles(c, v) for v, c in candles_per_venue.items()}
        vols = {v: _volumes_from_candles(c, v) for v, c in candles_per_venue.items()}

        n_per_venue = {v: len(m) for v, m in mids.items()}
        if any(n == 0 for n in n_per_venue.values()):
            print(f"  skip: empty venue data {n_per_venue}")
            continue

        ts, spread_bps = compute_spread_series(mids)
        if ts.size < 24:
            print(f"  skip: only {ts.size} aligned hours")
            continue

        med = float(np.median(spread_bps))
        p95 = float(np.percentile(spread_bps, 95))
        hl = ar1_half_life(spread_bps)
        vol_daily = {v: venue_daily_volume_usd(vm) for v, vm in vols.items()}

        rows.append({
            "base": base,
            "quote_cb": venues["coinbase"]["quote"],
            "quote_kr": venues["kraken"]["quote"],
            "quote_okx": venues["okx"]["quote"],
            "n_hours": int(ts.size),
            "median_bps": med,
            "p95_bps": p95,
            "half_life_hr": hl,
            "vol_cb_usd": vol_daily["coinbase"],
            "vol_kr_usd": vol_daily["kraken"],
            "vol_okx_usd": vol_daily["okx"],
        })
        print(f"  n={ts.size}  median={med:.1f}bps  p95={p95:.1f}bps  "
              f"hl={hl:.1f}hr  vol_cb/kr/okx=${vol_daily['coinbase']/1e3:.0f}k/"
              f"${vol_daily['kraken']/1e3:.0f}k/${vol_daily['okx']/1e3:.0f}k")

    if not rows:
        print("\nno rows collected.")
        return

    # Sort by median spread descending.
    rows.sort(key=lambda r: -r["median_bps"])

    print("\n\n=== full results ===")
    print(f"{'base':<8} {'quotes(cb/kr/okx)':<20} {'n':>5} {'med bps':>9} "
          f"{'p95 bps':>9} {'half-life hr':>14} "
          f"{'vol_cb $k':>11} {'vol_kr $k':>11} {'vol_okx $k':>11}")
    for r in rows:
        q = f"{r['quote_cb']}/{r['quote_kr']}/{r['quote_okx']}"
        print(f"{r['base']:<8} {q:<20} {r['n_hours']:>5} "
              f"{r['median_bps']:>9.1f} {r['p95_bps']:>9.1f} "
              f"{r['half_life_hr']:>14.1f} "
              f"{r['vol_cb_usd']/1e3:>11.0f} {r['vol_kr_usd']/1e3:>11.0f} "
              f"{r['vol_okx_usd']/1e3:>11.0f}")

    tradeable = [
        r for r in rows
        if r["median_bps"] >= args.min_median_spread_bps
        and min(r["vol_cb_usd"], r["vol_kr_usd"], r["vol_okx_usd"])
            >= args.min_venue_volume_usd
    ]
    print(f"\n\n=== candidates (median>={args.min_median_spread_bps:.0f}bps & "
          f"min venue vol>=${args.min_venue_volume_usd/1e3:.0f}k/day): "
          f"{len(tradeable)} ===")
    for r in tradeable:
        q = f"{r['quote_cb']}/{r['quote_kr']}/{r['quote_okx']}"
        print(f"  {r['base']:<8} {q:<16} med={r['median_bps']:.0f}bps "
              f"p95={r['p95_bps']:.0f}bps hl={r['half_life_hr']:.1f}hr "
              f"min_vol=${min(r['vol_cb_usd'], r['vol_kr_usd'], r['vol_okx_usd'])/1e3:.0f}k/d")


if __name__ == "__main__":
    main()
