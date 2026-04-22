#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Live REST ticker poll for top thin-alt candidates across CB/Kraken/OKX.

Reality-checks the candle-derived spread estimate from
scripts/thin_alt_spread_scan.py. For each polling tick we record per-venue
best bid/ask. This lets us compute:

    - simultaneous mid-to-mid spread (the real arb opportunity)
    - per-venue bid/ask spread (taker cost per leg)
    - USDT/USD basis correction for OKX (we poll Coinbase USDT-USD too)

Output: rows to data/altscan_cache/ticker_poll_<ts>.jsonl, plus a summary
table at exit.

The script is deliberately dumb: sequential GETs per tick. At 5s interval
and 5 symbols across 3 venues that's 15 req per tick (~3 req/s per venue),
comfortably under public rate limits.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

CB_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
CB_TICKER_URL = "https://api.exchange.coinbase.com/products/{id}/ticker"
KR_ASSETPAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
KR_TICKER_URL = "https://api.kraken.com/0/public/Ticker"
OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"

CACHE_DIR = Path("data/altscan_cache")

# Retail taker fees, bps per leg
CB_TAKER_BPS = 6.0
KR_TAKER_BPS = 25.0   # Kraken pro default; volume tiers bring it down
OKX_TAKER_BPS = 10.0  # default taker


def http_get(url: str, params: dict | None = None, timeout: int = 10) -> dict | list:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "icarus-altscan/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def load_ids_for_bases(bases: list[str]) -> dict[str, dict[str, str]]:
    """Return {base: {venue: inst_id}} using cached product-list JSON from stage-1."""
    cb_raw = json.loads((CACHE_DIR / "cb_products.json").read_text())
    kr_raw = json.loads((CACHE_DIR / "kr_assetpairs.json").read_text())
    okx_raw = json.loads((CACHE_DIR / "okx_instruments.json").read_text())

    out: dict[str, dict[str, str]] = {b: {} for b in bases}
    bases_up = {b.upper() for b in bases}

    # Coinbase: prefer USD quote
    cb_candidates: dict[str, dict] = {}
    for p in cb_raw:
        if p.get("trading_disabled") or p.get("status") != "online":
            continue
        b = p.get("base_currency", "")
        if b not in bases_up:
            continue
        q = p.get("quote_currency", "")
        if q not in ("USD", "USDC", "USDT"):
            continue
        prio = {"USD": 0, "USDC": 1, "USDT": 2}[q]
        cur = cb_candidates.get(b)
        if cur is None or prio < cur["prio"]:
            cb_candidates[b] = {"id": p["id"], "quote": q, "prio": prio}
    for b, info in cb_candidates.items():
        out[b]["coinbase"] = info["id"]

    # Kraken: prefer USD
    kr_candidates: dict[str, dict] = {}
    for pair_key, info in kr_raw.get("result", {}).items():
        if info.get("status", "online") != "online":
            continue
        wsname = info.get("wsname", "")
        if "/" not in wsname:
            continue
        base, quote = wsname.split("/", 1)
        if base == "XBT":
            base = "BTC"
        if base == "XDG":
            base = "DOGE"
        if base not in bases_up or quote not in ("USD", "USDC", "USDT"):
            continue
        prio = {"USD": 0, "USDC": 1, "USDT": 2}[quote]
        cur = kr_candidates.get(base)
        if cur is None or prio < cur["prio"]:
            kr_candidates[base] = {"id": pair_key, "quote": quote, "prio": prio}
    for b, info in kr_candidates.items():
        out[b]["kraken"] = info["id"]

    # OKX: prefer USDT
    okx_candidates: dict[str, dict] = {}
    for i in okx_raw.get("data", []):
        if i.get("state") != "live":
            continue
        b = i.get("baseCcy", "")
        q = i.get("quoteCcy", "")
        if b not in bases_up or q not in ("USDT", "USDC", "USD"):
            continue
        prio = {"USDT": 0, "USDC": 1, "USD": 2}[q]
        cur = okx_candidates.get(b)
        if cur is None or prio < cur["prio"]:
            okx_candidates[b] = {"id": i["instId"], "quote": q, "prio": prio}
    for b, info in okx_candidates.items():
        out[b]["okx"] = info["id"]
        # Also tag the quote so we know to USDT-correct
        out[b]["okx_quote"] = info["quote"]

    # Validate
    missing = [b for b in bases if not all(k in out[b] for k in ("coinbase", "kraken", "okx"))]
    if missing:
        print(f"WARN: missing venue ids for {missing}; skipping", file=sys.stderr)
    return {b: v for b, v in out.items() if all(k in v for k in ("coinbase", "kraken", "okx"))}


def poll_coinbase(inst_id: str) -> tuple[float, float] | None:
    try:
        r = http_get(CB_TICKER_URL.format(id=inst_id))
    except Exception as e:
        print(f"  cb {inst_id}: {e}", file=sys.stderr)
        return None
    try:
        return float(r["bid"]), float(r["ask"])
    except (KeyError, ValueError, TypeError):
        return None


def poll_kraken(pair_id: str) -> tuple[float, float] | None:
    try:
        r = http_get(KR_TICKER_URL, {"pair": pair_id})
    except Exception as e:
        print(f"  kr {pair_id}: {e}", file=sys.stderr)
        return None
    res = r.get("result", {}) if isinstance(r, dict) else {}
    for _, v in res.items():
        try:
            bid = float(v["b"][0])
            ask = float(v["a"][0])
            return bid, ask
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    return None


def poll_okx(inst_id: str) -> tuple[float, float] | None:
    try:
        r = http_get(OKX_TICKER_URL, {"instId": inst_id})
    except Exception as e:
        print(f"  okx {inst_id}: {e}", file=sys.stderr)
        return None
    data = r.get("data", []) if isinstance(r, dict) else []
    if not data:
        return None
    try:
        return float(data[0]["bidPx"]), float(data[0]["askPx"])
    except (KeyError, ValueError, TypeError):
        return None


def mid(ba: tuple[float, float] | None) -> float | None:
    if ba is None:
        return None
    b, a = ba
    if b <= 0 or a <= 0:
        return None
    return 0.5 * (b + a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str,
                   default="ARKM,BLUR,KSM,PNUT,OP,EIGEN,BIO,FLOKI,TURBO,WIF",
                   help="comma-separated base tickers to poll")
    p.add_argument("--duration-minutes", type=float, default=30.0)
    p.add_argument("--interval-seconds", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=None,
                   help="output JSONL; default data/altscan_cache/ticker_poll_<ts>.jsonl")
    args = p.parse_args()

    bases = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"symbols: {bases}")

    ids = load_ids_for_bases(bases)
    if not ids:
        raise SystemExit("no symbols resolved across all 3 venues")
    print(f"resolved {len(ids)} / {len(bases)} symbols with ids on all venues:")
    for b, v in ids.items():
        print(f"  {b}: cb={v['coinbase']}  kr={v['kraken']}  okx={v['okx']} ({v['okx_quote']})")

    # USDT/USD reference from Coinbase
    usdt_usd_id = "USDT-USD"

    out_path = args.output or CACHE_DIR / f"ticker_poll_{int(time.time())}.jsonl"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")
    print(f"writing {out_path}")

    stop_flag = {"stop": False}
    def _handle(signum, frame):
        stop_flag["stop"] = True
        print("\nstopping on signal...")
    signal.signal(signal.SIGINT, _handle)

    start_ts = time.time()
    end_ts = start_ts + args.duration_minutes * 60
    tick_count = 0

    # Per-(symbol,venue) mid time series
    mids: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)  # (ts, mid)
    bas:  dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)  # (ts, bid, ask)
    usdt_series: list[tuple[float, float]] = []  # (ts, usdt_usd_mid)

    try:
        while not stop_flag["stop"] and time.time() < end_ts:
            tick_start = time.time()
            ts_iso = int(tick_start)

            # USDT/USD reference
            ba = poll_coinbase(usdt_usd_id)
            m = mid(ba)
            if m is not None:
                usdt_series.append((tick_start, m))
                fh.write(json.dumps({"ts": tick_start, "venue": "coinbase",
                                     "symbol": "USDT", "bid": ba[0], "ask": ba[1]}) + "\n")

            for base, venue_ids in ids.items():
                # Sequential polls - cheap enough
                cb_ba = poll_coinbase(venue_ids["coinbase"])
                kr_ba = poll_kraken(venue_ids["kraken"])
                okx_ba = poll_okx(venue_ids["okx"])

                for venue, ba_val in [("coinbase", cb_ba), ("kraken", kr_ba), ("okx", okx_ba)]:
                    if ba_val is None:
                        continue
                    m2 = mid(ba_val)
                    if m2 is None:
                        continue
                    mids[(base, venue)].append((tick_start, m2))
                    bas[(base, venue)].append((tick_start, ba_val[0], ba_val[1]))
                    fh.write(json.dumps({"ts": tick_start, "venue": venue,
                                         "symbol": base, "bid": ba_val[0], "ask": ba_val[1]}) + "\n")

            fh.flush()
            tick_count += 1
            elapsed = time.time() - tick_start
            remaining = args.interval_seconds - elapsed
            if tick_count % 6 == 0:
                print(f"  tick {tick_count}  ({int(time.time() - start_ts)}s elapsed, "
                      f"last tick took {elapsed:.1f}s)")
            if remaining > 0 and not stop_flag["stop"]:
                time.sleep(remaining)
    finally:
        fh.close()

    print(f"\ncaptured {tick_count} ticks over {int(time.time() - start_ts)}s")
    print(f"output: {out_path}")

    # ----- analysis -----

    if not usdt_series:
        print("no USDT/USD samples; cannot correct OKX")
        return

    # Timestamp-align: for each ticket tick, find USDT/USD mid closest in time.
    usdt_ts = np.asarray([t for t, _ in usdt_series])
    usdt_mid_arr = np.asarray([m for _, m in usdt_series])

    def usdt_at(ts: float) -> float:
        i = int(np.argmin(np.abs(usdt_ts - ts)))
        return float(usdt_mid_arr[i])

    print("\n\n=== per-venue bid/ask spread (taker cost, bps) ===")
    print(f"{'symbol':<10} {'venue':<10} {'n':>5} {'median_bps':>11} {'p95_bps':>9}")
    for (base, venue), rows in sorted(bas.items()):
        if not rows:
            continue
        arr = np.asarray([(a - b) / (0.5 * (a + b)) * 10_000 for (_, b, a) in rows if a > 0 and b > 0])
        if arr.size == 0:
            continue
        print(f"{base:<10} {venue:<10} {len(arr):>5} "
              f"{float(np.median(arr)):>11.1f} {float(np.percentile(arr, 95)):>9.1f}")

    # Pairwise taker arb: for each venue pair (v1, v2) compute
    #   edge_bps = max(bid_v1 - ask_v2, bid_v2 - ask_v1) / avg_mid * 1e4
    # This is the *gross* profit from a simultaneous cross-book taker fill
    # (before fees). Positive means books are crossed — free money. Any
    # persistent negative value means you're paying the combined bid/ask
    # spreads to arb, which is the normal state.
    print("\n\n=== pairwise cross-venue taker arb edge (bps, pre-fee) ===")
    print("(positive = crossed book / free money; OKX ask*usdt_usd, bid*usdt_usd)")
    print(f"{'symbol':<10} {'venue_pair':<20} {'n':>5} {'median':>8} {'p95':>8} "
          f"{'max':>8} {'frac>0':>8} {'frac>round_trip_fee':>20}")

    def _series(base: str, venue: str):
        rows = bas.get((base, venue), [])
        ts = np.asarray([r[0] for r in rows])
        bid = np.asarray([r[1] for r in rows])
        ask = np.asarray([r[2] for r in rows])
        return ts, bid, ask

    fee_pair = {
        ("coinbase", "kraken"): CB_TAKER_BPS + KR_TAKER_BPS,
        ("coinbase", "okx"): CB_TAKER_BPS + OKX_TAKER_BPS,
        ("kraken", "okx"): KR_TAKER_BPS + OKX_TAKER_BPS,
    }

    for base in ids:
        for v1, v2 in [("coinbase", "kraken"), ("coinbase", "okx"), ("kraken", "okx")]:
            t1, b1, a1 = _series(base, v1)
            t2, b2, a2 = _series(base, v2)
            if t1.size == 0 or t2.size == 0:
                continue
            edges: list[float] = []
            for i, ts in enumerate(t1):
                j = int(np.argmin(np.abs(t2 - ts)))
                if abs(t2[j] - ts) > 8.0:
                    continue
                # USDT-correct OKX legs (bid/ask are in USDT, convert to USD-equivalent)
                if v1 == "okx":
                    u = usdt_at(ts)
                    bid1 = b1[i] * u
                    ask1 = a1[i] * u
                else:
                    bid1, ask1 = b1[i], a1[i]
                if v2 == "okx":
                    u = usdt_at(ts)
                    bid2 = b2[j] * u
                    ask2 = a2[j] * u
                else:
                    bid2, ask2 = b2[j], a2[j]
                avg = 0.25 * (bid1 + ask1 + bid2 + ask2)
                edge = max(bid1 - ask2, bid2 - ask1) / avg * 10_000.0
                edges.append(edge)
            if not edges:
                continue
            arr = np.asarray(edges)
            fee = fee_pair[(v1, v2)]
            frac_pos = float((arr > 0).mean()) * 100
            frac_over_fee = float((arr > fee).mean()) * 100
            print(f"{base:<10} {v1[:2] + '-' + v2[:2]:<20} {len(arr):>5} "
                  f"{float(np.median(arr)):>8.1f} {float(np.percentile(arr, 95)):>8.1f} "
                  f"{float(arr.max()):>8.1f} {frac_pos:>7.1f}% {frac_over_fee:>19.1f}%")

    print("\ntaker fee assumptions (bps per leg): "
          f"CB={CB_TAKER_BPS}, Kraken={KR_TAKER_BPS}, OKX={OKX_TAKER_BPS}")


if __name__ == "__main__":
    main()
