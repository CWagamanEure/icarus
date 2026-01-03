#!/usr/bin/env python3
"""
discover_markets.py — print top Polymarket markets by liquidity (Gamma Markets API)

pip install requests
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List
import requests

GAMMA = "https://gamma-api.polymarket.com"
URL = f"{GAMMA}/markets"

def fnum(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def fetch_pages(params: Dict[str, Any], pages: int, timeout: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    limit = int(params.get("limit", 200))
    offset = int(params.get("offset", 0))

    for _ in range(pages):
        params["limit"] = limit
        params["offset"] = offset

        r = requests.get(URL, params=params, timeout=timeout)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        out.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-liq", type=float, default=0.0)
    ap.add_argument("--include-closed", action="store_true")
    ap.add_argument("--include-inactive", action="store_true")
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    params: Dict[str, Any] = {
        "limit": 200,
        "offset": 0,
        "archived": "false",
    }
    if not args.include_inactive:
        params["active"] = "true"
    if not args.include_closed:
        params["closed"] = "false"
    if args.min_liq > 0:
        params["liquidity_num_min"] = args.min_liq

    if args.debug:
        print("GET", URL)
        print("params =", params)

    markets = fetch_pages(params, pages=args.pages)

    if args.debug:
        print(f"fetched {len(markets)} markets")

    if not markets:
        print("No markets returned. Try --include-closed, --include-inactive, lower --min-liq, or run with --debug.")
        return

    markets.sort(key=lambda m: fnum(m.get("liquidityNum", 0.0)), reverse=True)

    shown = 0
    for m in markets:
        liq = fnum(m.get("liquidityNum", 0.0))
        if liq < args.min_liq:
            continue

        title = m.get("question") or m.get("title") or m.get("slug") or ""
        vol = fnum(m.get("volumeNum", 0.0))
        print(f"{m.get('id')} {title}")
        print(f"Condition ID {m.get("conditionId")}")
        print(f"  liq: {liq:.2f}  vol: {vol:.2f}  active:{m.get('active')} closed:{m.get('closed')}")
        print(m["clobTokenIds"])
        shown += 1
        if shown >= args.top:
            break

    if shown == 0:
        print("No markets matched after filtering. Lower --min-liq or use --include-closed.")

if __name__ == "__main__":
    main()

