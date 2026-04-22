#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Pull Coinbase ETH-USD 1-minute candles as external LVR reference.

30d × 24h × 60m = 43,200 bars. Coinbase caps each request at 300 candles,
so ~145 paged requests, rate-limited.

Output: JSONL at data/dex_cache/cex_eth_1m_<days>d.jsonl
    {ts_s, open, high, low, close, volume}
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

CB_URL = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
CACHE_DIR = Path("data/dex_cache")


def fetch_page(start_s: int, end_s: int) -> list[list]:
    params = {
        "granularity": 60,
        "start": datetime.fromtimestamp(start_s, UTC).isoformat(),
        "end": datetime.fromtimestamp(end_s, UTC).isoformat(),
    }
    req = urllib.request.Request(
        f"{CB_URL}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "icarus-research/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or CACHE_DIR / f"cex_eth_1m_{int(args.days)}d.jsonl"
    now = int(time.time())
    start = now - int(args.days * 86_400)
    print(f"pulling ETH-USD 1m from {start} to {now}")
    print(f"output: {out}")

    # page forward by 300 minutes
    cursor = start
    step = 300 * 60
    seen: dict[int, list] = {}
    page = 0
    while cursor < now:
        page_end = min(cursor + step, now)
        try:
            batch = fetch_page(cursor, page_end)
        except Exception as e:
            print(f"  err at {cursor}: {e}", file=sys.stderr)
            time.sleep(2.0)
            cursor = page_end
            continue
        for c in batch:
            seen[int(c[0])] = c
        page += 1
        if page % 20 == 0:
            print(f"  page {page}  cursor_ts={cursor}  candles={len(seen):,}")
        cursor = page_end
        time.sleep(0.15)

    rows = sorted(seen.values(), key=lambda c: c[0])
    with out.open("w") as fh:
        for c in rows:
            fh.write(json.dumps({
                "ts_s": int(c[0]), "low": float(c[1]), "high": float(c[2]),
                "open": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
            }) + "\n")
    print(f"wrote {len(rows):,} minute bars")


if __name__ == "__main__":
    main()
