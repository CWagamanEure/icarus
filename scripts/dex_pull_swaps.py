#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Pull swap + modifyLiquidity events for a Uniswap V4 pool on Base.

Reads subgraph credentials from .env (GRAPH_API_KEY, UNISWAPV4_BASE_SUBGRAPH_ID)
and paginates swaps by ascending timestamp into a gzipped JSONL cache.

Resumable: if cache exists, resumes from the last captured timestamp. Useful
for 30-day pulls that take ~10 minutes wall-time.

Target pool (default): ETH/USDC 0.05%, no hooks,
    0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_POOL = "0x96d4b53a38337a5733179751781178a2613306063c511b78cd02684739288c0a"
GRAPH_URL_TEMPLATE = "https://gateway.thegraph.com/api/{key}/subgraphs/id/{sg}"
CACHE_DIR = Path("data/dex_cache")


def _load_env() -> None:
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def graph_url(env_key: str = "UNISWAPV4_BASE_SUBGRAPH_ID") -> str:
    key = os.environ.get("GRAPH_API_KEY")
    sg = os.environ.get(env_key)
    if not key or not sg:
        raise SystemExit(f"GRAPH_API_KEY and {env_key} must be set "
                         "in environment or .env")
    return GRAPH_URL_TEMPLATE.format(key=key, sg=sg)


def post(url: str, query: str, retries: int = 6, sleep_s: float = 2.0) -> dict:
    """POST a GraphQL query with simple retry on 5xx / indexer errors."""
    body = json.dumps({"query": query}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 icarus/0.1"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                r = json.loads(resp.read())
            if "errors" in r:
                # indexer transient errors — retry
                msg = str(r["errors"])[:300]
                if "indexer" in msg or "bad indexer" in msg:
                    print(f"  retry (indexer): {msg}", file=sys.stderr)
                    time.sleep(sleep_s * (2 ** attempt))
                    continue
                raise RuntimeError(f"GraphQL errors: {msg}")
            return r["data"]
        except urllib.error.HTTPError as e:
            if 500 <= e.code < 600:
                print(f"  retry ({e.code})", file=sys.stderr)
                time.sleep(sleep_s * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  retry (network): {e}", file=sys.stderr)
            time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError("exhausted retries")


def build_swap_query(pool: str, since_ts: int, first: int) -> str:
    return (
        "{ swaps("
        f'where: {{pool: "{pool}", timestamp_gt: "{since_ts}"}} '
        "orderBy: timestamp orderDirection: asc "
        f"first: {first}"
        ") { id timestamp sender origin amount0 amount1 amountUSD "
        "sqrtPriceX96 tick logIndex } }"
    )


def pull_swaps(pool: str, since_ts: int, until_ts: int, cache_path: Path,
               page_size: int = 1000,
               env_key: str = "UNISWAPV4_BASE_SUBGRAPH_ID") -> int:
    """Append new swaps to cache_path (gzipped JSONL). Return count written."""
    url = graph_url(env_key)
    written = 0
    cursor = since_ts
    with gzip.open(cache_path, "at") as fh:
        while True:
            q = build_swap_query(pool, cursor, page_size)
            data = post(url, q)
            swaps = data.get("swaps", []) if data else []
            if not swaps:
                break
            for s in swaps:
                ts = int(s["timestamp"])
                if ts > until_ts:
                    return written
                fh.write(json.dumps(s) + "\n")
                written += 1
                cursor = max(cursor, ts)
            print(f"  +{len(swaps):>5}  total {written:>7}  "
                  f"cursor_ts={cursor}  last_log={swaps[-1]['logIndex']}")
            if len(swaps) < page_size:
                break
    return written


def last_timestamp_in_cache(path: Path) -> int:
    """Stream the gzip once to find the max timestamp seen. Cheap enough."""
    if not path.exists():
        return 0
    last = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                ts = int(json.loads(line)["timestamp"])
                if ts > last:
                    last = ts
            except Exception:
                continue
    return last


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--days", type=float, default=30.0,
                    help="pull swaps from now-days to now")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--subgraph-env", default="UNISWAPV4_BASE_SUBGRAPH_ID",
                    help="env var holding the subgraph id (use "
                         "UNISWAPV3_BASE_SUBGRAPH_ID for V3)")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output or CACHE_DIR / f"swaps_{args.pool[:10]}_{int(args.days)}d.jsonl.gz"
    print(f"pool: {args.pool}")
    print(f"output: {out}")

    now = int(time.time())
    target_since = now - int(args.days * 86_400)
    resume_from = last_timestamp_in_cache(out)
    start = max(target_since, resume_from)
    print(f"target window: since={target_since} ({args.days}d ago)  "
          f"now={now}  resume_from={resume_from}  start={start}")

    written = pull_swaps(args.pool, start, now, out, args.page_size,
                         env_key=args.subgraph_env)
    print(f"\nwrote {written:,} swaps to {out}")


if __name__ == "__main__":
    main()
