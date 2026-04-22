#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001
"""List candidate Base pools from V3 subgraph: 0.3% ETH/USDC and LSTs."""
from __future__ import annotations
import os
import json
import urllib.request
from pathlib import Path


def _load_env():
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def post(url, q):
    body = json.dumps({"query": q}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 icarus/0.1",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    _load_env()
    key = os.environ["GRAPH_API_KEY"]
    sg = os.environ["UNISWAPV3_BASE_SUBGRAPH_ID"]
    url = f"https://gateway.thegraph.com/api/{key}/subgraphs/id/{sg}"

    # Base tokens
    WETH = "0x4200000000000000000000000000000000000006"
    USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

    # 1) ETH/USDC pools all fee tiers, ordered by TVL
    q = (
        '{pools(first: 20, where: {'
        f'token0_in: ["{WETH}","{USDC}"], token1_in: ["{WETH}","{USDC}"]'
        '}, orderBy: totalValueLockedUSD, orderDirection: desc) {'
        ' id feeTier totalValueLockedUSD volumeUSD token0{symbol} token1{symbol}}}'
    )
    print("=== ETH/USDC pools on Base (V3) by TVL ===")
    r = post(url, q)
    for p in r.get("data", {}).get("pools", []) or []:
        print(f"  {p['id']}  fee={int(p['feeTier'])/10000:.3f}%  "
              f"TVL=${float(p['totalValueLockedUSD']):,.0f}  "
              f"vol=${float(p['volumeUSD']):,.0f}  "
              f"{p['token0']['symbol']}/{p['token1']['symbol']}")

    # 2) LST pairs: search pools where one token symbol looks like an LST
    # Use symbol filters since addresses vary per LST
    q2 = (
        '{pools(first: 30, where: {'
        ' totalValueLockedUSD_gt: "1000000"'
        '}, orderBy: totalValueLockedUSD, orderDirection: desc) {'
        ' id feeTier totalValueLockedUSD volumeUSD token0{symbol} token1{symbol}}}'
    )
    print("\n=== top-30 pools by TVL, filtered to LST-looking pairs ===")
    r = post(url, q2)
    pools = r.get("data", {}).get("pools", []) or []
    lst_syms = {"CBETH", "WSTETH", "STETH", "RETH", "EZETH", "WEETH"}
    for p in pools:
        s0 = (p["token0"]["symbol"] or "").upper()
        s1 = (p["token1"]["symbol"] or "").upper()
        if s0 in lst_syms or s1 in lst_syms:
            print(f"  {p['id']}  fee={int(p['feeTier'])/10000:.3f}%  "
                  f"TVL=${float(p['totalValueLockedUSD']):,.0f}  "
                  f"vol=${float(p['volumeUSD']):,.0f}  "
                  f"{p['token0']['symbol']}/{p['token1']['symbol']}")


if __name__ == "__main__":
    main()
