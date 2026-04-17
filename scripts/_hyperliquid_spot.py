from __future__ import annotations

import json
from typing import Any, cast
from urllib import request


def resolve_hyperliquid_spot_subscription_coin(
    market: str,
    *,
    testnet: bool = False,
) -> str:
    meta = _fetch_spot_meta(testnet=testnet)
    universe = meta.get("universe")
    tokens = meta.get("tokens")
    if not isinstance(universe, list) or not isinstance(tokens, list):
        raise ValueError("Unexpected Hyperliquid spotMeta response shape.")

    market = market.upper()
    if "/" in market:
        base_symbol, quote_symbol = market.split("/", 1)
    else:
        base_symbol, quote_symbol = market, "USDC"

    base_candidates = {base_symbol}
    if base_symbol == "BTC":
        base_candidates.add("UBTC")
    if base_symbol == "ETH":
        base_candidates.add("UETH")

    for pair in universe:
        if not isinstance(pair, dict):
            continue
        pair_name = pair.get("name")
        token_indexes = pair.get("tokens")
        if (
            not isinstance(pair_name, str)
            or not isinstance(token_indexes, list)
            or len(token_indexes) != 2
        ):
            continue

        base_name = _token_name(tokens, token_indexes[0])
        quote_name = _token_name(tokens, token_indexes[1])
        if base_name in base_candidates and quote_name == quote_symbol:
            return pair_name

    raise ValueError(f"Could not resolve Hyperliquid spot subscription coin for {market}.")


def _fetch_spot_meta(*, testnet: bool) -> dict[str, Any]:
    url = "https://api.hyperliquid-testnet.xyz/info" if testnet else "https://api.hyperliquid.xyz/info"
    payload = json.dumps({"type": "spotMeta"}).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return cast(dict[str, Any], json.loads(response.read().decode("utf-8")))


def _token_name(tokens: list[Any], index: Any) -> str | None:
    if not isinstance(index, int):
        return None
    if index < 0 or index >= len(tokens):
        return None
    token = tokens[index]
    if not isinstance(token, dict):
        return None
    name = token.get("name")
    return name if isinstance(name, str) else None
