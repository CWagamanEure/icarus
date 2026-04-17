#!/usr/bin/env -S poetry run python

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if TYPE_CHECKING:
    from icarus.sockets.base import BaseSocket


def ensure_project_src_on_path() -> None:
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect to a websocket feed and print live JSON updates.",
    )
    parser.add_argument(
        "exchange",
        nargs="?",
        choices=["hyperliquid", "coinbase"],
        help="Which websocket feed to connect to.",
        default="coinbase",
    )
    parser.add_argument(
        "market",
        nargs="*",
        help=(
            "Market identifier. Use a coin symbol like BTC for Hyperliquid or one or more "
            "product ids like BTC-USD ETH-USD for Coinbase."
        ),
        default=None,
    )
    parser.add_argument(
        "--channel",
        action="append",
        dest="channels",
        help="Coinbase channel to subscribe to. Repeat to provide multiple channels.",
    )
    parser.add_argument(
        "--candle-interval",
        default="1m",
        help="Hyperliquid candle interval. Pass an empty string to disable candles.",
    )
    parser.add_argument(
        "--disable-trades",
        action="store_true",
        help="Disable Hyperliquid trades subscription.",
    )
    parser.add_argument(
        "--disable-l2-book",
        action="store_true",
        help="Disable Hyperliquid level-2 book subscription.",
    )
    parser.add_argument(
        "--disable-bbo",
        action="store_true",
        help="Disable Hyperliquid best bid/offer subscription.",
    )
    parser.add_argument(
        "--disable-active-asset-ctx",
        action="store_true",
        help="Disable Hyperliquid active asset context subscription.",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Coinbase sandbox websocket.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Hyperliquid testnet websocket.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for printed updates.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after printing this many messages. Zero means stream forever.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity for reconnect messages.",
    )
    return parser


def build_socket(args: argparse.Namespace) -> BaseSocket:
    ensure_project_src_on_path()

    from icarus.sockets.coinbase import CoinbaseSocket
    from icarus.sockets.hyperliquid import HyperliquidSocket

    markets = args.market or (["BTC"] if args.exchange == "hyperliquid" else ["BTC-USD"])

    if args.exchange == "hyperliquid":
        if len(markets) != 1:
            raise ValueError("Hyperliquid accepts exactly one market symbol.")

        candle_interval = args.candle_interval or None
        return HyperliquidSocket(
            markets[0],
            testnet=args.testnet,
            candle_interval=candle_interval,
            include_trades=not args.disable_trades,
            include_l2_book=not args.disable_l2_book,
            include_bbo=not args.disable_bbo,
            include_active_asset_ctx=not args.disable_active_asset_ctx,
        )

    if args.testnet:
        raise ValueError("--testnet only applies to Hyperliquid.")
    if any(
        [
            args.disable_trades,
            args.disable_l2_book,
            args.disable_bbo,
            args.disable_active_asset_ctx,
            args.candle_interval != "1m",
        ]
    ):
        raise ValueError("Hyperliquid-only flags cannot be used with Coinbase.")

    return CoinbaseSocket(
        markets,
        channels=args.channels,
        sandbox=args.sandbox,
    )


async def stream_updates(socket: BaseSocket, *, indent: int, limit: int) -> None:
    count = 0
    try:
        async for message in socket.stream_messages():
            print(json.dumps(message, indent=indent, sort_keys=True))
            count += 1
            if limit and count >= limit:
                break
    finally:
        await socket.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        socket = build_socket(args)
    except ValueError as exc:
        parser.error(str(exc))

    asyncio.run(stream_updates(socket, indent=args.indent, limit=args.limit))


if __name__ == "__main__":
    main()
