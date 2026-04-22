#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Simulate taker-arb PnL against basis-filter fair values from a captured DB.

For each non-anchor venue at each tick, compute the dislocation
    signal_V = venue_mid_V - (basis_common_price + basis_estimate_V)
and when |signal_V| > threshold, open a position against the dislocation
(sell if venue overpriced, buy if underpriced). Exit `horizon_ms` later at
that venue's mid. Fills are marked at venue mid, so this is an upper bound
on achievable edge — real execution pays half-spread + taker fees.
"""

from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VenueTick:
    timestamp_ms: int
    update_id: int
    venue_mid: float
    reconstructed: float
    age_ms: float


@dataclass(frozen=True, slots=True)
class Trade:
    venue: str
    entry_ts: int
    exit_ts: int
    side: int
    entry_price: float
    exit_price: float
    signal: float
    gross_pnl: float
    fee: float
    net_pnl: float


def load_venue_ticks(
    db_path: Path, max_age_ms: float
) -> dict[str, list[VenueTick]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = """
        SELECT
            u.id AS update_id,
            u.timestamp_ms,
            u.basis_common_price,
            u.anchor_exchange,
            v.exchange,
            v.fair_value,
            v.age_ms,
            b.basis_estimate
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        LEFT JOIN basis_states b
          ON b.update_id = u.id AND b.exchange = v.exchange
        WHERE u.basis_is_live = 1
          AND u.basis_common_price IS NOT NULL
          AND b.basis_estimate IS NOT NULL
        ORDER BY u.timestamp_ms, u.id
    """
    ticks: dict[str, list[VenueTick]] = defaultdict(list)
    for row in conn.execute(query):
        if row["exchange"] == row["anchor_exchange"]:
            continue
        if row["age_ms"] is None or row["age_ms"] > max_age_ms:
            continue
        reconstructed = row["basis_common_price"] + row["basis_estimate"]
        ticks[row["exchange"]].append(
            VenueTick(
                timestamp_ms=row["timestamp_ms"],
                update_id=row["update_id"],
                venue_mid=row["fair_value"],
                reconstructed=reconstructed,
                age_ms=row["age_ms"],
            )
        )
    conn.close()
    return ticks


def find_exit_tick(
    venue_ticks: list[VenueTick], entry_idx: int, target_ts: int
) -> VenueTick | None:
    timestamps = [t.timestamp_ms for t in venue_ticks]
    idx = bisect_left(timestamps, target_ts, lo=entry_idx + 1)
    if idx >= len(venue_ticks):
        return None
    return venue_ticks[idx]


def simulate_venue(
    venue: str,
    ticks: list[VenueTick],
    threshold: float,
    horizon_ms: int,
    fee_bps: float,
) -> list[Trade]:
    trades: list[Trade] = []
    fee_rate = fee_bps / 10_000.0
    i = 0
    while i < len(ticks):
        tick = ticks[i]
        signal = tick.venue_mid - tick.reconstructed
        if abs(signal) < threshold:
            i += 1
            continue
        side = -1 if signal > 0 else 1
        exit_target = tick.timestamp_ms + horizon_ms
        exit_tick = find_exit_tick(ticks, i, exit_target)
        if exit_tick is None:
            break
        entry_price = tick.venue_mid
        exit_price = exit_tick.venue_mid
        gross = side * (exit_price - entry_price)
        fee = fee_rate * (entry_price + exit_price)
        net = gross - fee
        trades.append(
            Trade(
                venue=venue,
                entry_ts=tick.timestamp_ms,
                exit_ts=exit_tick.timestamp_ms,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                signal=signal,
                gross_pnl=gross,
                fee=fee,
                net_pnl=net,
            )
        )
        exit_ts = exit_tick.timestamp_ms
        while i < len(ticks) and ticks[i].timestamp_ms <= exit_ts:
            i += 1
    return trades


def summarize(trades: list[Trade]) -> dict[str, float]:
    if not trades:
        return {"n": 0}
    gross = [t.gross_pnl for t in trades]
    net = [t.net_pnl for t in trades]
    wins = sum(1 for p in net if p > 0)
    n = len(trades)
    total_gross = sum(gross)
    total_net = sum(net)
    mean_gross = total_gross / n
    mean_net = total_net / n
    var = sum((p - mean_net) ** 2 for p in net) / max(n - 1, 1)
    stddev = var ** 0.5
    sharpe = mean_net / stddev if stddev > 0 else 0.0
    return {
        "n": n,
        "win_rate": wins / n * 100,
        "mean_gross": mean_gross,
        "mean_net": mean_net,
        "total_gross": total_gross,
        "total_net": total_net,
        "stddev": stddev,
        "sharpe_per_trade": sharpe,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/filter_eval_no_perp.sqlite3"),
    )
    parser.add_argument(
        "--horizons-ms",
        type=str,
        default="100,250,500,1000",
        help="comma-separated exit horizons in ms",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,2,5,10,20",
        help="comma-separated dislocation thresholds in USD",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="per-side taker fee in bps (round trip = 2 * fee_bps)",
    )
    parser.add_argument(
        "--max-venue-age-ms",
        type=float,
        default=500.0,
        help="skip ticks where the venue observation is older than this",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    horizons = [int(x) for x in args.horizons_ms.split(",") if x]
    thresholds = [float(x) for x in args.thresholds.split(",") if x]

    ticks_by_venue = load_venue_ticks(args.db_path, args.max_venue_age_ms)
    print(
        f"loaded ticks: "
        + ", ".join(f"{v}={len(ts)}" for v, ts in sorted(ticks_by_venue.items()))
    )
    print(
        f"fee_bps/side={args.fee_bps}  max_venue_age_ms={args.max_venue_age_ms}"
    )
    for horizon in horizons:
        print(f"\nhorizon={horizon}ms")
        header = (
            f"{'venue':<14}{'thr$':>6}{'n':>7}{'win%':>8}"
            f"{'mean_gross':>12}{'mean_net':>12}{'total_net':>14}{'sharpe':>10}"
        )
        print(header)
        for venue, ticks in sorted(ticks_by_venue.items()):
            for threshold in thresholds:
                trades = simulate_venue(
                    venue, ticks, threshold, horizon, args.fee_bps
                )
                s = summarize(trades)
                if s["n"] == 0:
                    print(f"{venue:<14}{threshold:>6.0f}{0:>7}")
                    continue
                print(
                    f"{venue:<14}{threshold:>6.0f}{s['n']:>7}"
                    f"{s['win_rate']:>8.2f}{s['mean_gross']:>12.3f}"
                    f"{s['mean_net']:>12.3f}{s['total_net']:>14.2f}"
                    f"{s['sharpe_per_trade']:>10.3f}"
                )


if __name__ == "__main__":
    main()
