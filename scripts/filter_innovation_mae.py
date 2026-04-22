#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Replay a capture through VenueBasisKalmanFilter and report per-venue
one-step-ahead MAE / RMSE on the fair-value innovation.

For each filter update, the filter reports `predicted_fair_value` (its
pre-update belief about each venue's mid, given the prior state and all
other venues' histories) and the realized observation. The innovation
(obs - predicted) is the one-step-ahead prediction error; aggregate |innov|
per venue to get a calibration-free accuracy metric.

This is the standard way to score a Kalman filter's predictive quality.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (
    VenueBasisKalmanConfig,
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)


PERP_EXCHANGES = {"hyperliquid_perp"}


def is_perp(exchange: str) -> bool:
    return exchange in PERP_EXCHANGES or exchange.endswith("_perp")


def load_updates(db_path: Path) -> list[tuple[int, int, list[VenueBasisObservation]]]:
    """Return list of (update_id, timestamp_ms, [observations]) ordered by id."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT u.id, u.timestamp_ms, v.exchange, v.venue_kind, v.fair_value,
               v.variance, v.age_ms
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        ORDER BY u.id
        """
    ).fetchall()
    conn.close()

    grouped: dict[int, tuple[int, list[VenueBasisObservation]]] = {}
    for r in rows:
        obs = VenueBasisObservation(
            name=r["exchange"],
            fair_value=float(r["fair_value"]),
            local_variance=float(r["variance"]),
            age_ms=float(r["age_ms"]),
            venue_kind=r["venue_kind"] if r["venue_kind"] in ("spot", "perp") else "spot",
        )
        if r["id"] not in grouped:
            grouped[r["id"]] = (int(r["timestamp_ms"]), [])
        grouped[r["id"]][1].append(obs)
    return [(uid, ts, obs) for uid, (ts, obs) in sorted(grouped.items())]


def build_config(args: argparse.Namespace, venues_present: set[str]) -> VenueBasisKalmanConfig:
    spot_order: list[str] = []
    perp_order: list[str] = []
    # Match the Makefile default order: coinbase anchor first.
    for v in ["coinbase", "hyperliquid", "okx", "kraken"]:
        if v in venues_present:
            spot_order.append(v)
    for v in sorted(venues_present):
        if is_perp(v):
            perp_order.append(v)
    return VenueBasisKalmanConfig(
        anchor_exchange=args.anchor_exchange,
        venue_order=tuple(spot_order),
        perp_exchange_order=tuple(perp_order),
        common_price_process_var_per_sec=args.common_price_process_var_per_sec,
        default_basis_process_var_per_sec=args.basis_process_var_per_sec,
        default_basis_rho_per_second=args.basis_rho_per_second,
        default_perp_basis_process_var_per_sec=args.perp_process_var_per_sec,
        default_perp_basis_rho_per_second=args.perp_rho_per_second,
        min_live_spot_venues=args.min_live_spot_venues,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", type=Path, default=Path("data/capture/2026-04-21.sqlite3"))
    p.add_argument("--anchor-exchange", default="coinbase")
    p.add_argument("--common-price-process-var-per-sec", type=float, default=20.0)
    p.add_argument("--basis-process-var-per-sec", type=float, default=0.005)
    p.add_argument("--basis-rho-per-second", type=float, default=0.995)
    p.add_argument("--perp-process-var-per-sec", type=float, default=0.05)
    p.add_argument("--perp-rho-per-second", type=float, default=0.997)
    p.add_argument("--min-live-spot-venues", type=int, default=1)
    p.add_argument("--warmup-minutes", type=float, default=5.0,
                   help="discard innovations in the first N minutes")
    args = p.parse_args()

    print(f"loading {args.db_path}")
    updates = load_updates(args.db_path)
    if not updates:
        raise SystemExit("no updates loaded")
    print(f"loaded {len(updates):,} updates")

    venues_present: set[str] = set()
    for _, _, obs_list in updates:
        for o in obs_list:
            venues_present.add(o.name)
    print(f"venues: {sorted(venues_present)}")

    config = build_config(args, venues_present)
    filt = VenueBasisKalmanFilter(config=config)

    ts0_ms = updates[0][1]
    warmup_cutoff_ms = ts0_ms + int(args.warmup_minutes * 60_000)

    abs_innovations: dict[str, list[float]] = defaultdict(list)
    sq_innovations: dict[str, list[float]] = defaultdict(list)
    counts_pre_warmup: dict[str, int] = defaultdict(int)

    bps_denoms: dict[str, list[float]] = defaultdict(list)  # for bps conversion

    for _, ts_ms, obs_list in updates:
        result = filt.update(timestamp_s=ts_ms / 1000.0, observations=obs_list)
        if result is None:
            continue
        for d in result.observation_diagnostics:
            if ts_ms < warmup_cutoff_ms:
                counts_pre_warmup[d.name] += 1
                continue
            abs_innovations[d.name].append(abs(d.innovation))
            sq_innovations[d.name].append(d.innovation ** 2)
            if d.fair_value > 0:
                bps_denoms[d.name].append(d.fair_value)

    # Report.
    print(f"\nwarmup skipped: first {args.warmup_minutes:.1f} min "
          f"({sum(counts_pre_warmup.values()):,} observations)")
    print(f"\n{'venue':<16} {'n':>10} {'MAE ($)':>10} {'RMSE ($)':>10} "
          f"{'MAE (bps)':>12} {'RMSE (bps)':>12}")
    print("-" * 76)
    for venue in sorted(abs_innovations.keys()):
        absv = np.asarray(abs_innovations[venue])
        sqv = np.asarray(sq_innovations[venue])
        mid_mean = float(np.mean(bps_denoms[venue])) if bps_denoms[venue] else float("nan")
        mae_usd = float(absv.mean())
        rmse_usd = float(np.sqrt(sqv.mean()))
        mae_bps = mae_usd / mid_mean * 10_000.0 if mid_mean > 0 else float("nan")
        rmse_bps = rmse_usd / mid_mean * 10_000.0 if mid_mean > 0 else float("nan")
        print(f"{venue:<16} {len(absv):>10,} {mae_usd:>10.3f} {rmse_usd:>10.3f} "
              f"{mae_bps:>12.3f} {rmse_bps:>12.3f}")


if __name__ == "__main__":
    main()
