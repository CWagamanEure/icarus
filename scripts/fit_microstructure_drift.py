#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Fit a per-venue linear model predicting near-term mid drift from microstructure.

For each venue, pair every row (t, mid_t, features_t) with the nearest row
at t + horizon_ms to get target = mid_{t+h} - mid_t. Fit OLS on features:
    [microprice - mid, depth_imbalance, mid_volatility_bps]
Reports coefficients, t-stats, R^2, and out-of-sample R^2 (first-70 / last-30).
"""

from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Row:
    ts: int
    mid: float
    microprice: float
    depth_imbalance: float
    vol_bps: float
    trade_net_flow: float
    trade_total_size: float


def load(db: Path, horizon_ms: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    conn = sqlite3.connect(str(db))
    query = """
        SELECT u.timestamp_ms, v.exchange, v.fair_value, v.microprice,
               v.depth_imbalance, v.mid_volatility_bps,
               COALESCE(v.trade_net_flow, 0.0),
               COALESCE(v.trade_buy_size, 0.0) + COALESCE(v.trade_sell_size, 0.0)
        FROM updates u
        JOIN venue_states v ON v.update_id = u.id
        WHERE v.microprice IS NOT NULL
          AND v.depth_imbalance IS NOT NULL
          AND v.mid_volatility_bps IS NOT NULL
        ORDER BY v.exchange, u.timestamp_ms, u.id
    """
    per_venue: dict[str, list[Row]] = defaultdict(list)
    for ts, exchange, mid, microprice, imb, vol, net_flow, total_size in conn.execute(query):
        per_venue[exchange].append(
            Row(ts, mid, microprice, imb, vol, net_flow, total_size)
        )
    conn.close()

    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for venue, rows in per_venue.items():
        if len(rows) < 100:
            continue
        timestamps = [r.ts for r in rows]
        features = []
        targets = []
        for i, row in enumerate(rows):
            target_ts = row.ts + horizon_ms
            j = bisect_left(timestamps, target_ts, lo=i + 1)
            if j >= len(rows):
                break
            future = rows[j]
            if future.ts - target_ts > horizon_ms:
                continue
            features.append(
                (
                    row.microprice - row.mid,
                    row.depth_imbalance,
                    row.vol_bps,
                    row.trade_net_flow,
                    row.trade_total_size,
                )
            )
            targets.append(future.mid - row.mid)
        if len(targets) < 50:
            continue
        datasets[venue] = (np.asarray(features), np.asarray(targets))
    return datasets


def ols_with_stats(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    n, k = x.shape
    xi = np.hstack([np.ones((n, 1)), x])
    beta, *_ = np.linalg.lstsq(xi, y, rcond=None)
    resid = y - xi @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma2 = ss_res / max(n - k - 1, 1)
    xtx_inv = np.linalg.pinv(xi.T @ xi)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    t_stats = beta / np.where(se > 0, se, np.nan)
    return beta, t_stats, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=Path("data/filter_eval_maker.sqlite3"))
    parser.add_argument("--horizon-ms", type=int, default=500)
    args = parser.parse_args()

    datasets = load(args.db_path, args.horizon_ms)
    print(f"horizon={args.horizon_ms}ms  venues_with_data={sorted(datasets.keys())}")
    feature_names = [
        "intercept",
        "microprice-mid",
        "depth_imbalance",
        "vol_bps",
        "trade_net_flow",
        "trade_total_size",
    ]
    for venue, (x, y) in sorted(datasets.items()):
        n = len(y)
        beta, t, r2 = ols_with_stats(x, y)
        split = int(n * 0.7)
        beta_in, _, _ = ols_with_stats(x[:split], y[:split])
        xi_out = np.hstack([np.ones((n - split, 1)), x[split:]])
        pred_out = xi_out @ beta_in
        ss_res_out = float(((y[split:] - pred_out) ** 2).sum())
        ss_tot_out = float(((y[split:] - y[split:].mean()) ** 2).sum())
        r2_out = 1 - ss_res_out / ss_tot_out if ss_tot_out > 0 else 0.0
        print(f"\n{venue}  n={n}  r2_in_sample={r2:.4f}  r2_oos(70/30)={r2_out:.4f}")
        for name, b, tval in zip(feature_names, beta, t):
            print(f"  {name:<20} beta={b:+.6f}  t={tval:+.2f}")


if __name__ == "__main__":
    main()
