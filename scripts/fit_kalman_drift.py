#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Time-varying parameter regression for short-horizon mid drift.

State = coefficient vector beta.
Observation: y_t = x_t . beta_t + v_t,  v_t ~ N(0, R)
Transition:  beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)

At each tick we emit a prediction using beta BEFORE seeing y_t, then update.
So the "online R^2" over the full sample is an honest out-of-sample measure.

Compares against static OLS (with train-period standardization) so we can see
whether letting beta drift buys anything beyond a scale fix.
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


FEATURE_NAMES = [
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
]


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
        features: list[tuple[float, ...]] = []
        targets: list[float] = []
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


def standardize_with_train_stats(
    x: np.ndarray, train_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[:train_end].mean(axis=0)
    std = x[:train_end].std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std, mean, std


def ols_with_split(
    x: np.ndarray, y: np.ndarray, train_frac: float
) -> tuple[float, float, np.ndarray]:
    n = len(y)
    split = int(n * train_frac)
    xi_train = np.hstack([np.ones((split, 1)), x[:split]])
    beta, *_ = np.linalg.lstsq(xi_train, y[:split], rcond=None)
    xi_full = np.hstack([np.ones((n, 1)), x])
    pred = xi_full @ beta
    r2_in = r2_score(y[:split], pred[:split])
    r2_out = r2_score(y[split:], pred[split:])
    return r2_in, r2_out, beta


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def kalman_on_beta(
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray | float,
    r_init: float | None,
    initial_beta: np.ndarray | None,
    initial_P_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Kalman filter on beta. Returns (predictions, betas_over_time).

    Predictions are emitted BEFORE each update, so they are truly online.
    q may be a scalar (uniform) or a per-coefficient vector of length k+1.
    """
    n, k = x.shape
    xi = np.hstack([np.ones((n, 1)), x])
    kd = k + 1

    beta = np.zeros(kd) if initial_beta is None else initial_beta.copy()
    P = np.eye(kd) * initial_P_scale
    q_vec = np.full(kd, q) if np.isscalar(q) else np.asarray(q)
    Q = np.diag(q_vec)

    # Observation noise: estimate from initial OLS residuals over first 20%
    if r_init is None:
        n_warm = max(200, n // 5)
        beta_ols, *_ = np.linalg.lstsq(xi[:n_warm], y[:n_warm], rcond=None)
        resid = y[:n_warm] - xi[:n_warm] @ beta_ols
        r = float(np.var(resid)) or 1.0
    else:
        r = r_init

    predictions = np.zeros(n)
    betas = np.zeros((n, kd))

    for t in range(n):
        P_pred = P + Q
        h = xi[t]
        y_pred = float(h @ beta)
        predictions[t] = y_pred

        innovation = y[t] - y_pred
        S = float(h @ P_pred @ h + r)
        K = (P_pred @ h) / S
        beta = beta + K * innovation
        P = P_pred - np.outer(K, h) @ P_pred

        betas[t] = beta

    return predictions, betas


def estimate_q_from_warmup(
    x: np.ndarray,
    y: np.ndarray,
    warm_end: int,
    q_init: float,
    initial_P_scale: float,
    q_floor: float,
    q_ceiling: float,
) -> np.ndarray:
    """Run a loose Kalman over warmup, then set per-coefficient Q from observed beta drift.

    Using a deliberately loose q_init means beta is free to move during warmup, so the
    observed increment variance reflects how fast each coefficient actually wants to drift.
    """
    _, betas = kalman_on_beta(
        x[:warm_end],
        y[:warm_end],
        q=q_init,
        r_init=None,
        initial_beta=None,
        initial_P_scale=initial_P_scale,
    )
    # Drop the very first few samples — beta is still slewing from zero.
    burn = min(200, warm_end // 4)
    deltas = np.diff(betas[burn:], axis=0)
    q_est = np.var(deltas, axis=0)
    return np.clip(q_est, q_floor, q_ceiling)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", type=Path, default=Path("data/filter_eval_maker.sqlite3")
    )
    parser.add_argument("--horizon-ms", type=int, default=500)
    parser.add_argument(
        "--q-init",
        type=float,
        default=1e-4,
        help="loose uniform Q used in warmup pass to observe natural beta drift",
    )
    parser.add_argument(
        "--q-floor",
        type=float,
        default=1e-9,
        help="minimum per-coefficient Q (prevents zero lock-in)",
    )
    parser.add_argument(
        "--q-ceiling",
        type=float,
        default=1e-3,
        help="maximum per-coefficient Q (prevents runaway)",
    )
    parser.add_argument(
        "--q-estimation-frac",
        type=float,
        default=0.3,
        help="fraction of series used for Q estimation pass (also skipped from R^2)",
    )
    parser.add_argument(
        "--initial-P-scale",
        type=float,
        default=1.0,
        help="initial covariance scale for beta",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="static OLS train fraction; also used for feature standardization",
    )
    args = parser.parse_args()

    datasets = load(args.db_path, args.horizon_ms)
    print(
        f"horizon={args.horizon_ms}ms  q_init={args.q_init}  "
        f"q_bounds=[{args.q_floor}, {args.q_ceiling}]  "
        f"venues={sorted(datasets.keys())}"
    )
    print(f"  features: {FEATURE_NAMES}")

    beta_names = ["intercept"] + FEATURE_NAMES
    for venue, (x, y) in sorted(datasets.items()):
        n = len(y)
        train_end = int(n * args.train_frac)
        q_end = int(n * args.q_estimation_frac)
        x_std, _mean, _std = standardize_with_train_stats(x, train_end)

        ols_r2_in, ols_r2_out, _ = ols_with_split(x_std, y, args.train_frac)

        q_vec = estimate_q_from_warmup(
            x_std,
            y,
            warm_end=q_end,
            q_init=args.q_init,
            initial_P_scale=args.initial_P_scale,
            q_floor=args.q_floor,
            q_ceiling=args.q_ceiling,
        )

        predictions, betas = kalman_on_beta(
            x_std,
            y,
            q=q_vec,
            r_init=None,
            initial_beta=None,
            initial_P_scale=args.initial_P_scale,
        )
        # Score only on the held-out portion AFTER the Q-estimation window.
        r2_post_qwin = r2_score(y[q_end:], predictions[q_end:])
        r2_train_slice = r2_score(y[q_end:train_end], predictions[q_end:train_end])
        r2_test_slice = r2_score(y[train_end:], predictions[train_end:])

        print(
            f"\n{venue}  n={n}  q_end={q_end}  train_end={train_end}\n"
            f"  OLS (z-scored): in={ols_r2_in:+.4f}  out={ols_r2_out:+.4f}\n"
            f"  Kalman online:  post_q_window={r2_post_qwin:+.4f}  "
            f"train_slice={r2_train_slice:+.4f}  "
            f"test_slice={r2_test_slice:+.4f}"
        )
        print("  estimated per-coefficient Q  |  final beta  |  beta drift over full run:")
        beta_mid = betas[train_end] if train_end < n else betas[-1]
        beta_end = betas[-1]
        for name, qi, bm, be in zip(beta_names, q_vec, beta_mid, beta_end):
            print(
                f"    {name:<20} Q={qi:.2e}  "
                f"mid={bm:+.4f}  end={be:+.4f}  drift={be - bm:+.4f}"
            )


if __name__ == "__main__":
    main()
