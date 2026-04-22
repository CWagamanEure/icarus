#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Walk-forward evaluation of the drift Kalman predictor.

For each fold we:
  1. Load (features, target) pairs for the fold's train/test windows per venue.
  2. Fit mean/std + per-coefficient Q + β₀ using the train window (same as
     simulate_basis_maker.calibrate_predictor).
  3. Run the Kalman online over the test window. Score:
       - OOS R² (vs y's own variance)
       - directional accuracy (sign(pred) == sign(y))
       - residual mean (prediction bias)
  4. Print per-fold and summary (mean, std, min, max, # folds where R² > 0).

This lets us distinguish "the drift filter has real edge" from "we got lucky
on one sample." If R² is positive on ≥70% of folds across multiple days of
data, the signal is real. If it's positive on 1/3 folds, we should stop
building on it.
"""

from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
)


@dataclass(frozen=True, slots=True)
class VenueDataset:
    venue: str
    x: np.ndarray  # (n, k) raw features
    y: np.ndarray  # (n,) realized drift over horizon
    ts: np.ndarray  # (n,) timestamp_ms per row


@dataclass(frozen=True, slots=True)
class Fold:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class FoldResult:
    fold: Fold
    venue: str
    r2: float
    directional_acc: float
    residual_mean: float
    residual_std: float
    n_test: int
    n_train: int


def load_datasets(
    db_paths: list[Path], horizon_ms: int, tolerance_ms: int
) -> dict[str, VenueDataset]:
    per_venue_rows: dict[str, list[tuple[int, float, tuple[float, ...]]]] = defaultdict(list)
    for db_path in db_paths:
        conn = sqlite3.connect(str(db_path))
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
        for ts, exchange, mid, microprice, imb, vol, net_flow, total_size in conn.execute(query):
            per_venue_rows[exchange].append(
                (
                    int(ts),
                    float(mid),
                    (
                        float(microprice) - float(mid),
                        float(imb),
                        float(vol),
                        float(net_flow),
                        float(total_size),
                    ),
                )
            )
        conn.close()

    datasets: dict[str, VenueDataset] = {}
    for venue, rows in per_venue_rows.items():
        rows.sort(key=lambda r: r[0])
        ts_list = [r[0] for r in rows]
        mids = [r[1] for r in rows]
        features = [r[2] for r in rows]
        xs: list[tuple[float, ...]] = []
        ys: list[float] = []
        row_ts: list[int] = []
        for i in range(len(rows)):
            target_ts = ts_list[i] + horizon_ms
            j = bisect_left(ts_list, target_ts, lo=i + 1)
            if j >= len(rows):
                break
            if ts_list[j] - target_ts > tolerance_ms:
                continue
            xs.append(features[i])
            ys.append(mids[j] - mids[i])
            row_ts.append(ts_list[i])
        if len(ys) < 200:
            continue
        datasets[venue] = VenueDataset(
            venue=venue,
            x=np.asarray(xs),
            y=np.asarray(ys),
            ts=np.asarray(row_ts),
        )
    return datasets


def make_folds(n: int, train_size: int, test_size: int, step: int) -> list[Fold]:
    folds: list[Fold] = []
    start = 0
    idx = 0
    while start + train_size + test_size <= n:
        folds.append(
            Fold(
                fold_index=idx,
                train_start=start,
                train_end=start + train_size,
                test_start=start + train_size,
                test_end=start + train_size + test_size,
            )
        )
        start += step
        idx += 1
    return folds


def fit_predictor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    q_floor: float,
    q_ceiling: float,
    q_init: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0)
    x_std = (x_train - mean) / std
    xi = np.hstack([np.ones((len(y_train), 1)), x_std])
    beta0, *_ = np.linalg.lstsq(xi, y_train, rcond=None)
    resid = y_train - xi @ beta0
    r = float(np.var(resid)) or 1.0

    beta = beta0.copy()
    P = np.eye(xi.shape[1])
    Q_init = np.eye(xi.shape[1]) * q_init
    betas = np.zeros_like(xi)
    for t in range(len(y_train)):
        P_pred = P + Q_init
        h = xi[t]
        innovation = y_train[t] - h @ beta
        S = float(h @ P_pred @ h + r)
        K = (P_pred @ h) / S
        beta = beta + K * innovation
        P = P_pred - np.outer(K, h) @ P_pred
        betas[t] = beta
    burn = min(200, len(y_train) // 4)
    q_vec = np.clip(np.var(np.diff(betas[burn:], axis=0), axis=0), q_floor, q_ceiling)
    return beta0.copy(), q_vec, mean, std, r


def run_online_kalman(
    beta0: np.ndarray,
    q_vec: np.ndarray,
    r: float,
    mean: np.ndarray,
    std: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    clip_sigma: float = 5.0,
) -> np.ndarray:
    x_z = np.clip((x_test - mean) / std, -clip_sigma, clip_sigma)
    xi = np.hstack([np.ones((len(y_test), 1)), x_z])
    beta = beta0.copy()
    P = np.eye(xi.shape[1])
    Q = np.diag(q_vec)
    preds = np.zeros(len(y_test))
    for t in range(len(y_test)):
        P_pred = P + Q
        h = xi[t]
        preds[t] = float(h @ beta)
        S = float(h @ P_pred @ h + r)
        innovation = y_test[t] - h @ beta
        K = (P_pred @ h) / S
        beta = beta + K * innovation
        P = P_pred - np.outer(K, h) @ P_pred
    return preds


def score_fold(preds: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    directional = float(np.mean((np.sign(preds) == np.sign(y)) | (preds == 0) & (y == 0)))
    resid = y - preds
    return {
        "r2": r2,
        "directional": directional,
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std()),
    }


def evaluate_venue(
    dataset: VenueDataset,
    folds: list[Fold],
    q_floor: float,
    q_ceiling: float,
    q_init: float,
) -> list[FoldResult]:
    results: list[FoldResult] = []
    for fold in folds:
        x_train = dataset.x[fold.train_start : fold.train_end]
        y_train = dataset.y[fold.train_start : fold.train_end]
        x_test = dataset.x[fold.test_start : fold.test_end]
        y_test = dataset.y[fold.test_start : fold.test_end]
        if len(y_train) < 200 or len(y_test) < 50:
            continue
        beta0, q_vec, mean, std, r = fit_predictor(
            x_train, y_train, q_floor, q_ceiling, q_init
        )
        preds = run_online_kalman(beta0, q_vec, r, mean, std, x_test, y_test)
        scores = score_fold(preds, y_test)
        results.append(
            FoldResult(
                fold=fold,
                venue=dataset.venue,
                r2=scores["r2"],
                directional_acc=scores["directional"],
                residual_mean=scores["residual_mean"],
                residual_std=scores["residual_std"],
                n_test=len(y_test),
                n_train=len(y_train),
            )
        )
    return results


def summarize(venue: str, results: list[FoldResult]) -> None:
    if not results:
        print(f"{venue}: no folds")
        return
    r2s = np.array([r.r2 for r in results])
    dirs = np.array([r.directional_acc for r in results])
    positive = int((r2s > 0).sum())
    print(
        f"{venue:<13}  folds={len(results)}  "
        f"R² mean={r2s.mean():+.3f}  std={r2s.std():.3f}  "
        f"min={r2s.min():+.3f}  max={r2s.max():+.3f}  "
        f"pos={positive}/{len(results)}  "
        f"dir mean={dirs.mean():.3f}"
    )
    for r in results:
        print(
            f"  fold {r.fold.fold_index:2d} "
            f"[{r.fold.train_start:>6}:{r.fold.train_end:>6} / "
            f"{r.fold.test_start:>6}:{r.fold.test_end:>6}]  "
            f"R²={r.r2:+.4f}  dir={r.directional_acc:.3f}  "
            f"resid μ={r.residual_mean:+.3f} σ={r.residual_std:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-paths",
        nargs="+",
        type=Path,
        default=[Path("data/filter_eval_maker.sqlite3")],
        help="One or more SQLite captures to concatenate in time order.",
    )
    parser.add_argument("--horizon-ms", type=int, default=500)
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=2000,
        help="max slack past target_ts when matching horizon row (venue-gap threshold)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=20000,
        help="rows per training window",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="rows per test window",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10000,
        help="rows to slide between folds",
    )
    parser.add_argument("--q-init", type=float, default=1e-4)
    parser.add_argument("--q-floor", type=float, default=1e-9)
    parser.add_argument("--q-ceiling", type=float, default=1e-3)
    args = parser.parse_args()

    datasets = load_datasets(args.db_paths, args.horizon_ms, args.tolerance_ms)
    print(
        f"horizon={args.horizon_ms}ms  "
        f"train/test/step={args.train_size}/{args.test_size}/{args.step}  "
        f"venues={sorted(datasets.keys())}"
    )
    for venue in sorted(datasets.keys()):
        ds = datasets[venue]
        folds = make_folds(len(ds.y), args.train_size, args.test_size, args.step)
        results = evaluate_venue(ds, folds, args.q_floor, args.q_ceiling, args.q_init)
        summarize(venue, results)


if __name__ == "__main__":
    main()
