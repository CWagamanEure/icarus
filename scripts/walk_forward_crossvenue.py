#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Walk-forward drift predictor with cross-venue lead-lag features.

Previous drift/classifier results showed R² positive but directional accuracy
≤ 0.5 on 3/4 venues — the Kalman fits magnitude but not sign. One plausible
reason: venue-local features at 500ms don't contain directional info, but
*other venues* may lead. Coinbase is the natural anchor (largest USD venue);
hyperliquid often reacts to spot prints. This script tests that hypothesis.

For each tick on venue V at time t, feature vector =
    (V's own features) + (most-recent features from every other venue at ≤ t)

Targets:
    y = mid_V(t + horizon) - mid_V(t)   (drift, for R² and directional accuracy)

Horizon is configurable so we can sweep 500ms, 2s, 5s and see where signal lives.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


OWN_FEATURE_NAMES = (
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
    "basis_dislocation",
)


@dataclass(frozen=True, slots=True)
class VenueSeries:
    venue: str
    ts: np.ndarray
    mid: np.ndarray
    features: np.ndarray  # (n, k_own) raw features per tick


@dataclass(frozen=True, slots=True)
class Fold:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def load_series(db_paths: list[Path]) -> dict[str, VenueSeries]:
    per_venue: dict[str, list[tuple[int, float, tuple[float, ...]]]] = defaultdict(list)
    for db_path in db_paths:
        conn = sqlite3.connect(str(db_path))
        query = """
            SELECT u.timestamp_ms, v.exchange, u.basis_common_price,
                   b.basis_estimate, v.fair_value, v.microprice,
                   v.depth_imbalance, v.mid_volatility_bps,
                   COALESCE(v.trade_net_flow, 0.0),
                   COALESCE(v.trade_buy_size, 0.0) + COALESCE(v.trade_sell_size, 0.0)
            FROM updates u
            JOIN venue_states v ON v.update_id = u.id
            LEFT JOIN basis_states b
              ON b.update_id = u.id AND b.exchange = v.exchange
            WHERE v.microprice IS NOT NULL
              AND v.depth_imbalance IS NOT NULL
              AND v.mid_volatility_bps IS NOT NULL
              AND u.basis_common_price IS NOT NULL
              AND b.basis_estimate IS NOT NULL
            ORDER BY v.exchange, u.timestamp_ms, u.id
        """
        for (
            ts,
            exchange,
            basis_common,
            basis_est,
            mid,
            microprice,
            imb,
            vol,
            net_flow,
            total_size,
        ) in conn.execute(query):
            reconstructed = float(basis_common) + float(basis_est)
            per_venue[exchange].append(
                (
                    int(ts),
                    float(mid),
                    (
                        float(microprice) - float(mid),
                        float(imb),
                        float(vol),
                        float(net_flow),
                        float(total_size),
                        float(mid) - reconstructed,
                    ),
                )
            )
        conn.close()

    out: dict[str, VenueSeries] = {}
    for venue, rows in per_venue.items():
        rows.sort(key=lambda r: r[0])
        if len(rows) < 200:
            continue
        out[venue] = VenueSeries(
            venue=venue,
            ts=np.asarray([r[0] for r in rows]),
            mid=np.asarray([r[1] for r in rows]),
            features=np.asarray([r[2] for r in rows]),
        )
    return out


def build_feature_names(predict_venue: str, other_venues: list[str]) -> list[str]:
    names = [f"own_{n}" for n in OWN_FEATURE_NAMES]
    for ov in other_venues:
        names.extend(f"{ov}_{n}" for n in OWN_FEATURE_NAMES)
    return names


def build_crossvenue_dataset(
    predict_venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """Build (X, y) where X[i] = own features + latest-known features from every other venue.

    Alignment: for each tick t_i on predict_venue, use most-recent snapshot from each
    other venue with ts ≤ t_i. Skip a row if any other venue has no prior data at t_i.
    Target y[i] = mid_V(t_j) - mid_V(t_i) where t_j is the first tick with
    timestamp ≥ t_i + horizon_ms and t_j - (t_i + horizon_ms) ≤ tolerance_ms.
    """
    if predict_venue not in series:
        return None
    own = series[predict_venue]
    other_venues = sorted(v for v in series if v != predict_venue)
    if not other_venues:
        return None
    feature_names = build_feature_names(predict_venue, other_venues)

    ts_own = own.ts
    # Pre-convert other venues' ts to lists once (bisect on ndarray is slow / wrong).
    other_ts_lists = {ov: series[ov].ts.tolist() for ov in other_venues}
    other_features = {ov: series[ov].features for ov in other_venues}

    # Vectorized target lookup: for each i, find j[i] via np.searchsorted.
    target_ts = ts_own + horizon_ms
    j_idx = np.searchsorted(ts_own, target_ts, side="left")
    # Ensure j > i.
    j_idx = np.maximum(j_idx, np.arange(len(ts_own)) + 1)

    # Pre-compute k_idx[i] per other venue: most-recent other-venue tick at t_i.
    other_k: dict[str, np.ndarray] = {}
    for ov in other_venues:
        other_k[ov] = np.searchsorted(other_ts_lists[ov], ts_own, side="right") - 1

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(len(ts_own)):
        j = int(j_idx[i])
        if j >= len(ts_own):
            break
        if int(ts_own[j]) - (int(ts_own[i]) + horizon_ms) > tolerance_ms:
            continue
        skip = False
        other_feats: list[np.ndarray] = []
        for ov in other_venues:
            k = int(other_k[ov][i])
            if k < 0:
                skip = True
                break
            other_feats.append(other_features[ov][k])
        if skip:
            continue
        combined = np.concatenate([own.features[i]] + other_feats)
        xs.append(combined)
        ys.append(float(own.mid[j] - own.mid[i]))
    if len(ys) < 200:
        return None
    return np.asarray(xs), np.asarray(ys), feature_names


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


def run_fold(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
    ridge_alpha: float,
) -> tuple[dict[str, float], np.ndarray]:
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0)
    xt = (x_train - mean) / std
    xs = np.clip((x_test - mean) / std, -5.0, 5.0)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(xt, y_train)
    preds = model.predict(xs)
    ss_res = float(((y_test - preds) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))
    mag_cut = np.quantile(np.abs(preds), 0.5)
    mask = np.abs(preds) >= mag_cut
    conf_dir = (
        float(np.mean(np.sign(preds[mask]) == np.sign(y_test[mask])))
        if mask.sum() > 0
        else 0.0
    )
    return {"r2": r2, "dir": dir_acc, "conf_dir": conf_dir}, model.coef_


def evaluate_venue(
    venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
    train_size: int,
    test_size: int,
    step: int,
    ridge_alpha: float,
    top_n_coefs: int,
) -> None:
    data = build_crossvenue_dataset(venue, series, horizon_ms, tolerance_ms)
    if data is None:
        print(f"{venue:<13}  SKIP  insufficient aligned data")
        return
    x, y, feature_names = data
    folds = make_folds(len(y), train_size, test_size, step)
    if not folds:
        print(f"{venue:<13}  SKIP  only {len(y)} rows")
        return
    r2s, dirs, confs = [], [], []
    coefs: list[np.ndarray] = []
    for f in folds:
        scores, coef = run_fold(
            x[f.train_start : f.train_end],
            y[f.train_start : f.train_end],
            x[f.test_start : f.test_end],
            y[f.test_start : f.test_end],
            ridge_alpha,
        )
        r2s.append(scores["r2"])
        dirs.append(scores["dir"])
        confs.append(scores["conf_dir"])
        coefs.append(coef)
    r2_arr = np.asarray(r2s)
    dir_arr = np.asarray(dirs)
    conf_arr = np.asarray(confs)
    pos = int((r2_arr > 0).sum())
    print(
        f"{venue:<13}  n={len(y)}  folds={len(folds)}  "
        f"R²={r2_arr.mean():+.3f}±{r2_arr.std():.2f}  "
        f"pos={pos}/{len(folds)}  "
        f"dir={dir_arr.mean():.3f}  conf_dir={conf_arr.mean():.3f}"
    )
    if top_n_coefs > 0 and coefs:
        coef_stack = np.vstack(coefs)
        mean_coef = coef_stack.mean(axis=0)
        std_coef = coef_stack.std(axis=0)
        order = np.argsort(-np.abs(mean_coef))[:top_n_coefs]
        ranked = "  ".join(
            f"{feature_names[i]}={mean_coef[i]:+.3e}(±{std_coef[i]:.1e})"
            for i in order
        )
        print(f"               top coefs: {ranked}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-paths", nargs="+", type=Path,
        default=[Path("data/capture/2026-04-21.sqlite3")],
    )
    parser.add_argument(
        "--horizons-ms",
        default="500,2000,5000",
        help="comma-separated horizons to sweep (ms)",
    )
    parser.add_argument("--tolerance-ms", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--top-n-coefs", type=int, default=6)
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons_ms.split(",") if x]
    series = load_series(args.db_paths)
    print(f"loaded series: {[(v, len(s.ts)) for v, s in series.items()]}")
    for horizon in horizons:
        print(f"\n=== horizon={horizon}ms  tolerance={args.tolerance_ms}ms ===")
        for venue in sorted(series.keys()):
            evaluate_venue(
                venue, series, horizon, args.tolerance_ms,
                args.train_size, args.test_size, args.step,
                args.ridge_alpha, args.top_n_coefs,
            )


if __name__ == "__main__":
    main()
