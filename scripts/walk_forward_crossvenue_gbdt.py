#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Walk-forward drift predictor: GBDT (LightGBM) on cross-venue raw features.

Companion to walk_forward_crossvenue.py. Same feature set (own 6 features +
most-recent snapshot of every other venue's 6 features = 24 features for a
4-venue dataset), same walk-forward protocol, same metrics (R², dir, conf_dir),
same horizon sweep.

Difference: model is a gradient-boosted tree ensemble instead of Ridge. The
motivation: earlier work showed coefficient std ±5x mean on microstructure
features in the Ridge fit, and dropping those features to an engineered
basis-only set lost 4-8 directional-accuracy points on coinbase/hyperliquid.
Interpretation: the alpha is a nonlinear interaction between basis spread and
microstructure that Ridge can only approximate with high-variance coefficients.
GBDT should handle that interaction natively.

Bar to pass: directional accuracy meaningfully > Ridge's 0.553 (coinbase) /
0.539 (hyperliquid) at horizon=5s, measured on identical folds. If GBDT
matches rather than beats, the feature set is the ceiling.
"""

from __future__ import annotations

import argparse
import sqlite3
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np


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
    features: np.ndarray


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
            ts, exchange, basis_common, basis_est, mid, microprice,
            imb, vol, net_flow, total_size,
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


def build_feature_names(other_venues: list[str]) -> list[str]:
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
    if predict_venue not in series:
        return None
    own = series[predict_venue]
    other_venues = sorted(v for v in series if v != predict_venue)
    if not other_venues:
        return None
    feature_names = build_feature_names(other_venues)

    ts_own = own.ts
    other_ts_lists = {ov: series[ov].ts.tolist() for ov in other_venues}
    other_features = {ov: series[ov].features for ov in other_venues}

    target_ts = ts_own + horizon_ms
    j_idx = np.searchsorted(ts_own, target_ts, side="left")
    j_idx = np.maximum(j_idx, np.arange(len(ts_own)) + 1)

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
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, float], np.ndarray]:
    # Hold out the tail of the train window as an internal validation set for
    # early stopping — cannot use test data, that's the walk-forward test.
    val_frac = 0.15
    n_train = len(y_train)
    cut = int(n_train * (1 - val_frac))
    x_tr, x_val = x_train[:cut], x_train[cut:]
    y_tr, y_val = y_train[:cut], y_train[cut:]

    train_ds = lgb.Dataset(x_tr, label=y_tr, feature_name=feature_names)
    val_ds = lgb.Dataset(x_val, label=y_val, feature_name=feature_names, reference=train_ds)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.train(
            params,
            train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

    preds = model.predict(x_test, num_iteration=model.best_iteration)
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
    importance = model.feature_importance(importance_type="gain")
    return {"r2": r2, "dir": dir_acc, "conf_dir": conf_dir}, importance


def evaluate_venue(
    venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
    train_size: int,
    test_size: int,
    step: int,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    top_n_feats: int,
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
    importances: list[np.ndarray] = []
    for f in folds:
        scores, importance = run_fold(
            x[f.train_start : f.train_end],
            y[f.train_start : f.train_end],
            x[f.test_start : f.test_end],
            y[f.test_start : f.test_end],
            feature_names, params, num_boost_round, early_stopping_rounds,
        )
        r2s.append(scores["r2"])
        dirs.append(scores["dir"])
        confs.append(scores["conf_dir"])
        importances.append(importance)
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
    if top_n_feats > 0 and importances:
        imp_stack = np.vstack(importances)
        imp_norm = imp_stack / imp_stack.sum(axis=1, keepdims=True).clip(min=1)
        mean_imp = imp_norm.mean(axis=0)
        order = np.argsort(-mean_imp)[:top_n_feats]
        ranked = "  ".join(
            f"{feature_names[i]}={mean_imp[i]*100:.1f}%" for i in order
        )
        print(f"               top gains: {ranked}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-paths", nargs="+", type=Path,
        default=[Path("data/capture/2026-04-21.sqlite3")],
    )
    parser.add_argument(
        "--horizons-ms", default="500,2000,5000",
        help="comma-separated horizons to sweep (ms)",
    )
    parser.add_argument("--tolerance-ms", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--top-n-feats", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    args = parser.parse_args()

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "verbose": -1,
    }

    horizons = [int(x) for x in args.horizons_ms.split(",") if x]
    series = load_series(args.db_paths)
    print(f"loaded series: {[(v, len(s.ts)) for v, s in series.items()]}")
    print(f"params: {params}")
    for horizon in horizons:
        print(f"\n=== horizon={horizon}ms  tolerance={args.tolerance_ms}ms ===")
        for venue in sorted(series.keys()):
            evaluate_venue(
                venue, series, horizon, args.tolerance_ms,
                args.train_size, args.test_size, args.step,
                params, args.num_boost_round, args.early_stopping_rounds,
                args.top_n_feats,
            )


if __name__ == "__main__":
    main()
