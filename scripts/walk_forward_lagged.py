#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Walk-forward drift predictor: cross-venue raw + lagged diff features, Ridge vs GBDT.

Context: walk_forward_crossvenue.py (Ridge) hit dir 0.55-0.59 at horizon=5s on
raw snapshot features; walk_forward_crossvenue_gbdt.py was 4-8pp worse. The
snapshot features only say "where are we now" — they can't distinguish
"dislocation is 0.8 and widening" from "dislocation is 0.8 and just reverted".
That trajectory info should help GBDT more than Ridge, because tree splits can
condition "if dislocation > X AND its 2s change is positive, drift is Y" in
ways Ridge cannot without explicit interaction terms.

Feature set (per own-venue tick at time t, with 4 venues):
    24 raw snapshot features: own 6 + other-3 × 6 (same as crossvenue scripts)
    16 lagged diffs: for each of basis_dislocation and microprice-mid, for each
        of the 4 venues, for each of lag-windows {2000ms, 5000ms}:
        feature(t) - feature(lag_venue_tick nearest to t - lag_window_ms)
    Total: 40 features.

Rows where any lag lookup returns no tick or a stale tick (> 2x lag window)
are skipped.

Runs both Ridge(α=1) and LightGBM on the same folds for direct comparison.
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
from sklearn.linear_model import Ridge


OWN_FEATURE_NAMES = (
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
    "basis_dislocation",
)

# Indices into the 6-feature vector that we compute lagged diffs for.
LAGGED_FEATURE_INDICES = (0, 5)  # microprice-mid, basis_dislocation
LAGGED_FEATURE_NAMES = tuple(OWN_FEATURE_NAMES[i] for i in LAGGED_FEATURE_INDICES)


@dataclass(frozen=True, slots=True)
class VenueSeries:
    venue: str
    ts: np.ndarray
    mid: np.ndarray
    features: np.ndarray  # (n, 6)


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


def build_feature_names(
    predict_venue: str, other_venues: list[str], lag_windows_ms: list[int]
) -> list[str]:
    names = [f"own_{n}" for n in OWN_FEATURE_NAMES]
    for ov in other_venues:
        names.extend(f"{ov}_{n}" for n in OWN_FEATURE_NAMES)
    venues_in_order = [predict_venue] + other_venues
    for v in venues_in_order:
        v_tag = "own" if v == predict_venue else v
        for fname in LAGGED_FEATURE_NAMES:
            for w in lag_windows_ms:
                names.append(f"{v_tag}_{fname}_diff_{w}ms")
    return names


def build_dataset(
    predict_venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
    lag_windows_ms: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if predict_venue not in series:
        return None
    own = series[predict_venue]
    other_venues = sorted(v for v in series if v != predict_venue)
    if not other_venues:
        return None
    feature_names = build_feature_names(predict_venue, other_venues, lag_windows_ms)

    ts_own = own.ts
    ts_own_list = ts_own.tolist()
    other_ts_lists = {ov: series[ov].ts.tolist() for ov in other_venues}
    other_features = {ov: series[ov].features for ov in other_venues}
    other_ts_arrays = {ov: series[ov].ts for ov in other_venues}

    target_ts = ts_own + horizon_ms
    j_idx = np.searchsorted(ts_own, target_ts, side="left")
    j_idx = np.maximum(j_idx, np.arange(len(ts_own)) + 1)

    # Current (latest-known) indices for other venues.
    other_k: dict[str, np.ndarray] = {}
    for ov in other_venues:
        other_k[ov] = np.searchsorted(other_ts_lists[ov], ts_own, side="right") - 1

    # Lagged indices per venue per window.
    own_lag_k: dict[int, np.ndarray] = {}
    other_lag_k: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for w in lag_windows_ms:
        target_lag = ts_own - w
        own_lag_k[w] = np.searchsorted(ts_own_list, target_lag, side="right") - 1
        for ov in other_venues:
            other_lag_k[ov][w] = (
                np.searchsorted(other_ts_lists[ov], target_lag, side="right") - 1
            )

    # Order venues for lagged-feature emission (own first, then others in sorted order).
    venue_order = [predict_venue] + other_venues

    max_lag_gap_ms = {w: 2 * w for w in lag_windows_ms}

    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(len(ts_own)):
        j = int(j_idx[i])
        if j >= len(ts_own):
            break
        if int(ts_own[j]) - (int(ts_own[i]) + horizon_ms) > tolerance_ms:
            continue
        t_i = int(ts_own[i])

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

        # Compute lagged diffs, ordered by venue then lagged-feature then window.
        lagged: list[float] = []
        for v in venue_order:
            if v == predict_venue:
                current_feats = own.features[i]
                for fi in LAGGED_FEATURE_INDICES:
                    for w in lag_windows_ms:
                        k = int(own_lag_k[w][i])
                        if k < 0:
                            skip = True
                            break
                        lag_ts = int(ts_own[k])
                        if t_i - lag_ts > max_lag_gap_ms[w]:
                            skip = True
                            break
                        lagged.append(
                            float(current_feats[fi]) - float(own.features[k][fi])
                        )
                    if skip:
                        break
                if skip:
                    break
            else:
                k_now = int(other_k[v][i])
                current_feats = other_features[v][k_now]
                for fi in LAGGED_FEATURE_INDICES:
                    for w in lag_windows_ms:
                        k = int(other_lag_k[v][w][i])
                        if k < 0:
                            skip = True
                            break
                        lag_ts = int(other_ts_arrays[v][k])
                        if t_i - lag_ts > max_lag_gap_ms[w]:
                            skip = True
                            break
                        lagged.append(
                            float(current_feats[fi])
                            - float(other_features[v][k][fi])
                        )
                    if skip:
                        break
                if skip:
                    break
        if skip:
            continue

        combined = np.concatenate([own.features[i]] + other_feats + [np.asarray(lagged)])
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


def _metrics(preds: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
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
    return {"r2": r2, "dir": dir_acc, "conf_dir": conf_dir}


def _ridge_fit_predict(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, ridge_alpha: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = np.where(x_train.std(axis=0) > 0, x_train.std(axis=0), 1.0)
    xt = (x_train - mean) / std
    xs = np.clip((x_test - mean) / std, -5.0, 5.0)
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(xt, y_train)
    return model.predict(xs)


def run_fold_ridge(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    ridge_alpha: float,
) -> dict[str, float]:
    preds = _ridge_fit_predict(x_train, y_train, x_test, ridge_alpha)
    return _metrics(preds, y_test)


def _run_fold_ridge_with_preds(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    ridge_alpha: float,
) -> tuple[dict[str, float], np.ndarray]:
    preds = _ridge_fit_predict(x_train, y_train, x_test, ridge_alpha)
    return _metrics(preds, y_test), preds


def run_fold_gbdt(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[dict[str, float], np.ndarray]:
    val_frac = 0.15
    n_train = len(y_train)
    cut = int(n_train * (1 - val_frac))
    train_ds = lgb.Dataset(x_train[:cut], label=y_train[:cut], feature_name=feature_names)
    val_ds = lgb.Dataset(
        x_train[cut:], label=y_train[cut:],
        feature_name=feature_names, reference=train_ds,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = lgb.train(
            params, train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[val_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    preds = model.predict(x_test, num_iteration=model.best_iteration)
    return _metrics(preds, y_test), model.feature_importance(importance_type="gain")


def build_dataset_with_ts(
    predict_venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
    lag_windows_ms: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray] | None:
    """Same as build_dataset, plus the per-row source timestamp for the own venue."""
    if predict_venue not in series:
        return None
    own = series[predict_venue]
    result = build_dataset(
        predict_venue, series, horizon_ms, tolerance_ms, lag_windows_ms,
    )
    if result is None:
        return None
    x, y, feature_names = result
    # Re-derive kept timestamps by replaying alignment and re-filtering against
    # the same skip rules. Simpler: rebuild with ts tracking inline.
    other_venues = sorted(v for v in series if v != predict_venue)
    ts_own = own.ts
    ts_own_list = ts_own.tolist()
    other_ts_lists = {ov: series[ov].ts.tolist() for ov in other_venues}
    other_features = {ov: series[ov].features for ov in other_venues}
    other_ts_arrays = {ov: series[ov].ts for ov in other_venues}

    target_ts = ts_own + horizon_ms
    j_idx = np.searchsorted(ts_own, target_ts, side="left")
    j_idx = np.maximum(j_idx, np.arange(len(ts_own)) + 1)
    other_k = {
        ov: np.searchsorted(other_ts_lists[ov], ts_own, side="right") - 1
        for ov in other_venues
    }
    own_lag_k = {}
    other_lag_k: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for w in lag_windows_ms:
        target_lag = ts_own - w
        own_lag_k[w] = np.searchsorted(ts_own_list, target_lag, side="right") - 1
        for ov in other_venues:
            other_lag_k[ov][w] = (
                np.searchsorted(other_ts_lists[ov], target_lag, side="right") - 1
            )
    venue_order = [predict_venue] + other_venues
    max_lag_gap_ms = {w: 2 * w for w in lag_windows_ms}

    kept_ts: list[int] = []
    for i in range(len(ts_own)):
        j = int(j_idx[i])
        if j >= len(ts_own):
            break
        if int(ts_own[j]) - (int(ts_own[i]) + horizon_ms) > tolerance_ms:
            continue
        t_i = int(ts_own[i])
        skip = False
        for ov in other_venues:
            if int(other_k[ov][i]) < 0:
                skip = True
                break
        if skip:
            continue
        for v in venue_order:
            if v == predict_venue:
                for _fi in LAGGED_FEATURE_INDICES:
                    for w in lag_windows_ms:
                        k = int(own_lag_k[w][i])
                        if k < 0 or t_i - int(ts_own[k]) > max_lag_gap_ms[w]:
                            skip = True
                            break
                    if skip:
                        break
            else:
                for _fi in LAGGED_FEATURE_INDICES:
                    for w in lag_windows_ms:
                        k = int(other_lag_k[v][w][i])
                        if k < 0 or t_i - int(other_ts_arrays[v][k]) > max_lag_gap_ms[w]:
                            skip = True
                            break
                    if skip:
                        break
            if skip:
                break
        if skip:
            continue
        kept_ts.append(t_i)
    kept_ts_arr = np.asarray(kept_ts)
    assert len(kept_ts_arr) == len(y), (
        f"kept ts ({len(kept_ts_arr)}) != dataset rows ({len(y)}) — skip rules drifted"
    )
    return x, y, feature_names, kept_ts_arr


def evaluate_venue(
    venue: str,
    series: dict[str, VenueSeries],
    horizon_ms: int,
    tolerance_ms: int,
    lag_windows_ms: list[int],
    train_size: int,
    test_size: int,
    step: int,
    ridge_alpha: float,
    gbdt_params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    top_n_feats: int,
    emit_predictions_path: Path | None,
    ridge_only: bool,
) -> None:
    if emit_predictions_path is not None:
        data = build_dataset_with_ts(
            venue, series, horizon_ms, tolerance_ms, lag_windows_ms,
        )
        if data is None:
            print(f"{venue:<13}  SKIP  insufficient aligned data")
            return
        x, y, feature_names, kept_ts = data
    else:
        raw = build_dataset(
            venue, series, horizon_ms, tolerance_ms, lag_windows_ms,
        )
        if raw is None:
            print(f"{venue:<13}  SKIP  insufficient aligned data")
            return
        x, y, feature_names = raw
        kept_ts = None
    folds = make_folds(len(y), train_size, test_size, step)
    if not folds:
        print(f"{venue:<13}  SKIP  only {len(y)} rows")
        return

    r2_ridge, dir_ridge, conf_ridge = [], [], []
    r2_gbdt, dir_gbdt, conf_gbdt = [], [], []
    importances: list[np.ndarray] = []
    emitted_ts: list[int] = []
    emitted_pred: list[float] = []
    for f in folds:
        xtr, ytr = x[f.train_start : f.train_end], y[f.train_start : f.train_end]
        xte, yte = x[f.test_start : f.test_end], y[f.test_start : f.test_end]

        if emit_predictions_path is not None:
            rs, ridge_preds = _run_fold_ridge_with_preds(xtr, ytr, xte, yte, ridge_alpha)
            if kept_ts is not None:
                emitted_ts.extend(kept_ts[f.test_start : f.test_end].tolist())
                emitted_pred.extend(ridge_preds.tolist())
        else:
            rs = run_fold_ridge(xtr, ytr, xte, yte, ridge_alpha)
        r2_ridge.append(rs["r2"])
        dir_ridge.append(rs["dir"])
        conf_ridge.append(rs["conf_dir"])

        if not ridge_only:
            gs, imp = run_fold_gbdt(
                xtr, ytr, xte, yte, feature_names,
                gbdt_params, num_boost_round, early_stopping_rounds,
            )
            r2_gbdt.append(gs["r2"])
            dir_gbdt.append(gs["dir"])
            conf_gbdt.append(gs["conf_dir"])
            importances.append(imp)

    if emit_predictions_path is not None and emitted_ts:
        emit_predictions_path.mkdir(parents=True, exist_ok=True)
        out_path = emit_predictions_path / f"{venue}_{horizon_ms}ms.csv"
        with out_path.open("w") as f:
            f.write("timestamp_ms,drift_hat\n")
            for ts, p in zip(emitted_ts, emitted_pred, strict=True):
                f.write(f"{int(ts)},{p:.8f}\n")
        print(f"               wrote {len(emitted_ts)} predictions to {out_path}")

    r_r2 = np.asarray(r2_ridge)
    r_dir = np.asarray(dir_ridge)
    r_conf = np.asarray(conf_ridge)
    print(f"{venue:<13}  n={len(y)}  folds={len(folds)}  n_features={x.shape[1]}")
    print(
        f"               Ridge  R²={r_r2.mean():+.3f}±{r_r2.std():.2f}  "
        f"dir={r_dir.mean():.3f}  conf_dir={r_conf.mean():.3f}"
    )
    if not ridge_only:
        g_r2 = np.asarray(r2_gbdt)
        g_dir = np.asarray(dir_gbdt)
        g_conf = np.asarray(conf_gbdt)
        print(
            f"               GBDT   R²={g_r2.mean():+.3f}±{g_r2.std():.2f}  "
            f"dir={g_dir.mean():.3f}  conf_dir={g_conf.mean():.3f}"
        )
    if not ridge_only and top_n_feats > 0 and importances:
        imp_stack = np.vstack(importances)
        imp_norm = imp_stack / imp_stack.sum(axis=1, keepdims=True).clip(min=1)
        mean_imp = imp_norm.mean(axis=0)
        order = np.argsort(-mean_imp)[:top_n_feats]
        ranked = "  ".join(
            f"{feature_names[i]}={mean_imp[i]*100:.1f}%" for i in order
        )
        print(f"               GBDT top gains: {ranked}")


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
    parser.add_argument(
        "--lag-windows-ms", default="2000,5000",
        help="comma-separated lag windows for diff features (ms)",
    )
    parser.add_argument("--tolerance-ms", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--top-n-feats", type=int, default=10)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=5)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument(
        "--emit-predictions-dir", type=Path, default=None,
        help="if set, write per-venue walk-forward Ridge predictions to "
        "<dir>/<venue>_<horizon>ms.csv with columns timestamp_ms,drift_hat",
    )
    parser.add_argument(
        "--ridge-only", action="store_true",
        help="skip GBDT (useful when generating predictions without comparison)",
    )
    args = parser.parse_args()

    gbdt_params = {
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
    lag_windows = [int(x) for x in args.lag_windows_ms.split(",") if x]
    series = load_series(args.db_paths)
    print(f"loaded series: {[(v, len(s.ts)) for v, s in series.items()]}")
    print(f"lag windows (ms): {lag_windows}")
    print(f"gbdt params: {gbdt_params}")

    for horizon in horizons:
        print(f"\n=== horizon={horizon}ms  tolerance={args.tolerance_ms}ms ===")
        for venue in sorted(series.keys()):
            evaluate_venue(
                venue, series, horizon, args.tolerance_ms, lag_windows,
                args.train_size, args.test_size, args.step,
                args.ridge_alpha, gbdt_params, args.num_boost_round,
                args.early_stopping_rounds, args.top_n_feats,
                args.emit_predictions_dir, args.ridge_only,
            )


if __name__ == "__main__":
    main()
