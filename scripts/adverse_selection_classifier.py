#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Walk-forward adverse-selection classifier.

Predicts P(the next-horizon mid move is "large") from microstructure features.
This is a *magnitude* predictor, not a direction predictor — the walk-forward
drift R² showed direction is at-or-below random for 3/4 venues, so directional
skew is a dead end. But if we can flag "big move coming" (regardless of sign),
we can pull or widen quotes ahead of adverse flow.

Target:
    y = 1  if  |mid(t+H) - mid(t)| > move_bps_threshold * mid(t) / 10000
    y = 0  otherwise

Features (per tick):
    microprice-mid, depth_imbalance, vol_bps, trade_net_flow, trade_total_size,
    basis_dislocation = mid - basis_reconstructed

Model: L2-regularized logistic regression per (train-fold). Walk-forward eval
matches scripts/walk_forward_drift.py structure.

Per-fold metrics: AUC, base-rate, precision@top-5%, precision@top-20%, and
"pull economics": if we pull the top-p fraction of highest-probability ticks,
what fraction of toxicity do we capture?
"""

from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


FEATURE_NAMES = (
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
    "basis_dislocation",
)


@dataclass(frozen=True, slots=True)
class VenueDataset:
    venue: str
    x: np.ndarray  # (n, k) raw features
    y_bps: np.ndarray  # (n,) signed mid move over horizon in bps
    mid: np.ndarray  # (n,) reference mid at t
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
    auc: float
    base_rate: float
    precision_top5: float
    precision_top20: float
    capture_at_pull20: float  # % of toxicity caught by pulling top 20%
    n_test: int
    n_train: int


def load_datasets(
    db_paths: list[Path], horizon_ms: int, tolerance_ms: int
) -> dict[str, VenueDataset]:
    per_venue_rows: dict[
        str, list[tuple[int, float, tuple[float, ...]]]
    ] = defaultdict(list)
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
                        float(mid) - reconstructed,
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
        ys_bps: list[float] = []
        mids_out: list[float] = []
        row_ts: list[int] = []
        for i in range(len(rows)):
            target_ts = ts_list[i] + horizon_ms
            j = bisect_left(ts_list, target_ts, lo=i + 1)
            if j >= len(rows):
                break
            if ts_list[j] - target_ts > tolerance_ms:
                continue
            xs.append(features[i])
            ys_bps.append((mids[j] - mids[i]) / mids[i] * 10_000.0)
            mids_out.append(mids[i])
            row_ts.append(ts_list[i])
        if len(ys_bps) < 200:
            continue
        datasets[venue] = VenueDataset(
            venue=venue,
            x=np.asarray(xs),
            y_bps=np.asarray(ys_bps),
            mid=np.asarray(mids_out),
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


def score_fold(
    probs: np.ndarray, y_bin: np.ndarray
) -> dict[str, float]:
    n = len(y_bin)
    base_rate = float(y_bin.mean())
    auc = (
        float(roc_auc_score(y_bin, probs))
        if 0 < y_bin.sum() < n
        else 0.5
    )
    # Precision @ top k%
    order = np.argsort(-probs)
    top5 = order[: max(1, n // 20)]
    top20 = order[: max(1, n // 5)]
    prec5 = float(y_bin[top5].mean())
    prec20 = float(y_bin[top20].mean())
    capture20 = (
        float(y_bin[top20].sum()) / float(y_bin.sum()) if y_bin.sum() > 0 else 0.0
    )
    return {
        "auc": auc,
        "base_rate": base_rate,
        "precision_top5": prec5,
        "precision_top20": prec20,
        "capture_at_pull20": capture20,
    }


def evaluate_venue(
    dataset: VenueDataset,
    folds: list[Fold],
    move_bps_threshold: float,
) -> list[FoldResult]:
    results: list[FoldResult] = []
    for fold in folds:
        x_train = dataset.x[fold.train_start : fold.train_end]
        y_train_bps = dataset.y_bps[fold.train_start : fold.train_end]
        x_test = dataset.x[fold.test_start : fold.test_end]
        y_test_bps = dataset.y_bps[fold.test_start : fold.test_end]
        if len(y_train_bps) < 200 or len(y_test_bps) < 50:
            continue

        y_train_bin = (np.abs(y_train_bps) > move_bps_threshold).astype(int)
        y_test_bin = (np.abs(y_test_bps) > move_bps_threshold).astype(int)

        # Skip folds where class is degenerate (all 0 or all 1).
        if y_train_bin.sum() == 0 or y_train_bin.sum() == len(y_train_bin):
            continue

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        x_train_s = (x_train - mean) / std
        x_test_s = np.clip((x_test - mean) / std, -5.0, 5.0)

        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(x_train_s, y_train_bin)
        probs = model.predict_proba(x_test_s)[:, 1]

        scores = score_fold(probs, y_test_bin)
        results.append(
            FoldResult(
                fold=fold,
                venue=dataset.venue,
                auc=scores["auc"],
                base_rate=scores["base_rate"],
                precision_top5=scores["precision_top5"],
                precision_top20=scores["precision_top20"],
                capture_at_pull20=scores["capture_at_pull20"],
                n_test=len(y_test_bin),
                n_train=len(y_train_bin),
            )
        )
    return results


def summarize(venue: str, results: list[FoldResult]) -> None:
    if not results:
        print(f"{venue}: no folds")
        return
    aucs = np.array([r.auc for r in results])
    base = np.array([r.base_rate for r in results])
    prec5 = np.array([r.precision_top5 for r in results])
    prec20 = np.array([r.precision_top20 for r in results])
    cap20 = np.array([r.capture_at_pull20 for r in results])
    pos = int((aucs > 0.55).sum())
    print(
        f"{venue:<13}  folds={len(results)}  "
        f"AUC mean={aucs.mean():.3f}  std={aucs.std():.3f}  "
        f"min={aucs.min():.3f}  max={aucs.max():.3f}  "
        f"AUC>.55: {pos}/{len(results)}  "
        f"base_rate={base.mean():.3f}"
    )
    print(
        f"               prec@top5%={prec5.mean():.3f}  "
        f"prec@top20%={prec20.mean():.3f}  "
        f"capture@pull20%={cap20.mean():.3f}"
    )
    for r in results:
        lift5 = (r.precision_top5 / r.base_rate) if r.base_rate > 0 else 0.0
        print(
            f"  fold {r.fold.fold_index:2d} "
            f"[{r.fold.train_start:>6}:{r.fold.train_end:>6} / "
            f"{r.fold.test_start:>6}:{r.fold.test_end:>6}]  "
            f"AUC={r.auc:.3f}  base={r.base_rate:.3f}  "
            f"p@5%={r.precision_top5:.3f} (lift {lift5:.2f}x)  "
            f"cap@20%={r.capture_at_pull20:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-paths",
        nargs="+",
        type=Path,
        default=[Path("data/filter_eval_maker.sqlite3")],
    )
    parser.add_argument("--horizon-ms", type=int, default=500)
    parser.add_argument("--tolerance-ms", type=int, default=5000)
    parser.add_argument(
        "--move-bps-threshold",
        type=float,
        default=1.0,
        help="absolute mid move > this many bps over horizon = 'toxic'",
    )
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--step", type=int, default=2000)
    args = parser.parse_args()

    datasets = load_datasets(args.db_paths, args.horizon_ms, args.tolerance_ms)
    print(
        f"horizon={args.horizon_ms}ms  threshold={args.move_bps_threshold}bps  "
        f"train/test/step={args.train_size}/{args.test_size}/{args.step}  "
        f"venues={sorted(datasets.keys())}"
    )
    for venue in sorted(datasets.keys()):
        ds = datasets[venue]
        folds = make_folds(len(ds.y_bps), args.train_size, args.test_size, args.step)
        results = evaluate_venue(ds, folds, args.move_bps_threshold)
        summarize(venue, results)


if __name__ == "__main__":
    main()
