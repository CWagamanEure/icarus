#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Train toxic-swap classifier from dex_toxic_features.py output.

Time-based 80/20 split. LightGBM primary, logistic baseline.
Evaluates:
  - AUC on test set
  - Precision at top-k% predicted-toxic
  - P&L impact: gated vs. passive LP (skip swaps above predicted-toxic threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import csv as _csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


ALL_FEATURES = [
    "size_usd", "signed_usd", "basis_bps", "abs_basis_bps",
    "rv_5m", "rv_15m", "rv_60m",
    "flow_5m", "flow_30m",
    "rate_5m", "p95_size_usd", "time_since_big_s",
]
BASIS_FEATURES = {"basis_bps", "abs_basis_bps"}


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> float:
    n = int(len(y_score) * k_frac)
    if n <= 0:
        return float("nan")
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].sum() / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("data/dex_cache/toxic_features_5pct.csv"))
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--gate-frac", type=float, default=0.10,
                    help="simulate skipping top-k%% of predicted-toxic swaps")
    ap.add_argument("--drop-basis", action="store_true",
                    help="exclude basis features to test residual edge")
    ap.add_argument("--only-basis", action="store_true",
                    help="use ONLY basis features as a baseline")
    args = ap.parse_args()

    if args.only_basis:
        FEATURE_COLS = [f for f in ALL_FEATURES if f in BASIS_FEATURES]
        tag = "ONLY-BASIS"
    elif args.drop_basis:
        FEATURE_COLS = [f for f in ALL_FEATURES if f not in BASIS_FEATURES]
        tag = "NO-BASIS"
    else:
        FEATURE_COLS = ALL_FEATURES
        tag = "ALL"
    print(f"feature set: {tag}  ({len(FEATURE_COLS)} features)")
    print(f"  {FEATURE_COLS}")

    print(f"loading {args.input}")
    with args.input.open() as fh:
        reader = _csv.DictReader(fh)
        rows = list(reader)
    print(f"  {len(rows):,} rows")

    def to_f(r, k):
        try:
            v = float(r[k])
            return v if np.isfinite(v) else np.nan
        except (ValueError, TypeError):
            return np.nan

    # Filter in-range and drop any NaN features
    clean: list[dict] = []
    for r in rows:
        if int(r["in_range"]) != 1:
            continue
        feats = [to_f(r, c) for c in FEATURE_COLS]
        if any(not np.isfinite(x) for x in feats):
            continue
        clean.append(r)
    print(f"  {len(clean):,} in-range rows after dropna")
    clean.sort(key=lambda r: int(r["ts"]))

    n_train = int(len(clean) * args.train_frac)
    train = clean[:n_train]
    test = clean[n_train:]
    y_tr = np.asarray([int(r["is_toxic"]) for r in train])
    y_te = np.asarray([int(r["is_toxic"]) for r in test])
    print(f"  train {len(train):,}  test {len(test):,}")
    print(f"  test toxic rate: {y_te.mean()*100:.2f}%")

    X_tr = np.asarray([[to_f(r, c) for c in FEATURE_COLS] for r in train])
    X_te = np.asarray([[to_f(r, c) for c in FEATURE_COLS] for r in test])
    test_size = np.asarray([to_f(r, "size_usd") for r in test])
    test_tox = np.asarray([to_f(r, "toxicity_raw") for r in test])
    train_size = np.asarray([to_f(r, "size_usd") for r in train])
    train_tox = np.asarray([to_f(r, "toxicity_raw") for r in train])

    # --- Logistic baseline (standardize first) ---
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-9
    Xs_tr = (X_tr - mu) / sd
    Xs_te = (X_te - mu) / sd
    clf_lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_lr.fit(Xs_tr, y_tr)
    p_lr = clf_lr.predict_proba(Xs_te)[:, 1]

    print("\n--- Logistic Regression ---")
    print(f"  AUC: {roc_auc_score(y_te, p_lr):.4f}")
    print(f"  AP:  {average_precision_score(y_te, p_lr):.4f}")
    print(f"  P@top-5%:  {precision_at_k(y_te, p_lr, 0.05):.3f}")
    print(f"  P@top-10%: {precision_at_k(y_te, p_lr, 0.10):.3f}")

    if HAVE_LGB:
        print("\n--- LightGBM ---")
        dtr = lgb.Dataset(X_tr, y_tr)
        dte = lgb.Dataset(X_te, y_te, reference=dtr)
        params = dict(
            objective="binary", metric="auc",
            learning_rate=0.05, num_leaves=31,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_data_in_leaf=100, verbose=-1,
        )
        model = lgb.train(params, dtr, num_boost_round=500,
                          valid_sets=[dte],
                          callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)])
        p_gb = model.predict(X_te)
        print(f"  AUC: {roc_auc_score(y_te, p_gb):.4f}")
        print(f"  AP:  {average_precision_score(y_te, p_gb):.4f}")
        print(f"  P@top-5%:  {precision_at_k(y_te, p_gb, 0.05):.3f}")
        print(f"  P@top-10%: {precision_at_k(y_te, p_gb, 0.10):.3f}")

        imp = sorted(zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
                     key=lambda x: -x[1])
        print("\n  feature gain (lgb):")
        for name, g in imp:
            print(f"    {name:<20} {g:>12,.0f}")
    else:
        p_gb = p_lr
        print("\n(lightgbm not installed, using logistic for gating sim)")

    # --- Simulate gated strategy on TEST set ---
    # "Gate" = skip the top-k% predicted-toxic swaps (assume we pulled liquidity
    # or at least forfeited the fee + LVR on those). Real-world, pulling liq has
    # gas cost, but as an upper bound:
    gate_k = args.gate_frac
    n_skip = int(len(test) * gate_k)
    skip_idx = np.argsort(-p_gb)[:n_skip]
    gate_mask = np.ones(len(test), dtype=bool)
    gate_mask[skip_idx] = False

    # For each test swap: fee and tox are raw. A fair comparison: "dv_ext at L=1"
    # doesn't give us USD P&L directly, but we can use the fee/LVR ratio:
    # Passive  = sum(fee_share_usd) - sum(toxicity)
    # We don't have fee_share per swap in this CSV; use amount_usd * 5bps as
    # upper-bound (ignoring share). Then we compare passive vs gated magnitudes.
    fee_rate = 5e-4
    fees_ub = test_size * fee_rate
    tox_raw = test_tox

    # Normalize tox to same scale as fees: both proportional to L_pos. So
    # fee_per_unit_L = fee_ub * (L_pos / L_pool_at_swap) — we don't know L_pool.
    # Instead: use the *ratio* fee_ub / tox_raw. Since both scale with L_pos,
    # the ratio is invariant. We just want the COMPARATIVE P&L (passive vs gated).
    # Scale: pick C so that sum(tox_raw * C) equals sum(fees_ub) on the training
    # portion (i.e., passive baseline P&L ≈ 0). Then departures show up clearly.
    tox_train = train_tox
    fees_train = train_size * fee_rate
    # We want C such that scale of tox × C ≈ scale of fees
    C = fees_train.sum() / (tox_train[tox_train > 0].sum() + 1e-30)
    print(f"\nscale factor (fees/tox) from train: C={C:.3e}")
    tox_scaled = np.where(tox_raw > 0, tox_raw * C, 0.0)

    passive_pnl = fees_ub.sum() - tox_scaled.sum()
    gated_pnl = fees_ub[gate_mask].sum() - tox_scaled[gate_mask].sum()
    lost_fees = fees_ub[~gate_mask].sum()
    avoided_tox = tox_scaled[~gate_mask].sum()

    print(f"\n--- Gated strategy (skip top {gate_k*100:.0f}% predicted-toxic) ---")
    print(f"  passive pnl (test):   {passive_pnl:+.2f}  (arbitrary units)")
    print(f"  gated pnl:            {gated_pnl:+.2f}")
    print(f"  improvement:          {gated_pnl - passive_pnl:+.2f}  "
          f"({(gated_pnl/passive_pnl-1)*100 if passive_pnl else 0:+.1f}%)")
    print(f"  lost fees (skipped):  {lost_fees:+.2f}")
    print(f"  avoided toxicity:     {avoided_tox:+.2f}")
    print(f"  avoid/lost ratio:     {avoided_tox/lost_fees if lost_fees else 0:.2f}x")

    # Also show the oracle upper bound (perfect foresight)
    oracle_idx = np.argsort(-tox_scaled)[:n_skip]
    oracle_mask = np.ones(len(test), dtype=bool)
    oracle_mask[oracle_idx] = False
    oracle_pnl = fees_ub[oracle_mask].sum() - tox_scaled[oracle_mask].sum()
    print(f"\n  ORACLE gated pnl:     {oracle_pnl:+.2f}  "
          f"(perfect foresight top {gate_k*100:.0f}%)")
    print(f"  classifier/oracle:    "
          f"{(gated_pnl-passive_pnl)/(oracle_pnl-passive_pnl)*100 if oracle_pnl != passive_pnl else 0:.1f}% of oracle edge captured")


if __name__ == "__main__":
    main()
