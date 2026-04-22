#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Hourly regime-gated LP backtest.

Pipeline:
  1. Load per-swap attribution (from dex_lp_backtest.py --attrib-out)
  2. Bucket into fixed windows (default 1h)
  3. Per window compute:
       label    = sum(fee) + sum(dv_ext)  [net LP P&L at ext price, in-range only]
       features = prior-window realized quantities (no lookahead):
                  basis_vol, pool_rv, cex_rv, flow_imb,
                  tox_rate_prev, vol_usd_prev, swap_rate_prev, hour, dow
  4. Time-split train/test, fit LightGBM regressor
  5. Gate: LP only when predicted P&L > threshold (0 by default)
  6. Report gated vs always-on P&L, both in raw units and annualized

Retail-feasibility: decisions at hourly cadence are well within retail
execution latency. Gas cost per mint/burn is modeled separately.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False


def load_attrib(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        reader = _csv.DictReader(fh)
        for r in reader:
            rows.append({
                "ts": int(r["ts"]),
                "amount_usd": float(r["amount_usd"]),
                "pool_pre": float(r["pool_pre"]),
                "pool_post": float(r["pool_post"]),
                "ext": float(r["ext"]),
                "dist_pre": float(r["dist_pre"]),
                "dist_post": float(r["dist_post"]),
                "toward": int(r["toward"]),
                "in_range": int(r["in_range"]),
                "share": float(r["share"]),
                "fee": float(r["fee"]),
                "dv_ext": float(r["dv_ext"]),
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def load_cex(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts: list[int] = []
    px: list[float] = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            ts.append(int(r["ts_s"]))
            px.append(float(r["close"]))
    a = np.asarray(ts)
    p = np.asarray(px)
    order = np.argsort(a)
    return a[order], p[order]


def bucketize(rows: list[dict], cex_ts: np.ndarray, cex_px: np.ndarray,
              bucket_s: int) -> list[dict]:
    """Group swaps into windows of bucket_s seconds.

    Emit per-window dict with:
      t_start, t_end, n_swaps, vol_usd, fees, dv_ext_sum,
      basis_mean, basis_abs_mean, basis_std,
      pool_rv, flow_imb, n_in_range, toxic_rate
    plus label `pnl` = fees + dv_ext_sum (in-range only).
    """
    if not rows:
        return []
    t0 = (rows[0]["ts"] // bucket_s) * bucket_s
    t_end_all = rows[-1]["ts"]
    windows: list[dict] = []
    cur_start = t0
    cur: list[dict] = []
    for r in rows:
        if r["ts"] >= cur_start + bucket_s:
            while r["ts"] >= cur_start + bucket_s:
                windows.append(_summarize(cur, cur_start, cur_start + bucket_s,
                                          cex_ts, cex_px))
                cur_start += bucket_s
                cur = []
        cur.append(r)
    if cur:
        windows.append(_summarize(cur, cur_start, cur_start + bucket_s,
                                  cex_ts, cex_px))
    return windows


def _summarize(rs: list[dict], t_start: int, t_end: int,
               cex_ts: np.ndarray, cex_px: np.ndarray) -> dict:
    in_rng = [r for r in rs if r["in_range"]]
    n = len(rs)
    vol = sum(r["amount_usd"] for r in rs)
    fees = sum(r["fee"] for r in in_rng)
    dv_sum = sum(r["dv_ext"] for r in in_rng)

    # Basis stats: pool_pre vs ext at each swap
    if rs:
        basis = np.asarray([(r["pool_pre"] - r["ext"]) / r["ext"] * 1e4 for r in rs])
        basis_mean = float(basis.mean())
        basis_abs_mean = float(np.abs(basis).mean())
        basis_std = float(basis.std())
    else:
        basis_mean = basis_abs_mean = basis_std = 0.0

    # Pool realized log-return vol from pool_post sequence
    if n >= 2:
        prices = np.asarray([r["pool_post"] for r in rs])
        lr = np.diff(np.log(prices))
        pool_rv = float(np.sqrt(np.sum(lr * lr)) * 1e4)
    else:
        pool_rv = 0.0

    # CEX realized vol over the window
    mask = (cex_ts >= t_start) & (cex_ts < t_end)
    cp = cex_px[mask]
    if cp.size >= 2:
        lr = np.diff(np.log(cp))
        cex_rv = float(np.sqrt(np.sum(lr * lr)) * 1e4)
    else:
        cex_rv = 0.0

    # Flow imbalance: net signed volume
    flow_imb = sum(
        r["amount_usd"] * (1 if r["pool_post"] < r["pool_pre"] else -1)
        for r in rs
    )

    # Toxic labels from attribution: "toward" flag weighted by |dv_ext|
    tox_gross = sum(-r["dv_ext"] for r in in_rng if r["toward"] == 1 and r["dv_ext"] < 0)
    toxic_rate = (sum(1 for r in in_rng if r["toward"] == 1)
                  / max(1, len(in_rng)))

    return {
        "t_start": t_start, "t_end": t_end,
        "n_swaps": n, "n_in_range": len(in_rng),
        "vol_usd": vol, "fees": fees, "dv_ext_sum": dv_sum,
        "basis_mean": basis_mean, "basis_abs_mean": basis_abs_mean,
        "basis_std": basis_std,
        "pool_rv": pool_rv, "cex_rv": cex_rv,
        "flow_imb": flow_imb,
        "tox_gross": tox_gross, "toxic_rate": toxic_rate,
    }


def build_xy(windows: list[dict], lookback: int):
    """For each window w_i, features come from windows [i-lookback .. i-1].

    Label = windows[i]["fees"] + windows[i]["dv_ext_sum"].
    Also return per-window realized fees and dv_ext for P&L simulation.
    """
    feat_rows: list[list[float]] = []
    y: list[float] = []
    ts: list[int] = []
    fees_real: list[float] = []
    dv_real: list[float] = []
    hours_of_day: list[int] = []

    def agg(ws: list[dict]) -> list[float]:
        if not ws:
            return [0.0] * 10
        arr_pool_rv = np.asarray([w["pool_rv"] for w in ws])
        arr_cex_rv = np.asarray([w["cex_rv"] for w in ws])
        arr_basis_abs = np.asarray([w["basis_abs_mean"] for w in ws])
        arr_basis_std = np.asarray([w["basis_std"] for w in ws])
        arr_vol = np.asarray([w["vol_usd"] for w in ws])
        arr_flow = np.asarray([w["flow_imb"] for w in ws])
        arr_tox = np.asarray([w["toxic_rate"] for w in ws])
        arr_n = np.asarray([w["n_swaps"] for w in ws])
        return [
            float(arr_pool_rv.mean()), float(arr_pool_rv.max()),
            float(arr_cex_rv.mean()), float(arr_cex_rv.max()),
            float(arr_basis_abs.mean()), float(arr_basis_std.mean()),
            float(arr_vol.sum()), float(arr_flow.sum()),
            float(arr_tox.mean()), float(arr_n.sum()),
        ]

    for i in range(lookback, len(windows)):
        w = windows[i]
        hist = windows[i - lookback:i]
        feats = agg(hist)
        # time-of-day and day-of-week
        hod = (w["t_start"] // 3600) % 24
        dow = (w["t_start"] // 86400) % 7
        feats += [float(hod), float(dow)]
        feat_rows.append(feats)
        label = w["fees"] + w["dv_ext_sum"]  # in-range net
        y.append(label)
        ts.append(w["t_start"])
        fees_real.append(w["fees"])
        dv_real.append(w["dv_ext_sum"])
        hours_of_day.append(hod)

    cols = [
        "pool_rv_mean", "pool_rv_max", "cex_rv_mean", "cex_rv_max",
        "basis_abs_mean", "basis_std_mean", "vol_sum", "flow_sum",
        "tox_rate_mean", "n_swaps_sum", "hour", "dow",
    ]
    return (np.asarray(feat_rows), np.asarray(y), np.asarray(ts),
            np.asarray(fees_real), np.asarray(dv_real), cols)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", type=Path,
                    default=Path("data/dex_cache/attrib_5pct.csv"))
    ap.add_argument("--cex", type=Path,
                    default=Path("data/dex_cache/cex_eth_1m_30d.jsonl"))
    ap.add_argument("--bucket-hours", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=6,
                    help="use prior N windows as features")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--gas-cost-usd", type=float, default=0.50,
                    help="cost of a single modifyLiquidity (mint or burn)")
    ap.add_argument("--deposit-usd", type=float, default=10_000.0)
    args = ap.parse_args()

    print(f"loading {args.attrib}")
    rows = load_attrib(args.attrib)
    cex_ts, cex_px = load_cex(args.cex)
    print(f"  {len(rows):,} swaps")

    bucket_s = int(args.bucket_hours * 3600)
    windows = bucketize(rows, cex_ts, cex_px, bucket_s)
    print(f"  {len(windows):,} windows of {args.bucket_hours}h")

    X, y, ts, fees_real, dv_real, cols = build_xy(windows, args.lookback)
    print(f"  {len(y):,} labeled windows  (dropped first {args.lookback})")
    print(f"  label (window pnl): mean={y.mean():.3e}  "
          f"std={y.std():.3e}  positive={np.mean(y > 0)*100:.1f}%")
    print(f"  in-range-fees mean/window: {fees_real.mean():.2f}")
    print(f"  dv_ext_sum   mean/window: {dv_real.mean():.3e}")

    # y is already in USD: fees(usd) + dv_ext(usd). No rescaling needed.
    y_usd = y
    n_tr = int(len(y_usd) * args.train_frac)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr_usd, yte_usd = y_usd[:n_tr], y_usd[n_tr:]
    fees_tr, fees_te = fees_real[:n_tr], fees_real[n_tr:]
    dv_tr, dv_te = dv_real[:n_tr], dv_real[n_tr:]
    ts_te = ts[n_tr:]
    print(f"\ntrain {len(ytr_usd)}  test {len(yte_usd)}")
    test_hours = len(yte_usd) * args.bucket_hours

    print(f"\n=== always-on LP (baseline) ===")
    print(f"  train pnl (sum):  ${ytr_usd.sum():+.2f}  "
          f"(fees ${fees_tr.sum():.2f}  dv ${dv_tr.sum():+.2f})")
    print(f"  test pnl (sum):   ${yte_usd.sum():+.2f}  "
          f"(fees ${fees_te.sum():.2f}  dv ${dv_te.sum():+.2f})")
    print(f"  test annualized on ${args.deposit_usd:.0f}: "
          f"{yte_usd.sum()/args.deposit_usd * 365/(test_hours/24) * 100:+.1f}%")

    if not HAVE_LGB:
        print("\nlightgbm missing; stopping.")
        return

    # Train regressor to predict y (original mixed units) — signs only matter.
    dtr = lgb.Dataset(Xtr, ytr_usd, feature_name=cols)
    dte = lgb.Dataset(Xte, yte_usd, reference=dtr)
    params = dict(
        objective="regression", metric="mae",
        learning_rate=0.05, num_leaves=15,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        min_data_in_leaf=25, verbose=-1,
    )
    model = lgb.train(params, dtr, num_boost_round=400,
                      valid_sets=[dte],
                      callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)])
    pred = model.predict(Xte)

    # IC = spearman-ish correlation
    if yte_usd.std() > 0 and pred.std() > 0:
        ic = float(np.corrcoef(pred, yte_usd)[0, 1])
    else:
        ic = float("nan")
    imp = sorted(zip(cols, model.feature_importance(importance_type="gain")),
                 key=lambda x: -x[1])
    print("\n=== regime regressor ===")
    print(f"  IC (pearson) on test: {ic:.4f}")
    print(f"  feature gain:")
    for name, g in imp:
        print(f"    {name:<20} {g:>10,.0f}")

    # Simulate gating: LP when pred > threshold
    def simulate(mask: np.ndarray, label: str) -> float:
        # Gas: pay when state changes (in<->out)
        prev = False
        gas = 0.0
        for m in mask:
            if bool(m) != prev:
                gas += args.gas_cost_usd
                prev = bool(m)
        pnl_gated = yte_usd[mask].sum() - gas
        flip_count = int(np.sum(np.diff(mask.astype(int)) != 0))
        print(f"\n  [{label}]  pnl_test=${pnl_gated:+.2f}  "
              f"flips={flip_count}  gas=${gas:.2f}  "
              f"on={mask.mean()*100:.1f}%  "
              f"annualized={pnl_gated/args.deposit_usd * 365/(test_hours/24)*100:+.1f}%")
        return pnl_gated

    print("\n=== gating strategies (test) ===")
    simulate(np.ones(len(yte_usd), dtype=bool), "always-on")
    simulate(pred > 0, "gate: pred>0")
    # Top-half
    thresh = float(np.median(pred))
    simulate(pred > thresh, "gate: pred>median")
    # Top quartile
    thresh = float(np.percentile(pred, 75))
    simulate(pred > thresh, "gate: pred>p75")
    # Top decile
    thresh = float(np.percentile(pred, 90))
    simulate(pred > thresh, "gate: pred>p90")

    # Oracle upper bound: perfect foresight
    simulate(yte_usd > 0, "ORACLE: y_true>0")


if __name__ == "__main__":
    main()
