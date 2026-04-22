#!/usr/bin/env -S poetry run python
# ruff: noqa: E402, I001

"""Does cross-venue lead-lag survive at hourly horizons?

At 5-second horizons we measured ~40bps of drift alpha per fill on coinbase,
but 40bps of retail fee killed it. At 1-hour horizons, 19bps round-trip is
tiny relative to typical hourly price moves. This script asks: is there any
hourly predictability in cross-venue returns that we can trade with retail
fees?

Approach:
    1. Fetch 6-12 months of hourly closes for BTC from Coinbase (spot),
       Hyperliquid (perp), Kraken (spot).
    2. Compute hourly log returns per venue, aligned on hour.
    3. Report cross-correlation at lags -3 to +3 between every pair.
    4. For each (leader, follower) pair: walk-forward Ridge regressing
       follower[t] on leader[t-1], follower[t-1], follower[t-2]. Report
       out-of-sample IC and the Sharpe of a simple sign-based strategy
       with 19bps round-trip fees applied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HL_URL = "https://api.hyperliquid.xyz/info"
CB_URL_TEMPLATE = "https://api.exchange.coinbase.com/products/{product}/candles"
KR_URL = "https://api.kraken.com/0/public/OHLC"

ROUND_TRIP_FEE_BPS = 19.0  # 6 spot taker × 2 + 3.5 perp taker × 2, conservative retail


def post_json(url: str, body: dict) -> list | dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_json(url: str, params: dict) -> list | dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{qs}",
        headers={"User-Agent": "icarus-research/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_hl_hourly(coin: str, start_ms: int, end_ms: int) -> dict[int, float]:
    out: dict[int, float] = {}
    cursor = start_ms
    step_ms = 500 * 3_600_000
    while cursor < end_ms:
        page_end = min(cursor + step_ms, end_ms)
        batch = post_json(HL_URL, {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": "1h",
                    "startTime": cursor, "endTime": page_end},
        })
        assert isinstance(batch, list)
        if not batch:
            cursor = page_end
            continue
        for c in batch:
            out[int(c["t"])] = float(c["c"])
        cursor = int(batch[-1]["t"]) + 3_600_000
    return out


def fetch_coinbase_hourly(product: str, start_ms: int, end_ms: int) -> dict[int, float]:
    out: dict[int, float] = {}
    step_s = 300 * 3600
    cursor = start_ms // 1000
    end_s = end_ms // 1000
    url = CB_URL_TEMPLATE.format(product=product)
    while cursor < end_s:
        page_end = min(cursor + step_s, end_s)
        params = {
            "granularity": 3600,
            "start": datetime.fromtimestamp(cursor, UTC).isoformat(),
            "end": datetime.fromtimestamp(page_end, UTC).isoformat(),
        }
        try:
            batch = get_json(url, params)
        except Exception as e:
            print(f"coinbase fail ({cursor}): {e}", file=sys.stderr)
            cursor = page_end
            time.sleep(1.0)
            continue
        assert isinstance(batch, list)
        for row in batch:
            out[int(row[0]) * 1000] = float(row[4])  # close
        cursor = page_end
        time.sleep(0.25)
    return out


def fetch_kraken_hourly(pair: str, start_ms: int, end_ms: int) -> dict[int, float]:
    """Kraken returns up to 720 OHLC rows; walk window in chunks."""
    out: dict[int, float] = {}
    cursor = start_ms // 1000
    end_s = end_ms // 1000
    step_s = 720 * 3600
    while cursor < end_s:
        try:
            data = get_json(KR_URL, {"pair": pair, "interval": 60, "since": cursor})
        except Exception as e:
            print(f"kraken fail: {e}", file=sys.stderr)
            cursor += step_s
            time.sleep(1.0)
            continue
        result = data.get("result", {}) if isinstance(data, dict) else {}
        # Kraken result is dict with one pair key + "last"
        rows: list = []
        for k, v in result.items():
            if k != "last" and isinstance(v, list):
                rows = v
                break
        if not rows:
            break
        for row in rows:
            t = int(row[0]) * 1000
            if t <= end_ms:
                out[t] = float(row[4])  # close
        last_t = int(rows[-1][0])
        if last_t <= cursor:
            break
        cursor = last_t + 1
        time.sleep(0.3)
    return out


def cached(path: Path, producer):  # noqa: ANN001
    if path.exists():
        return {int(k): float(v) for k, v in json.loads(path.read_text()).items()}
    data = producer()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def align_returns(
    prices: dict[str, dict[int, float]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Return (R[T,V] log returns, P[T,V] prices, ts list, venue list) aligned on common hours."""
    venues = sorted(prices.keys())
    common = set.intersection(*[set(prices[v].keys()) for v in venues])
    ts_sorted = sorted(common)
    rets = np.full((len(ts_sorted), len(venues)), np.nan)
    px = np.full((len(ts_sorted), len(venues)), np.nan)
    for vi, v in enumerate(venues):
        p_arr = np.asarray([prices[v][t] for t in ts_sorted])
        px[:, vi] = p_arr
        rets[1:, vi] = np.log(p_arr[1:]) - np.log(p_arr[:-1])
    return rets[1:], px[1:], ts_sorted[1:], venues


def crosscorr(rets: np.ndarray, venues: list[str], max_lag: int = 3) -> None:
    print(f"\ncross-correlation of hourly returns, leader lagged behind follower")
    print(f"  (positive lag = leader predicts follower k hours later)")
    print(f"\n{'lag':>4}  " + "  ".join(f"{a:>5}→{b:<5}" for a in venues for b in venues if a != b))
    for lag in range(-max_lag, max_lag + 1):
        row = [f"{lag:>+4d}"]
        for a in venues:
            for b in venues:
                if a == b:
                    continue
                a_idx = venues.index(a)
                b_idx = venues.index(b)
                x = rets[:, a_idx]
                y = rets[:, b_idx]
                if lag > 0:
                    xs, ys = x[:-lag], y[lag:]
                elif lag < 0:
                    xs, ys = x[-lag:], y[:lag]
                else:
                    xs, ys = x, y
                mask = ~(np.isnan(xs) | np.isnan(ys))
                if mask.sum() < 100:
                    row.append("  n/a ")
                else:
                    c = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
                    row.append(f"{c:+7.4f}")
        print("  ".join(row))


def build_basis_features(
    px: np.ndarray, rets: np.ndarray, venues: list[str],
    leader: str, follower: str, basis_window: int = 24,
) -> np.ndarray:
    """Per-hour basis features: level, change, z-score vs rolling window."""
    li = venues.index(leader)
    fi = venues.index(follower)
    basis = np.log(px[:, li]) - np.log(px[:, fi])  # log-ratio basis (dimensionless)
    n = len(basis)
    # Rolling mean/std over [t-window, t-1]
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    for t in range(basis_window, n):
        window = basis[t - basis_window:t]
        rolling_mean[t] = window.mean()
        rolling_std[t] = window.std() if window.std() > 1e-12 else 1e-12
    z = (basis - rolling_mean) / rolling_std
    d_basis = np.concatenate([[np.nan], np.diff(basis)])
    return np.column_stack([basis, d_basis, z])


def walk_forward_ridge(
    rets: np.ndarray, px: np.ndarray, venues: list[str],
    leader: str, follower: str,
    train_hours: int, test_hours: int,
    lags: int = 3, use_basis: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict follower[t] from lagged returns (both venues) and optional basis features.
    Returns (actual, predicted) aligned arrays from the test windows only.
    """
    from sklearn.linear_model import Ridge  # type: ignore

    li = venues.index(leader)
    fi = venues.index(follower)
    n = rets.shape[0]

    basis_feats = (
        build_basis_features(px, rets, venues, leader, follower)
        if use_basis else None
    )

    # Build full feature matrix.
    X_list, y_list = [], []
    for t in range(lags, n):
        feat: list[float] = []
        for k in range(1, lags + 1):
            feat.append(rets[t - k, li])
            feat.append(rets[t - k, fi])
        if basis_feats is not None:
            feat.extend(basis_feats[t - 1].tolist())  # basis[t-1], d_basis[t-1], z[t-1]
        X_list.append(feat)
        y_list.append(rets[t, fi])
    X = np.asarray(X_list)
    y = np.asarray(y_list)

    actual_list, pred_list = [], []
    step = test_hours
    start = train_hours
    while start + test_hours <= len(X):
        Xtr, ytr = X[start - train_hours:start], y[start - train_hours:start]
        Xte, yte = X[start:start + test_hours], y[start:start + test_hours]
        mask_tr = ~(np.isnan(Xtr).any(axis=1) | np.isnan(ytr))
        mask_te = ~(np.isnan(Xte).any(axis=1) | np.isnan(yte))
        if mask_tr.sum() < 100 or mask_te.sum() < 10:
            start += step
            continue
        model = Ridge(alpha=1.0)
        model.fit(Xtr[mask_tr], ytr[mask_tr])
        pred = model.predict(Xte[mask_te])
        actual_list.append(yte[mask_te])
        pred_list.append(pred)
        start += step
    if not actual_list:
        return np.array([]), np.array([])
    return np.concatenate(actual_list), np.concatenate(pred_list)


def evaluate_strategy(actual: np.ndarray, pred: np.ndarray, label: str) -> None:
    if len(actual) == 0:
        print(f"{label}: no out-of-sample predictions")
        return
    ic = float(np.corrcoef(actual, pred)[0, 1])
    oos_r2 = 1.0 - np.var(actual - pred) / np.var(actual)

    # Simple sign strategy: long if pred > 0, short if pred < 0, 1h hold.
    # Round-trip fee per trade = 19 bps = 0.0019 applied to every entry+exit.
    fee = ROUND_TRIP_FEE_BPS / 10_000.0
    strat_returns = np.sign(pred) * actual - fee  # every hour enters+exits
    gross_returns = np.sign(pred) * actual
    sharpe_net = strat_returns.mean() / (strat_returns.std() + 1e-12)
    sharpe_net_ann = sharpe_net * np.sqrt(24 * 365)
    sharpe_gross = gross_returns.mean() / (gross_returns.std() + 1e-12)
    sharpe_gross_ann = sharpe_gross * np.sqrt(24 * 365)
    wins = (strat_returns > 0).mean()

    print(f"\n{label}:")
    print(f"  n_oos={len(actual):,}  IC={ic:+.4f}  OOS R²={oos_r2:+.4f}")
    print(f"  strategy (long/short 1h by sign of pred, fee={ROUND_TRIP_FEE_BPS} bps rt):")
    print(f"    gross mean ret={gross_returns.mean()*10000:+.2f} bps/hr  "
          f"std={gross_returns.std()*10000:.2f} bps  "
          f"Sharpe(ann)={sharpe_gross_ann:+.2f}")
    print(f"    net   mean ret={strat_returns.mean()*10000:+.2f} bps/hr  "
          f"std={strat_returns.std()*10000:.2f} bps  "
          f"Sharpe(ann)={sharpe_net_ann:+.2f}  win%={wins:.1%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--coin", default="BTC")
    p.add_argument("--cache-dir", type=Path, default=Path("data/hourly_cache"))
    p.add_argument("--train-hours", type=int, default=24 * 30)  # 30 days
    p.add_argument("--test-hours", type=int, default=24 * 7)    # 7 days
    p.add_argument("--include-kraken", action="store_true")
    args = p.parse_args()

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86_400 * 1000
    print(f"fetching {args.days}d  "
          f"{datetime.fromtimestamp(start_ms/1000, UTC):%Y-%m-%d} → "
          f"{datetime.fromtimestamp(end_ms/1000, UTC):%Y-%m-%d}")

    cbp = f"{args.coin}-USD"
    kraken_pair = f"X{args.coin}ZUSD" if args.coin == "BTC" else f"{args.coin}USD"
    # Kraken's BTC code is XBT in their pair naming.
    if args.coin == "BTC":
        kraken_pair = "XXBTZUSD"

    prices: dict[str, dict[int, float]] = {}
    prices["coinbase"] = cached(
        args.cache_dir / f"cb_{cbp}_{start_ms}_{end_ms}.json",
        lambda: {str(k): v for k, v in fetch_coinbase_hourly(cbp, start_ms, end_ms).items()},
    )
    prices["hyperliquid_perp"] = cached(
        args.cache_dir / f"hl_{args.coin}_{start_ms}_{end_ms}.json",
        lambda: {str(k): v for k, v in fetch_hl_hourly(args.coin, start_ms, end_ms).items()},
    )
    if args.include_kraken:
        prices["kraken"] = cached(
            args.cache_dir / f"kr_{kraken_pair}_{start_ms}_{end_ms}.json",
            lambda: {str(k): v for k, v in fetch_kraken_hourly(kraken_pair, start_ms, end_ms).items()},
        )

    for v, d in prices.items():
        print(f"  {v}: {len(d):,} hourly bars")

    rets, px, ts, venues = align_returns(prices)
    print(f"\naligned {len(rets):,} hours across {venues}")

    crosscorr(rets, venues)

    # Walk-forward every pair, with and without basis features.
    for use_basis in [False, True]:
        label_suffix = " (with basis)" if use_basis else " (returns only)"
        print(f"\n{'=' * 20}{label_suffix}{'=' * 20}")
        for leader in venues:
            for follower in venues:
                if leader == follower:
                    continue
                actual, pred = walk_forward_ridge(
                    rets, px, venues, leader, follower,
                    train_hours=args.train_hours, test_hours=args.test_hours,
                    use_basis=use_basis,
                )
                evaluate_strategy(actual, pred, f"{leader} → {follower}{label_suffix}")


if __name__ == "__main__":
    main()
