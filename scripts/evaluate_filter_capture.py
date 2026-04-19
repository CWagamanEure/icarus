#!/usr/bin/env -S poetry run python

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate captured filter snapshots from SQLite.",
    )
    parser.add_argument(
        "--db-path",
        default="data/filter_eval.sqlite3",
        help="SQLite file produced by capture_filter_eval.py.",
    )
    parser.add_argument(
        "--anchor-exchange",
        default="coinbase",
        help="Exchange to use as the future target reference.",
    )
    parser.add_argument(
        "--horizons-ms",
        type=int,
        nargs="+",
        default=[100, 250, 500, 1000],
        help="Prediction horizons in milliseconds for future-anchor MAE/RMSE.",
    )
    return parser


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def rmse(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def format_metric(label: str, values: list[float]) -> str:
    return f"  {label:<18} mae={mean(values):>8.4f} rmse={rmse(values):>8.4f} n={len(values)}"


def correlation(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return float("nan")
    mean_x = mean(xs)
    mean_y = mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


def sign_hit_rate(signals: list[float], returns: list[float]) -> float:
    if not signals or len(signals) != len(returns):
        return float("nan")
    hits = 0
    total = 0
    for signal, future_return in zip(signals, returns):
        if signal == 0.0 or future_return == 0.0:
            continue
        total += 1
        if (signal > 0.0 and future_return > 0.0) or (signal < 0.0 and future_return < 0.0):
            hits += 1
    return (hits / total) if total else float("nan")


def avg_by_signal_sign(signals: list[float], returns: list[float], positive: bool) -> float:
    selected = [
        future_return
        for signal, future_return in zip(signals, returns)
        if (signal > 0.0 if positive else signal < 0.0)
    ]
    return mean(selected)


def target_label(target_kind: str, target_name: str) -> str:
    if target_kind == "exchange":
        return target_name
    return "composite"


def load_rows(conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], dict[int, dict[str, sqlite3.Row]], dict[int, dict[str, sqlite3.Row]]]:
    conn.row_factory = sqlite3.Row
    updates = conn.execute(
        "SELECT * FROM updates ORDER BY timestamp_ms, id"
    ).fetchall()

    venues_by_update: dict[int, dict[str, sqlite3.Row]] = defaultdict(dict)
    for row in conn.execute("SELECT * FROM venue_states ORDER BY update_id, exchange"):
        venues_by_update[int(row["update_id"])][str(row["exchange"])] = row

    basis_by_update: dict[int, dict[str, sqlite3.Row]] = defaultdict(dict)
    for row in conn.execute("SELECT * FROM basis_states ORDER BY update_id, exchange"):
        basis_by_update[int(row["update_id"])][str(row["exchange"])] = row

    return updates, venues_by_update, basis_by_update


def find_future_index(
    timestamps: list[int],
    updates: list[sqlite3.Row],
    start_index: int,
    target_time: int,
    *,
    target_kind: str,
    target_name: str,
    venues_by_update: dict[int, dict[str, sqlite3.Row]],
) -> int | None:
    future_index = bisect.bisect_left(timestamps, target_time, lo=start_index + 1)
    while future_index < len(updates):
        future_update_id = int(updates[future_index]["id"])
        if target_kind == "exchange":
            if target_name in venues_by_update.get(future_update_id, {}):
                return future_index
        else:
            if updates[future_index]["composite_price"] is not None:
                return future_index
        future_index += 1
    return None


def get_target_price(
    update: sqlite3.Row,
    *,
    update_id: int,
    target_kind: str,
    target_name: str,
    venues_by_update: dict[int, dict[str, sqlite3.Row]],
) -> float | None:
    if target_kind == "exchange":
        row = venues_by_update.get(update_id, {}).get(target_name)
        return float(row["fair_value"]) if row is not None else None
    value = update["composite_price"]
    return float(value) if value is not None else None


def main() -> None:
    args = build_parser().parse_args()
    conn = sqlite3.connect(Path(args.db_path))
    updates, venues_by_update, basis_by_update = load_rows(conn)
    conn.close()

    if not updates:
        raise SystemExit(f"no captured updates found in {args.db_path}")

    timestamps = [int(row["timestamp_ms"]) for row in updates]
    prediction_targets = [
        ("exchange", args.anchor_exchange),
        ("exchange", "kraken"),
        ("composite", "composite"),
    ]

    outside_counts = {"composite": 0, "kalman": 0, "basis": 0}
    outside_denoms = {"composite": 0, "kalman": 0, "basis": 0}

    basis_reconstruction_errors: list[float] = []
    horizon_errors: dict[tuple[str, str, int], dict[str, list[float]]] = {
        (target_kind, target_name, horizon): {
            "composite": [],
            "kalman": [],
            "basis": [],
            "basis_live": [],
            "basis_held": [],
            "composite_with_perp": [],
            "composite_without_perp": [],
            "kalman_with_perp": [],
            "kalman_without_perp": [],
            "basis_with_perp": [],
            "basis_without_perp": [],
        }
        for target_kind, target_name in prediction_targets
        for horizon in args.horizons_ms
    }
    horizon_signals: dict[int, dict[str, list[float]]] = {
        horizon: {
            "composite": [],
            "composite_return": [],
            "kalman": [],
            "kalman_return": [],
            "basis": [],
            "basis_return": [],
        }
        for horizon in args.horizons_ms
    }

    for index, update in enumerate(updates):
        update_id = int(update["id"])
        venue_rows = venues_by_update.get(update_id, {})
        if not venue_rows:
            continue

        spot_prices = [
            float(row["fair_value"])
            for row in venue_rows.values()
            if row["venue_kind"] == "spot"
        ]
        if spot_prices:
            low = min(spot_prices)
            high = max(spot_prices)
            for key, column in (
                ("composite", "composite_price"),
                ("kalman", "kalman_filtered_price"),
                ("basis", "basis_common_price"),
            ):
                value = update[column]
                if value is None:
                    continue
                outside_denoms[key] += 1
                value_f = float(value)
                if value_f < low or value_f > high:
                    outside_counts[key] += 1

        basis_common = update["basis_common_price"]
        if basis_common is not None:
            basis_common_f = float(basis_common)
            basis_rows = basis_by_update.get(update_id, {})
            for exchange, venue_row in venue_rows.items():
                basis_row = basis_rows.get(exchange)
                basis_estimate = float(basis_row["basis_estimate"]) if basis_row is not None else 0.0
                predicted = basis_common_f + basis_estimate
                basis_reconstruction_errors.append(abs(float(venue_row["fair_value"]) - predicted))

        anchor_row = venue_rows.get(args.anchor_exchange)
        anchor_price_now = float(anchor_row["fair_value"]) if anchor_row is not None else None
        has_perp = "hyperliquid_perp" in venue_rows

        for horizon in args.horizons_ms:
            target_time = int(update["timestamp_ms"]) + horizon
            for target_kind, target_name in prediction_targets:
                future_index = find_future_index(
                    timestamps,
                    updates,
                    index,
                    target_time,
                    target_kind=target_kind,
                    target_name=target_name,
                    venues_by_update=venues_by_update,
                )
                if future_index is None:
                    continue
                future_update = updates[future_index]
                future_update_id = int(future_update["id"])
                target_price = get_target_price(
                    future_update,
                    update_id=future_update_id,
                    target_kind=target_kind,
                    target_name=target_name,
                    venues_by_update=venues_by_update,
                )
                if target_price is None:
                    continue
                bucket = horizon_errors[(target_kind, target_name, horizon)]
                suffix = "with_perp" if has_perp else "without_perp"
                if update["composite_price"] is not None:
                    value = abs(float(update["composite_price"]) - target_price)
                    bucket["composite"].append(value)
                    bucket[f"composite_{suffix}"].append(value)
                if update["kalman_filtered_price"] is not None:
                    value = abs(float(update["kalman_filtered_price"]) - target_price)
                    bucket["kalman"].append(value)
                    bucket[f"kalman_{suffix}"].append(value)
                if update["basis_common_price"] is not None:
                    basis_error = abs(float(update["basis_common_price"]) - target_price)
                    bucket["basis"].append(basis_error)
                    bucket[f"basis_{suffix}"].append(basis_error)
                    if int(update["basis_is_live"]) == 1:
                        bucket["basis_live"].append(basis_error)
                    else:
                        bucket["basis_held"].append(basis_error)

            future_index = find_future_index(
                timestamps,
                updates,
                index,
                target_time,
                target_kind="exchange",
                target_name=args.anchor_exchange,
                venues_by_update=venues_by_update,
            )
            if future_index is not None and anchor_price_now is not None:
                future_update = updates[future_index]
                future_update_id = int(future_update["id"])
                target_price = get_target_price(
                    future_update,
                    update_id=future_update_id,
                    target_kind="exchange",
                    target_name=args.anchor_exchange,
                    venues_by_update=venues_by_update,
                )
                if target_price is not None:
                    future_return = target_price - anchor_price_now
                    if update["composite_price"] is not None:
                        horizon_signals[horizon]["composite"].append(
                            float(update["composite_price"]) - anchor_price_now
                        )
                        horizon_signals[horizon]["composite_return"].append(future_return)
                    if update["kalman_filtered_price"] is not None:
                        horizon_signals[horizon]["kalman"].append(
                            float(update["kalman_filtered_price"]) - anchor_price_now
                        )
                        horizon_signals[horizon]["kalman_return"].append(future_return)
                    if update["basis_common_price"] is not None:
                        horizon_signals[horizon]["basis"].append(
                            float(update["basis_common_price"]) - anchor_price_now
                        )
                        horizon_signals[horizon]["basis_return"].append(future_return)

    print(f"rows: {len(updates)}")
    print(f"anchor target: {args.anchor_exchange}")
    print(f"horizons_ms: {', '.join(str(h) for h in args.horizons_ms)}")
    print()
    print("Future-target absolute error by horizon:")
    for target_kind, target_name in prediction_targets:
        print(f"Target: future {target_label(target_kind, target_name)}")
        for horizon in args.horizons_ms:
            bucket = horizon_errors[(target_kind, target_name, horizon)]
            print(f"  horizon={horizon}ms")
            print(format_metric("raw_composite", bucket["composite"]))
            print(format_metric("kalman_1d", bucket["kalman"]))
            print(format_metric("basis_common", bucket["basis"]))
            if target_kind == "exchange" and target_name == args.anchor_exchange:
                print(format_metric("basis_common live", bucket["basis_live"]))
                print(format_metric("basis_common held", bucket["basis_held"]))
            print(format_metric("basis with_perp", bucket["basis_with_perp"]))
            print(format_metric("basis without_perp", bucket["basis_without_perp"]))
            print()
    print("Signal vs future anchor return:")
    for horizon in args.horizons_ms:
        print(f"  horizon={horizon}ms")
        for label, key, return_key in (
            ("raw_composite", "composite", "composite_return"),
            ("kalman_1d", "kalman", "kalman_return"),
            ("basis_common", "basis", "basis_return"),
        ):
            signals = horizon_signals[horizon][key]
            returns = horizon_signals[horizon][return_key]
            n = min(len(signals), len(returns))
            signals = signals[:n]
            returns = returns[:n]
            print(
                f"  {label:<18} corr={correlation(signals, returns):>8.4f} "
                f"hit_rate={sign_hit_rate(signals, returns):>8.4%} "
                f"avg_ret_pos={avg_by_signal_sign(signals, returns, positive=True):>8.4f} "
                f"avg_ret_neg={avg_by_signal_sign(signals, returns, positive=False):>8.4f} "
                f"n={n}"
            )
        print()
    print("Outside current spot venue cloud:")
    for key in ("composite", "kalman", "basis"):
        denom = outside_denoms[key]
        freq = (outside_counts[key] / denom) if denom else float("nan")
        print(f"  {key}_outside_freq={freq:.4%} n={denom}")
    print()
    print("Basis reconstruction:")
    print(
        f"  basis_reconstruction_mae={mean(basis_reconstruction_errors):.4f} n={len(basis_reconstruction_errors)}"
    )


if __name__ == "__main__":
    main()
