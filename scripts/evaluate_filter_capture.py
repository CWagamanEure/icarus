#!/usr/bin/env -S poetry run python

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from icarus.strategy.fair_value.weighting import cap_and_renormalize


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
    parser.add_argument(
        "--max-venue-age-ms",
        type=float,
        default=750.0,
        help=(
            "Only treat exchange venue states as valid evaluation targets / cloud bounds "
            "when their captured age is at or below this threshold."
        ),
    )
    parser.add_argument(
        "--composite-max-weight",
        type=float,
        default=0.75,
        help=(
            "Per-venue max weight when reconstructing a spot-only composite "
            "from the basis filter state."
        ),
    )
    parser.add_argument(
        "--min-consensus-venues",
        type=int,
        default=2,
        help=(
            "Minimum fresh spot venues required to compute a consensus target "
            "(median / variance-weighted) and its corresponding basis prediction."
        ),
    )
    return parser


def median_value(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        raise ValueError("cannot take median of empty list")
    ordered = sorted(values)
    if n % 2 == 1:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


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
    return target_kind


def paired_diff_stats(a: list[float], b: list[float]) -> dict[str, float]:
    """Summarize pairwise (a[i] - b[i]) differences.

    Returns mean, sem, t-statistic, fraction where a < b (a beats b),
    fraction where a == b (ties), and n.
    """
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"),
                "t": float("nan"), "a_better_frac": float("nan"),
                "tie_frac": float("nan")}
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = sum(diffs) / n
    if n > 1:
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        sem = (variance / n) ** 0.5
    else:
        sem = float("nan")
    t_stat = mean_diff / sem if sem and sem > 0 else float("inf" if mean_diff != 0 else "nan")
    wins = sum(1 for x, y in zip(a, b) if x < y)
    ties = sum(1 for x, y in zip(a, b) if x == y)
    return {
        "n": n,
        "mean": mean_diff,
        "sem": sem,
        "t": t_stat,
        "a_better_frac": wins / n,
        "tie_frac": ties / n,
    }


def is_fresh_venue_row(row: sqlite3.Row | None, *, max_venue_age_ms: float) -> bool:
    if row is None:
        return False
    age_ms = row["age_ms"]
    if age_ms is None:
        return False
    return float(age_ms) <= max_venue_age_ms


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
    target_prices_for_target: dict[int, float],
) -> int | None:
    future_index = bisect.bisect_left(timestamps, target_time, lo=start_index + 1)
    while future_index < len(updates):
        if int(updates[future_index]["id"]) in target_prices_for_target:
            return future_index
        future_index += 1
    return None


def compute_target_prices(
    updates: list[sqlite3.Row],
    venues_by_update: dict[int, dict[str, sqlite3.Row]],
    *,
    prediction_targets: list[tuple[str, str]],
    max_venue_age_ms: float,
    min_consensus_venues: int,
) -> dict[tuple[str, str], dict[int, float]]:
    """Precompute per-update target price for every requested target kind."""
    targets: dict[tuple[str, str], dict[int, float]] = {
        target: {} for target in prediction_targets
    }
    for update in updates:
        uid = int(update["id"])
        venue_rows = venues_by_update.get(uid, {})
        fresh_spot_rows = [
            row
            for row in venue_rows.values()
            if row["venue_kind"] == "spot"
            and is_fresh_venue_row(row, max_venue_age_ms=max_venue_age_ms)
        ]

        for target in prediction_targets:
            target_kind, target_name = target
            if target_kind == "exchange":
                row = venue_rows.get(target_name)
                if is_fresh_venue_row(row, max_venue_age_ms=max_venue_age_ms):
                    targets[target][uid] = float(row["fair_value"])
            elif target_kind == "composite":
                value = update["composite_price"]
                if value is not None:
                    targets[target][uid] = float(value)
            elif target_kind == "consensus_median":
                if len(fresh_spot_rows) >= min_consensus_venues:
                    targets[target][uid] = median_value(
                        [float(row["fair_value"]) for row in fresh_spot_rows]
                    )
            elif target_kind == "consensus_weighted":
                if len(fresh_spot_rows) >= min_consensus_venues:
                    total_weight = 0.0
                    weighted_sum = 0.0
                    for row in fresh_spot_rows:
                        variance = float(row["variance"])
                        if not math.isfinite(variance) or variance <= 0.0:
                            continue
                        weight = 1.0 / variance
                        total_weight += weight
                        weighted_sum += weight * float(row["fair_value"])
                    if total_weight > 0.0:
                        targets[target][uid] = weighted_sum / total_weight
    return targets


def basis_predicted_consensus_price(
    update: sqlite3.Row,
    *,
    update_id: int,
    venues_by_update: dict[int, dict[str, sqlite3.Row]],
    basis_by_update: dict[int, dict[str, sqlite3.Row]],
    max_venue_age_ms: float,
    min_consensus_venues: int,
    mode: str,
) -> float | None:
    """Filter's prediction of a consensus target, using the same venue set."""
    basis_common = update["basis_common_price"]
    if basis_common is None:
        return None
    basis_common_f = float(basis_common)
    venue_rows = venues_by_update.get(update_id, {})
    basis_rows = basis_by_update.get(update_id, {})

    per_venue: list[tuple[float, float]] = []  # (reconstructed_price, variance)
    for exchange, venue_row in venue_rows.items():
        if venue_row["venue_kind"] != "spot":
            continue
        if not is_fresh_venue_row(venue_row, max_venue_age_ms=max_venue_age_ms):
            continue
        basis_row = basis_rows.get(exchange)
        basis_estimate = float(basis_row["basis_estimate"]) if basis_row is not None else 0.0
        predicted_price = basis_common_f + basis_estimate
        variance = float(venue_row["variance"])
        per_venue.append((predicted_price, variance))

    if len(per_venue) < min_consensus_venues:
        return None

    if mode == "consensus_median":
        return median_value([price for price, _ in per_venue])
    if mode == "consensus_weighted":
        total_weight = 0.0
        weighted_sum = 0.0
        for price, variance in per_venue:
            if not math.isfinite(variance) or variance <= 0.0:
                continue
            weight = 1.0 / variance
            total_weight += weight
            weighted_sum += weight * price
        if total_weight <= 0.0:
            return None
        return weighted_sum / total_weight
    raise ValueError(f"unknown consensus mode: {mode!r}")


def basis_predicted_exchange_price(
    update: sqlite3.Row,
    *,
    update_id: int,
    exchange: str,
    anchor_exchange: str,
    basis_by_update: dict[int, dict[str, sqlite3.Row]],
) -> float | None:
    basis_common = update["basis_common_price"]
    if basis_common is None:
        return None
    if exchange == anchor_exchange:
        return float(basis_common)

    basis_row = basis_by_update.get(update_id, {}).get(exchange)
    if basis_row is None:
        return None
    return float(basis_common) + float(basis_row["basis_estimate"])


def basis_predicted_composite_price(
    update: sqlite3.Row,
    *,
    update_id: int,
    anchor_exchange: str,
    venues_by_update: dict[int, dict[str, sqlite3.Row]],
    basis_by_update: dict[int, dict[str, sqlite3.Row]],
    max_venue_age_ms: float,
    composite_max_weight: float,
) -> float | None:
    venue_rows = venues_by_update.get(update_id, {})
    if not venue_rows:
        return None

    raw_weights: list[float] = []
    predicted_prices: list[float] = []
    for exchange, venue_row in venue_rows.items():
        if venue_row["venue_kind"] != "spot":
            continue
        if not is_fresh_venue_row(venue_row, max_venue_age_ms=max_venue_age_ms):
            continue

        predicted_price = basis_predicted_exchange_price(
            update,
            update_id=update_id,
            exchange=exchange,
            anchor_exchange=anchor_exchange,
            basis_by_update=basis_by_update,
        )
        if predicted_price is None:
            continue

        variance = float(venue_row["variance"])
        if not math.isfinite(variance) or variance <= 0.0:
            continue

        raw_weights.append(1.0 / variance)
        predicted_prices.append(predicted_price)

    if not raw_weights:
        return None

    weight_sum = sum(raw_weights)
    if weight_sum <= 0.0:
        return None

    normalized_weights = [weight / weight_sum for weight in raw_weights]
    capped_weights = cap_and_renormalize(
        normalized_weights,
        max_weight=composite_max_weight,
    )
    return sum(weight * price for weight, price in zip(capped_weights, predicted_prices))


def main() -> None:
    args = build_parser().parse_args()
    conn = sqlite3.connect(Path(args.db_path))
    updates, venues_by_update, basis_by_update = load_rows(conn)
    conn.close()

    if not updates:
        raise SystemExit(f"no captured updates found in {args.db_path}")

    timestamps = [int(row["timestamp_ms"]) for row in updates]
    prediction_targets: list[tuple[str, str]] = [
        ("exchange", args.anchor_exchange),
        ("exchange", "kraken"),
        ("composite", "composite"),
        ("consensus_median", "consensus_median"),
        ("consensus_weighted", "consensus_weighted"),
    ]
    target_prices = compute_target_prices(
        updates,
        venues_by_update,
        prediction_targets=prediction_targets,
        max_venue_age_ms=args.max_venue_age_ms,
        min_consensus_venues=args.min_consensus_venues,
    )

    outside_counts = {"composite": 0, "kalman": 0, "basis": 0}
    outside_denoms = {"composite": 0, "kalman": 0, "basis": 0}

    basis_reconstruction_errors: list[float] = []
    horizon_errors: dict[tuple[str, str, int], dict[str, list[float]]] = {
        (target_kind, target_name, horizon): {
            "composite": [],
            "kalman": [],
            "basis_live": [],
            "basis_held": [],
            "composite_with_perp": [],
            "composite_without_perp": [],
            "kalman_with_perp": [],
            "kalman_without_perp": [],
            "basis_live_with_perp": [],
            "basis_live_without_perp": [],
            "basis_held_with_perp": [],
            "basis_held_without_perp": [],
        }
        for target_kind, target_name in prediction_targets
        for horizon in args.horizons_ms
    }
    # Paired per-row error tuples for rows where composite, kalman, and
    # basis_live all produced a prediction and the target is available.
    paired_errors: dict[
        tuple[str, str, int], list[tuple[float, float, float]]
    ] = defaultdict(list)
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
            and is_fresh_venue_row(row, max_venue_age_ms=args.max_venue_age_ms)
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

        basis_is_live = int(update["basis_is_live"]) == 1
        basis_common = update["basis_common_price"]
        if basis_common is not None and basis_is_live:
            basis_common_f = float(basis_common)
            basis_rows = basis_by_update.get(update_id, {})
            for exchange, venue_row in venue_rows.items():
                if not is_fresh_venue_row(venue_row, max_venue_age_ms=args.max_venue_age_ms):
                    continue
                basis_row = basis_rows.get(exchange)
                basis_estimate = float(basis_row["basis_estimate"]) if basis_row is not None else 0.0
                predicted = basis_common_f + basis_estimate
                basis_reconstruction_errors.append(abs(float(venue_row["fair_value"]) - predicted))

        anchor_row = venue_rows.get(args.anchor_exchange)
        anchor_price_now = (
            float(anchor_row["fair_value"])
            if is_fresh_venue_row(anchor_row, max_venue_age_ms=args.max_venue_age_ms)
            else None
        )
        has_perp = is_fresh_venue_row(
            venue_rows.get("hyperliquid_perp"),
            max_venue_age_ms=args.max_venue_age_ms,
        )
        basis_exchange_predictions = {
            target_name: basis_predicted_exchange_price(
                update,
                update_id=update_id,
                exchange=target_name,
                anchor_exchange=args.anchor_exchange,
                basis_by_update=basis_by_update,
            )
            for target_kind, target_name in prediction_targets
            if target_kind == "exchange"
        }
        basis_composite_prediction = basis_predicted_composite_price(
            update,
            update_id=update_id,
            anchor_exchange=args.anchor_exchange,
            venues_by_update=venues_by_update,
            basis_by_update=basis_by_update,
            max_venue_age_ms=args.max_venue_age_ms,
            composite_max_weight=args.composite_max_weight,
        )
        basis_consensus_median_prediction = basis_predicted_consensus_price(
            update,
            update_id=update_id,
            venues_by_update=venues_by_update,
            basis_by_update=basis_by_update,
            max_venue_age_ms=args.max_venue_age_ms,
            min_consensus_venues=args.min_consensus_venues,
            mode="consensus_median",
        )
        basis_consensus_weighted_prediction = basis_predicted_consensus_price(
            update,
            update_id=update_id,
            venues_by_update=venues_by_update,
            basis_by_update=basis_by_update,
            max_venue_age_ms=args.max_venue_age_ms,
            min_consensus_venues=args.min_consensus_venues,
            mode="consensus_weighted",
        )

        def basis_prediction_for(target_kind: str, target_name: str) -> float | None:
            if target_kind == "exchange":
                return basis_exchange_predictions.get(target_name)
            if target_kind == "composite":
                return basis_composite_prediction
            if target_kind == "consensus_median":
                return basis_consensus_median_prediction
            if target_kind == "consensus_weighted":
                return basis_consensus_weighted_prediction
            return None

        for horizon in args.horizons_ms:
            target_time = int(update["timestamp_ms"]) + horizon
            for target_kind, target_name in prediction_targets:
                target_map = target_prices[(target_kind, target_name)]
                future_index = find_future_index(
                    timestamps, updates, index, target_time, target_map
                )
                if future_index is None:
                    continue
                future_uid = int(updates[future_index]["id"])
                target_price = target_map[future_uid]

                bucket = horizon_errors[(target_kind, target_name, horizon)]
                suffix = "with_perp" if has_perp else "without_perp"
                composite_err: float | None = None
                kalman_err: float | None = None
                basis_live_err: float | None = None

                if update["composite_price"] is not None:
                    composite_err = abs(float(update["composite_price"]) - target_price)
                    bucket["composite"].append(composite_err)
                    bucket[f"composite_{suffix}"].append(composite_err)
                if update["kalman_filtered_price"] is not None:
                    kalman_err = abs(float(update["kalman_filtered_price"]) - target_price)
                    bucket["kalman"].append(kalman_err)
                    bucket[f"kalman_{suffix}"].append(kalman_err)

                basis_prediction = basis_prediction_for(target_kind, target_name)
                if basis_prediction is not None:
                    basis_error = abs(basis_prediction - target_price)
                    if basis_is_live:
                        bucket["basis_live"].append(basis_error)
                        bucket[f"basis_live_{suffix}"].append(basis_error)
                        basis_live_err = basis_error
                    else:
                        bucket["basis_held"].append(basis_error)
                        bucket[f"basis_held_{suffix}"].append(basis_error)

                if (
                    basis_live_err is not None
                    and kalman_err is not None
                    and composite_err is not None
                ):
                    paired_errors[(target_kind, target_name, horizon)].append(
                        (basis_live_err, kalman_err, composite_err)
                    )

            anchor_target_map = target_prices[("exchange", args.anchor_exchange)]
            future_index = find_future_index(
                timestamps, updates, index, target_time, anchor_target_map
            )
            if future_index is not None and anchor_price_now is not None:
                future_uid = int(updates[future_index]["id"])
                target_price = anchor_target_map[future_uid]
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
                if update["basis_common_price"] is not None and basis_is_live:
                    horizon_signals[horizon]["basis"].append(
                        float(update["basis_common_price"]) - anchor_price_now
                    )
                    horizon_signals[horizon]["basis_return"].append(future_return)

    live_rows = sum(1 for row in updates if int(row["basis_is_live"]) == 1)
    held_rows = len(updates) - live_rows
    print(f"rows: {len(updates)}  basis_live={live_rows}  basis_held={held_rows}")
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
            basis_label = (
                "basis_common"
                if target_kind == "exchange" and target_name == args.anchor_exchange
                else "basis_reconstructed"
            )
            print(format_metric(f"{basis_label} live", bucket["basis_live"]))
            print(format_metric(f"{basis_label} held", bucket["basis_held"]))
            print(format_metric(f"{basis_label} live with_perp", bucket["basis_live_with_perp"]))
            print(
                format_metric(
                    f"{basis_label} live without_perp",
                    bucket["basis_live_without_perp"],
                )
            )
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
    print("Paired error tests (rows where basis_live + kalman + composite all emitted):")
    for target_kind, target_name in prediction_targets:
        label = target_label(target_kind, target_name)
        print(f"Target: future {label}")
        for horizon in args.horizons_ms:
            rows = paired_errors[(target_kind, target_name, horizon)]
            basis_errs = [r[0] for r in rows]
            kalman_errs = [r[1] for r in rows]
            composite_errs = [r[2] for r in rows]
            bk = paired_diff_stats(basis_errs, kalman_errs)
            bc = paired_diff_stats(basis_errs, composite_errs)
            print(f"  horizon={horizon}ms  n={bk['n']}")
            print(
                f"    basis - kalman_1d   mean_Δ={bk['mean']:>+9.4f} "
                f"sem={bk['sem']:>7.4f} t={bk['t']:>+7.2f} "
                f"basis_better_frac={bk['a_better_frac']:>7.2%}"
            )
            print(
                f"    basis - composite   mean_Δ={bc['mean']:>+9.4f} "
                f"sem={bc['sem']:>7.4f} t={bc['t']:>+7.2f} "
                f"basis_better_frac={bc['a_better_frac']:>7.2%}"
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
