from __future__ import annotations

from decimal import Decimal

import pytest

from icarus.strategy.fair_value.filters.venue_basis_kalman_filter import (
    VenueBasisKalmanConfig,
    VenueBasisKalmanFilter,
    VenueBasisObservation,
)


def _obs(
    name: str,
    price: float | Decimal,
    variance: float | Decimal = 1.0,
    *,
    age_ms: float = 0.0,
    venue_kind: str = "spot",
) -> VenueBasisObservation:
    return VenueBasisObservation(
        name=name,
        fair_value=price,
        local_variance=variance,
        age_ms=age_ms,
        venue_kind=venue_kind,
    )


def test_two_venue_anchor_and_persistent_positive_offset() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
            common_price_process_var_per_sec=1e-4,
            default_basis_process_var_per_sec=1e-3,
            default_basis_rho_per_second=0.999,
            initial_common_price_variance=4.0,
            initial_basis_variance=100.0,
        )
    )

    result = None
    for i in range(200):
        result = filt.update(
            i * 0.1,
            [
                _obs("coinbase", 100.0, 1.0),
                _obs("kraken", 105.0, 1.0),
            ],
        )

    assert result is not None
    assert result.common_price == pytest.approx(100.0, abs=0.75)
    assert result.basis_estimates["coinbase"] == 0.0
    assert result.basis_estimates["kraken"] == pytest.approx(5.0, abs=0.75)


def test_three_venue_layout_is_stable_and_anchor_is_excluded() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken", "okx"),
        )
    )

    result = filt.update(
        0.0,
        [
            _obs("okx", 101.0, 2.0),
            _obs("coinbase", 100.0, 2.0),
            _obs("kraken", 100.5, 2.0),
        ],
    )

    assert result is not None
    assert result.basis_state_indices == {"kraken": 1, "okx": 2}
    assert "coinbase" not in result.basis_state_indices
    assert result.basis_estimates["coinbase"] == 0.0


def test_perp_basis_layout_is_stable_and_separate_from_spot_basis() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
            perp_exchange_order=("hyperliquid_perp",),
        )
    )

    result = filt.update(
        0.0,
        [
            _obs("kraken", 100.5, 2.0),
            _obs("coinbase", 100.0, 2.0),
            _obs("hyperliquid_perp", 101.5, 2.0, venue_kind="perp"),
        ],
    )

    assert result is not None
    assert result.basis_state_indices == {"kraken": 1, "hyperliquid_perp": 2}
    assert result.basis_state_kinds == {
        "kraken": "spot",
        "hyperliquid_perp": "perp",
    }


def test_partial_updates_handle_missing_venues_without_dropping_state() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken", "okx"),
            default_basis_rho_per_second=1.0,
            default_basis_process_var_per_sec=1e-6,
        )
    )

    first = filt.update(
        0.0,
        [
            _obs("coinbase", 100.0, 1.0),
            _obs("kraken", 103.0, 1.0),
            _obs("okx", 99.0, 1.0),
        ],
    )
    second = filt.update(
        1.0,
        [
            _obs("coinbase", 100.0, 1.0),
            _obs("okx", 99.0, 1.0),
        ],
    )

    assert first is not None
    assert second is not None
    assert second.active_venues == ["coinbase", "okx"]
    assert "kraken" not in second.observation_variances
    assert second.basis_estimates["kraken"] == pytest.approx(
        first.basis_estimates["kraken"],
        abs=0.25,
    )


def test_missing_anchor_skips_measurement_update_by_default() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
        )
    )

    first = filt.update(
        0.0,
        [
            _obs("coinbase", 100.0, 1.0),
            _obs("kraken", 98.0, 1.0),
        ],
    )
    second = filt.update(
        1.0,
        [
            _obs("kraken", 97.0, 1.0),
        ],
    )

    assert first is not None
    assert second is None


def test_converges_to_stable_basis_when_venue_is_persistently_shifted() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="A",
            venue_order=("A", "B"),
            common_price_process_var_per_sec=1e-4,
            default_basis_process_var_per_sec=1e-3,
            default_basis_rho_per_second=0.999,
        )
    )

    result = None
    for i in range(300):
        result = filt.update(
            i * 0.05,
            [
                _obs("A", 100.0, 1.0),
                _obs("B", 107.0, 1.0),
            ],
        )

    assert result is not None
    assert result.basis_estimates["B"] == pytest.approx(7.0, abs=0.75)
    assert result.basis_stddevs["B"] < 2.0


def test_common_price_tracks_shared_market_move_instead_of_lagging_below_all_venues() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
        )
    )

    result = None
    for i in range(80):
        result = filt.update(
            i * 0.1,
            [
                _obs("coinbase", 100.0, 1.0),
                _obs("kraken", 101.0, 1.0),
            ],
        )

    for i in range(80, 120):
        result = filt.update(
            i * 0.1,
            [
                _obs("coinbase", 110.0, 1.0),
                _obs("kraken", 111.0, 1.0),
            ],
        )

    assert result is not None
    assert result.common_price > 109.0
    assert result.common_price < 110.5
    assert result.basis_estimates["kraken"] == pytest.approx(1.0, abs=1.0)


def test_distinguishes_noisy_observations_from_persistent_basis() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="anchor",
            venue_order=("anchor", "noisy", "shifted"),
            common_price_process_var_per_sec=1e-4,
            default_basis_process_var_per_sec=1e-3,
            default_basis_rho_per_second=0.999,
        )
    )

    result = None
    for i in range(200):
        noisy_price = 105.0 if i % 2 == 0 else 95.0
        result = filt.update(
            i * 0.1,
            [
                _obs("anchor", 100.0, 1.0),
                _obs("noisy", noisy_price, 25.0),
                _obs("shifted", 104.0, 1.0),
            ],
        )

    assert result is not None
    assert abs(result.basis_estimates["noisy"]) < 1.5
    assert result.basis_estimates["shifted"] == pytest.approx(4.0, abs=0.75)


def test_persistent_perp_premium_is_learned_as_perp_basis() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
            perp_exchange_order=("hyperliquid_perp",),
            common_price_process_var_per_sec=1e-4,
            default_basis_process_var_per_sec=1e-4,
            default_basis_rho_per_second=0.999,
            default_perp_basis_process_var_per_sec=5e-4,
            default_perp_basis_rho_per_second=0.9995,
            initial_common_price_variance=4.0,
            initial_basis_variance=100.0,
        )
    )

    result = None
    for i in range(250):
        result = filt.update(
            i * 0.1,
            [
                _obs("coinbase", 100.0, 1.0),
                _obs("kraken", 100.5, 1.0),
                _obs("hyperliquid_perp", 103.0, 1.0, venue_kind="perp"),
            ],
        )

    assert result is not None
    assert result.common_price == pytest.approx(100.0, abs=1.0)
    assert result.basis_estimates["kraken"] == pytest.approx(0.5, abs=1.0)
    assert result.basis_estimates["hyperliquid_perp"] == pytest.approx(3.0, abs=1.0)


def test_decimal_inputs_are_deterministic() -> None:
    cfg = VenueBasisKalmanConfig(
        anchor_exchange="coinbase",
        venue_order=("coinbase", "kraken"),
    )
    filt_a = VenueBasisKalmanFilter(config=cfg)
    filt_b = VenueBasisKalmanFilter(config=cfg)

    sequence = [
        [
            _obs("coinbase", Decimal("100.0"), Decimal("1.0")),
            _obs("kraken", Decimal("102.5"), Decimal("2.0")),
        ],
        [
            _obs("coinbase", Decimal("100.1"), Decimal("1.0")),
            _obs("kraken", Decimal("102.6"), Decimal("2.0")),
        ],
    ]

    result_a = None
    result_b = None
    for i, observations in enumerate(sequence):
        result_a = filt_a.update(float(i), observations)
        result_b = filt_b.update(float(i), observations)

    assert result_a is not None
    assert result_b is not None
    assert result_a.common_price == result_b.common_price
    assert result_a.basis_estimates == result_b.basis_estimates
    assert result_a.innovations == result_b.innovations


def test_anchor_identifiability_prevents_free_basis_estimation() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
        )
    )

    result = filt.update(
        0.0,
        [
            _obs("coinbase", 100.0, 1.0),
            _obs("kraken", 101.0, 1.0),
        ],
    )

    assert result is not None
    assert result.anchor_exchange == "coinbase"
    assert "coinbase" not in result.basis_state_indices
    assert result.basis_estimates["coinbase"] == 0.0
    assert set(result.basis_estimates) == {"coinbase", "kraken"}


def test_motivating_tight_quote_scenario_is_absorbed_as_basis() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
            common_price_process_var_per_sec=1e-4,
            default_basis_process_var_per_sec=5e-4,
            default_basis_rho_per_second=0.9995,
            initial_common_price_variance=4.0,
            initial_basis_variance=400.0,
        )
    )

    result = None
    for i in range(300):
        result = filt.update(
            i * 0.1,
            [
                _obs("coinbase", 76912.0, 4.0),
                _obs("kraken", 76897.0, 0.05),
            ],
        )

    assert result is not None
    assert result.common_price > 76905.0
    assert result.common_price < 76913.0
    assert result.basis_estimates["kraken"] < -10.0
    assert result.basis_estimates["kraken"] == pytest.approx(-15.0, abs=2.0)


def test_observation_variance_is_not_age_penalized_twice() -> None:
    filt = VenueBasisKalmanFilter(
        config=VenueBasisKalmanConfig(
            anchor_exchange="coinbase",
            venue_order=("coinbase", "kraken"),
        )
    )

    result = filt.update(
        0.0,
        [
            _obs("coinbase", 100.0, 9.0),
            _obs("kraken", 101.0, 16.0, age_ms=700.0),
        ],
    )

    assert result is not None
    assert result.observation_variances["coinbase"] == pytest.approx(9.0, abs=1e-9)
    assert result.observation_variances["kraken"] == pytest.approx(16.0, abs=1e-9)
