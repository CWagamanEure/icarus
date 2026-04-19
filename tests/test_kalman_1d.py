from __future__ import annotations

import math

import pytest

from icarus.strategy.fair_value.filters.kalman_1d import (
    AdaptiveEfficientPriceKalman,
    KalmanFilterConfig,
    VenueObservation,
)


def _obs(name: str, price: float, variance: float = 100.0) -> VenueObservation:
    return VenueObservation(name=name, fair_value=price, local_variance=variance)


def _run(
    kalman: AdaptiveEfficientPriceKalman,
    steps: int,
    observations_fn,
    dt: float = 0.1,
):
    result = None
    for i in range(steps):
        result = kalman.update(i * dt, observations_fn(i))
    return result


def test_basic_bootstrap_and_convergence_to_single_venue() -> None:
    k = AdaptiveEfficientPriceKalman(config=KalmanFilterConfig())
    result = k.update(0.0, [_obs("A", 100.0, 1.0)])
    assert result is not None
    assert result.filtered_price == 100.0
    # On bootstrap there is no prior so the innovation diagnostic is zero.
    d = result.venue_diagnostics[0]
    assert d.innovation == 0.0
    assert d.standardized_z2 == 0.0


def test_max_weight_cap_prevents_domination_under_two_venues() -> None:
    # One venue has huge precision, the other tiny. Without a cap this
    # would be ~99.9% vs 0.1%. With cap 0.75 it should be clipped.
    k = AdaptiveEfficientPriceKalman(
        config=KalmanFilterConfig(
            max_venue_weight=0.75,
            reliability_enabled=False,
        ),
    )
    result = k.update(
        0.0,
        [
            VenueObservation(name="tight", fair_value=100.0, local_variance=0.01),
            VenueObservation(name="wide", fair_value=102.0, local_variance=10000.0),
        ],
    )
    assert result is not None
    assert result.weights["tight"] == pytest.approx(0.75, abs=1e-9)
    assert result.weights["wide"] == pytest.approx(0.25, abs=1e-9)
    # Composite is 0.75*100 + 0.25*102 = 100.5 — meaningfully influenced
    # by the wide venue, not pinned to the tight one.
    assert result.raw_fused_price == pytest.approx(100.5, abs=1e-9)


def test_reliability_uses_prior_not_post_update_fused() -> None:
    """Venue A consistently offsets from the latent state; venue B matches the
    latent state. A's reliability score should grow relative to B's even
    though, in a 2-venue case with no cap, the post-update composite sits
    midway between them. This is only possible if the reliability update
    uses x_prior (the pre-update predicted state) and NOT the post-update
    fused price as the innovation reference.
    """
    k = AdaptiveEfficientPriceKalman(
        config=KalmanFilterConfig(
            max_venue_weight=1.0,  # disable cap to isolate the mechanism
            reliability_ewma_lambda=0.8,  # fast adaptation for a short test
            q_base_per_sec=1e-6,
            q_vol_scale=0.0,
        ),
    )

    # Bootstrap both venues near the true price.
    k.update(0.0, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])
    # Then A starts reporting +5 while B stays at 100. B is the anchor.
    for i in range(1, 30):
        k.update(i * 0.1, [_obs("A", 105.0, 1.0), _obs("B", 100.0, 1.0)])

    q_a = k._venue_reliability["A"]
    q_b = k._venue_reliability["B"]
    assert q_a > q_b
    # A should inflate its effective variance; B should be near 1.0.
    assert q_a > 1.0
    assert q_b < q_a


def test_reliability_decays_when_venue_behaves() -> None:
    cfg = KalmanFilterConfig(
        max_venue_weight=1.0,
        reliability_ewma_lambda=0.8,
        q_base_per_sec=1e-6,
        q_vol_scale=0.0,
    )
    k = AdaptiveEfficientPriceKalman(config=cfg)

    k.update(0.0, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])
    # Burn in disagreement.
    for i in range(1, 30):
        k.update(i * 0.1, [_obs("A", 105.0, 1.0), _obs("B", 100.0, 1.0)])

    q_bad = k._venue_reliability["A"]
    assert q_bad > 1.5

    # Now A behaves for a while.
    for i in range(30, 80):
        k.update(i * 0.1, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])

    q_good = k._venue_reliability["A"]
    assert q_good < q_bad
    # Converges back toward 1 (not exactly — finite horizon EWMA).
    assert q_good < 1.5


def test_reliability_multiplier_is_clipped_at_max() -> None:
    cfg = KalmanFilterConfig(
        max_venue_weight=1.0,
        reliability_ewma_lambda=0.5,  # adapt very fast
        reliability_alpha=100.0,       # blow up the multiplier
        reliability_multiplier_max=5.0,
        reliability_z2_clip=1000.0,
        q_base_per_sec=1e-6,
        q_vol_scale=0.0,
    )
    k = AdaptiveEfficientPriceKalman(config=cfg)
    k.update(0.0, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])
    for i in range(1, 50):
        k.update(i * 0.1, [_obs("A", 200.0, 1.0), _obs("B", 100.0, 1.0)])

    result = k.update(5.0, [_obs("A", 200.0, 1.0), _obs("B", 100.0, 1.0)])
    assert result is not None
    a_diag = next(d for d in result.venue_diagnostics if d.name == "A")
    assert a_diag.reliability_multiplier <= 5.0 + 1e-9


def test_reliability_disabled_keeps_multiplier_at_one() -> None:
    cfg = KalmanFilterConfig(
        max_venue_weight=1.0,
        reliability_enabled=False,
    )
    k = AdaptiveEfficientPriceKalman(config=cfg)
    k.update(0.0, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])
    for i in range(1, 20):
        k.update(i * 0.1, [_obs("A", 200.0, 1.0), _obs("B", 100.0, 1.0)])

    result = k.update(2.0, [_obs("A", 200.0, 1.0), _obs("B", 100.0, 1.0)])
    for d in result.venue_diagnostics:
        assert d.reliability_multiplier == 1.0


def test_stale_venue_is_dropped_from_fuse() -> None:
    cfg = KalmanFilterConfig(stale_cutoff_ms=500.0, max_venue_weight=1.0)
    k = AdaptiveEfficientPriceKalman(config=cfg)
    result = k.update(
        0.0,
        [
            VenueObservation(name="fresh", fair_value=100.0, local_variance=1.0, age_ms=10),
            VenueObservation(name="stale", fair_value=200.0, local_variance=1.0, age_ms=1000),
        ],
    )
    assert result is not None
    assert "fresh" in result.weights
    assert "stale" not in result.weights


def test_missing_all_venues_returns_none() -> None:
    k = AdaptiveEfficientPriceKalman()
    assert k.update(0.0, []) is None


def test_venue_disappear_and_reappear_preserves_reliability() -> None:
    cfg = KalmanFilterConfig(
        max_venue_weight=1.0,
        reliability_ewma_lambda=0.8,
        q_base_per_sec=1e-6,
        q_vol_scale=0.0,
    )
    k = AdaptiveEfficientPriceKalman(config=cfg)
    k.update(0.0, [_obs("A", 100.0, 1.0), _obs("B", 100.0, 1.0)])
    for i in range(1, 20):
        k.update(i * 0.1, [_obs("A", 110.0, 1.0), _obs("B", 100.0, 1.0)])

    q_before = k._venue_reliability["A"]
    assert q_before > 1.0

    # A goes away for a while.
    for i in range(20, 40):
        k.update(i * 0.1, [_obs("B", 100.0, 1.0)])

    # A's state is untouched while absent.
    assert k._venue_reliability["A"] == q_before


def test_scenario_motivating_fix() -> None:
    """Motivating scenario: one venue has a tight inside on a thin book
    and tiny base variance; the other has a wider, deeper quote. They
    trade with a persistent basis. Without the fixes the tight venue
    dominates the composite. With the fixes (floor + cap + reliability),
    both venues keep meaningful influence on the fused result.
    """
    cfg = KalmanFilterConfig()
    k = AdaptiveEfficientPriceKalman(config=cfg)

    # Tight = very low local variance (simulates floor-hitting quote);
    # Wide = much higher variance. Tight is biased +2 relative to wide.
    result = None
    for i in range(200):
        result = k.update(
            i * 0.1,
            [
                VenueObservation(name="tight", fair_value=102.0, local_variance=0.5),
                VenueObservation(name="wide", fair_value=100.0, local_variance=50.0),
            ],
        )
    assert result is not None

    # Without the cap, tight's weight would be ~100/0.5 vs 100/50 = 100:1.
    # With cap 0.75 and adaptive reliability, tight should be <= 0.75 and
    # the composite should be strictly between 100 and 102.
    w_tight = result.weights["tight"]
    w_wide = result.weights["wide"]
    assert w_tight <= 0.75 + 1e-9
    assert w_wide >= 0.25 - 1e-9
    assert 100.0 < result.filtered_price < 102.0
    # Filtered price should not be pinned to tight. Expect meaningfully
    # below the pure cap result (which would give 0.75*102 + 0.25*100 = 101.5).
    # With reliability pulling tight down further, we expect closer to 100.
    assert result.filtered_price < 101.5 + 1e-9


def test_filtered_price_is_finite_under_extreme_observations() -> None:
    cfg = KalmanFilterConfig()
    k = AdaptiveEfficientPriceKalman(config=cfg)
    result = k.update(
        0.0,
        [
            VenueObservation(name="A", fair_value=100.0, local_variance=1.0),
            VenueObservation(name="B", fair_value=1e9, local_variance=1e-8),
        ],
    )
    assert result is not None
    assert math.isfinite(result.filtered_price)


def test_process_variance_tracks_current_fused_move() -> None:
    cfg = KalmanFilterConfig(
        q_base_per_sec=0.0,
        q_vol_scale=1.0,
        obs_var_ewma_alpha=1.0,
        reliability_enabled=False,
        max_venue_weight=1.0,
    )
    k = AdaptiveEfficientPriceKalman(config=cfg)

    first = k.update(0.0, [_obs("A", 100.0, 1.0)])
    second = k.update(1.0, [_obs("A", 110.0, 1.0)])

    assert first is not None
    assert second is not None
    assert second.raw_fused_price == pytest.approx(110.0, abs=1e-9)
    assert second.process_variance_q == pytest.approx(100.0, abs=1e-9)
