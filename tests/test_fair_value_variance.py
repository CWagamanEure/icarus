from __future__ import annotations

from decimal import Decimal

import pytest

from icarus.strategy.fair_value.types import FairValueFeatures
from icarus.strategy.fair_value.variance import (
    DEFAULT_VARIANCE_CONFIG,
    FairValueVarianceConfig,
    compute_measurement_variance,
)


def _features(
    *,
    spread_bps: Decimal = Decimal("1"),
    top_depth_btc: Decimal = Decimal("1"),
    midprice: Decimal = Decimal("60000"),
    quote_age_ms: int = 50,
    mid_vol_bps: Decimal = Decimal("1"),
    micro_vol_bps: Decimal = Decimal("1"),
) -> FairValueFeatures:
    half = top_depth_btc / Decimal("2")
    return FairValueFeatures(
        midprice=midprice,
        microprice=None,
        spread_bps=spread_bps,
        depth_imbalance=Decimal("0"),
        quote_age_ms=quote_age_ms,
        mid_volatility_bps=mid_vol_bps,
        micro_volatility_bps=micro_vol_bps,
        top_bid_depth=half,
        top_ask_depth=half,
    )


def test_noise_floor_default_is_1_5_bps() -> None:
    assert DEFAULT_VARIANCE_CONFIG.min_noise_bps == Decimal("1.5")


def test_floor_applies_when_spread_is_below_floor() -> None:
    # Ultra tight inside, deep top book — without the floor this would
    # produce near-zero variance and monopolize inverse-variance fusion.
    features = _features(
        spread_bps=Decimal("0.05"),
        top_depth_btc=Decimal("10"),
        mid_vol_bps=Decimal("0"),
        micro_vol_bps=Decimal("0"),
    )
    variance = compute_measurement_variance(features)

    # Compute what the floor alone would imply:
    # noise = 1.5 bps => variance = (1.5/10000)^2 * 60000^2 = 81
    assert variance is not None
    assert variance == Decimal("81.0000000000")


def test_floor_respects_config_override() -> None:
    config = FairValueVarianceConfig(min_noise_bps=Decimal("0.25"))
    features = _features(
        spread_bps=Decimal("0.05"),
        top_depth_btc=Decimal("10"),
        mid_vol_bps=Decimal("0"),
        micro_vol_bps=Decimal("0"),
    )
    variance = compute_measurement_variance(features, config=config)

    # noise = 0.25 bps => variance = (0.25/10000)^2 * 60000^2 = 2.25
    assert variance is not None
    assert variance == Decimal("2.2500000000")


def test_wide_spread_passes_through_without_floor() -> None:
    features = _features(
        spread_bps=Decimal("5"),
        top_depth_btc=Decimal("10"),
        mid_vol_bps=Decimal("0"),
        micro_vol_bps=Decimal("0"),
    )
    variance = compute_measurement_variance(features)

    # noise = 5 bps => variance = (5/10000)^2 * 60000^2 = 900
    assert variance is not None
    assert variance == Decimal("900.0000000000")


def test_depth_penalty_bites_below_reference_notional() -> None:
    deep = _features(spread_bps=Decimal("2"), top_depth_btc=Decimal("10"))
    thin = _features(spread_bps=Decimal("2"), top_depth_btc=Decimal("0.1"))

    v_deep = compute_measurement_variance(deep)
    v_thin = compute_measurement_variance(thin)

    assert v_deep is not None and v_thin is not None
    # Thin book should be much higher variance.
    assert v_thin > v_deep * Decimal("5")


def test_returns_none_on_missing_midprice() -> None:
    features = FairValueFeatures(
        midprice=None,
        microprice=None,
        spread_bps=Decimal("1"),
        depth_imbalance=None,
        quote_age_ms=0,
        mid_volatility_bps=None,
        micro_volatility_bps=None,
        top_bid_depth=None,
        top_ask_depth=None,
    )
    assert compute_measurement_variance(features) is None


def test_variance_is_decimal_type() -> None:
    v = compute_measurement_variance(_features())
    assert isinstance(v, Decimal)


@pytest.mark.parametrize("age_ms,expected_factor", [
    (50, Decimal("1.0")),
    (200, Decimal("1.2")),
    (400, Decimal("1.5")),
    (1000, Decimal("2.0")),
])
def test_age_factor_tiers(age_ms: int, expected_factor: Decimal) -> None:
    f_fresh = _features(quote_age_ms=0)
    f_aged = _features(quote_age_ms=age_ms)

    v_fresh = compute_measurement_variance(f_fresh)
    v_aged = compute_measurement_variance(f_aged)
    assert v_fresh is not None and v_aged is not None
    assert v_aged == v_fresh * expected_factor
