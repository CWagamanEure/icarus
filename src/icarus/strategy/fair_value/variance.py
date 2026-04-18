from __future__ import annotations

from decimal import Decimal

from .types import FairValueFeatures


MIN_VAR = Decimal("1e-10")

# Floor on combined (spread, short-horizon vol) noise, in bps. A pathologically tight
# top-of-book — frozen book, rounding-level quotes, venues that publish an indicative
# inside — should not translate into near-zero variance. Without this floor, a single
# venue reporting ~0.04 bps of noise can drive its inverse-variance weight above 95%
# and pin the cross-venue composite to it even when its price is structurally wrong.
MIN_NOISE_BPS = Decimal("0.25")


def compute_measurement_variance(features: FairValueFeatures) -> Decimal | None:
    midprice = features.midprice
    if midprice is None or midprice <= 0:
        return None

    spread_bps = features.spread_bps or Decimal("0")
    quote_age_ms = Decimal(features.quote_age_ms or 0)
    mid_vol_bps = features.mid_volatility_bps or Decimal("0")
    micro_vol_bps = features.micro_volatility_bps or Decimal("0")
    imbalance = abs(features.depth_imbalance) if features.depth_imbalance is not None else Decimal("0")

    top_depth = (features.top_bid_depth or Decimal("0")) + (features.top_ask_depth or Decimal("0"))

    # base noise from spread and short-horizon movement
    spread_component = spread_bps / Decimal("10000")
    vol_component = max(mid_vol_bps, micro_vol_bps) / Decimal("10000")
    min_component = MIN_NOISE_BPS / Decimal("10000")

    # Enforce a minimum combined-noise magnitude so no venue can underreport itself
    # into domination of the cross-venue fuse.
    noise_squared = spread_component**2 + vol_component**2
    min_noise_squared = min_component**2
    if noise_squared < min_noise_squared:
        noise_squared = min_noise_squared

    # thin book penalty
    if top_depth <= 0:
        depth_factor = Decimal("2.0")
    elif top_depth < Decimal("1"):
        depth_factor = Decimal("1.75")
    elif top_depth < Decimal("5"):
        depth_factor = Decimal("1.3")
    else:
        depth_factor = Decimal("1.0")

    # stale quote penalty
    if quote_age_ms <= 100:
        age_factor = Decimal("1.0")
    elif quote_age_ms <= 250:
        age_factor = Decimal("1.2")
    elif quote_age_ms <= 500:
        age_factor = Decimal("1.5")
    else:
        age_factor = Decimal("2.0")

    # slight confidence boost if imbalance is strong and microprice is informative
    imbalance_factor = Decimal("1.0") - Decimal("0.2") * imbalance
    if imbalance_factor < Decimal("0.8"):
        imbalance_factor = Decimal("0.8")

    variance = noise_squared * depth_factor * age_factor * imbalance_factor * midprice**2

    if variance < MIN_VAR:
        return MIN_VAR
    return variance
