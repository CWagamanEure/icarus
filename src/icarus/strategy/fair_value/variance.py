from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .types import FairValueFeatures


@dataclass(frozen=True, slots=True)
class FairValueVarianceConfig:
    """
    Configuration for the microstructure measurement-variance heuristic.

    This variance is meant to express *quote precision* — how precisely the
    venue's top-of-book pins down its own local fair value. It intentionally
    does NOT try to express closeness to the latent common cross-venue fair
    value; that is handled downstream by the adaptive venue-reliability layer
    in the Kalman filter.
    """

    # Floor on combined (spread, short-horizon vol) noise in bps. A venue that
    # posts an ultra-tight inside quote on a thin book can produce a near-zero
    # raw noise estimate, which under inverse-variance weighting would let it
    # monopolize the cross-venue composite even when it is structurally offset
    # from the common fair value. This floor prevents the precision metric
    # from understating real short-horizon uncertainty.
    min_noise_bps: Decimal = Decimal("1.5")

    # Reference top-of-book notional (in quote currency). Venues with top-of-
    # book notional below this get quadratic variance inflation. Notional
    # (depth * midprice) is used rather than raw base quantity so the penalty
    # is scale-invariant across assets and across venues that report depth
    # differently.
    reference_top_notional: Decimal = Decimal("250000")

    # Lower bound on returned variance to avoid degenerate zero-variance
    # outputs feeding inverse-variance weighting.
    min_variance: Decimal = Decimal("1e-10")


DEFAULT_VARIANCE_CONFIG = FairValueVarianceConfig()

# Kept for backward compatibility / external inspection. Mutating these has
# no effect on compute_measurement_variance; pass a config instead.
MIN_NOISE_BPS = DEFAULT_VARIANCE_CONFIG.min_noise_bps
REFERENCE_TOP_NOTIONAL = DEFAULT_VARIANCE_CONFIG.reference_top_notional
MIN_VAR = DEFAULT_VARIANCE_CONFIG.min_variance


def compute_measurement_variance(
    features: FairValueFeatures,
    *,
    config: FairValueVarianceConfig = DEFAULT_VARIANCE_CONFIG,
) -> Decimal | None:
    midprice = features.midprice
    if midprice is None or midprice <= 0:
        return None

    spread_bps = features.spread_bps or Decimal("0")
    quote_age_ms = Decimal(features.quote_age_ms or 0)
    mid_vol_bps = features.mid_volatility_bps or Decimal("0")
    micro_vol_bps = features.micro_volatility_bps or Decimal("0")

    top_depth = (features.top_bid_depth or Decimal("0")) + (features.top_ask_depth or Decimal("0"))

    # Base noise from spread and short-horizon movement.
    spread_component = spread_bps / Decimal("10000")
    vol_component = max(mid_vol_bps, micro_vol_bps) / Decimal("10000")
    min_component = config.min_noise_bps / Decimal("10000")

    noise_squared = spread_component**2 + vol_component**2
    min_noise_squared = min_component**2
    if noise_squared < min_noise_squared:
        noise_squared = min_noise_squared

    # Thin-book penalty based on top-of-book notional.
    top_notional = top_depth * midprice
    if top_notional <= 0:
        depth_factor = Decimal("10000")
    elif top_notional >= config.reference_top_notional:
        depth_factor = Decimal("1.0")
    else:
        ratio = config.reference_top_notional / top_notional
        depth_factor = ratio * ratio

    # Stale quote penalty.
    if quote_age_ms <= 100:
        age_factor = Decimal("1.0")
    elif quote_age_ms <= 250:
        age_factor = Decimal("1.2")
    elif quote_age_ms <= 500:
        age_factor = Decimal("1.5")
    else:
        age_factor = Decimal("2.0")

    variance = noise_squared * depth_factor * age_factor * midprice**2

    if variance < config.min_variance:
        return config.min_variance
    return variance
