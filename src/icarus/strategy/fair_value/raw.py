from __future__ import annotations

from decimal import Decimal

from .types import FairValueFeatures


ZERO = Decimal("0")
ONE = Decimal("1")


def compute_micro_alpha(features: FairValueFeatures) -> Decimal:
    if features.midprice is None or features.microprice is None:
        return ZERO

    if features.spread_bps is None or features.spread_bps <= 0:
        return ZERO

    if features.depth_imbalance is None:
        return ZERO

    spread_bps = features.spread_bps
    imbalance = abs(features.depth_imbalance)
    quote_age_ms = features.quote_age_ms or 0

    total_depth = (features.top_bid_depth or ZERO) + (features.top_ask_depth or ZERO)

    # spread gate: trust micro more when spread is tight
    if spread_bps <= Decimal("1"):
        spread_factor = Decimal("1")
    elif spread_bps <= Decimal("2"):
        spread_factor = Decimal("0.6")
    elif spread_bps <= Decimal("5"):
        spread_factor = Decimal("0.25")
    else:
        spread_factor = Decimal("0")

    # age penalty
    if quote_age_ms <= 100:
        age_factor = Decimal("1")
    elif quote_age_ms <= 250:
        age_factor = Decimal("0.7")
    elif quote_age_ms <= 500:
        age_factor = Decimal("0.4")
    else:
        age_factor = Decimal("0")

    # depth bonus, capped
    if total_depth <= 0:
        depth_factor = Decimal("0.5")
    elif total_depth < Decimal("1"):
        depth_factor = Decimal("0.6")
    elif total_depth < Decimal("5"):
        depth_factor = Decimal("0.8")
    else:
        depth_factor = Decimal("1")

    alpha = Decimal("0.5") * imbalance * spread_factor * age_factor * depth_factor

    if alpha < ZERO:
        return ZERO
    if alpha > Decimal("0.5"):
        return Decimal("0.5")
    return alpha


def compute_raw_fair_value(features: FairValueFeatures) -> tuple[Decimal | None, Decimal | None, str]:
    if features.midprice is None and features.microprice is None:
        return None, None, "no_price"

    if features.midprice is None:
        return features.microprice, None, "micro_only"

    if features.microprice is None:
        return features.midprice, None, "mid_only"

    alpha = compute_micro_alpha(features)
    raw_fair = features.midprice + alpha * (features.microprice - features.midprice)
    return raw_fair, alpha, "mid_plus_micro_adjustment"
