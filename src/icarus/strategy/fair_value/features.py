from __future__ import annotations

from icarus.measurements.types import MarketMeasurement

from .types import FairValueFeatures


def features_from_measurement(measurement: MarketMeasurement) -> FairValueFeatures:
    return FairValueFeatures(
        midprice=measurement.midprice,
        microprice=measurement.microprice,
        spread_bps=measurement.spread_bps,
        depth_imbalance=measurement.depth_imbalance,
        quote_age_ms=measurement.quote_age_ms,
        mid_volatility_bps=measurement.mid_volatility_bps,
        micro_volatility_bps=measurement.micro_volatility_bps,
        top_bid_depth=measurement.top_bid_depth,
        top_ask_depth=measurement.top_ask_depth,
    )
