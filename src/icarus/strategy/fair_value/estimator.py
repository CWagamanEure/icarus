from __future__ import annotations

from icarus.measurements.types import MarketMeasurement

from .features import features_from_measurement
from .raw import compute_raw_fair_value
from .types import RawFairValueEstimate
from .variance import compute_measurement_variance


class RawFairValueEstimator:
    def estimate(self, measurement: MarketMeasurement) -> RawFairValueEstimate:
        features = features_from_measurement(measurement)
        raw_fair_value, micro_alpha, note = compute_raw_fair_value(features)
        measurement_variance = compute_measurement_variance(features)

        return RawFairValueEstimate(
            timestamp_ms=measurement.timestamp_ms,
            exchange=measurement.exchange,
            market=measurement.market,
            raw_fair_value=raw_fair_value,
            measurement_variance=measurement_variance,
            micro_alpha=micro_alpha,
            used_midprice=features.midprice,
            used_microprice=features.microprice,
        )
