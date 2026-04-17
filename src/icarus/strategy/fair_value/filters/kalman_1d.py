from __future__ import annotations

from decimal import Decimal


class Kalman1DFairValueFilter:
    def __init__(
        self,
        *,
        process_variance_per_second: float = 1e-6,
        initial_variance: float = 1e-4,
    ) -> None:
        self.process_variance_per_second = process_variance_per_second
        self.x: float | None = None
        self.p: float = initial_variance
        self.last_timestamp_ms: int | None = None

    def update(
        self,
        *,
        measurement: Decimal,
        measurement_variance: Decimal,
        timestamp_ms: int,
    ) -> tuple[Decimal, Decimal]:
        z = float(measurement)
        r = float(measurement_variance)

        if self.x is None:
            self.x = z
            self.p = max(r, 1e-12)
            self.last_timestamp_ms = timestamp_ms
            return Decimal(str(self.x)), Decimal(str(self.p))

        dt_ms = 0 if self.last_timestamp_ms is None else max(timestamp_ms - self.last_timestamp_ms, 0)
        dt_sec = max(dt_ms / 1000.0, 1e-6)

        # predict
        q = self.process_variance_per_second * dt_sec
        x_pred = self.x
        p_pred = self.p + q

        # update
        k = p_pred / (p_pred + r)
        self.x = x_pred + k * (z - x_pred)
        self.p = (1.0 - k) * p_pred
        self.last_timestamp_ms = timestamp_ms

        return Decimal(str(self.x)), Decimal(str(self.p))
