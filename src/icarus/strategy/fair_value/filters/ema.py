from __future__ import annotations

from decimal import Decimal

from .base import BaseFairValueFilter


class EMAFairValueFilter(BaseFairValueFilter):
    def __init__(self, alpha: Decimal = Decimal("0.2")) -> None:
        if not (Decimal("0") < alpha <= Decimal("1")):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self._value: Decimal | None = None

    def update(
        self,
        *,
        measurement: Decimal,
        measurement_variance: Decimal,
        timestamp_ms: int,
    ) -> tuple[Decimal, Decimal]:
        if self._value is None:
            self._value = measurement
        else:
            self._value = self.alpha * measurement + (Decimal("1") - self.alpha) * self._value
        return self._value, measurement_variance
