from __future__ import annotations

import abc
from decimal import Decimal


class BaseFairValueFilter(abc.ABC):
    @abc.abstractmethod
    def update(
        self,
        *,
        measurement: Decimal,
        measurement_variance: Decimal,
        timestamp_ms: int,
    ) -> tuple[Decimal, Decimal]:
        raise NotImplementedError
