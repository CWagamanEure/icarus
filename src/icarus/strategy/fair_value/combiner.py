from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .types import CombinedFairValueEstimate, VenueFairValueState


@dataclass(frozen=True, slots=True)
class CrossVenueCombinerConfig:
    stale_after_ms: int = 500
    age_penalty_per_second: Decimal = Decimal("1")


class CrossVenueFairValueCombiner:
    def __init__(self, market: str, config: CrossVenueCombinerConfig | None = None) -> None:
        self.market = market
        self.config = config or CrossVenueCombinerConfig()
        self._states: dict[str, VenueFairValueState] = {}

    def update(
        self,
        state: VenueFairValueState,
        *,
        now_ms: int | None = None,
    ) -> CombinedFairValueEstimate | None:
        if state.market != self.market:
            raise ValueError(
                f"State market {state.market} does not match combiner market {self.market}."
            )

        self._states[state.exchange] = state
        return self.combine(now_ms=now_ms if now_ms is not None else state.timestamp_ms)

    def combine(self, *, now_ms: int) -> CombinedFairValueEstimate | None:
        weighted_value_sum = Decimal("0")
        weight_sum = Decimal("0")
        contributing_exchanges: list[str] = []
        latest_timestamp_ms = 0

        for state in self._states.values():
            age_ms = now_ms - state.timestamp_ms
            if age_ms < 0:
                age_ms = 0
            if age_ms > self.config.stale_after_ms:
                continue

            effective_variance = self._effective_variance(state.variance, age_ms)
            if effective_variance <= 0:
                continue

            weight = Decimal("1") / effective_variance
            weighted_value_sum += state.fair_value * weight
            weight_sum += weight
            contributing_exchanges.append(state.exchange)
            latest_timestamp_ms = max(latest_timestamp_ms, state.timestamp_ms)

        if weight_sum <= 0 or not contributing_exchanges:
            return None

        return CombinedFairValueEstimate(
            market=self.market,
            timestamp_ms=latest_timestamp_ms,
            fair_value=weighted_value_sum / weight_sum,
            variance=Decimal("1") / weight_sum,
            contributing_exchanges=tuple(sorted(contributing_exchanges)),
        )

    def _effective_variance(self, variance: Decimal, age_ms: int) -> Decimal:
        if variance <= 0:
            return Decimal("0")

        age_seconds = Decimal(age_ms) / Decimal("1000")
        return variance * (Decimal("1") + (self.config.age_penalty_per_second * age_seconds))
