from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .types import CombinedFairValueEstimate, VenueFairValueState


@dataclass(frozen=True, slots=True)
class CrossVenueCombinerConfig:
    stale_after_ms: int = 2000
    age_penalty_per_second: Decimal = Decimal("1")
    disagreement_scale: Decimal = Decimal("1")
    variance_floor: Decimal = Decimal("0.01")


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
        live_states: list[tuple[VenueFairValueState, Decimal]] = []

        for state in self._states.values():
            age_ms = now_ms - state.timestamp_ms
            if age_ms < 0:
                age_ms = 0
            if age_ms > self.config.stale_after_ms:
                continue

            effective_variance = self._effective_variance(state.variance, age_ms)
            if effective_variance <= 0:
                continue

            live_states.append((state, effective_variance))

        if not live_states:
            return None

        raw_weights: list[Decimal] = []
        fair_values: list[Decimal] = []
        variances: list[Decimal] = []
        exchanges: list[str] = []

        for state, effective_variance in live_states:
            weight = Decimal("1") / effective_variance
            raw_weights.append(weight)
            fair_values.append(state.fair_value)
            variances.append(effective_variance)
            exchanges.append(state.exchange)

        if not raw_weights:
            return None

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            return None

        normalized_weights = [w / weight_sum for w in raw_weights]

        combined_fv = sum(
            w * fv for w, fv in zip(normalized_weights, fair_values)
        )

        intrinsic_variance = sum(
            (w * w) * v for w, v in zip(normalized_weights, variances)
        )

        disagreement_variance = sum(
            w * ((fv - combined_fv) ** 2)
            for w, fv in zip(normalized_weights, fair_values)
        )

        combined_variance = intrinsic_variance + (
            self.config.disagreement_scale * disagreement_variance
        )

        if combined_variance < self.config.variance_floor:
            combined_variance = self.config.variance_floor

        return CombinedFairValueEstimate(
            market=self.market,
            timestamp_ms=now_ms,
            fair_value=combined_fv,
            variance=combined_variance,
            contributing_exchanges=tuple(sorted(exchanges)),
        )

    def _effective_variance(self, variance: Decimal, age_ms: int) -> Decimal:
        if variance <= 0:
            return Decimal("0")

        age_seconds = Decimal(age_ms) / Decimal("1000")
        age_multiplier = Decimal("1") + (
            self.config.age_penalty_per_second * age_seconds
        )
        return variance * age_multiplier
