from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .types import CombinedFairValueEstimate, VenueFairValueState
from .weighting import cap_and_renormalize


@dataclass(frozen=True, slots=True)
class CrossVenueCombinerConfig:
    stale_after_ms: int = 2000
    age_penalty_per_second: Decimal = Decimal("1")
    disagreement_scale: Decimal = Decimal("1")
    variance_floor: Decimal = Decimal("0.01")

    # Cap on any single venue's normalized weight in the fused composite.
    # Prevents a venue with pathologically tight quote precision from
    # monopolizing the composite when venues have a persistent basis.
    # Set to 1 to disable.
    max_venue_weight: Decimal = Decimal("0.75")


@dataclass(frozen=True, slots=True)
class VenueCombinerDiagnostic:
    exchange: str
    base_variance: Decimal
    effective_variance: Decimal
    raw_weight: Decimal
    capped_weight: Decimal
    fair_value: Decimal
    age_ms: int


@dataclass(frozen=True, slots=True)
class CombinerDiagnostics:
    timestamp_ms: int
    venues: tuple[VenueCombinerDiagnostic, ...] = field(default_factory=tuple)


class CrossVenueFairValueCombiner:
    def __init__(self, market: str, config: CrossVenueCombinerConfig | None = None) -> None:
        self.market = market
        self.config = config or CrossVenueCombinerConfig()
        self._states: dict[str, VenueFairValueState] = {}
        self._last_diagnostics: CombinerDiagnostics | None = None

    @property
    def last_diagnostics(self) -> CombinerDiagnostics | None:
        """Per-venue weight/variance breakdown from the most recent combine()."""
        return self._last_diagnostics

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
        live_states: list[tuple[VenueFairValueState, Decimal, int]] = []

        for state in self._states.values():
            age_ms = now_ms - state.timestamp_ms
            if age_ms < 0:
                age_ms = 0
            if age_ms > self.config.stale_after_ms:
                continue

            effective_variance = self._effective_variance(state.variance, age_ms)
            if effective_variance <= 0:
                continue

            live_states.append((state, effective_variance, age_ms))

        if not live_states:
            self._last_diagnostics = CombinerDiagnostics(timestamp_ms=now_ms, venues=())
            return None

        raw_weights: list[Decimal] = []
        fair_values: list[Decimal] = []
        variances: list[Decimal] = []
        exchanges: list[str] = []
        ages_ms: list[int] = []
        base_variances: list[Decimal] = []

        for state, effective_variance, age_ms in live_states:
            weight = Decimal("1") / effective_variance
            raw_weights.append(weight)
            fair_values.append(state.fair_value)
            variances.append(effective_variance)
            exchanges.append(state.exchange)
            ages_ms.append(age_ms)
            base_variances.append(state.variance)

        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            self._last_diagnostics = CombinerDiagnostics(timestamp_ms=now_ms, venues=())
            return None

        normalized_raw = [w / weight_sum for w in raw_weights]
        capped_weights = cap_and_renormalize(
            normalized_raw,
            max_weight=self.config.max_venue_weight,
        )

        combined_fv = sum(
            w * fv for w, fv in zip(capped_weights, fair_values)
        )

        intrinsic_variance = sum(
            (w * w) * v for w, v in zip(capped_weights, variances)
        )

        disagreement_variance = sum(
            w * ((fv - combined_fv) ** 2)
            for w, fv in zip(capped_weights, fair_values)
        )

        combined_variance = intrinsic_variance + (
            self.config.disagreement_scale * disagreement_variance
        )

        if combined_variance < self.config.variance_floor:
            combined_variance = self.config.variance_floor

        self._last_diagnostics = CombinerDiagnostics(
            timestamp_ms=now_ms,
            venues=tuple(
                VenueCombinerDiagnostic(
                    exchange=exchanges[i],
                    base_variance=base_variances[i],
                    effective_variance=variances[i],
                    raw_weight=normalized_raw[i],
                    capped_weight=capped_weights[i],
                    fair_value=fair_values[i],
                    age_ms=ages_ms[i],
                )
                for i in range(len(exchanges))
            ),
        )

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
