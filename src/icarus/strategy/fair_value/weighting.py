from __future__ import annotations

from decimal import Decimal
from typing import List, Sequence, TypeVar

Number = TypeVar("Number", Decimal, float)


def cap_and_renormalize(
    normalized_weights: Sequence[Number],
    *,
    max_weight: Number,
) -> List[Number]:
    """
    Cap each entry of an already-normalized weight vector (sum == 1) at
    ``max_weight`` and fairly redistribute the removed mass proportionally
    across uncapped entries.

    Iterates to a fixed point: redistributing can push another entry over
    the cap, which is then also capped. Terminates in at most len(weights)
    passes because each pass caps at least one new entry.

    Edge cases:
      * Empty input: returns [].
      * Single venue: cap is moot, returns the input unchanged.
      * max_weight * n < 1: cap infeasible (even a uniform distribution
        violates it); returns the input unchanged.
      * All weights non-positive: returns the input unchanged.

    Generic in numeric type. Decimal inputs yield Decimal outputs; float
    inputs yield float outputs. No mixing.
    """
    n = len(normalized_weights)
    if n == 0:
        return []
    if n == 1:
        return list(normalized_weights)

    is_decimal = isinstance(max_weight, Decimal)
    zero: Number = Decimal(0) if is_decimal else 0.0  # type: ignore[assignment]
    one: Number = Decimal(1) if is_decimal else 1.0  # type: ignore[assignment]

    if max_weight * n < one:
        return list(normalized_weights)

    current: List[Number] = list(normalized_weights)
    total = _sum(current, zero)
    if total <= zero:
        return current

    capped = [False] * n

    for _ in range(n):
        if all(c for c in capped):
            break

        capped_mass = _sum(
            (w for w, c in zip(current, capped) if c),
            zero,
        )
        uncapped_sum = _sum(
            (w for w, c in zip(current, capped) if not c),
            zero,
        )
        if uncapped_sum <= zero:
            break

        target_remaining = one - capped_mass
        scale = target_remaining / uncapped_sum

        new_weights: List[Number] = [
            max_weight if c else w * scale
            for w, c in zip(current, capped)
        ]

        newly_capped = [
            (not c) and (new_weights[i] > max_weight)
            for i, c in enumerate(capped)
        ]
        if not any(newly_capped):
            return new_weights

        current = [
            max_weight if newly_capped[i] or capped[i] else new_weights[i]
            for i in range(n)
        ]
        for i, nc in enumerate(newly_capped):
            if nc:
                capped[i] = True

    return current


def _sum(values, zero: Number) -> Number:
    total: Number = zero
    for v in values:
        total = total + v  # type: ignore[assignment]
    return total
