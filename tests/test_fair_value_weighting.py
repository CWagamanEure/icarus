from __future__ import annotations

from decimal import Decimal

from icarus.strategy.fair_value.weighting import cap_and_renormalize


def _approx_sum(weights: list[Decimal]) -> Decimal:
    total = Decimal(0)
    for w in weights:
        total += w
    return total


def test_cap_does_nothing_when_already_below_cap() -> None:
    w = [Decimal("0.5"), Decimal("0.3"), Decimal("0.2")]
    out = cap_and_renormalize(w, max_weight=Decimal("0.75"))
    assert out == w


def test_cap_reduces_dominant_and_redistributes_proportionally() -> None:
    w = [Decimal("0.9"), Decimal("0.07"), Decimal("0.03")]
    out = cap_and_renormalize(w, max_weight=Decimal("0.75"))
    assert out[0] == Decimal("0.75")
    # 0.07 and 0.03 had ratio 7:3. Remaining mass 0.25 should split 7:3 too.
    assert out[1] == Decimal("0.175")
    assert out[2] == Decimal("0.075")
    assert _approx_sum(out) == Decimal(1)


def test_cap_iterates_when_redistribution_pushes_another_over() -> None:
    # After capping the first at 0.4, redistribution pushes the second past 0.4 too.
    w = [Decimal("0.8"), Decimal("0.15"), Decimal("0.05")]
    out = cap_and_renormalize(w, max_weight=Decimal("0.4"))
    assert out == [Decimal("0.4"), Decimal("0.4"), Decimal("0.2")]


def test_cap_returns_input_when_infeasible() -> None:
    # 2 venues * 0.3 cap = 0.6, cannot sum to 1.
    w = [Decimal("0.9"), Decimal("0.1")]
    out = cap_and_renormalize(w, max_weight=Decimal("0.3"))
    assert out == w


def test_cap_noop_for_single_venue() -> None:
    out = cap_and_renormalize([Decimal("1")], max_weight=Decimal("0.75"))
    assert out == [Decimal("1")]


def test_cap_empty() -> None:
    assert cap_and_renormalize([], max_weight=Decimal("0.75")) == []


def test_cap_preserves_decimal_type() -> None:
    w = [Decimal("0.99"), Decimal("0.01")]
    out = cap_and_renormalize(w, max_weight=Decimal("0.75"))
    for x in out:
        assert isinstance(x, Decimal)


def test_cap_works_with_float_inputs() -> None:
    out = cap_and_renormalize([0.99, 0.01], max_weight=0.75)
    assert out[0] == 0.75
    assert abs(out[1] - 0.25) < 1e-12
