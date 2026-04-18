from __future__ import annotations

from decimal import Decimal

from icarus.strategy.fair_value.combiner import (
    CrossVenueCombinerConfig,
    CrossVenueFairValueCombiner,
)
from icarus.strategy.fair_value.types import VenueFairValueState


def test_cross_venue_combiner_combines_multiple_fresh_venues() -> None:
    combiner = CrossVenueFairValueCombiner(
        "BTC-USD",
        config=CrossVenueCombinerConfig(age_penalty_per_second=Decimal("0")),
    )
    venue_a = VenueFairValueState(
        exchange="coinbase",
        market="BTC-USD",
        timestamp_ms=1000,
        fair_value=Decimal("100"),
        variance=Decimal("1"),
    )
    venue_b = VenueFairValueState(
        exchange="hyperliquid",
        market="BTC-USD",
        timestamp_ms=1005,
        fair_value=Decimal("102"),
        variance=Decimal("4"),
    )

    first = combiner.update(venue_a)
    second = combiner.update(venue_b, now_ms=1010)

    assert first is not None
    assert first.fair_value == Decimal("100")
    assert second is not None
    assert second.fair_value == Decimal("100.4")
    # intrinsic = 0.8^2*1 + 0.2^2*4 = 0.8
    # disagreement = 0.8*(100-100.4)^2 + 0.2*(102-100.4)^2 = 0.64
    # combined = 0.8 + 1*0.64 = 1.44
    assert second.variance == Decimal("1.44")
    assert second.contributing_exchanges == ("coinbase", "hyperliquid")


def test_cross_venue_combiner_drops_stale_venues() -> None:
    combiner = CrossVenueFairValueCombiner(
        "BTC-USD",
        config=CrossVenueCombinerConfig(stale_after_ms=100),
    )
    combiner.update(
        VenueFairValueState(
            exchange="coinbase",
            market="BTC-USD",
            timestamp_ms=1000,
            fair_value=Decimal("100"),
            variance=Decimal("1"),
        )
    )

    combined = combiner.update(
        VenueFairValueState(
            exchange="hyperliquid",
            market="BTC-USD",
            timestamp_ms=1200,
            fair_value=Decimal("110"),
            variance=Decimal("1"),
        ),
        now_ms=1200,
    )

    assert combined is not None
    assert combined.fair_value == Decimal("110")
    assert combined.contributing_exchanges == ("hyperliquid",)


def test_cross_venue_combiner_penalizes_older_venues_without_dropping_them() -> None:
    combiner = CrossVenueFairValueCombiner(
        "BTC-USD",
        config=CrossVenueCombinerConfig(
            stale_after_ms=500,
            age_penalty_per_second=Decimal("1"),
        ),
    )
    combiner.update(
        VenueFairValueState(
            exchange="coinbase",
            market="BTC-USD",
            timestamp_ms=1000,
            fair_value=Decimal("100"),
            variance=Decimal("1"),
        )
    )

    combined = combiner.update(
        VenueFairValueState(
            exchange="hyperliquid",
            market="BTC-USD",
            timestamp_ms=1200,
            fair_value=Decimal("110"),
            variance=Decimal("1"),
        ),
        now_ms=1200,
    )

    assert combined is not None
    assert combined.fair_value > Decimal("105")
    assert combined.fair_value < Decimal("110")
    # Variance is inflated by the disagreement term (venues are $10 apart)
    assert combined.variance > Decimal("1")
