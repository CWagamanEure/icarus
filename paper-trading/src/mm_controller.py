from .quote import DesiredQuote
from .quoting.fair_price_engine import FairPriceEngine

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from typing import Optional

@dataclass(frozen=True)
class ControllerParams:
    size: Decimal = Decimal("50")
    tick: Decimal = Decimal("0.001")

    half_spread_ticks: int=1

    min_requote_ms: int = 250
    requote_if_fair_moves_ticks: int = 1



class MMController:
    def __init__(self, fair_engine: FairPriceEngine, params: ControllerParams):
        self.fair_engine = fair_engine
        self.params = params


        self._las_quote: Optional[DesiredQuote] = None
        self._las_quote_ts_ms: Optional[int] = None
        self._last_fair_tick: Optional[int] = None


    def on_book(self, book) -> Optional[DesiredQuote]:
        """
        Returns a desired quote when decides a requote is needed, else None
        """

        snap = self.fair_engine.on_event(book)
        if snap is None:
            return None

        ts_ms = snap.ts_ms or book.ts_ms
        if ts_ms is None:
            ts_ms = 0

        fair = Decimal(str(snap.p_fair))
        fair_tick = self._to_tick_floor(fair)

        # rigth now we just always return if new snap for testing

        half = Decimal(self.params.half_spread_ticks) * self.params.tick
        bid = self._round_down_to_tick(fair - half)
        ask = self._round_up_to_tick(fair + half)

        if ask <= bid:
            ask = bid + self.params.tick

        dq = DesiredQuote(
            bid=bid,
            ask=ask,
            size=self.params.size,
            ts_ms=ts_ms,
            p_fair=float(snap.p_fair)
        )

        self._las_quote = dq
        self._las_quote_ts_ms = ts_ms
        self._last_fair_tick = fair_tick
        return dq


    #------ Helpers -------------
    def _to_tick_floor(self, px: Decimal) -> int:
        t = (px / self.params.tick).to_integral_value(rounding=ROUND_FLOOR)
        return int(t)

    def _round_down_to_tick(self, px: Decimal) -> Decimal:
        t = (px / self.params.tick).to_integral_value(rounding=ROUND_FLOOR)
        return t * self.params.tick

    def _round_up_to_tick(self, px: Decimal) -> Decimal:
        t = (px / self.params.tick).to_integral_value(rounding=ROUND_CEILING)
        return t * self.params.tick
