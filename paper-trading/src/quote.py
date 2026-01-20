from dataclasses import dataclass
from typing import Optional
from decimal import Decimal



@dataclass(frozen=True)
class DesiredQuote:
    bid: Decimal
    ask: Decimal
    size: Decimal
    ts_ms: Optional[int] = None
    p_fair: Optional[float] = None
