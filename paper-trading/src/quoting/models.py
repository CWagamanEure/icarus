from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MicroFeatures:
    spread: float
    depth_bid: float
    depth_ask: float
    imbalance: float
    mid: float
    microprice: float

@dataclass(frozen=True)
class NoiseModelParams:
    a0: float
    a_spread: float
    a_depth: float
    a_imb: float


@dataclass(frozen=True)
class KFState:
    x: float # logit space mean
    P: float  # logit space variance


@dataclass(frozen=True)
class FairPriceSnapshot:
    ts_ms: Optional[int]
    p_obs: float
    p_fair: float
    x_fair: float
    P: float
    R: float
    features: MicroFeatures


