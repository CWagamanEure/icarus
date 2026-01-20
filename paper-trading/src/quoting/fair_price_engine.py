from decimal import Decimal
from typing import Optional

from ..orderbook_state import OrderBookState
from .math_helpers import sigmoid, clamp, logit 
from .models import MicroFeatures, FairPriceSnapshot, NoiseModelParams, KFState
from .kalman_filter import obs_var_Rt, kf_step_random_walk 

def microfeatures_from_orderbook_state(ob, depth_levels: int=5) -> MicroFeatures:
    """
    ob is orderbookstate

    """
    if ob.best_bid is None or ob.best_ask is None:
        raise ValueError("NO best bid/ask yet")

    bid_px = float(ob._tick_to_price(ob.best_bid))
    ask_px = float(ob._tick_to_price(ob.best_ask))

    mid = 0.5 * (bid_px + ask_px)
    spread = max(ask_px - bid_px, 0.0)


    # best level sizes for microprice
    bid_sz0 = float(ob.bids.get(ob.best_bid, Decimal("0")))
    ask_sz0 = float(ob.asks.get(ob.best_ask, Decimal("0")))
    denom0 = max(bid_sz0 + ask_sz0, 1e-9)

    #microprice (weighted towards side with LESS size at touch)
    #microprice (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)
    microprice = (bid_px * ask_sz0 + ask_px * bid_sz0) / denom0

    # depth + imbalance using top N levels 
    depth_bid = float(ob.depth_sum(side="BUY", n=depth_levels))
    depth_ask = float(ob.depth_sum(side="ASK", n=depth_levels))

    denom = max(depth_bid + depth_ask, 1e-9)
    imbalance = (depth_bid - depth_ask) / denom

    return MicroFeatures(
        spread=spread,
        depth_bid=depth_bid,
        depth_ask=depth_ask,
        imbalance=imbalance,
        mid=clamp(mid, 1e-6, 1.0 - 1e-6),
        microprice=clamp(microprice, 1e-6, 1- 1e-6)
    )


class FairPriceEngine:
    """
    Call engine.on_event(orderbook_state) after each applied event.
    Keeps internal KF state and outputs a FairPriceSnapshot
    """

    def __init__(self,
                 *,
                 Q: float,
                 x0: float,
                 P0: float,
                 noise_params: NoiseModelParams,
                 depth_levels: int=5,
                 obs_mode: str= "microprice",  # make this either microprice, mid, or blend
                 blend_alpha: float = 0.7,):
        self.state = KFState(x=x0, P=P0)
        self.Q = Q
        self.noise_params = noise_params
        self.depth_levels = depth_levels
        self.obs_mode = obs_mode
        self.blend_alpha = blend_alpha


    def _choose_obs(self, feats: MicroFeatures) -> float:
        if self.obs_mode == "mid":
            return feats.mid
        if self.obs_mode == "microprice": return feats.microprice
        
        # blend
        a = clamp(self.blend_alpha, 0.0, 1.0)
        return a * feats.microprice + (1.0 - a) * feats.mid


    def on_event(self, ob) -> Optional[FairPriceSnapshot]:

        # only produce estimate when we have valid top of book
        if ob.best_bid is None or ob.best_ask is None:
            return None
        
        feats = microfeatures_from_orderbook_state(ob, depth_levels=self.depth_levels)
        p_obs = self._choose_obs(feats)

        y = logit(p_obs)
        R = obs_var_Rt(feats, self.noise_params)

        self.state = kf_step_random_walk(self.state, y=y, R=R, Q=self.Q)

        p_fair = sigmoid(self.state.x)

        return FairPriceSnapshot(
            ts_ms=ob.ts_ms,
            p_obs=p_obs,
            p_fair=p_fair,
            x_fair=self.state.x,
            P=self.state.P,
            R=R,
            features=feats,
        )



