from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math


@dataclass
class VenueObservation:
    name: str
    fair_value: float              # price in dollars
    local_variance: float          # variance in dollars^2
    age_ms: float = 0.0            # how old this venue observation is
    valid: bool = True


@dataclass
class FilterResult:
    timestamp_s: float
    filtered_price: float
    raw_fused_price: float
    kalman_gain: float
    process_variance_q: float
    measurement_variance_r: float
    posterior_variance_p: float
    weights: Dict[str, float]
    used_venues: List[str]


@dataclass(frozen=True, slots=True)
class KalmanFilterConfig:
    """
    Tunable parameters for AdaptiveEfficientPriceKalman.

    Jumpiness knobs (quick reference):
      - LOWER q_base_per_sec / q_vol_scale => smoother (less process noise => lower gain).
      - HIGHER r_floor => smoother (observations trusted less).
      - HIGHER disagreement_scale => smoother when venues disagree.
      - LOWER obs_var_ewma_alpha => slower adaptation of volatility estimate.
    """

    initial_variance: float = 25.0         # P_0 in dollars^2
    q_base_per_sec: float = 0.05           # baseline process variance per second
    q_vol_scale: float = 1.0               # multiplies EWMA observed move variance
    r_floor: float = 0.01                  # minimum R_t
    local_var_floor: float = 1e-4          # minimum per-venue local variance
    disagreement_scale: float = 1.0        # inflates R when venues disagree
    stale_cutoff_ms: float = 750.0         # drop venue obs older than this
    age_variance_scale: float = 2.0        # quadratic penalty on older venue obs
    obs_var_ewma_alpha: float = 0.05       # EWMA alpha for process noise proxy


class AdaptiveEfficientPriceKalman:
    """
    1D Kalman filter on a fused cross-venue fair value.

    State:
        x_t = latent global efficient price

    Observation:
        z_t = fused fair value from venues

    Transition:
        x_t = x_{t-1} + eta_t

    Observation:
        z_t = x_t + eps_t
    """

    def __init__(
        self,
        *,
        config: KalmanFilterConfig | None = None,
        initial_price: Optional[float] = None,
    ) -> None:
        self.config = config or KalmanFilterConfig()

        self.x: Optional[float] = initial_price
        self.p: float = self.config.initial_variance

        self.last_timestamp_s: Optional[float] = None
        self.last_raw_fused_price: Optional[float] = None
        self.obs_move_var_ewma: float = 0.0

    def update(self, timestamp_s: float, observations: List[VenueObservation]) -> Optional[FilterResult]:
        live = self._select_live_observations(observations)
        if not live:
            return None

        raw_fused_price, r_t, weights = self._fuse_observations(live)

        # Initialize state on first update
        if self.x is None:
            self.x = raw_fused_price
            self.last_timestamp_s = timestamp_s
            self.last_raw_fused_price = raw_fused_price
            return FilterResult(
                timestamp_s=timestamp_s,
                filtered_price=self.x,
                raw_fused_price=raw_fused_price,
                kalman_gain=1.0,
                process_variance_q=0.0,
                measurement_variance_r=r_t,
                posterior_variance_p=self.p,
                weights=weights,
                used_venues=list(weights.keys()),
            )

        dt = 0.0
        if self.last_timestamp_s is not None:
            dt = max(timestamp_s - self.last_timestamp_s, 1e-3)

        q_t = self._compute_process_variance(dt, raw_fused_price)

        # Predict
        p_prior = self.p + q_t
        x_prior = self.x

        # Update
        k_t = p_prior / (p_prior + r_t)
        x_post = x_prior + k_t * (raw_fused_price - x_prior)
        p_post = (1.0 - k_t) * p_prior

        # Save state
        self.x = x_post
        self.p = p_post
        self.last_timestamp_s = timestamp_s
        self.last_raw_fused_price = raw_fused_price

        return FilterResult(
            timestamp_s=timestamp_s,
            filtered_price=x_post,
            raw_fused_price=raw_fused_price,
            kalman_gain=k_t,
            process_variance_q=q_t,
            measurement_variance_r=r_t,
            posterior_variance_p=p_post,
            weights=weights,
            used_venues=list(weights.keys()),
        )

    def _select_live_observations(self, observations: List[VenueObservation]) -> List[VenueObservation]:
        live: List[VenueObservation] = []
        for obs in observations:
            if not obs.valid:
                continue
            if not math.isfinite(obs.fair_value):
                continue
            if not math.isfinite(obs.local_variance):
                continue
            if obs.age_ms > self.config.stale_cutoff_ms:
                continue
            live.append(obs)
        return live

    def _effective_local_variance(self, obs: VenueObservation) -> float:
        """
        Inflate local variance for stale observations.
        """
        cfg = self.config
        base_var = max(obs.local_variance, cfg.local_var_floor)
        age_factor = 1.0 + cfg.age_variance_scale * (obs.age_ms / max(cfg.stale_cutoff_ms, 1.0)) ** 2
        return base_var * age_factor

    def _fuse_observations(
        self,
        live: List[VenueObservation],
    ) -> tuple[float, float, Dict[str, float]]:
        """
        Returns:
            z_t: fused observation
            R_t: measurement variance for z_t
            weights: normalized venue weights
        """
        effective_vars = [self._effective_local_variance(obs) for obs in live]

        # Inverse-variance weights
        inv_vars = [1.0 / v for v in effective_vars]
        inv_sum = sum(inv_vars)
        alphas = [iv / inv_sum for iv in inv_vars]

        z_t = sum(a * obs.fair_value for a, obs in zip(alphas, live))

        # Variance of weighted average under independent noise
        intrinsic_fused_var = sum((a ** 2) * v for a, v in zip(alphas, effective_vars))

        # Cross-venue disagreement term
        disagreement_var = sum(a * ((obs.fair_value - z_t) ** 2) for a, obs in zip(alphas, live))

        r_t = max(
            self.config.r_floor,
            intrinsic_fused_var + self.config.disagreement_scale * disagreement_var,
        )

        weights = {obs.name: a for obs, a in zip(live, alphas)}
        return z_t, r_t, weights

    def _compute_process_variance(self, dt: float, raw_fused_price: float) -> float:
        """
        Q_t controls how much the latent efficient price is allowed to move.
        Use an EWMA of fused observation moves as a proxy for current market speed.
        """
        cfg = self.config
        if self.last_raw_fused_price is not None:
            move = raw_fused_price - self.last_raw_fused_price
            move_var_per_sec = (move * move) / max(dt, 1e-3)

            a = cfg.obs_var_ewma_alpha
            self.obs_move_var_ewma = a * move_var_per_sec + (1.0 - a) * self.obs_move_var_ewma

        q_per_sec = cfg.q_base_per_sec + cfg.q_vol_scale * self.obs_move_var_ewma
        return max(1e-8, q_per_sec * dt)
