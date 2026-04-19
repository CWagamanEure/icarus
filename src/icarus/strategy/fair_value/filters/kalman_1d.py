from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..weighting import cap_and_renormalize


@dataclass
class VenueObservation:
    name: str
    fair_value: float              # price in dollars
    local_variance: float          # microstructure measurement variance (dollars^2)
    age_ms: float = 0.0            # how old this venue observation is
    valid: bool = True


@dataclass
class VenueFilterDiagnostic:
    """
    Per-venue diagnostic information for a single filter update.

    The two variance quantities matter:
      * ``base_variance`` — microstructure measurement variance from the
        spread/depth/age heuristic. Expresses *quote precision*.
      * ``effective_variance`` — base variance inflated by the adaptive
        venue-reliability score. Expresses our best estimate of the
        variance of the venue's offset from the latent common fair value,
        given how surprising the venue has been historically.

    Until a full latent venue-basis state model is in place, the gap
    between base and effective variance is how we compensate for
    unmodeled persistent venue deviations.
    """

    name: str
    base_variance: float
    reliability_score: float          # q_j: EWMA of clipped standardized z^2
    reliability_multiplier: float     # m_j: multiplier applied to base_variance
    effective_variance: float         # base_variance * m_j
    raw_weight: float                 # inverse-variance normalized, pre-cap
    capped_weight: float              # after max-weight cap + renormalization
    innovation: float                 # obs.fair_value - x_prior
    standardized_z2: float            # clipped (innovation^2 / (base_variance + p_prior))


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
    venue_diagnostics: List[VenueFilterDiagnostic] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KalmanFilterConfig:
    """
    Tunable parameters for AdaptiveEfficientPriceKalman.

    Base jumpiness knobs:
      - LOWER q_base_per_sec / q_vol_scale => smoother.
      - HIGHER r_floor => smoother.
      - HIGHER disagreement_scale => smoother on venue disagreement.
      - LOWER obs_var_ewma_alpha => slower vol adaptation.

    Adaptive venue-reliability knobs (apply across updates, not within one):
      - HIGHER reliability_ewma_lambda => slower adaptation (longer memory).
      - HIGHER reliability_alpha => more aggressive variance inflation once
        a venue is flagged as unreliable.
      - HIGHER reliability_z2_clip => allow larger shocks to register
        (but also blow up the EWMA further on extremes).
      - HIGHER reliability_multiplier_max => harder penalty cap for a
        venue that is consistently surprising.
    """

    initial_variance: float = 25.0
    q_base_per_sec: float = 0.05
    q_vol_scale: float = 1.0
    r_floor: float = 0.01
    local_var_floor: float = 1e-4
    disagreement_scale: float = 1.0
    stale_cutoff_ms: float = 750.0
    age_variance_scale: float = 2.0
    obs_var_ewma_alpha: float = 0.05

    # Max per-venue normalized weight in the fused observation. Guards against
    # any single venue monopolizing the composite regardless of how tight its
    # quote looks. 1.0 disables the cap.
    max_venue_weight: float = 0.75

    # --- Adaptive venue reliability ---
    # EWMA smoothing of standardized-innovation-squared. Closer to 1 is
    # slower. 0.98 ≈ 50-sample memory, appropriate for ~10-50 Hz update
    # cadence over multi-second scales.
    reliability_ewma_lambda: float = 0.98
    # How aggressively q_j > 1 inflates base variance.
    # R_eff = R_base * (1 + alpha * max(q_j - 1, 0)).
    reliability_alpha: float = 1.0
    # Winsorize clipped-z^2 per update. Caps the effect of a single extreme
    # innovation on the EWMA. 25 == 5 sigma.
    reliability_z2_clip: float = 25.0
    # Maximum effective-variance multiplier. Prevents a persistently offset
    # venue from being weighted to zero; we still want the signal from it.
    reliability_multiplier_max: float = 20.0
    # Set to 0 to disable adaptive reliability entirely.
    reliability_enabled: bool = True


class AdaptiveEfficientPriceKalman:
    """
    1D Kalman filter on a fused cross-venue fair value.

    State:
        x_t = latent common fair value

    Transition:
        x_t = x_{t-1} + eta_t,  eta_t ~ N(0, Q_t)

    Observation:
        z_t = H x_t + eps_t,    eps_t ~ N(0, R_t)

    Per-venue observations are combined into z_t via inverse-variance
    weighting with a max-weight cap. Each venue's effective variance is
    its microstructure base variance inflated by an adaptive reliability
    score built from its history of *pre-update* innovations against the
    latent state (not the post-update composite — see ``_update_reliability``
    for the non-circularity rationale).
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
        # Per-venue reliability EWMA q_j. Seeded at 1.0 (neutral —
        # expectation of a properly-standardized z^2 is 1) so new or
        # returning venues do not start over- or under-penalized.
        self._venue_reliability: Dict[str, float] = {}

    def update(
        self,
        timestamp_s: float,
        observations: List[VenueObservation],
    ) -> Optional[FilterResult]:
        live = self._select_live_observations(observations)
        if not live:
            return None

        # Step 1: predict (so x_prior and p_prior exist for innovations)
        if self.x is None:
            # First ever observation — fuse with zero reliability history
            # and seed the state. No innovation update possible yet.
            raw_fused_price, r_t, weights, diagnostics = self._fuse_observations(
                live,
                x_prior=None,
                p_prior=self.config.initial_variance,
            )
            self.x = raw_fused_price
            self.p = self.config.initial_variance
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
                venue_diagnostics=diagnostics,
            )

        dt = 0.0
        if self.last_timestamp_s is not None:
            dt = max(timestamp_s - self.last_timestamp_s, 1e-3)
        x_prior = self.x

        # Preview the current fused observation so the process-noise proxy
        # can respond to actual observation moves instead of comparing the
        # previous fused value to itself.
        raw_fused_preview, _, _, _ = self._fuse_observations(
            live,
            x_prior=x_prior,
            p_prior=self.p,
        )
        q_t = self._compute_process_variance(dt, raw_fused_preview)

        # Predict. Under the random-walk transition, x_prior == self.x.
        p_prior = self.p + q_t

        # Step 2: fuse observations using the current reliability scores.
        # The reliability scores here are from the PREVIOUS call — they
        # reflect how surprising each venue has been against prior x's,
        # never including the current observation. This is what breaks
        # the circularity.
        raw_fused_price, r_t, weights, diagnostics = self._fuse_observations(
            live,
            x_prior=x_prior,
            p_prior=p_prior,
        )

        # Step 3: Kalman update using the fused observation.
        k_t = p_prior / (p_prior + r_t)
        x_post = x_prior + k_t * (raw_fused_price - x_prior)
        p_post = (1.0 - k_t) * p_prior

        # Step 4: update per-venue reliability EWMA using the PRE-UPDATE
        # prior x_prior as the reference, not x_post or raw_fused_price.
        # This ensures venue j's reliability is computed against a
        # quantity independent of venue j's current observation.
        self._update_reliability(live, x_prior=x_prior, p_prior=p_prior)

        # Persist state
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
            venue_diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_live_observations(
        self,
        observations: List[VenueObservation],
    ) -> List[VenueObservation]:
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
        """Microstructure base variance, inflated only by staleness."""
        cfg = self.config
        base_var = max(obs.local_variance, cfg.local_var_floor)
        age_factor = 1.0 + cfg.age_variance_scale * (
            obs.age_ms / max(cfg.stale_cutoff_ms, 1.0)
        ) ** 2
        return base_var * age_factor

    def _reliability_multiplier(self, name: str) -> float:
        """Current variance multiplier for a venue given its reliability EWMA."""
        if not self.config.reliability_enabled:
            return 1.0
        q = self._venue_reliability.get(name, 1.0)
        multiplier = 1.0 + self.config.reliability_alpha * max(q - 1.0, 0.0)
        if multiplier > self.config.reliability_multiplier_max:
            return self.config.reliability_multiplier_max
        if multiplier < 1.0:
            return 1.0
        return multiplier

    def _fuse_observations(
        self,
        live: List[VenueObservation],
        *,
        x_prior: Optional[float],
        p_prior: float,
    ) -> tuple[float, float, Dict[str, float], List[VenueFilterDiagnostic]]:
        """
        Returns (z_t, R_t, weights, per-venue diagnostics).

        Weights are inverse-effective-variance with a per-venue max-weight
        cap. Effective variance = base variance * reliability multiplier.
        """
        base_vars = [self._effective_local_variance(obs) for obs in live]
        multipliers = [self._reliability_multiplier(obs.name) for obs in live]
        effective_vars = [b * m for b, m in zip(base_vars, multipliers)]

        inv_vars = [1.0 / v for v in effective_vars]
        inv_sum = sum(inv_vars)
        raw_weights = [iv / inv_sum for iv in inv_vars]

        capped_weights = cap_and_renormalize(
            raw_weights,
            max_weight=self.config.max_venue_weight,
        )

        z_t = sum(a * obs.fair_value for a, obs in zip(capped_weights, live))

        intrinsic_fused_var = sum(
            (a ** 2) * v for a, v in zip(capped_weights, effective_vars)
        )
        disagreement_var = sum(
            a * ((obs.fair_value - z_t) ** 2) for a, obs in zip(capped_weights, live)
        )
        r_t = max(
            self.config.r_floor,
            intrinsic_fused_var + self.config.disagreement_scale * disagreement_var,
        )

        # Build per-venue diagnostic snapshot. Innovation is against the
        # pre-update prior (or raw_fused on bootstrap when there's no prior).
        diagnostics: List[VenueFilterDiagnostic] = []
        for i, obs in enumerate(live):
            if x_prior is None:
                innovation = 0.0
                z2 = 0.0
            else:
                innovation = obs.fair_value - x_prior
                denom = max(base_vars[i] + p_prior, 1e-12)
                z2 = min(
                    (innovation * innovation) / denom,
                    self.config.reliability_z2_clip,
                )
            diagnostics.append(
                VenueFilterDiagnostic(
                    name=obs.name,
                    base_variance=base_vars[i],
                    reliability_score=self._venue_reliability.get(obs.name, 1.0),
                    reliability_multiplier=multipliers[i],
                    effective_variance=effective_vars[i],
                    raw_weight=raw_weights[i],
                    capped_weight=capped_weights[i],
                    innovation=innovation,
                    standardized_z2=z2,
                )
            )

        weights = {obs.name: a for obs, a in zip(live, capped_weights)}
        return z_t, r_t, weights, diagnostics

    def _update_reliability(
        self,
        live: List[VenueObservation],
        *,
        x_prior: float,
        p_prior: float,
    ) -> None:
        """
        Update each venue's reliability EWMA.

        Uses x_prior (the predicted latent state BEFORE incorporating the
        current batch of observations) as the innovation reference. This is
        deliberately NOT x_post or the post-update fused price, either of
        which would contain venue j's own current observation and create
        self-reinforcing circularity (venue j would appear reliable against
        a composite it just moved).

        Standardization: z^2 = innovation^2 / (R_base_j + P_prior).
        Under the model the innovation is ~ N(0, R_base_j + P_prior), so
        z^2 has expectation 1 when the venue is well-calibrated. Persistent
        z^2 > 1 is evidence of unmodeled local deviation.

        z^2 is clipped to ``reliability_z2_clip`` to prevent a single
        outlier tick from dominating the EWMA.
        """
        cfg = self.config
        if not cfg.reliability_enabled:
            return

        lam = cfg.reliability_ewma_lambda
        if lam >= 1.0:
            return  # no adaptation
        z2_clip = cfg.reliability_z2_clip

        for obs in live:
            base_var = self._effective_local_variance(obs)
            denom = max(base_var + p_prior, 1e-12)
            innovation = obs.fair_value - x_prior
            z2 = (innovation * innovation) / denom
            if z2 > z2_clip:
                z2 = z2_clip
            prev = self._venue_reliability.get(obs.name, 1.0)
            self._venue_reliability[obs.name] = lam * prev + (1.0 - lam) * z2

    def _compute_process_variance(self, dt: float, reference_price: float) -> float:
        cfg = self.config
        if self.last_raw_fused_price is not None:
            move = reference_price - self.last_raw_fused_price
            move_var_per_sec = (move * move) / max(dt, 1e-3)
            a = cfg.obs_var_ewma_alpha
            self.obs_move_var_ewma = (
                a * move_var_per_sec + (1.0 - a) * self.obs_move_var_ewma
            )

        q_per_sec = cfg.q_base_per_sec + cfg.q_vol_scale * self.obs_move_var_ewma
        return max(1e-8, q_per_sec * dt)
