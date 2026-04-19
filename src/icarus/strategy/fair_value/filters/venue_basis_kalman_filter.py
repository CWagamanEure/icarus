from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Literal, Optional

import numpy as np

type VenueBasisKind = Literal["spot", "perp"]


@dataclass
class VenueBasisObservation:
    name: str
    fair_value: float | Decimal
    local_variance: float | Decimal
    age_ms: float = 0.0
    valid: bool = True
    venue_kind: VenueBasisKind = "spot"


@dataclass
class VenueBasisObservationDiagnostic:
    name: str
    fair_value: float
    predicted_fair_value: float
    innovation: float
    observation_variance: float
    basis_estimate: float
    basis_stddev: float


@dataclass
class VenueBasisFilterResult:
    timestamp_s: float
    common_price: float
    common_price_stddev: float
    anchor_exchange: str
    active_venues: List[str]
    basis_estimates: Dict[str, float]
    basis_stddevs: Dict[str, float]
    observation_variances: Dict[str, float]
    innovations: Dict[str, float]
    state_covariance_trace: float
    basis_state_indices: Dict[str, int]
    basis_state_kinds: Dict[str, VenueBasisKind]
    observation_diagnostics: List[VenueBasisObservationDiagnostic] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VenueBasisKalmanConfig:
    anchor_exchange: str
    venue_order: tuple[str, ...] = ()
    perp_exchange_order: tuple[str, ...] = ()
    require_anchor_observation: bool = True
    # The common latent price should be materially more mobile than venue-basis
    # states; otherwise shared market moves get misattributed to basis drift.
    common_price_process_var_per_sec: float = 2.0
    default_basis_process_var_per_sec: float = 0.01
    default_perp_basis_process_var_per_sec: float = 0.25
    # Persistence over one second for the AR(1) basis process. The per-update
    # transition coefficient is computed as rho_per_second ** dt_seconds.
    default_basis_rho_per_second: float = 0.98
    default_perp_basis_rho_per_second: float = 0.995
    basis_process_var_per_sec_overrides: dict[str, float] = field(default_factory=dict)
    basis_rho_per_second_overrides: dict[str, float] = field(default_factory=dict)
    perp_basis_process_var_per_sec_overrides: dict[str, float] = field(default_factory=dict)
    perp_basis_rho_per_second_overrides: dict[str, float] = field(default_factory=dict)
    initial_common_price_variance: float = 25.0
    initial_basis_variance: float = 100.0
    stale_cutoff_ms: float = 750.0
    local_var_floor: float = 1e-4
    innovation_var_floor: float = 1e-8
    covariance_floor: float = 1e-10


class VenueBasisKalmanFilter:
    """
    Experimental cross-venue state-space filter with anchored venue basis states.

    State:
        [x_t, b_{j1,t}, b_{j2,t}, ...]

    where x_t is the common efficient price and each b_j,t is a non-anchor
    venue's persistent local basis relative to the anchor venue, whose basis is
    fixed at zero for identifiability.
    """

    def __init__(
        self,
        *,
        config: VenueBasisKalmanConfig,
        initial_price: float | Decimal | None = None,
    ) -> None:
        self.config = config
        self._validate_config()

        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._known_spot_venues: set[str] = set()
        self._known_perp_venues: set[str] = set()
        self._basis_state_indices: dict[str, int] = {}
        self._basis_state_kinds: dict[str, VenueBasisKind] = {}
        self.last_timestamp_s: float | None = None

        if initial_price is not None:
            self._state = np.array([float(initial_price)], dtype=np.float64)
            self._covariance = np.array(
                [[max(config.initial_common_price_variance, config.covariance_floor)]],
                dtype=np.float64,
            )

    @property
    def basis_state_indices(self) -> Dict[str, int]:
        return dict(self._basis_state_indices)

    @property
    def basis_state_kinds(self) -> Dict[str, VenueBasisKind]:
        return dict(self._basis_state_kinds)

    def update(
        self,
        timestamp_s: float,
        observations: List[VenueBasisObservation],
    ) -> Optional[VenueBasisFilterResult]:
        live = self._select_live_observations(observations)
        if not live:
            return None
        if self.config.require_anchor_observation and not any(
            obs.name == self.config.anchor_exchange for obs in live
        ):
            return None

        self._ensure_state_layout(live)

        if self._state is None or self._covariance is None:
            self._bootstrap_state(live)

        assert self._state is not None
        assert self._covariance is not None

        dt = 0.0
        if self.last_timestamp_s is not None:
            dt = max(timestamp_s - self.last_timestamp_s, 1e-3)

        state_prior, covariance_prior = self._predict(
            self._state,
            self._covariance,
            dt,
        )
        measurement_model = self._build_measurement_model(live, state_prior)
        if measurement_model is None:
            return None

        h_matrix, y_vector, r_matrix, diagnostics_seed = measurement_model
        innovation = y_vector - (h_matrix @ state_prior)
        s_matrix = (h_matrix @ covariance_prior @ h_matrix.T) + r_matrix
        s_matrix = self._stabilize_innovation_covariance(s_matrix)
        kalman_gain = covariance_prior @ h_matrix.T @ np.linalg.inv(s_matrix)

        state_post = state_prior + (kalman_gain @ innovation)
        identity = np.eye(len(state_prior), dtype=np.float64)
        kh = kalman_gain @ h_matrix
        i_minus_kh = identity - kh
        covariance_post = (
            (i_minus_kh @ covariance_prior @ i_minus_kh.T)
            + (kalman_gain @ r_matrix @ kalman_gain.T)
        )
        covariance_post = self._stabilize_covariance(covariance_post)

        self._state = state_post
        self._covariance = covariance_post
        self.last_timestamp_s = timestamp_s

        basis_estimates = self._basis_estimates(state_post)
        basis_stddevs = self._basis_stddevs(covariance_post)
        observation_variances = {
            diag.name: diag.observation_variance for diag in diagnostics_seed
        }
        innovations = {
            diagnostics_seed[i].name: innovation[i] for i in range(len(diagnostics_seed))
        }

        diagnostics = [
            VenueBasisObservationDiagnostic(
                name=seed.name,
                fair_value=seed.fair_value,
                predicted_fair_value=seed.predicted_fair_value,
                innovation=innovations[seed.name],
                observation_variance=seed.observation_variance,
                basis_estimate=basis_estimates[seed.name],
                basis_stddev=basis_stddevs[seed.name],
            )
            for seed in diagnostics_seed
        ]

        return VenueBasisFilterResult(
            timestamp_s=timestamp_s,
            common_price=state_post[0],
            common_price_stddev=math.sqrt(max(float(covariance_post[0, 0]), 0.0)),
            anchor_exchange=self.config.anchor_exchange,
            active_venues=[diag.name for diag in diagnostics],
            basis_estimates=basis_estimates,
            basis_stddevs=basis_stddevs,
            observation_variances=observation_variances,
            innovations=innovations,
            state_covariance_trace=float(np.trace(covariance_post)),
            basis_state_indices=self.basis_state_indices,
            basis_state_kinds=self.basis_state_kinds,
            observation_diagnostics=diagnostics,
        )

    def _validate_config(self) -> None:
        if not self.config.anchor_exchange:
            raise ValueError("anchor_exchange must be non-empty.")
        if not (0.0 <= self.config.default_basis_rho_per_second <= 1.0):
            raise ValueError("default_basis_rho_per_second must be in [0, 1].")
        if not (0.0 <= self.config.default_perp_basis_rho_per_second <= 1.0):
            raise ValueError("default_perp_basis_rho_per_second must be in [0, 1].")
        for exchange, rho in self.config.basis_rho_per_second_overrides.items():
            if not (0.0 <= rho <= 1.0):
                raise ValueError(
                    f"basis_rho_per_second_overrides[{exchange!r}] must be in [0, 1]."
                )
        for exchange, rho in self.config.perp_basis_rho_per_second_overrides.items():
            if not (0.0 <= rho <= 1.0):
                raise ValueError(
                    f"perp_basis_rho_per_second_overrides[{exchange!r}] must be in [0, 1]."
                )
        if self.config.anchor_exchange in self.config.perp_exchange_order:
            raise ValueError("anchor_exchange cannot also be configured as a perp exchange.")

    def _select_live_observations(
        self,
        observations: List[VenueBasisObservation],
    ) -> List[VenueBasisObservation]:
        live: List[VenueBasisObservation] = []
        for obs in observations:
            fair_value = _to_float(obs.fair_value)
            local_variance = _to_float(obs.local_variance)
            if not obs.valid:
                continue
            if not math.isfinite(fair_value):
                continue
            if not math.isfinite(local_variance):
                continue
            if obs.age_ms > self.config.stale_cutoff_ms:
                continue
            live.append(
                VenueBasisObservation(
                    name=obs.name,
                    fair_value=fair_value,
                    local_variance=local_variance,
                    age_ms=float(obs.age_ms),
                    valid=obs.valid,
                    venue_kind=obs.venue_kind,
                )
            )
        return live

    def _ensure_state_layout(self, live: List[VenueBasisObservation]) -> None:
        seen_spot = {obs.name for obs in live if obs.venue_kind != "perp"}
        seen_perp = {obs.name for obs in live if obs.venue_kind == "perp"}
        if self.config.anchor_exchange not in self._known_spot_venues:
            self._known_spot_venues.add(self.config.anchor_exchange)
        if seen_spot.issubset(self._known_spot_venues) and seen_perp.issubset(
            self._known_perp_venues
        ):
            return

        self._known_spot_venues.update(seen_spot)
        self._known_perp_venues.update(seen_perp)
        ordered_spot = self._ordered_non_anchor_spot_venues(self._known_spot_venues)
        ordered_perp = self._ordered_perp_venues(self._known_perp_venues)
        ordered_basis = ordered_spot + ordered_perp
        new_indices = {exchange: idx + 1 for idx, exchange in enumerate(ordered_basis)}
        new_kinds = {
            exchange: ("perp" if exchange in set(ordered_perp) else "spot")
            for exchange in ordered_basis
        }

        if self._state is None or self._covariance is None:
            self._basis_state_indices = new_indices
            self._basis_state_kinds = new_kinds
            return

        new_dim = 1 + len(ordered_basis)
        old_state = self._state
        old_covariance = self._covariance
        old_indices = self._basis_state_indices

        new_state = np.zeros(new_dim, dtype=np.float64)
        new_state[0] = old_state[0]
        for exchange, old_idx in old_indices.items():
            new_state[new_indices[exchange]] = old_state[old_idx]

        new_covariance = np.zeros((new_dim, new_dim), dtype=np.float64)
        new_covariance[0, 0] = old_covariance[0, 0]

        for exchange_i, new_i in new_indices.items():
            if exchange_i in old_indices:
                old_i = old_indices[exchange_i]
                new_covariance[new_i, 0] = old_covariance[old_i, 0]
                new_covariance[0, new_i] = old_covariance[0, old_i]
                for exchange_j, new_j in new_indices.items():
                    if exchange_j in old_indices:
                        old_j = old_indices[exchange_j]
                        new_covariance[new_i, new_j] = old_covariance[old_i, old_j]
                    elif new_i == new_j:
                        new_covariance[new_i, new_j] = self.config.initial_basis_variance
            else:
                new_covariance[new_i, new_i] = self.config.initial_basis_variance

        self._state = new_state
        self._covariance = self._stabilize_covariance(new_covariance)
        self._basis_state_indices = new_indices
        self._basis_state_kinds = new_kinds

    def _ordered_non_anchor_spot_venues(self, venues: set[str]) -> list[str]:
        configured = [venue for venue in self.config.venue_order if venue != self.config.anchor_exchange]
        configured_set = set(configured)
        discovered = sorted(
            venue
            for venue in venues
            if venue != self.config.anchor_exchange and venue not in configured_set
        )
        return configured + discovered

    def _ordered_perp_venues(self, venues: set[str]) -> list[str]:
        configured = list(self.config.perp_exchange_order)
        configured_set = set(configured)
        discovered = sorted(venue for venue in venues if venue not in configured_set)
        return configured + discovered

    def _bootstrap_state(self, live: List[VenueBasisObservation]) -> None:
        dim = 1 + len(self._basis_state_indices)
        state = np.zeros(dim, dtype=np.float64)

        anchor_obs = next((obs for obs in live if obs.name == self.config.anchor_exchange), None)
        if anchor_obs is not None:
            state[0] = anchor_obs.fair_value
        else:
            weights = [
                1.0 / self._effective_local_variance(obs)
                for obs in live
            ]
            total_weight = sum(weights)
            if total_weight <= 0.0:
                state[0] = live[0].fair_value
            else:
                state[0] = sum(
                    weight * obs.fair_value for weight, obs in zip(weights, live)
                ) / total_weight

        covariance = np.zeros((dim, dim), dtype=np.float64)
        covariance[0, 0] = max(
            self.config.initial_common_price_variance,
            self.config.covariance_floor,
        )
        for idx in range(1, dim):
            covariance[idx, idx] = max(
                self.config.initial_basis_variance,
                self.config.covariance_floor,
            )

        self._state = state
        self._covariance = covariance

    def _predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dim = len(state)
        transition = np.eye(dim, dtype=np.float64)
        process_noise = np.zeros((dim, dim), dtype=np.float64)
        process_noise[0, 0] = max(
            self.config.common_price_process_var_per_sec * dt,
            self.config.covariance_floor,
        )

        state_prior = state.copy()
        for exchange, idx in self._basis_state_indices.items():
            venue_kind = self._basis_state_kinds.get(exchange, "spot")
            rho = self._basis_rho(exchange, venue_kind, dt)
            transition[idx, idx] = rho
            state_prior[idx] = rho * state[idx]
            process_noise[idx, idx] = max(
                self._basis_process_var_per_sec(exchange, venue_kind) * dt,
                self.config.covariance_floor,
            )

        covariance_prior = (transition @ covariance @ transition.T) + process_noise
        return state_prior, self._stabilize_covariance(covariance_prior)

    def _build_measurement_model(
        self,
        live: List[VenueBasisObservation],
        state_prior: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[VenueBasisObservationDiagnostic],
    ] | None:
        h_rows: list[list[float]] = []
        y_values: list[float] = []
        r_diag: list[float] = []
        diagnostics: list[VenueBasisObservationDiagnostic] = []

        for obs in live:
            row = [0.0] * len(state_prior)
            row[0] = 1.0
            basis_idx = self._basis_state_indices.get(obs.name)
            if basis_idx is not None:
                row[basis_idx] = 1.0

            predicted = float(np.dot(np.asarray(row, dtype=np.float64), state_prior))
            variance = self._effective_local_variance(obs)

            h_rows.append(row)
            y_values.append(obs.fair_value)
            r_diag.append(variance)
            diagnostics.append(
                VenueBasisObservationDiagnostic(
                    name=obs.name,
                    fair_value=obs.fair_value,
                    predicted_fair_value=predicted,
                    innovation=obs.fair_value - predicted,
                    observation_variance=variance,
                    basis_estimate=0.0,
                    basis_stddev=0.0,
                )
            )

        if not h_rows:
            return None
        return (
            np.asarray(h_rows, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
            np.diag(np.asarray(r_diag, dtype=np.float64)),
            diagnostics,
        )

    def _effective_local_variance(self, obs: VenueBasisObservation) -> float:
        # local_variance is already the microstructure heuristic R_j,t, including
        # spread/depth/age effects from variance.py. Keep it as the observation
        # noise input here rather than re-applying an age penalty.
        return max(obs.local_variance, self.config.innovation_var_floor, self.config.local_var_floor)

    def _basis_rho(self, exchange: str, venue_kind: VenueBasisKind, dt: float) -> float:
        if venue_kind == "perp":
            rho_per_sec = self.config.perp_basis_rho_per_second_overrides.get(
                exchange,
                self.config.default_perp_basis_rho_per_second,
            )
        else:
            rho_per_sec = self.config.basis_rho_per_second_overrides.get(
                exchange,
                self.config.default_basis_rho_per_second,
            )
        if dt <= 0.0:
            return 1.0
        return rho_per_sec ** dt

    def _basis_process_var_per_sec(self, exchange: str, venue_kind: VenueBasisKind) -> float:
        if venue_kind == "perp":
            return self.config.perp_basis_process_var_per_sec_overrides.get(
                exchange,
                self.config.default_perp_basis_process_var_per_sec,
            )
        return self.config.basis_process_var_per_sec_overrides.get(
            exchange,
            self.config.default_basis_process_var_per_sec,
        )

    def _basis_estimates(self, state: np.ndarray) -> dict[str, float]:
        estimates = {self.config.anchor_exchange: 0.0}
        for exchange, idx in self._basis_state_indices.items():
            estimates[exchange] = float(state[idx])
        return estimates

    def _basis_stddevs(self, covariance: np.ndarray) -> dict[str, float]:
        stddevs = {self.config.anchor_exchange: 0.0}
        for exchange, idx in self._basis_state_indices.items():
            stddevs[exchange] = math.sqrt(max(float(covariance[idx, idx]), 0.0))
        return stddevs

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        stabilized = 0.5 * (covariance + covariance.T)
        diag = np.diag(stabilized).copy()
        diag = np.maximum(diag, self.config.covariance_floor)
        np.fill_diagonal(stabilized, diag)
        return stabilized

    def _stabilize_innovation_covariance(self, covariance: np.ndarray) -> np.ndarray:
        stabilized = 0.5 * (covariance + covariance.T)
        diag = np.diag(stabilized).copy()
        diag = np.maximum(diag, self.config.innovation_var_floor)
        np.fill_diagonal(stabilized, diag)
        return stabilized


def _to_float(value: float | Decimal) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)
