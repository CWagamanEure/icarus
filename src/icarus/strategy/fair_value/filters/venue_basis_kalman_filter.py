from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional


@dataclass
class VenueBasisObservation:
    name: str
    fair_value: float | Decimal
    local_variance: float | Decimal
    age_ms: float = 0.0
    valid: bool = True


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
    observation_diagnostics: List[VenueBasisObservationDiagnostic] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class VenueBasisKalmanConfig:
    anchor_exchange: str
    venue_order: tuple[str, ...] = ()
    require_anchor_observation: bool = True
    # The common latent price should be materially more mobile than venue-basis
    # states; otherwise shared market moves get misattributed to basis drift.
    common_price_process_var_per_sec: float = 2.0
    default_basis_process_var_per_sec: float = 0.01
    # Persistence over one second for the AR(1) basis process. The per-update
    # transition coefficient is computed as rho_per_second ** dt_seconds.
    default_basis_rho_per_second: float = 0.98
    basis_process_var_per_sec_overrides: dict[str, float] = field(default_factory=dict)
    basis_rho_per_second_overrides: dict[str, float] = field(default_factory=dict)
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

        self._state: list[float] | None = None
        self._covariance: list[list[float]] | None = None
        self._known_venues: set[str] = set()
        self._basis_state_indices: dict[str, int] = {}
        self.last_timestamp_s: float | None = None

        if initial_price is not None:
            self._state = [float(initial_price)]
            self._covariance = [[max(config.initial_common_price_variance, config.covariance_floor)]]

    @property
    def basis_state_indices(self) -> Dict[str, int]:
        return dict(self._basis_state_indices)

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
        innovation = _vector_sub(y_vector, _matvec(h_matrix, state_prior))
        s_matrix = _matrix_add(
            _matrix_multiply(_matrix_multiply(h_matrix, covariance_prior), _transpose(h_matrix)),
            r_matrix,
        )
        s_inv = _invert_matrix(s_matrix, self.config.innovation_var_floor)
        kalman_gain = _matrix_multiply(
            _matrix_multiply(covariance_prior, _transpose(h_matrix)),
            s_inv,
        )

        state_post = _vector_add(state_prior, _matvec(kalman_gain, innovation))
        identity = _identity(len(state_prior))
        kh = _matrix_multiply(kalman_gain, h_matrix)
        i_minus_kh = _matrix_sub(identity, kh)
        covariance_post = _matrix_add(
            _matrix_multiply(
                _matrix_multiply(i_minus_kh, covariance_prior),
                _transpose(i_minus_kh),
            ),
            _matrix_multiply(
                _matrix_multiply(kalman_gain, r_matrix),
                _transpose(kalman_gain),
            ),
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
            common_price_stddev=math.sqrt(max(covariance_post[0][0], 0.0)),
            anchor_exchange=self.config.anchor_exchange,
            active_venues=[diag.name for diag in diagnostics],
            basis_estimates=basis_estimates,
            basis_stddevs=basis_stddevs,
            observation_variances=observation_variances,
            innovations=innovations,
            state_covariance_trace=sum(covariance_post[i][i] for i in range(len(covariance_post))),
            basis_state_indices=self.basis_state_indices,
            observation_diagnostics=diagnostics,
        )

    def _validate_config(self) -> None:
        if not self.config.anchor_exchange:
            raise ValueError("anchor_exchange must be non-empty.")
        if not (0.0 <= self.config.default_basis_rho_per_second <= 1.0):
            raise ValueError("default_basis_rho_per_second must be in [0, 1].")
        for exchange, rho in self.config.basis_rho_per_second_overrides.items():
            if not (0.0 <= rho <= 1.0):
                raise ValueError(
                    f"basis_rho_per_second_overrides[{exchange!r}] must be in [0, 1]."
                )

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
                )
            )
        return live

    def _ensure_state_layout(self, live: List[VenueBasisObservation]) -> None:
        seen_names = {obs.name for obs in live}
        if self.config.anchor_exchange not in self._known_venues:
            self._known_venues.add(self.config.anchor_exchange)
        if seen_names.issubset(self._known_venues):
            return

        self._known_venues.update(seen_names)
        ordered_non_anchor = self._ordered_non_anchor_venues(self._known_venues)
        new_indices = {exchange: idx + 1 for idx, exchange in enumerate(ordered_non_anchor)}

        if self._state is None or self._covariance is None:
            self._basis_state_indices = new_indices
            return

        new_dim = 1 + len(ordered_non_anchor)
        old_state = self._state
        old_covariance = self._covariance
        old_indices = self._basis_state_indices

        new_state = [0.0] * new_dim
        new_state[0] = old_state[0]
        for exchange, old_idx in old_indices.items():
            new_state[new_indices[exchange]] = old_state[old_idx]

        new_covariance = [[0.0] * new_dim for _ in range(new_dim)]
        new_covariance[0][0] = old_covariance[0][0]

        for exchange_i, new_i in new_indices.items():
            if exchange_i in old_indices:
                old_i = old_indices[exchange_i]
                new_covariance[new_i][0] = old_covariance[old_i][0]
                new_covariance[0][new_i] = old_covariance[0][old_i]
                for exchange_j, new_j in new_indices.items():
                    if exchange_j in old_indices:
                        old_j = old_indices[exchange_j]
                        new_covariance[new_i][new_j] = old_covariance[old_i][old_j]
                    elif new_i == new_j:
                        new_covariance[new_i][new_j] = self.config.initial_basis_variance
            else:
                new_covariance[new_i][new_i] = self.config.initial_basis_variance

        self._state = new_state
        self._covariance = self._stabilize_covariance(new_covariance)
        self._basis_state_indices = new_indices

    def _ordered_non_anchor_venues(self, venues: set[str]) -> list[str]:
        configured = [venue for venue in self.config.venue_order if venue != self.config.anchor_exchange]
        configured_set = set(configured)
        discovered = sorted(
            venue
            for venue in venues
            if venue != self.config.anchor_exchange and venue not in configured_set
        )
        return configured + discovered

    def _bootstrap_state(self, live: List[VenueBasisObservation]) -> None:
        dim = 1 + len(self._basis_state_indices)
        state = [0.0] * dim

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

        covariance = [[0.0] * dim for _ in range(dim)]
        covariance[0][0] = max(
            self.config.initial_common_price_variance,
            self.config.covariance_floor,
        )
        for idx in range(1, dim):
            covariance[idx][idx] = max(
                self.config.initial_basis_variance,
                self.config.covariance_floor,
            )

        self._state = state
        self._covariance = covariance

    def _predict(
        self,
        state: list[float],
        covariance: list[list[float]],
        dt: float,
    ) -> tuple[list[float], list[list[float]]]:
        dim = len(state)
        transition = _identity(dim)
        process_noise = [[0.0] * dim for _ in range(dim)]
        process_noise[0][0] = max(
            self.config.common_price_process_var_per_sec * dt,
            self.config.covariance_floor,
        )

        state_prior = list(state)
        for exchange, idx in self._basis_state_indices.items():
            rho = self._basis_rho(exchange, dt)
            transition[idx][idx] = rho
            state_prior[idx] = rho * state[idx]
            process_noise[idx][idx] = max(
                self._basis_process_var_per_sec(exchange) * dt,
                self.config.covariance_floor,
            )

        covariance_prior = _matrix_add(
            _matrix_multiply(_matrix_multiply(transition, covariance), _transpose(transition)),
            process_noise,
        )
        return state_prior, self._stabilize_covariance(covariance_prior)

    def _build_measurement_model(
        self,
        live: List[VenueBasisObservation],
        state_prior: list[float],
    ) -> tuple[
        list[list[float]],
        list[float],
        list[list[float]],
        list[VenueBasisObservationDiagnostic],
    ] | None:
        h_matrix: list[list[float]] = []
        y_vector: list[float] = []
        r_diag: list[float] = []
        diagnostics: list[VenueBasisObservationDiagnostic] = []

        for obs in live:
            row = [0.0] * len(state_prior)
            row[0] = 1.0
            basis_idx = self._basis_state_indices.get(obs.name)
            if basis_idx is not None:
                row[basis_idx] = 1.0

            predicted = _dot(row, state_prior)
            variance = self._effective_local_variance(obs)

            h_matrix.append(row)
            y_vector.append(obs.fair_value)
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

        if not h_matrix:
            return None
        return h_matrix, y_vector, _diag(r_diag), diagnostics

    def _effective_local_variance(self, obs: VenueBasisObservation) -> float:
        # local_variance is already the microstructure heuristic R_j,t, including
        # spread/depth/age effects from variance.py. Keep it as the observation
        # noise input here rather than re-applying an age penalty.
        return max(obs.local_variance, self.config.innovation_var_floor, self.config.local_var_floor)

    def _basis_rho(self, exchange: str, dt: float) -> float:
        rho_per_sec = self.config.basis_rho_per_second_overrides.get(
            exchange,
            self.config.default_basis_rho_per_second,
        )
        if dt <= 0.0:
            return 1.0
        return rho_per_sec ** dt

    def _basis_process_var_per_sec(self, exchange: str) -> float:
        return self.config.basis_process_var_per_sec_overrides.get(
            exchange,
            self.config.default_basis_process_var_per_sec,
        )

    def _basis_estimates(self, state: list[float]) -> dict[str, float]:
        estimates = {self.config.anchor_exchange: 0.0}
        for exchange, idx in self._basis_state_indices.items():
            estimates[exchange] = state[idx]
        return estimates

    def _basis_stddevs(self, covariance: list[list[float]]) -> dict[str, float]:
        stddevs = {self.config.anchor_exchange: 0.0}
        for exchange, idx in self._basis_state_indices.items():
            stddevs[exchange] = math.sqrt(max(covariance[idx][idx], 0.0))
        return stddevs

    def _stabilize_covariance(self, covariance: list[list[float]]) -> list[list[float]]:
        dim = len(covariance)
        stabilized = [[0.0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                stabilized[i][j] = 0.5 * (covariance[i][j] + covariance[j][i])
            stabilized[i][i] = max(stabilized[i][i], self.config.covariance_floor)
        return stabilized


def _to_float(value: float | Decimal) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _diag(values: list[float]) -> list[list[float]]:
    return [
        [values[i] if i == j else 0.0 for j in range(len(values))]
        for i in range(len(values))
    ]


def _identity(dim: int) -> list[list[float]]:
    return [
        [1.0 if i == j else 0.0 for j in range(dim)]
        for i in range(dim)
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    b_t = _transpose(b)
    return [[_dot(row, col) for col in b_t] for row in a]


def _matrix_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[i][j] + b[i][j] for j in range(len(a[i]))]
        for i in range(len(a))
    ]


def _matrix_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [a[i][j] - b[i][j] for j in range(len(a[i]))]
        for i in range(len(a))
    ]


def _vector_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def _invert_matrix(matrix: list[list[float]], floor: float) -> list[list[float]]:
    dim = len(matrix)
    augmented = [
        list(row) + identity_row
        for row, identity_row in zip(matrix, _identity(dim))
    ]

    for col in range(dim):
        pivot_row = max(range(col, dim), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot_row][col]) < floor:
            augmented[pivot_row][col] = floor
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        if abs(pivot) < floor:
            pivot = floor
            augmented[col][col] = pivot

        inv_pivot = 1.0 / pivot
        for j in range(2 * dim):
            augmented[col][j] *= inv_pivot

        for row in range(dim):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(2 * dim):
                augmented[row][j] -= factor * augmented[col][j]

    return [row[dim:] for row in augmented]
