"""Shared drift Kalman predictor (script-local, not yet promoted to src/).

State = regression coefficient vector β. Emits drift_hat from standardized
microstructure features and walks β when realized drift arrives.

**Probation status:** walk-forward showed directional accuracy ≤ 0.5 on 3/4
venues on a 28-min sample. This lives under scripts/ until multi-day data
either validates it (→ promote to src/icarus/strategy/) or refutes it (→ delete).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = (
    "microprice-mid",
    "depth_imbalance",
    "vol_bps",
    "trade_net_flow",
    "trade_total_size",
)


@dataclass
class DriftPredictor:
    beta: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    r: float
    mean: np.ndarray
    std: np.ndarray

    clip_sigma: float = 5.0

    def standardize(self, features: tuple[float, ...]) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        z = np.clip((x - self.mean) / self.std, -self.clip_sigma, self.clip_sigma)
        return np.concatenate([[1.0], z])

    def predict(self, xi: np.ndarray) -> float:
        return float(xi @ self.beta)

    def update(self, xi: np.ndarray, y: float) -> None:
        P_pred = self.P + self.Q
        S = float(xi @ P_pred @ xi + self.r)
        innovation = y - xi @ self.beta
        K = (P_pred @ xi) / S
        self.beta = self.beta + K * innovation
        self.P = P_pred - np.outer(K, xi) @ P_pred
