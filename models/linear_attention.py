"""Minimal generic linear-attention recurrent state.

Implements:
    S_t = S_{t-1} + phi(k_t) v_t^T
    z_t = z_{t-1} + phi(k_t)

Retrieval:
    v_hat = S^T phi(q) / (z^T phi(q))

This is a mechanism-validation implementation, NOT a production
Transformer / RetNet / Mamba / RWKV model. See methodology disclaimer.
"""

import numpy as np


def elu_plus_one(x):
    """phi(x) = elu(x) + 1  (standard positive feature map)."""
    return np.where(x >= 0, x + 1.0, np.exp(x))


def relu_map(x):
    """phi(x) = relu(x) + epsilon (simple positive map)."""
    return np.maximum(x, 0.0) + 1e-6


def identity_pos_map(x):
    """phi(x) = exp(x) clipped — kept small for CPU stability."""
    return np.exp(np.clip(x, -5.0, 5.0))


# E[elu(X) + 1] for X ~ N(0,1): 1 + 1/sqrt(2*pi) + (e^0.5 * Phi(-1) - 0.5).
# Subtracting it centers the features, removing the dominant common-mode
# overlap that drives collisions under elu+1 (see orthkeys finding).
ELU1_MEAN = 1.1605


def centered_elu_map(x):
    """phi(x) = elu(x) + 1 - E[elu+1]: zero-mean features under N(0,1) keys."""
    return elu_plus_one(x) - ELU1_MEAN


FEATURE_MAPS = {
    "elu+1": elu_plus_one,
    "relu": relu_map,
    "exp": identity_pos_map,
    "centered_elu": centered_elu_map,
}


class LinearAttentionState:
    """Fixed-capacity recurrent KV state."""

    def __init__(self, dim, feature_map="elu+1", dtype=np.float64):
        if feature_map not in FEATURE_MAPS:
            raise ValueError(f"Unknown feature map: {feature_map}")
        self.dim = int(dim)
        self.phi = FEATURE_MAPS[feature_map]
        self.feature_map_name = feature_map
        self.dtype = np.dtype(dtype)
        self.reset()

    def reset(self):
        self.S = np.zeros((self.dim, self.dim), dtype=self.dtype)
        self.z = np.zeros((self.dim,), dtype=self.dtype)
        self.count = 0

    def update(self, key, value):
        """Single streaming update. key: (dim,), value: (dim,)."""
        k = np.asarray(key, dtype=self.dtype).reshape(-1)
        v = np.asarray(value, dtype=self.dtype).reshape(-1)
        fk = self.phi(k.astype(np.float64)).astype(self.dtype)
        # streaming: no history stored
        self.S += np.outer(fk, v)
        self.z += fk
        self.count += 1

    def query(self, query):
        """Retrieve value for query vector. Returns (dim,) array."""
        q = np.asarray(query, dtype=self.dtype).reshape(-1)
        fq = self.phi(q.astype(np.float64)).astype(self.dtype)
        denom = float(np.dot(self.z, fq))
        if abs(denom) < 1e-12:
            return np.zeros((self.dim,), dtype=self.dtype)
        return (self.S.T @ fq) / denom

    @property
    def state_norm(self):
        return float(np.linalg.norm(self.S.astype(np.float64)))

    @property
    def state_magnitude(self):
        return float(np.abs(self.S.astype(np.float64)).sum())
