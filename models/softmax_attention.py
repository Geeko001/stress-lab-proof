"""Tiny exact-attention reference baseline.

Retains individual key-value bindings and performs softmax-weighted
retrieval. Conceptual reference only — NOT a production Transformer.
Its purpose is to contrast explicit storage against fixed-state
compressed storage.
"""

import numpy as np


class SoftmaxReference:
    """Exact-ish explicit-storage baseline (small scale only)."""

    def __init__(self, scale=None):
        self.keys = []
        self.values = []
        self.scale = scale

    def reset(self):
        self.keys = []
        self.values = []

    def update(self, key, value):
        self.keys.append(np.asarray(key, dtype=np.float64).reshape(-1).copy())
        self.values.append(np.asarray(value, dtype=np.float64).reshape(-1).copy())

    def query(self, query):
        if not self.keys:
            raise ValueError("No bindings stored.")
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        K = np.stack(self.keys)  # (m, d)
        V = np.stack(self.values)  # (m, d)
        d = K.shape[1]
        scale = self.scale if self.scale is not None else 1.0 / np.sqrt(d)
        scores = (K @ q) * scale
        scores = scores - scores.max()  # numerical stability
        weights = np.exp(scores)
        weights = weights / weights.sum()
        return weights @ V

    def __len__(self):
        return len(self.keys)
