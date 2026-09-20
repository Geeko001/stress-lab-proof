"""Synthetic data generators — small, seeded, reproducible."""

import numpy as np


def seeded_rng(seed):
    return np.random.default_rng(int(seed))


def random_key_value_pairs(rng, m, dim, value_scale=1.0):
    """Generate m random key/value pairs.

    Keys: standard normal. Values: random one-hot-like codes scaled,
    so retrieval can be scored by nearest-code matching.
    Returns (keys (m,d), values (m,d), codes (m,) int labels).
    """
    keys = rng.standard_normal((m, dim))
    codes = rng.integers(0, m, size=m) if m > 0 else np.array([], dtype=int)
    # Unique code per binding: one-hot in value space (padded/truncated to dim)
    values = np.zeros((m, dim))
    for i in range(m):
        idx = i % dim
        values[i, idx] = value_scale * (1.0 + 0.1 * rng.standard_normal())
    return keys, values, codes


def decaying_update_stream(n_steps, start_magnitude=1.0, decay=0.5, dim=8, seed=0):
    """Stream of updates with geometrically shrinking magnitude."""
    rng = seeded_rng(seed)
    direction = rng.standard_normal((dim,))
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    updates = []
    mag = start_magnitude
    for _ in range(n_steps):
        updates.append(direction * mag)
        mag *= decay
    return updates


def estimate_memory_bytes(n, dim, precision="float64", n_arrays=3):
    """Rough memory estimate for planning safety warnings."""
    itemsize = {"float64": 8, "float32": 4, "float16": 2}.get(precision, 8)
    return n * dim * itemsize * n_arrays
