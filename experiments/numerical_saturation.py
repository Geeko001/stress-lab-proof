"""Experiment 3: numerical / floating-point state saturation.

Two parts:
  A. shrinking-update stream: S_t = S_{t-1} + u_t with |u_t| decaying;
     measures when updates become numerically ineffective per precision.
  B. decay stream: S_t = gamma * S_{t-1} + u_t for several gammas.
"""

import numpy as np

from utils.generators import seeded_rng
from utils.metrics import relative_change


def shrinking_update_experiment(n_steps=60, start_magnitude=1.0, decay=0.5,
                                dim=8, precisions=("float64", "float32", "float16"),
                                seed=0):
    """Returns rows: one per (precision, timestep)."""
    rows = []
    base_rng = seeded_rng(seed)
    direction = base_rng.standard_normal(dim)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    mags = [start_magnitude * (decay ** t) for t in range(n_steps)]

    for prec in precisions:
        try:
            dt = np.dtype(prec)
        except TypeError:
            continue  # e.g. bfloat16 unsupported
        state = np.zeros(dim, dtype=dt)
        ineffective = 0
        for t, mag in enumerate(mags):
            u = (direction * mag).astype(dt)
            old = state.copy()
            state = (state + u).astype(dt)
            change = float(np.abs(state.astype(np.float64) - old.astype(np.float64)).sum())
            rel = relative_change(float(np.abs(old.astype(np.float64)).sum()), float(np.abs(state.astype(np.float64)).sum()))
            is_ineffective = (change == 0.0)
            ineffective += int(is_ineffective)
            rows.append({
                "timestep": t,
                "precision": prec,
                "update_magnitude": float(mag),
                "state_magnitude": float(np.abs(state.astype(np.float64)).sum()),
                "effective_change": change,
                "relative_change": float(rel),
                "ineffective": bool(is_ineffective),
                "ineffective_cumulative": ineffective,
                "dim": dim,
            })
    return rows


def decay_experiment(n_steps=2000, gamma_values=(0.99, 0.999, 0.9999),
                     dim=8, update_scale=0.01, precision="float64", seed=0):
    """Constant-magnitude updates with exponential decay. Returns summary rows per gamma."""
    rng = seeded_rng(seed)
    rows = []
    for gamma in gamma_values:
        dt = np.dtype(precision)
        state = np.zeros(dim, dtype=np.float64)
        for _ in range(n_steps):
            u = rng.standard_normal(dim) * update_scale
            state = gamma * state + u
        rows.append({
            "gamma": float(gamma),
            "n_steps": int(n_steps),
            "precision": precision,
            "dim": int(dim),
            "final_state_magnitude": float(np.abs(state).sum()),
            "final_state_norm": float(np.linalg.norm(state)),
            # theoretical steady-state magnitude scale ~ update_scale*sqrt(dim)/(1-gamma)
            "theoretical_scale": float(update_scale * np.sqrt(dim) / (1.0 - gamma)),
        })
    return rows
