"""Shared metric helpers. All descriptive — no proof claims."""

import numpy as np


def exact_match_accuracy(retrieved, expected, tol=0.1):
    """Fraction of rows retrieved within tolerance (nearest-code match)."""
    retrieved = np.asarray(retrieved)
    expected = np.asarray(expected)
    if len(expected) == 0:
        return 0.0
    dists = np.linalg.norm(retrieved - expected, axis=1)
    return float(np.mean(dists < tol))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else (float(values.mean()), 0.0)


def candidate_breakpoint_region(x_vals, y_vals, drop_threshold=0.15):
    """Transparent heuristic: first x where accuracy drops >= threshold
    below the max observed so far. Returns index or None.

    This is a DESCRIPTIVE heuristic only, never proof of a breakpoint.
    """
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    if not y_vals:
        return None
    best = y_vals[0]
    for i, y in enumerate(y_vals):
        if y > best:
            best = y
        if best - y >= drop_threshold:
            return i
    return None


def relative_change(old, new):
    old = float(old)
    new = float(new)
    denom = abs(old) if abs(old) > 1e-300 else 1e-300
    return abs(new - old) / denom
