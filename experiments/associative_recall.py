"""Experiment 1: high-density associative recall.

Stores m synthetic key-value bindings, then queries each key and
measures retrieval accuracy for linear attention vs softmax reference.
Pure functions (no Streamlit) for testability.
"""

import time
import numpy as np

from models.linear_attention import LinearAttentionState
from models.softmax_attention import SoftmaxReference
from utils.generators import seeded_rng
from utils.metrics import mean_std


def _nearest_code_accuracy(outputs, values, tol_factor=0.5):
    """Score retrieval by nearest stored value (code matching).

    Robust to scale differences between linear/softmax outputs.
    """
    outputs = np.asarray(outputs)
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    # pairwise distances outputs[i] vs values[j]
    d2 = ((outputs[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
    pred = d2.argmin(axis=1)
    true = np.arange(len(values))
    return float(np.mean(pred == true))


def run_single_trial(m, dim, seed, feature_map="elu+1", dtype=np.float64):
    rng = seeded_rng(seed)
    keys = rng.standard_normal((m, dim))
    # distinct value codes: one-hot positions cycling through dim
    values = np.zeros((m, dim))
    for i in range(m):
        values[i, i % dim] = 1.0 + 0.1 * rng.standard_normal()

    # Linear
    lin = LinearAttentionState(dim, feature_map=feature_map, dtype=dtype)
    for k, v in zip(keys, values):
        lin.update(k, v)
    lin_out = np.stack([lin.query(k) for k in keys])
    lin_acc = _nearest_code_accuracy(lin_out, values)

    # Softmax reference
    sm = SoftmaxReference()
    for k, v in zip(keys, values):
        sm.update(k, v)
    sm_out = np.stack([sm.query(k) for k in keys])
    sm_acc = _nearest_code_accuracy(sm_out, values)

    return {
        "linear_accuracy": lin_acc,
        "softmax_accuracy": sm_acc,
        "state_norm": lin.state_norm,
    }


def run_associative_recall(m_values, dim=32, trials=10, seed=0,
                           feature_map="elu+1", dtype=np.float64,
                           progress_cb=None):
    """Sweep binding counts. Returns list of row dicts (one per m)."""
    rows = []
    for i, m in enumerate(m_values):
        t0 = time.perf_counter()
        lin_accs, sm_accs, norms = [], [], []
        for t in range(trials):
            r = run_single_trial(int(m), dim, seed=int(seed) + t,
                                 feature_map=feature_map, dtype=dtype)
            lin_accs.append(r["linear_accuracy"])
            sm_accs.append(r["softmax_accuracy"])
            norms.append(r["state_norm"])
        lin_mean, lin_std = mean_std(lin_accs)
        sm_mean, sm_std = mean_std(sm_accs)
        rows.append({
            "m": int(m),
            "dim": int(dim),
            "trials": int(trials),
            "feature_map": feature_map,
            "precision": np.dtype(dtype).name,
            "linear_mean": lin_mean,
            "linear_std": lin_std,
            "softmax_mean": sm_mean,
            "softmax_std": sm_std,
            "state_norm_mean": float(np.mean(norms)),
            "runtime_s": time.perf_counter() - t0,
        })
        if progress_cb is not None:
            progress_cb((i + 1) / len(m_values))
    return rows


def run_dimensionality_sweep(m_values, dim_values=(8, 16, 32, 64), trials=10,
                             seed=0, feature_map="elu+1", dtype=np.float64,
                             progress_cb=None):
    """Sweep binding counts across multiple state dimensions.

    Returns a flat list of row dicts (one per (d, m)) in the same format
    as run_associative_recall, so curves can be overlaid on one plot to
    visualise capacity expansion as d increases.
    """
    all_rows = []
    dim_values = [int(d) for d in dim_values]
    for di, d in enumerate(dim_values):
        all_rows.extend(
            run_associative_recall(list(m_values), dim=d, trials=trials,
                                   seed=seed, feature_map=feature_map,
                                   dtype=dtype))
        if progress_cb is not None:
            progress_cb((di + 1) / len(dim_values))
    return all_rows
