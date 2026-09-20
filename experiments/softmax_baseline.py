"""Experiment 4: Softmax vs Linear attention baseline benchmark.

Both models face the IDENTICAL marker-recall task (same pregenerated
key/value stream, same marker key, position, and value): a known marker
is injected at a fractional position, the full sequence of length N is
ingested, and the model is queried for the marker at the end.

The comparison contrasts:
  - SoftmaxReference: exact explicit storage of all N bindings
    (storage grows with N).
  - LinearAttentionState (elu+1): fixed-capacity recurrent state
    (storage constant in N).

Metrics per trial: marker recovered (bool), wall-clock ingest+query time
per model, and measured storage footprint in bytes.

Pure NumPy, matching the existing experiment structure. No PyTorch:
per project constraints, heavy frameworks are avoided unless necessary.
"""

import time
import numpy as np

from models.linear_attention import LinearAttentionState
from models.softmax_attention import SoftmaxReference
from utils.generators import seeded_rng
from utils.metrics import mean_std


def run_single_trial(N, dim, seed, position_frac=0.5, feature_map="elu+1",
                     dtype=np.float64, distractor_scale=0.01,
                     marker_value=739281.0):
    """One head-to-head trial on a shared pregenerated stream."""
    rng = seeded_rng(seed)
    N = int(N)
    marker_pos = int(N * position_frac)
    marker_key = rng.standard_normal(dim)
    marker_vec = np.zeros(dim)
    marker_vec[0] = marker_value / 1e6  # same encoding as Exp 2

    keys = rng.standard_normal((N, dim))
    vals = rng.standard_normal((N, dim)) * distractor_scale
    keys[marker_pos] = marker_key
    vals[marker_pos] = marker_vec

    # --- linear attention (streaming, fixed state) ---
    t0 = time.perf_counter()
    lin = LinearAttentionState(dim, feature_map=feature_map, dtype=dtype)
    for k, v in zip(keys, vals):
        lin.update(k, v)
    out_lin = lin.query(marker_key)
    t_lin = time.perf_counter() - t0
    lin_bytes = int(lin.S.nbytes + lin.z.nbytes)

    # --- softmax reference (explicit storage of all N bindings) ---
    t0 = time.perf_counter()
    sm = SoftmaxReference()
    for k, v in zip(keys, vals):
        sm.update(k, v)
    out_sm = sm.query(marker_key)
    t_sm = time.perf_counter() - t0
    sm_bytes = int(sum(k.nbytes + v.nbytes for k, v in zip(sm.keys, sm.values)))

    def _recovered(out):
        out = np.asarray(out, dtype=np.float64)
        return bool(int(np.argmax(out)) == 0 and float(out[0]) > 0)

    return {
        "N": N,
        "position_frac": float(position_frac),
        "linear_recovered": _recovered(out_lin),
        "softmax_recovered": _recovered(out_sm),
        "linear_time_s": float(t_lin),
        "softmax_time_s": float(t_sm),
        "linear_bytes": lin_bytes,
        "softmax_bytes": sm_bytes,
        "linear_state_norm": lin.state_norm,
    }


def run_softmax_baseline(N_values=(128, 512, 1024, 4096), dim=16, trials=10,
                         seed=42, position_fracs=(0.1, 0.5, 0.9),
                         feature_map="elu+1", dtype=np.float64,
                         progress_cb=None):
    """Aggregate head-to-head trials. Returns list of row dicts (one per N)."""
    rows = []
    total = len(N_values)
    for i, N in enumerate(N_values):
        lin_accs, sm_accs, lin_times, sm_times = [], [], [], []
        lin_b, sm_b, norms = [], [], []
        for t in range(trials):
            for pf in position_fracs:
                r = run_single_trial(int(N), dim, seed=int(seed) + t,
                                     position_frac=pf, feature_map=feature_map,
                                     dtype=dtype)
                lin_accs.append(1.0 if r["linear_recovered"] else 0.0)
                sm_accs.append(1.0 if r["softmax_recovered"] else 0.0)
                lin_times.append(r["linear_time_s"])
                sm_times.append(r["softmax_time_s"])
                lin_b.append(r["linear_bytes"])
                sm_b.append(r["softmax_bytes"])
                norms.append(r["linear_state_norm"])
        lin_mean, lin_std = mean_std(lin_accs)
        sm_mean, sm_std = mean_std(sm_accs)
        rows.append({
            "N": int(N),
            "dim": int(dim),
            "trials": int(trials * len(position_fracs)),
            "seed": int(seed),
            "feature_map": feature_map,
            "precision": np.dtype(dtype).name,
            "linear_accuracy": lin_mean,
            "linear_std": lin_std,
            "softmax_accuracy": sm_mean,
            "softmax_std": sm_std,
            "linear_time_mean_s": float(np.mean(lin_times)),
            "softmax_time_mean_s": float(np.mean(sm_times)),
            "linear_bytes": int(lin_b[0]),
            "softmax_bytes": int(sm_b[0]),
            "linear_state_norm_mean": float(np.mean(norms)),
            "experiment": "softmax_baseline",
        })
        if progress_cb is not None:
            progress_cb((i + 1) / total)
    return rows
