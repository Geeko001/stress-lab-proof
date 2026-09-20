"""Experiment 2: sequence-length scaling with marker recall.

Streams N synthetic vectors through the recurrent state, with a known
marker value injected at a given fractional position. At the end, query
for the marker and measure recoverability. Streaming — no full history.
"""

import time
import numpy as np

from models.linear_attention import LinearAttentionState
from utils.generators import seeded_rng


def run_single_length(N, dim, seed, position_frac=0.1, feature_map="elu+1",
                      dtype=np.float64, marker_value=739281.0,
                      distractor_scale=0.01):
    rng = seeded_rng(seed)
    state = LinearAttentionState(dim, feature_map=feature_map, dtype=dtype)
    marker_pos = int(N * position_frac)
    marker_key = rng.standard_normal(dim)
    # marker value: constant vector encoding the special value (scaled)
    marker_vec = np.zeros(dim)
    marker_vec[0] = marker_value / 1e6  # keep magnitudes sane

    t0 = time.perf_counter()
    for t in range(N):
        if t == marker_pos:
            state.update(marker_key, marker_vec)
        else:
            k = rng.standard_normal(dim)
            # small distractors so the marker stands out at short N but
            # washes out as N grows — the length-degradation curve of interest
            v = rng.standard_normal(dim) * distractor_scale
            state.update(k, v)
    out = state.query(marker_key)
    runtime = time.perf_counter() - t0

    # recoverability: is output's argmax at index 0 and positive?
    pred_idx = int(np.argmax(out))
    recovered = bool(pred_idx == 0 and out[0] > 0)
    # relative error on the marker component
    err = abs(float(out[0]) - float(marker_vec[0])) / (abs(float(marker_vec[0])) + 1e-12)
    return {
        "N": int(N),
        "position_frac": float(position_frac),
        "marker_pos": marker_pos,
        "recovered": recovered,
        "retrieval_error": float(err),
        "state_norm": state.state_norm,
        "state_magnitude": state.state_magnitude,
        "runtime_s": runtime,
    }


def run_length_sweep(N_values, dim=16, trials=5, seed=0, position_fracs=(0.1, 0.5, 0.9),
                     feature_map="elu+1", dtype=np.float64, progress_cb=None):
    rows = []
    total = len(N_values) * len(position_fracs)
    done = 0
    for N in N_values:
        for pf in position_fracs:
            recs, errs, norms, mags, runtimes = [], [], [], [], []
            for t in range(trials):
                r = run_single_length(int(N), dim, seed=int(seed) + t,
                                      position_frac=pf, feature_map=feature_map,
                                      dtype=dtype)
                recs.append(1.0 if r["recovered"] else 0.0)
                errs.append(r["retrieval_error"])
                norms.append(r["state_norm"])
                mags.append(r["state_magnitude"])
                runtimes.append(r["runtime_s"])
            rows.append({
                "N": int(N),
                "position_frac": float(pf),
                "dim": int(dim),
                "trials": int(trials),
                "accuracy": float(np.mean(recs)),
                "retrieval_error_mean": float(np.mean(errs)),
                "state_norm_mean": float(np.mean(norms)),
                "state_magnitude_mean": float(np.mean(mags)),
                "runtime_s": float(sum(runtimes)),
                "precision": np.dtype(dtype).name,
                "feature_map": feature_map,
            })
            done += 1
            if progress_cb is not None:
                progress_cb(done / total)
    return rows
