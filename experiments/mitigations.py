"""Mitigation probes for paper Table 2.

All probes run the same marker-recall task as Exp 2 (shared pregenerated
stream) so mitigation effects are directly comparable to the baseline:

- hybrid window: exact storage for the last w bindings + linear state
  for everything older (local-softmax hybrid row).
- chunked reset: state reset every C steps, bounding drift at the cost
  of cross-boundary information (chunked-reset row).
- selective decay: oracle salience latch (gamma=1 from the marker update
  onward, gamma_bg elsewhere) vs fixed gamma — upper bound on what learned
  selective gating (Mamba-style) can achieve (dynamic-decay row).
  A one-step gate cannot work: subsequent decay erases the update anyway;
  retention must persist, which is the mechanism tested here.
- mixed precision: fp16 compute with fp32 vs fp16 state accumulation
  (mixed-precision row).
"""

import numpy as np

from models.linear_attention import LinearAttentionState
from models.softmax_attention import SoftmaxReference
from utils.generators import seeded_rng
from utils.metrics import mean_std


def _make_stream(N, dim, seed, position_frac=0.1, distractor_scale=0.01,
                 marker_value=739281.0):
    rng = seeded_rng(seed)
    N = int(N)
    marker_pos = int(N * position_frac)
    marker_key = rng.standard_normal(dim)
    marker_vec = np.zeros(dim)
    marker_vec[0] = marker_value / 1e6
    keys = rng.standard_normal((N, dim))
    vals = rng.standard_normal((N, dim)) * distractor_scale
    keys[marker_pos] = marker_key
    vals[marker_pos] = marker_vec
    return keys, vals, marker_key, marker_pos


def _recovered(out):
    out = np.asarray(out, dtype=np.float64)
    return bool(int(np.argmax(out)) == 0 and float(out[0]) > 0)


# ------------------------------------------------------------- hybrid window
def run_hybrid_trial(N, dim, seed, window=32, position_frac=0.1,
                     feature_map="elu+1"):
    keys, vals, marker_key, marker_pos = _make_stream(N, dim, seed,
                                                      position_frac)
    lin = LinearAttentionState(dim, feature_map=feature_map, dtype=np.float64)
    for k, v in zip(keys, vals):
        lin.update(k, v)
    if 0 < window and marker_pos >= N - window:
        sm = SoftmaxReference()
        for k, v in zip(keys[N - window:], vals[N - window:]):
            sm.update(k, v)
        return {"recovered": _recovered(sm.query(marker_key)), "path": "exact-window"}
    return {"recovered": _recovered(lin.query(marker_key)), "path": "linear-state"}


def run_hybrid_sweep(windows=(0, 8, 32, 128, 1024), N=1024, dim=16, trials=8,
                     seed=42, position_fracs=(0.1, 0.9), progress_cb=None):
    rows = []
    total = len(windows) * len(position_fracs)
    done = 0
    for w in windows:
        for pf in position_fracs:
            accs = [1.0 if run_hybrid_trial(N, dim, int(seed) + t, window=w,
                                            position_frac=pf)["recovered"] else 0.0
                    for t in range(trials)]
            mean, std = mean_std(accs)
            rows.append({"window": int(w), "N": int(N), "dim": int(dim),
                         "position_frac": float(pf), "trials": int(trials),
                         "accuracy": mean, "std": std,
                         "experiment": "hybrid_window"})
            done += 1
            if progress_cb is not None:
                progress_cb(done / total)
    return rows


# ------------------------------------------------------------- chunked reset
def run_chunked_trial(N, dim, seed, chunk=1024, position_frac=0.1,
                      feature_map="elu+1"):
    keys, vals, marker_key, marker_pos = _make_stream(N, dim, seed,
                                                      position_frac)
    st = LinearAttentionState(dim, feature_map=feature_map, dtype=np.float64)
    max_norm = 0.0
    for t, (k, v) in enumerate(zip(keys, vals)):
        if t > 0 and t % chunk == 0:
            st.reset()
        st.update(k, v)
        max_norm = max(max_norm, st.state_norm)
    wiped = (marker_pos // chunk) != ((N - 1) // chunk)
    return {"recovered": _recovered(st.query(marker_key)),
            "wiped": bool(wiped), "max_norm": max_norm}


def run_chunked_sweep(chunks=(256, 1024, 4096), N=4096, dim=16, trials=8,
                      seed=42, position_frac=0.1, progress_cb=None):
    rows = []
    for i, c in enumerate(chunks):
        accs, norms = [], []
        for t in range(trials):
            r = run_chunked_trial(N, dim, int(seed) + t, chunk=c,
                                  position_frac=position_frac)
            accs.append(1.0 if r["recovered"] else 0.0)
            norms.append(r["max_norm"])
        mean, std = mean_std(accs)
        rows.append({"chunk": int(c), "N": int(N), "dim": int(dim),
                     "trials": int(trials), "accuracy": mean, "std": std,
                     "max_state_norm": float(np.mean(norms)),
                     "experiment": "chunked_reset"})
        if progress_cb is not None:
            progress_cb((i + 1) / len(chunks))
    return rows


# ---------------------------------------------------------- selective decay
def run_selective_trial(N, dim, seed, mode="fixed-0.99", gamma_bg=0.99,
                        position_frac=0.1, feature_map="elu+1"):
    """mode: fixed-1.0 | fixed-0.99 | oracle (gamma=1 latched from marker on)."""
    from models.linear_attention import FEATURE_MAPS
    phi = FEATURE_MAPS[feature_map]
    keys, vals, marker_key, marker_pos = _make_stream(N, dim, seed,
                                                      position_frac)
    S = np.zeros((dim, dim))
    z = np.zeros((dim,))
    for t, (k, v) in enumerate(zip(keys, vals)):
        if mode == "fixed-1.0":
            g = 1.0
        elif mode == "oracle" and t >= marker_pos:
            g = 1.0  # salience latch: retain from the critical update onward
        else:
            g = gamma_bg
        fk = phi(k)
        S = g * S + np.outer(fk, v)
        z = g * z + fk
    fq = phi(marker_key)
    denom = float(z @ fq)
    out = np.zeros(dim) if abs(denom) < 1e-12 else (S.T @ fq) / denom
    return {"recovered": _recovered(out)}


def run_selective_sweep(modes=("fixed-1.0", "fixed-0.99", "oracle"), N=4096,
                        dim=16, trials=8, seed=42, position_frac=0.1,
                        gamma_bg=0.99, progress_cb=None):
    rows = []
    for i, m in enumerate(modes):
        accs = [1.0 if run_selective_trial(N, dim, int(seed) + t, mode=m,
                                           gamma_bg=gamma_bg,
                                           position_frac=position_frac)["recovered"]
                else 0.0 for t in range(trials)]
        mean, std = mean_std(accs)
        rows.append({"mode": m, "N": int(N), "dim": int(dim),
                     "trials": int(trials), "accuracy": mean, "std": std,
                     "experiment": "selective_decay"})
        if progress_cb is not None:
            progress_cb((i + 1) / len(modes))
    return rows


# ---------------------------------------------------------- mixed precision
def run_mixed_precision(n_steps=60, start_magnitude=1.0, decay=0.5, dim=8,
                        seed=42):
    """fp16 compute, fp32 vs fp16 accumulation. Reference: float64."""
    rng = seeded_rng(seed)
    direction = rng.standard_normal(dim)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    mags = [start_magnitude * (decay ** t) for t in range(n_steps)]
    ref = np.zeros(dim)
    for mag in mags:
        ref = ref + direction * mag
    rows = []
    for acc in ("float16", "float32"):
        state = np.zeros(dim, dtype=np.dtype(acc))
        ineff = 0
        for mag in mags:
            u = (direction * mag).astype(np.float16)  # compute in fp16
            old = state.copy()
            state = state + u  # promoted to acc dtype
            if float(np.abs(state - old).sum()) == 0.0:
                ineff += 1
        rel_err = (float(np.abs(state.astype(np.float64) - ref).sum())
                   / (float(np.abs(ref).sum()) + 1e-300))
        rows.append({"acc_dtype": acc, "n_steps": n_steps,
                     "ineffective": int(ineff),
                     "rel_error_vs_fp64": float(rel_err),
                     "experiment": "mixed_precision"})
    return rows
