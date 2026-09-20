"""Fine-grained dissipation, density-vs-length controls, key-structure probes.

- coarse_vs_fine (§4.2c): a repeated (redundant, coarse) signal and a
  singleton (fine-grained) marker share one stream; measures which is
  lost first as N grows.
- m_x_N grid (§4.2a diagnostic): m bindings followed by N filler steps;
  accuracy as a function of m at fixed filler (and vice versa) separates
  density-driven from length-driven failure.
- orthogonal keys: one-hot + noise keys test whether near-orthogonal
  structure restores a rank-scaled plateau (bounds the §3.3 m* claim).
"""

import numpy as np

from models.linear_attention import LinearAttentionState
from models.softmax_attention import SoftmaxReference
from utils.generators import seeded_rng
from utils.metrics import mean_std


def _recovered_idx(out, idx):
    out = np.asarray(out, dtype=np.float64)
    return bool(int(np.argmax(out)) == idx and float(out[idx]) > 0)


def _nearest_code_accuracy(outputs, values):
    d2 = ((np.asarray(outputs)[:, None, :]
           - np.asarray(values)[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(d2.argmin(axis=1) == np.arange(len(values))))


# ------------------------------------------------------- coarse vs fine
def run_dissipation_trial(N, dim, seed, repeats=20, fine_pos=0.1,
                          distractor_scale=0.01):
    rng = seeded_rng(seed)
    N = int(N)
    coarse_key = rng.standard_normal(dim)
    coarse_vec = np.zeros(dim)
    coarse_vec[1] = 0.5
    fine_key = rng.standard_normal(dim)
    fine_vec = np.zeros(dim)
    fine_vec[0] = 739281.0 / 1e6

    keys = rng.standard_normal((N, dim))
    vals = rng.standard_normal((N, dim)) * distractor_scale
    keys[int(N * fine_pos)] = fine_key
    vals[int(N * fine_pos)] = fine_vec
    for j in range(repeats):
        pos = int((j + 1) * N / (repeats + 1))
        keys[pos] = coarse_key + 0.01 * rng.standard_normal(dim)
        vals[pos] = coarse_vec

    st = LinearAttentionState(dim, dtype=np.float64)
    for k, v in zip(keys, vals):
        st.update(k, v)
    return {
        "coarse_recovered": _recovered_idx(st.query(coarse_key), 1),
        "fine_recovered": _recovered_idx(st.query(fine_key), 0),
    }


def run_dissipation_sweep(N_values=(128, 512, 2048, 8192), dim=16, trials=8,
                          seed=42, progress_cb=None):
    rows = []
    for i, N in enumerate(N_values):
        coarse, fine = [], []
        for t in range(trials):
            r = run_dissipation_trial(int(N), dim, int(seed) + t)
            coarse.append(1.0 if r["coarse_recovered"] else 0.0)
            fine.append(1.0 if r["fine_recovered"] else 0.0)
        cm, cs = mean_std(coarse)
        fm, fs = mean_std(fine)
        rows.append({"N": int(N), "dim": int(dim), "trials": int(trials),
                     "coarse_accuracy": cm, "coarse_std": cs,
                     "fine_accuracy": fm, "fine_std": fs,
                     "experiment": "coarse_vs_fine"})
        if progress_cb is not None:
            progress_cb((i + 1) / len(N_values))
    return rows


# ------------------------------------------------------- m x N grid
def run_mN_trial(m, nfill, dim, seed, feature_map="elu+1",
                 distractor_scale=0.01):
    rng = seeded_rng(seed)
    keys = rng.standard_normal((m, dim))
    values = np.zeros((m, dim))
    for i in range(m):
        values[i, i % dim] = 1.0 + 0.1 * rng.standard_normal()
    st = LinearAttentionState(dim, feature_map=feature_map, dtype=np.float64)
    for k, v in zip(keys, values):
        st.update(k, v)
    for _ in range(int(nfill)):
        st.update(rng.standard_normal(dim),
                  rng.standard_normal(dim) * distractor_scale)
    out = np.stack([st.query(k) for k in keys])
    return {"accuracy": _nearest_code_accuracy(out, values)}


def run_mN_grid(m_values=(4, 16, 64), nfill_values=(0, 256, 2048), dim=16,
                trials=8, seed=42, progress_cb=None):
    rows = []
    total = len(m_values) * len(nfill_values)
    done = 0
    for m in m_values:
        for nf in nfill_values:
            accs = [run_mN_trial(int(m), int(nf), dim, int(seed) + t)["accuracy"]
                    for t in range(trials)]
            mean, std = mean_std(accs)
            rows.append({"m": int(m), "nfill": int(nf), "dim": int(dim),
                         "trials": int(trials), "accuracy": mean, "std": std,
                         "experiment": "m_x_N_grid"})
            done += 1
            if progress_cb is not None:
                progress_cb(done / total)
    return rows


# ------------------------------------------------------- orthogonal keys
def run_orth_trial(m, dim, seed, noise=0.05, feature_map="elu+1"):
    rng = seeded_rng(seed)
    eye = np.eye(dim)
    keys = np.stack([eye[i % dim] + noise * rng.standard_normal(dim)
                     for i in range(m)])
    values = np.zeros((m, dim))
    for i in range(m):
        values[i, i % dim] = 1.0 + 0.1 * rng.standard_normal()

    lin = LinearAttentionState(dim, feature_map=feature_map, dtype=np.float64)
    for k, v in zip(keys, values):
        lin.update(k, v)
    lin_acc = _nearest_code_accuracy(np.stack([lin.query(k) for k in keys]),
                                     values)
    sm = SoftmaxReference()
    for k, v in zip(keys, values):
        sm.update(k, v)
    sm_acc = _nearest_code_accuracy(np.stack([sm.query(k) for k in keys]),
                                    values)
    return {"linear_accuracy": lin_acc, "softmax_accuracy": sm_acc}


def run_orth_sweep(m_values=(2, 4, 8, 16, 32, 64, 128), dim=16, trials=10,
                   seed=42, feature_map="elu+1", progress_cb=None):
    rows = []
    for i, m in enumerate(m_values):
        la, sa = [], []
        for t in range(trials):
            r = run_orth_trial(int(m), dim, int(seed) + t,
                               feature_map=feature_map)
            la.append(r["linear_accuracy"])
            sa.append(r["softmax_accuracy"])
        lm, ls = mean_std(la)
        sm, ss = mean_std(sa)
        rows.append({"m": int(m), "dim": int(dim), "trials": int(trials),
                     "linear_mean": lm, "linear_std": ls,
                     "softmax_mean": sm, "softmax_std": ss,
                     "feature_map": feature_map,
                     "experiment": "orthogonal_keys"})
        if progress_cb is not None:
            progress_cb((i + 1) / len(m_values))
    return rows
