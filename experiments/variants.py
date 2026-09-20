"""Batch-3 variant probes (paper gaps audit).

- per-channel decay (RetNet-style gamma vector vs scalar gamma)
- novelty-gated write (selectivity toy: suppress near-duplicate updates)
- multi-head banks at equal parameter budget
- compressibility (low-entropy vs random streams at fixed N)
- distribution shift (second-half key drift, pre-shift marker)
- noisy queries (q = k + sigma * noise, linear vs softmax)
- centered feature map support lives in models.linear_attention
  ("centered_elu"); sweeps reuse associative_recall / dissipation helpers.

Every sweep returns (rows, trial_rows): aggregate rows plus per-trial
records enabling significance testing downstream.
"""

import numpy as np

from models.linear_attention import LinearAttentionState, FEATURE_MAPS
from models.softmax_attention import SoftmaxReference
from utils.generators import seeded_rng
from utils.metrics import mean_std


def _stream(N, dim, seed, position_frac=0.1, distractor_scale=0.01,
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
    return rng, keys, vals, marker_key, marker_pos


def _rec(out, idx=0):
    out = np.asarray(out, dtype=np.float64)
    return bool(int(np.argmax(out)) == idx and float(out[idx]) > 0)


def _nearest_code_accuracy(outputs, values):
    d2 = ((np.asarray(outputs)[:, None, :]
           - np.asarray(values)[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(d2.argmin(axis=1) == np.arange(len(values))))


# ------------------------------------------------------- per-channel decay
class PerChannelGatedState:
    """S rows decayed by a per-channel gamma vector (RWKV/RetNet-style)."""

    def __init__(self, dim, gamma, feature_map="elu+1"):
        self.phi = FEATURE_MAPS[feature_map]
        self.g = np.asarray(gamma, dtype=np.float64).reshape(-1)
        assert self.g.shape == (dim,) and bool(((self.g > 0) & (self.g <= 1)).all())
        self.dim = int(dim)
        self.reset()

    def reset(self):
        self.S = np.zeros((self.dim, self.dim))
        self.z = np.zeros((self.dim,))

    def update(self, key, value):
        fk = self.phi(np.asarray(key, dtype=float))
        v = np.asarray(value, dtype=float)
        self.S = self.S * self.g[:, None] + np.outer(fk, v)
        self.z = self.z * self.g + fk

    def query(self, query):
        fq = self.phi(np.asarray(query, dtype=float))
        denom = float(self.z @ fq)
        if abs(denom) < 1e-12:
            return np.zeros((self.dim,))
        return (self.S.T @ fq) / denom


def _perchannel_gammas(mode, dim):
    if mode == "scalar-0.99":
        return np.full(dim, 0.99)
    if mode == "scalar-0.999":
        return np.full(dim, 0.999)
    if mode == "perchannel":
        # slow channels (~1.0) preserve early info, fast channels (~0.9)
        # track recent input
        return 1.0 - np.logspace(-4, -1, dim)
    raise ValueError(mode)


def run_perchannel_trial(N, dim, seed, mode="perchannel", position_frac=0.1):
    _, keys, vals, marker_key, _ = _stream(N, dim, seed, position_frac)
    st = PerChannelGatedState(dim, _perchannel_gammas(mode, dim))
    for k, v in zip(keys, vals):
        st.update(k, v)
    return {"recovered": _rec(st.query(marker_key))}


def run_perchannel_sweep(modes=("scalar-0.99", "scalar-0.999", "perchannel"),
                         N=1024, dim=16, trials=8, seed=42,
                         position_frac=0.1):
    """NOTE: N=1024 keeps the slow channel above the forgetting floor.
    At N=4096 every mode sits at the noise floor and nothing differentiates.
    """
    rows, trial_rows = [], []
    for m in modes:
        accs = []
        for t in range(trials):
            r = run_perchannel_trial(N, dim, int(seed) + t, mode=m,
                                     position_frac=position_frac)
            accs.append(1.0 if r["recovered"] else 0.0)
            trial_rows.append({"mode": m, "trial": t, "seed": int(seed) + t,
                               "recovered": accs[-1]})
        mean, std = mean_std(accs)
        rows.append({"mode": m, "N": int(N), "dim": int(dim),
                     "trials": int(trials), "accuracy": mean, "std": std,
                     "experiment": "perchannel_decay"})
    return rows, trial_rows


# ------------------------------------------------------- novelty gating
class NoveltyGatedState:
    """Write scaled by novelty vs a buffer of recent keys (selectivity toy).

    thresh=None: soft scaling (1 - max|cos|). With random keys in low
    dim, chance overlaps (~0.5 at d=16) dampen EVERYTHING, so the soft
    gate attenuates rather than filters. thresh=float: hard cutoff —
    write fully unless near-duplicate (the margin a real gate needs).
    """

    def __init__(self, dim, buffer_k=16, thresh=None, feature_map="elu+1"):
        self.st = LinearAttentionState(dim, feature_map=feature_map,
                                       dtype=np.float64)
        self.buffer_k = int(buffer_k)
        self.thresh = thresh
        self.buf = []

    def update(self, key, value):
        k = np.asarray(key, dtype=np.float64)
        scale = 1.0
        if self.buffer_k > 0 and self.buf:
            B = np.stack(self.buf[-self.buffer_k:])
            bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            kn = k / (np.linalg.norm(k) + 1e-12)
            maxcos = float(np.abs(bn @ kn).max())
            if self.thresh is None:
                scale = float(np.clip(1.0 - maxcos, 0.0, 1.0))
            else:
                scale = 1.0 if maxcos < float(self.thresh) else 0.0
        fk = self.st.phi(k)
        v = np.asarray(value, dtype=np.float64)
        self.st.S += scale * np.outer(fk, v)
        self.st.z += scale * fk
        self.st.count += 1
        self.buf.append(k.copy())

    def query(self, q):
        return self.st.query(q)


def run_novelty_trial(m, dim, seed, buffer_k=16, thresh=None,
                      feature_map="elu+1"):
    rng = seeded_rng(seed)
    keys = rng.standard_normal((m, dim))
    values = np.zeros((m, dim))
    for i in range(m):
        values[i, i % dim] = 1.0 + 0.1 * rng.standard_normal()
    st = NoveltyGatedState(dim, buffer_k=buffer_k, thresh=thresh,
                           feature_map=feature_map)
    for k, v in zip(keys, values):
        st.update(k, v)
    out = np.stack([st.query(k) for k in keys])
    return {"accuracy": _nearest_code_accuracy(out, values)}


def run_novelty_sweep(m_values=(2, 4, 8, 16, 32, 64, 128), dim=16, trials=6,
                      seed=42, modes=(("off", 0, None), ("soft-16", 16, None),
                                      ("soft-64", 64, None),
                                      ("hard-16-0.8", 16, 0.8)),
                      feature_map="elu+1"):
    rows, trial_rows = [], []
    for name, b, th in modes:
        for m in m_values:
            accs = []
            for t in range(trials):
                r = run_novelty_trial(int(m), dim, int(seed) + t, buffer_k=b,
                                      thresh=th, feature_map=feature_map)
                accs.append(r["accuracy"])
                trial_rows.append({"mode": name, "m": int(m), "trial": t,
                                   "seed": int(seed) + t,
                                   "accuracy": r["accuracy"]})
            mean, std = mean_std(accs)
            rows.append({"mode": name, "buffer": int(b), "m": int(m),
                         "dim": int(dim), "trials": int(trials),
                         "accuracy": mean, "std": std,
                         "experiment": "novelty_gate"})
    return rows, trial_rows


# ------------------------------------------------------- multi-head banks
class MultiHeadLinearState:
    """h parallel banks; total dim D = h * dh. No learned projections."""

    def __init__(self, n_heads, head_dim, feature_map="elu+1"):
        self.heads = [LinearAttentionState(head_dim, feature_map=feature_map,
                                           dtype=np.float64)
                      for _ in range(int(n_heads))]
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)

    def update(self, key, value):
        for st, k, v in zip(self.heads, np.array_split(key, self.n_heads),
                            np.array_split(value, self.n_heads)):
            st.update(k, v)

    def query(self, query):
        return np.concatenate([st.query(k) for st, k in
                               zip(self.heads,
                                   np.array_split(query, self.n_heads))])


def run_multihead_trial(m, total_dim, seed, n_heads, feature_map="elu+1"):
    dh = total_dim // n_heads
    assert dh * n_heads == total_dim
    rng = seeded_rng(seed)
    keys = rng.standard_normal((m, total_dim))
    values = np.zeros((m, total_dim))
    for i in range(m):
        values[i, i % total_dim] = 1.0 + 0.1 * rng.standard_normal()
    st = MultiHeadLinearState(n_heads, dh, feature_map=feature_map)
    for k, v in zip(keys, values):
        st.update(k, v)
    out = np.stack([st.query(k) for k in keys])
    return {"accuracy": _nearest_code_accuracy(out, values)}


def run_multihead_sweep(head_configs=((1, 64), (4, 32), (16, 16)),
                        m_values=(2, 4, 8, 16, 32, 64, 128), trials=6,
                        seed=42, feature_map="elu+1"):
    """Equal-parameter configs: h*dh^2 = 4096 in all three cases."""
    rows, trial_rows = [], []
    for h, dh in head_configs:
        for m in m_values:
            accs = []
            for t in range(trials):
                r = run_multihead_trial(int(m), h * dh, int(seed) + t,
                                        n_heads=h, feature_map=feature_map)
                accs.append(r["accuracy"])
                trial_rows.append({"heads": int(h), "head_dim": int(dh),
                                   "m": int(m), "trial": t,
                                   "seed": int(seed) + t,
                                   "accuracy": r["accuracy"]})
            mean, std = mean_std(accs)
            rows.append({"heads": int(h), "head_dim": int(dh), "m": int(m),
                         "trials": int(trials), "accuracy": mean, "std": std,
                         "experiment": "multihead"})
    return rows, trial_rows


# ------------------------------------------------------- compressibility
def run_compress_trial(N, dim, seed, alphabet=8, position_frac=0.1,
                       distractor_scale=0.01):
    rng = seeded_rng(seed)
    N = int(N)
    marker_pos = int(N * position_frac)
    marker_key = rng.standard_normal(dim)
    marker_vec = np.zeros(dim)
    marker_vec[0] = 739281.0 / 1e6
    if alphabet is None:
        keys = rng.standard_normal((N, dim))
    else:
        bank = rng.standard_normal((int(alphabet), dim))
        keys = bank[rng.integers(0, int(alphabet), size=N)]
    vals = rng.standard_normal((N, dim)) * distractor_scale
    keys[marker_pos] = marker_key
    vals[marker_pos] = marker_vec
    st = LinearAttentionState(dim, dtype=np.float64)
    for k, v in zip(keys, vals):
        st.update(k, v)
    return {"recovered": _rec(st.query(marker_key))}


def run_compress_sweep(alphabets=(8, 64, None), N=2048, dim=16, trials=16,
                       seed=42, position_frac=0.1):
    rows, trial_rows = [], []
    for a in alphabets:
        accs = []
        for t in range(trials):
            r = run_compress_trial(N, dim, int(seed) + t, alphabet=a,
                                   position_frac=position_frac)
            accs.append(1.0 if r["recovered"] else 0.0)
            trial_rows.append({"alphabet": -1 if a is None else int(a),
                               "trial": t, "seed": int(seed) + t,
                               "recovered": accs[-1]})
        mean, std = mean_std(accs)
        rows.append({"alphabet": -1 if a is None else int(a), "N": int(N),
                     "dim": int(dim), "trials": int(trials),
                     "accuracy": mean, "std": std,
                     "experiment": "compressibility"})
    return rows, trial_rows


# ------------------------------------------------------- distribution shift
def run_shift_trial(N, dim, seed, shift=2.0, position_frac=0.1,
                    distractor_scale=0.01):
    rng = seeded_rng(seed)
    N = int(N)
    marker_pos = int(N * position_frac)
    marker_key = rng.standard_normal(dim)
    marker_vec = np.zeros(dim)
    marker_vec[0] = 739281.0 / 1e6
    keys = rng.standard_normal((N, dim))
    keys[N // 2:] += float(shift)  # second-half distribution shift
    vals = rng.standard_normal((N, dim)) * distractor_scale
    keys[marker_pos] = marker_key
    vals[marker_pos] = marker_vec
    st = LinearAttentionState(dim, dtype=np.float64)
    for k, v in zip(keys, vals):
        st.update(k, v)
    return {"recovered": _rec(st.query(marker_key))}


def run_shift_sweep(shifts=(0.0, 1.0, 2.0, 4.0), N=2048, dim=16, trials=8,
                    seed=42, position_frac=0.1):
    rows, trial_rows = [], []
    for s in shifts:
        accs = []
        for t in range(trials):
            r = run_shift_trial(N, dim, int(seed) + t, shift=s,
                                position_frac=position_frac)
            accs.append(1.0 if r["recovered"] else 0.0)
            trial_rows.append({"shift": float(s), "trial": t,
                               "seed": int(seed) + t, "recovered": accs[-1]})
        mean, std = mean_std(accs)
        rows.append({"shift": float(s), "N": int(N), "dim": int(dim),
                     "trials": int(trials), "accuracy": mean, "std": std,
                     "experiment": "shift_probe"})
    return rows, trial_rows


# ------------------------------------------------------- noisy queries
def run_noisy_trial(N, dim, seed, sigma=0.3, position_frac=0.5,
                    distractor_scale=0.01):
    rng = seeded_rng(seed)
    # identical stream for both models by construction (same seed)
    _, keys, vals, marker_key, _ = _stream(N, dim, seed, position_frac,
                                           distractor_scale)
    q = marker_key + float(sigma) * rng.standard_normal(dim)
    lin = LinearAttentionState(dim, dtype=np.float64)
    for k, v in zip(keys, vals):
        lin.update(k, v)
    sm = SoftmaxReference()
    for k, v in zip(keys, vals):
        sm.update(k, v)
    return {"linear_recovered": _rec(lin.query(q)),
            "softmax_recovered": _rec(sm.query(q))}


def run_noisy_sweep(sigmas=(0.0, 0.1, 0.3, 1.0), N=1024, dim=16, trials=10,
                    seed=42, position_frac=0.5):
    rows, trial_rows = [], []
    for s in sigmas:
        la, sa = [], []
        for t in range(trials):
            r = run_noisy_trial(N, dim, int(seed) + t, sigma=s,
                                position_frac=position_frac)
            la.append(1.0 if r["linear_recovered"] else 0.0)
            sa.append(1.0 if r["softmax_recovered"] else 0.0)
            trial_rows.append({"sigma": float(s), "trial": t,
                               "seed": int(seed) + t,
                               "linear": la[-1], "softmax": sa[-1]})
        lm, ls = mean_std(la)
        sm, ss = mean_std(sa)
        rows.append({"sigma": float(s), "N": int(N), "dim": int(dim),
                     "trials": int(trials), "linear_accuracy": lm,
                     "linear_std": ls, "softmax_accuracy": sm,
                     "softmax_std": ss, "experiment": "noisy_query"})
    return rows, trial_rows
