"""Gated (RetNet-style) recall: S_t = gamma * S_{t-1} + phi(k_t) v_t^T.

Tests the decay-horizon mechanism (paper §3.4): with a marker injected
early, retrieval accuracy vs gamma and N reveals the effective forgetting
horizon, compared against 1/(1-gamma) and the finite-precision bound
ln(1/eps)/(1-gamma). gamma=1.0 recovers the undecayed baseline (Exp 2).

Also supports precision x decay interaction (cancellation, §3.5).
"""

import numpy as np

from models.linear_attention import FEATURE_MAPS
from utils.generators import seeded_rng
from utils.metrics import mean_std


class GatedLinearState:
    """Fixed-capacity recurrent state with scalar decay."""

    def __init__(self, dim, gamma=1.0, feature_map="elu+1", dtype=np.float64):
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.dim = int(dim)
        self.gamma = float(gamma)
        self.phi = FEATURE_MAPS[feature_map]
        self.feature_map_name = feature_map
        self.dtype = np.dtype(dtype)
        self.reset()

    def reset(self):
        self.S = np.zeros((self.dim, self.dim), dtype=self.dtype)
        self.z = np.zeros((self.dim,), dtype=self.dtype)
        self.count = 0

    def update(self, key, value):
        k = np.asarray(key, dtype=self.dtype).reshape(-1)
        v = np.asarray(value, dtype=self.dtype).reshape(-1)
        g = self.dtype.type(self.gamma)
        fk = self.phi(k.astype(np.float64)).astype(self.dtype)
        self.S = (g * self.S + np.outer(fk, v)).astype(self.dtype)
        self.z = (g * self.z + fk).astype(self.dtype)
        self.count += 1

    def query(self, query):
        q = np.asarray(query, dtype=self.dtype).reshape(-1)
        fq = self.phi(q.astype(np.float64)).astype(self.dtype)
        denom = float(np.dot(self.z, fq))
        if abs(denom) < 1e-12:
            return np.zeros((self.dim,), dtype=self.dtype)
        return (self.S.T @ fq) / denom

    @property
    def state_norm(self):
        return float(np.linalg.norm(self.S.astype(np.float64)))


def theory_horizons(gamma):
    """Return (1/(1-g), fp32-zeroing bound ln(2^24)/(1-g)); None if g==1."""
    if float(gamma) >= 1.0:
        return None, None
    inv = 1.0 / (1.0 - float(gamma))
    return inv, 16.63553233 * inv  # ln(2^24) ~= 16.64


def run_gated_trial(N, dim, seed, gamma=0.99, position_frac=0.1,
                    feature_map="elu+1", dtype=np.float64,
                    distractor_scale=0.01, marker_value=739281.0):
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

    st = GatedLinearState(dim, gamma=gamma, feature_map=feature_map, dtype=dtype)
    for k, v in zip(keys, vals):
        st.update(k, v)
    out = np.asarray(st.query(marker_key), dtype=np.float64)
    recovered = bool(int(np.argmax(out)) == 0 and float(out[0]) > 0)
    return {"N": N, "gamma": float(gamma), "recovered": recovered,
            "output": out.tolist(), "state_norm": st.state_norm}


def run_gamma_sweep(gammas=(1.0, 0.9, 0.99, 0.999, 0.9999), N=4096, dim=16,
                    trials=10, seed=42, position_frac=0.1,
                    feature_map="elu+1", dtype=np.float64, progress_cb=None):
    rows = []
    for i, g in enumerate(gammas):
        accs, norms = [], []
        for t in range(trials):
            r = run_gated_trial(N, dim, int(seed) + t, gamma=g,
                                position_frac=position_frac,
                                feature_map=feature_map, dtype=dtype)
            accs.append(1.0 if r["recovered"] else 0.0)
            norms.append(r["state_norm"])
        mean, std = mean_std(accs)
        h1, h32 = theory_horizons(g)
        rows.append({
            "gamma": float(g), "N": int(N), "dim": int(dim),
            "trials": int(trials), "position_frac": float(position_frac),
            "accuracy": mean, "std": std,
            "state_norm_mean": float(np.mean(norms)),
            "horizon_1_over_1mg": None if h1 is None else float(h1),
            "horizon_fp32_zero": None if h32 is None else float(h32),
            "feature_map": feature_map,
            "precision": np.dtype(dtype).name,
            "experiment": "gated_decay",
        })
        if progress_cb is not None:
            progress_cb((i + 1) / len(gammas))
    return rows


def run_precision_decay(precisions=("float64", "float32", "float16"),
                        gammas=(0.99, 0.999), N=2048, dim=16, trials=8,
                        seed=42, position_frac=0.9, feature_map="elu+1",
                        progress_cb=None):
    """Retrieval accuracy per (precision, gamma): cancellation probe.

    NOTE: the marker is recent (position_frac=0.9). An early marker sits
    below the forgetting floor for all gamma < 1 and cannot differentiate
    precisions; recency keeps the update partially alive so numerical
    differences can manifest.
    """
    rows = []
    total = len(precisions) * len(gammas)
    done = 0
    for prec in precisions:
        for g in gammas:
            accs, drifts = [], []
            for t in range(trials):
                r = run_gated_trial(N, dim, int(seed) + t, gamma=g,
                                    position_frac=position_frac,
                                    feature_map=feature_map,
                                    dtype=np.dtype(prec))
                # float64 reference on the IDENTICAL stream (same seed):
                # drift is the continuous metric; binary recall is too
                # coarse to move under rounding-level differences.
                ref = run_gated_trial(N, dim, int(seed) + t, gamma=g,
                                      position_frac=position_frac,
                                      feature_map=feature_map,
                                      dtype=np.float64)
                accs.append(1.0 if r["recovered"] else 0.0)
                drifts.append(float(np.abs(np.asarray(r["output"])
                                           - np.asarray(ref["output"])).mean()))
            mean, std = mean_std(accs)
            rows.append({
                "precision": prec, "gamma": float(g), "N": int(N),
                "dim": int(dim), "trials": int(trials),
                "accuracy": mean, "std": std,
                "drift_vs_fp64": float(np.mean(drifts)),
                "experiment": "precision_decay",
            })
            done += 1
            if progress_cb is not None:
                progress_cb(done / total)
    return rows
