"""Batch-4 probes: closing the paper-gap audit.

A1  centered m* x d scaling          (runner-level, existing fns)
A2  recall density under decay       (run_gated_msweep)
A3  m-sweep with novelty gate on     (run_gate_msweep)
A4  oracle-latch x N                 (runner-level, run_selective_trial)
A5  per-binding survival pattern     (run_survival)
A6  cosine-similarity drift vs N     (run_cosine)
A8  compressibility top-up           (runner-level, trials=40)
A9  multi-seed batch2/3              (runner-level)
B11 bf16 shrinking stream            (run_bf16)
B12 chunk WITH summarization         (run_chunk_summary)
B13 per-trial logs of old Exps       (runner-level, single-trial fns)
B14 hybrid w-scaling + stacked       (runner-level + run_stacked)
C16 frozen-calibration OOD toy       (run_frozen_calib)
C18 random-gate control              (run_random_gate)
C19 softmax OOM boundary             (runner-level subprocess probe)

Sweeps return (rows, trial_rows) unless noted.
"""

import numpy as np

from models.linear_attention import LinearAttentionState
from experiments.gated_recall import GatedLinearState
from experiments.variants import NoveltyGatedState
from utils.generators import seeded_rng
from utils.metrics import mean_std


def _kv_codes(rng, m, dim):
    keys = rng.standard_normal((m, dim))
    values = np.zeros((m, dim))
    for i in range(m):
        values[i, i % dim] = 1.0 + 0.1 * rng.standard_normal()
    return keys, values


def _nearest_code_accuracy(outputs, values):
    d2 = ((np.asarray(outputs)[:, None, :]
           - np.asarray(values)[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(d2.argmin(axis=1) == np.arange(len(values))))


def _rec(out, idx=0):
    out = np.asarray(out, dtype=np.float64)
    return bool(int(np.argmax(out)) == idx and float(out[idx]) > 0)


# ------------------------------------------------- A2: density under decay
def run_gated_msweep(gammas=(1.0, 0.99, 0.9),
                     m_values=(2, 4, 8, 16, 32, 64, 128), dim=16, trials=8,
                     seed=42, feature_map="elu+1"):
    """m-sweep stored through a decaying state, all bindings queried fresh
    (recent). Tests Table 1's 'decay adds partial mitigation' cell."""
    rows, trial_rows = [], []
    for g in gammas:
        for m in m_values:
            accs = []
            for t in range(trials):
                rng = seeded_rng(int(seed) + t)
                keys, values = _kv_codes(rng, int(m), dim)
                st = GatedLinearState(dim, gamma=g, feature_map=feature_map,
                                      dtype=np.float64)
                for k, v in zip(keys, values):
                    st.update(k, v)
                out = np.stack([st.query(k) for k in keys])
                a = _nearest_code_accuracy(out, values)
                accs.append(a)
                trial_rows.append({"gamma": float(g), "m": int(m), "trial": t,
                                   "seed": int(seed) + t, "accuracy": a})
            mean, std = mean_std(accs)
            rows.append({"gamma": float(g), "m": int(m), "dim": int(dim),
                         "trials": int(trials), "accuracy": mean, "std": std,
                         "experiment": "gated_msweep"})
    return rows, trial_rows


# ------------------------------------------------- A3: m-sweep, gate on
def run_gate_msweep(modes=("off", "hard-16-0.8"),
                    m_values=(2, 4, 8, 16, 32, 64, 128), dim=16, trials=8,
                    seed=42, feature_map="elu+1"):
    """m-sweep with the novelty gate on vs off. Tests whether selectivity
    moves m* (Table 1, Mamba cell)."""
    rows, trial_rows = [], []
    for mode in modes:
        b, th = (0, None) if mode == "off" else (16, 0.8)
        for m in m_values:
            accs = []
            for t in range(trials):
                rng = seeded_rng(int(seed) + t)
                keys, values = _kv_codes(rng, int(m), dim)
                st = NoveltyGatedState(dim, buffer_k=b, thresh=th,
                                       feature_map=feature_map)
                for k, v in zip(keys, values):
                    st.update(k, v)
                out = np.stack([st.query(k) for k in keys])
                a = _nearest_code_accuracy(out, values)
                accs.append(a)
                trial_rows.append({"mode": mode, "m": int(m), "trial": t,
                                   "seed": int(seed) + t, "accuracy": a})
            mean, std = mean_std(accs)
            rows.append({"mode": mode, "m": int(m), "dim": int(dim),
                         "trials": int(trials), "accuracy": mean, "std": std,
                         "experiment": "gate_msweep"})
    return rows, trial_rows


# ------------------------------------------------- A5: survival pattern
def run_survival(m=32, dim=16, trials=20, seed=42, feature_map="elu+1"):
    """Per-binding correctness by stored position. Catastrophic (bimodal:
    early-dead/late-alive or vice versa) vs graceful (uniform partial)?"""
    trial_rows = []
    for t in range(trials):
        rng = seeded_rng(int(seed) + t)
        keys, values = _kv_codes(rng, m, dim)
        st = LinearAttentionState(dim, feature_map=feature_map,
                                  dtype=np.float64)
        for k, v in zip(keys, values):
            st.update(k, v)
        out = np.stack([st.query(k) for k in keys])
        d2 = ((out[:, None, :] - values[None, :, :]) ** 2).sum(axis=2)
        pred = d2.argmin(axis=1)
        for i in range(m):
            trial_rows.append({"pos": int(i), "pos_frac": float(i / max(m - 1, 1)),
                               "trial": t, "seed": int(seed) + t,
                               "correct": float(pred[i] == i)})
    rows = []
    for q in range(5):
        lo, hi = q / 5.0, (q + 1) / 5.0
        sel = [r["correct"] for r in trial_rows
               if lo <= r["pos_frac"] < hi or (q == 4 and r["pos_frac"] == 1.0)]
        mean, std = mean_std(sel)
        rows.append({"quintile": q, "m": m, "dim": dim, "trials": trials,
                     "accuracy": mean, "std": std,
                     "experiment": "survival_pattern"})
    return rows, trial_rows


# ------------------------------------------------- A6: cosine drift
def run_cosine_sweep(N_values=(128, 256, 512, 1024, 2048, 4096, 8192),
                     dim=16, trials=8, seed=42, position_frac=0.1,
                     distractor_scale=0.01):
    """Scale-invariant retrieval quality vs N. Works where relative error
    saturates (washed output ~= 0 gives err ~= 1 by construction)."""
    rows, trial_rows = [], []
    for N in N_values:
        cos, acc = [], []
        for t in range(trials):
            rng = seeded_rng(int(seed) + t)
            N = int(N)
            mp = int(N * position_frac)
            mk = rng.standard_normal(dim)
            mv = np.zeros(dim)
            mv[0] = 739281.0 / 1e6
            keys = rng.standard_normal((N, dim))
            vals = rng.standard_normal((N, dim)) * distractor_scale
            keys[mp] = mk
            vals[mp] = mv
            st = LinearAttentionState(dim, dtype=np.float64)
            for k, v in zip(keys, vals):
                st.update(k, v)
            o = np.asarray(st.query(mk), dtype=np.float64)
            c = float(o @ mv / ((np.linalg.norm(o) + 1e-300)
                                * (np.linalg.norm(mv) + 1e-300)))
            cos.append(c)
            a = 1.0 if _rec(o) else 0.0
            acc.append(a)
            trial_rows.append({"N": N, "trial": t, "seed": int(seed) + t,
                               "cosine": c, "recovered": a})
        cm, cs = mean_std(cos)
        am, _ = mean_std(acc)
        rows.append({"N": N, "dim": dim, "trials": trials,
                     "cosine_mean": cm, "cosine_std": cs,
                     "accuracy": am, "experiment": "cosine_drift"})
    return rows, trial_rows


# ------------------------------------------------- B11: bf16
def run_bf16(n_steps=60, start_magnitude=1.0, decay=0.5, dim=8, seed=42):
    """Shrinking-stream protocol with bf16 (ml_dtypes) if importable,
    else records an explicit skip row (documents the boundary)."""
    try:
        import ml_dtypes  # noqa: F401
        import numpy as _np  # noqa: F401
        bf16 = np.dtype("bfloat16")
    except Exception as e:  # pragma: no cover
        return [{"acc_dtype": "bfloat16", "status": f"skipped: {e}",
                 "experiment": "bf16"}], []
    from utils.generators import seeded_rng as _rng
    rng = _rng(seed)
    direction = rng.standard_normal(dim)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    mags = [start_magnitude * (decay ** t) for t in range(n_steps)]
    ref = np.zeros(dim)
    for mag in mags:
        ref = ref + direction * mag
    rows = []
    for acc in ("float16", "bfloat16", "float32"):
        dt = np.dtype("bfloat16") if acc == "bfloat16" else np.dtype(acc)
        state = np.zeros(dim, dtype=dt)
        ineff = 0
        for mag in mags:
            u = (direction * mag).astype(dt)
            old = state.copy()
            state = state + u
            if float(np.abs(state.astype(np.float64)
                            - old.astype(np.float64)).sum()) == 0.0:
                ineff += 1
        rel = (float(np.abs(state.astype(np.float64) - ref).sum())
               / (float(np.abs(ref).sum()) + 1e-300))
        rows.append({"acc_dtype": acc, "n_steps": n_steps,
                     "ineffective": int(ineff), "rel_error_vs_fp64": float(rel),
                     "experiment": "bf16"})
    return rows, []


# ------------------------------------------------- B12: chunk with summary
def run_chunk_summary(chunks=(1024,), N=4096, dim=16, trials=8, seed=42,
                      modes=("zero", "summary", "none"),
                      position_fracs=(0.1, 0.9)):
    """zero: reset to empty. summary: carry one mean-binding forward.
    none: no reset (baseline)."""
    rows = []
    for mode in modes:
        for pf in position_fracs:
            accs, norms = [], []
            for t in range(trials):
                rng = seeded_rng(int(seed) + t)
                Nn = int(N)
                mp = int(Nn * pf)
                mk = rng.standard_normal(dim)
                mv = np.zeros(dim)
                mv[0] = 739281.0 / 1e6
                keys = rng.standard_normal((Nn, dim))
                vals = rng.standard_normal((Nn, dim)) * 0.01
                keys[mp] = mk
                vals[mp] = mv
                st = LinearAttentionState(dim, dtype=np.float64)
                ck, cv, cn = [], [], 0
                maxn = 0.0
                for i, (k, v) in enumerate(zip(keys, vals)):
                    if mode in ("zero", "summary") and i > 0 and i % chunks[0] == 0:
                        if mode == "summary" and cn > 0:
                            kb = np.stack(ck).mean(axis=0)
                            vb = np.stack(cv).mean(axis=0)
                            st.reset()
                            st.update(kb, vb)
                        else:
                            st.reset()
                        ck, cv, cn = [], [], 0
                    st.update(k, v)
                    ck.append(k)
                    cv.append(v)
                    cn += 1
                    maxn = max(maxn, st.state_norm)
                accs.append(1.0 if _rec(st.query(mk)) else 0.0)
                norms.append(maxn)
            mean, std = mean_std(accs)
            rows.append({"mode": mode, "chunk": chunks[0], "N": Nn,
                         "dim": dim, "position_frac": float(pf),
                         "trials": trials, "accuracy": mean, "std": std,
                         "max_state_norm": float(np.mean(norms)),
                         "experiment": "chunk_summary"})
    return rows, []


# ------------------------------------------------- B14: stacked mitigations
def run_stacked(N=4096, dim=16, trials=8, seed=42, window=128, gamma_bg=0.99,
                position_fracs=(0.1, 0.9)):
    """linear-only | hybrid-only | decayed-only | stacked (decayed+hybrid)."""
    from models.softmax_attention import SoftmaxReference
    from models.linear_attention import FEATURE_MAPS
    phi = FEATURE_MAPS["elu+1"]
    rows = []
    for mode in ("linear-only", "hybrid-only", "decayed-only", "stacked"):
        for pf in position_fracs:
            accs = []
            for t in range(trials):
                rng = seeded_rng(int(seed) + t)
                Nn = int(N)
                mp = int(Nn * pf)
                mk = rng.standard_normal(dim)
                mv = np.zeros(dim)
                mv[0] = 739281.0 / 1e6
                keys = rng.standard_normal((Nn, dim))
                vals = rng.standard_normal((Nn, dim)) * 0.01
                keys[mp] = mk
                vals[mp] = mv
                use_decay = mode in ("decayed-only", "stacked")
                g = gamma_bg if use_decay else 1.0
                S = np.zeros((dim, dim))
                z = np.zeros((dim,))
                for k, v in zip(keys, vals):
                    fk = phi(k)
                    S = g * S + np.outer(fk, v)
                    z = g * z + fk
                if mode in ("hybrid-only", "stacked") and mp >= Nn - window:
                    sm = SoftmaxReference()
                    for k, v in zip(keys[Nn - window:], vals[Nn - window:]):
                        sm.update(k, v)
                    accs.append(1.0 if _rec(sm.query(mk)) else 0.0)
                else:
                    fq = phi(mk)
                    d = float(z @ fq)
                    o = np.zeros(dim) if abs(d) < 1e-12 else (S.T @ fq) / d
                    accs.append(1.0 if _rec(o) else 0.0)
            mean, std = mean_std(accs)
            rows.append({"mode": mode, "N": Nn, "dim": dim,
                         "position_frac": float(pf), "trials": trials,
                         "accuracy": mean, "std": std,
                         "experiment": "stacked"})
    return rows, []


# ------------------------------------------------- C18: random gate
def run_random_gate(N_values=(1024, 2048, 4096, 8192), dim=16, trials=8,
                    seed=42, gamma_bg=0.99, position_frac=0.1):
    """fixed-0.99 | oracle latch | coin-flip gate. Gating must be informed."""
    from models.linear_attention import FEATURE_MAPS
    phi = FEATURE_MAPS["elu+1"]
    rows = []
    for mode in ("fixed-0.99", "oracle", "random"):
        for N in N_values:
            accs = []
            for t in range(trials):
                rng = seeded_rng(int(seed) + t)
                Nn = int(N)
                mp = int(Nn * position_frac)
                mk = rng.standard_normal(dim)
                mv = np.zeros(dim)
                mv[0] = 739281.0 / 1e6
                keys = rng.standard_normal((Nn, dim))
                vals = rng.standard_normal((Nn, dim)) * 0.01
                keys[mp] = mk
                vals[mp] = mv
                grng = seeded_rng(10_000 + int(seed) + t)
                S = np.zeros((dim, dim))
                z = np.zeros((dim,))
                for i, (k, v) in enumerate(zip(keys, vals)):
                    if mode == "oracle" and i >= mp:
                        g = 1.0
                    elif mode == "random":
                        g = 1.0 if grng.random() < 0.5 else gamma_bg
                    else:
                        g = gamma_bg
                    fk = phi(k)
                    S = g * S + np.outer(fk, v)
                    z = g * z + fk
                fq = phi(mk)
                d = float(z @ fq)
                o = np.zeros(dim) if abs(d) < 1e-12 else (S.T @ fq) / d
                accs.append(1.0 if _rec(o) else 0.0)
            mean, std = mean_std(accs)
            rows.append({"mode": mode, "N": Nn, "dim": dim, "trials": trials,
                         "accuracy": mean, "std": std,
                         "experiment": "random_gate"})
    return rows, []


# ------------------------------------------------- C16: frozen calibration
def run_frozen_calib(N=2048, dim=16, trials=8, seed=42, shift=2.0,
                     calib_N=512):
    """Centering constant estimated on pre-shift calibration data then
    frozen (toy 'training boundary') vs oracle constant, marker post-shift.
    Honest miniature of train/test distribution mismatch."""
    rows = []
    for mode in ("oracle-c", "frozen-c"):
        accs = []
        for t in range(trials):
            rng = seeded_rng(int(seed) + t)
            Nn = int(N)
            mp = int(Nn * 0.75)
            mk = rng.standard_normal(dim)
            mv = np.zeros(dim)
            mv[0] = 739281.0 / 1e6
            keys = rng.standard_normal((Nn, dim))
            keys[Nn // 2:] += shift
            vals = rng.standard_normal((Nn, dim)) * 0.01
            keys[mp] = mk
            vals[mp] = mv
            if mode == "frozen-c":
                # constant estimated on pre-shift calibration data, then frozen
                cal = rng.standard_normal((calib_N, dim))
                from models.linear_attention import elu_plus_one
                c = float(elu_plus_one(cal).mean())
            else:
                # oracle: constant matched to the POST-shift distribution
                # (this is the mismatch the frozen condition suffers)
                from models.linear_attention import elu_plus_one
                post = rng.standard_normal((20000, dim)) + shift
                c = float(elu_plus_one(post).mean())
            from models.linear_attention import elu_plus_one
            phi = lambda x: elu_plus_one(x) - c  # noqa: E731
            S = np.zeros((dim, dim))
            z = np.zeros((dim,))
            for k, v in zip(keys, vals):
                fk = phi(k)
                S = S + np.outer(fk, v)
                z = z + fk
            fq = phi(mk)
            d = float(z @ fq)
            o = np.zeros(dim) if abs(d) < 1e-12 else (S.T @ fq) / d
            accs.append(1.0 if _rec(o) else 0.0)
        mean, std = mean_std(accs)
        rows.append({"mode": mode, "N": Nn, "dim": dim, "trials": trials,
                     "accuracy": mean, "std": std,
                     "experiment": "frozen_calib"})
    return rows, []
