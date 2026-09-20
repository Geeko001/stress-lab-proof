"""Batch-4 runs: final paper-gap audit. Staged for timeouts.

  stageC : centered-d scaling, gated/gate m-sweeps, latch-N, survival, cosine
  stageD : compress top-up, multi-seed, bf16, chunk-summary, old-exp trials,
           hybrid-N4096, stacked, frozen-calib, random-gate, OOM probe
  finalize : horizon fit, Welch t-tests, CI table, figures, bundle update

Usage: .\\.venv\\Scripts\\python run_batch4.py [stageC|stageD|finalize|all]
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting import save_high_res

ROOT = Path(__file__).parent
OUT = ROOT / "public_evidence"
FIG = OUT / "figures"
SEED = 42


def _save(rows, name):
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


# ================================================================ STAGE C
def stageC():
    from experiments.associative_recall import run_dimensionality_sweep
    from experiments.dissipation import run_orth_sweep
    from experiments.finalbatch import (run_gated_msweep, run_gate_msweep,
                                        run_survival, run_cosine_sweep)
    from experiments.mitigations import run_selective_trial

    print("[C] centered m* x d ...", flush=True)
    rows = run_dimensionality_sweep([2, 4, 8, 16, 32, 64, 128],
                                    dim_values=(8, 32, 64), trials=12,
                                    seed=SEED, feature_map="centered_elu")
    for r in rows:
        r["experiment"] = "centered_d_scaling"
    _save(rows, "exp12_centered_d.csv")
    orows = []
    for d in (8, 32, 64):
        print(f"  centered-orth d={d}", flush=True)
        for r in run_orth_sweep(feature_map="centered_elu", dim=d,
                                trials=10, seed=SEED):
            r["experiment"] = "centered_orth_d"
            orows.append(r)
    _save(orows, "exp12_centered_orth_d.csv")

    print("[C] gated m-sweep ...", flush=True)
    rows, tr = run_gated_msweep()
    _save(rows, "exp12_gatedm.csv")
    _save(tr, "exp12_gatedm_trials.csv")

    print("[C] gate m-sweep ...", flush=True)
    rows, tr = run_gate_msweep()
    _save(rows, "exp12_gatem.csv")
    _save(tr, "exp12_gatem_trials.csv")

    print("[C] latch x N ...", flush=True)
    lrows = []
    for mode in ("fixed-1.0", "fixed-0.99", "oracle"):
        for N in (512, 1024, 2048, 4096, 8192):
            accs = [1.0 if run_selective_trial(N, 16, SEED + t, mode=mode,
                                               position_frac=0.1)["recovered"]
                    else 0.0 for t in range(10)]
            lrows.append({"mode": mode, "N": N, "trials": 10,
                          "accuracy": float(np.mean(accs)),
                          "experiment": "latch_N"})
        print(f"  {mode} done", flush=True)
    _save(lrows, "exp12_latchN.csv")

    print("[C] survival pattern ...", flush=True)
    rows, tr = run_survival()
    _save(rows, "exp12_survival.csv")
    _save(tr, "exp12_survival_trials.csv")

    print("[C] cosine drift ...", flush=True)
    rows, tr = run_cosine_sweep()
    _save(rows, "exp12_cosine.csv")
    _save(tr, "exp12_cosine_trials.csv")
    print("[C] done.", flush=True)


# ================================================================ STAGE D
def stageD():
    from experiments.variants import run_compress_sweep
    from experiments.gated_recall import run_gamma_sweep
    from experiments.mitigations import run_selective_sweep
    from experiments.associative_recall import (run_single_trial,
                                                run_dimensionality_sweep)
    from experiments.dissipation import run_dissipation_sweep
    from experiments.variants import run_perchannel_sweep
    from experiments.context_length import run_single_length
    from experiments.softmax_baseline import run_single_trial as sm_single
    from experiments.numerical_saturation import shrinking_update_experiment
    from experiments.finalbatch import (run_bf16, run_chunk_summary,
                                        run_stacked, run_frozen_calib,
                                        run_random_gate)
    from experiments.mitigations import run_hybrid_sweep

    print("[D] compress top-up x40 ...", flush=True)
    rows, tr = run_compress_sweep(trials=40)
    _save(rows, "exp11_compress.csv")
    _save(tr, "exp11_compress_trials.csv")

    print("[D] multi-seed headline probes ...", flush=True)
    ms = []
    for s in (43, 44):
        for r in run_gamma_sweep(trials=10, seed=s):
            r["seed"] = s
            r["probe"] = "gated_gamma"
            ms.append(r)
        for r in run_selective_sweep(trials=10, seed=s):
            r["seed"] = s
            r["probe"] = "selective"
            ms.append(r)
        for r in run_perchannel_sweep(trials=10, seed=s)[0]:
            r["seed"] = s
            r["probe"] = "perchannel"
            ms.append(r)
        for r in run_dimensionality_sweep([2, 4, 8, 16, 32, 64, 128],
                                          dim_values=(16,), trials=10, seed=s,
                                          feature_map="centered_elu"):
            r["seed"] = s
            r["probe"] = "centered_d16"
            ms.append(r)
        for r in run_dissipation_sweep(trials=10, seed=s):
            r["seed"] = s
            r["probe"] = "dissipation"
            ms.append(r)
        print(f"  seed={s} done", flush=True)
    _save(ms, "exp12_multiseed.csv")

    print("[D] bf16 ...", flush=True)
    rows, _ = run_bf16()
    _save(rows, "exp12_bf16.csv")
    print(" ", rows)

    print("[D] chunk-with-summary ...", flush=True)
    rows, _ = run_chunk_summary()
    _save(rows, "exp12_chunksummary.csv")

    print("[D] old-exp trial logs ...", flush=True)
    t1 = []
    for d in (8, 16, 32, 64):
        for m in (2, 4, 8, 16, 32, 64, 128):
            for t in range(25):
                r = run_single_trial(m, d, SEED + t)
                t1.append({"m": m, "dim": d, "trial": t, "seed": SEED + t,
                           "linear": r["linear_accuracy"],
                           "softmax": r["softmax_accuracy"],
                           "state_norm": r["state_norm"]})
    _save(t1, "exp01_trials.csv")
    print("  exp01 done", flush=True)
    t2 = []
    for N in (128, 512, 1024, 4096):
        for pf in (0.1, 0.5, 0.9):
            for t in range(10):
                r = run_single_length(N, 16, SEED + t, position_frac=pf)
                t2.append({"N": N, "position_frac": pf, "trial": t,
                           "recovered": float(r["recovered"]),
                           "error": r["retrieval_error"],
                           "state_norm": r["state_norm"]})
    _save(t2, "exp02_trials.csv")
    print("  exp02 done", flush=True)
    t4 = []
    for N in (128, 512, 1024, 4096):
        for pf in (0.1, 0.5, 0.9):
            for t in range(10):
                r = sm_single(N, 16, SEED + t, position_frac=pf)
                t4.append({"N": N, "position_frac": pf, "trial": t,
                           "linear": float(r["linear_recovered"]),
                           "softmax": float(r["softmax_recovered"])})
    _save(t4, "exp04_trials.csv")
    print("  exp04 done", flush=True)
    t3 = []
    for s in range(42, 47):
        for r in shrinking_update_experiment(
                precisions=("float64", "float32", "float16"), seed=s):
            r["seed"] = s
            t3.append(r)
    _save(t3, "exp03_trials.csv")
    print("  exp03 done", flush=True)

    print("[D] hybrid @N4096 ...", flush=True)
    _save(run_hybrid_sweep(windows=(0, 32, 128, 512), N=4096, trials=6),
          "exp12_hybridN.csv")

    print("[D] stacked ...", flush=True)
    rows, _ = run_stacked()
    _save(rows, "exp12_stacked.csv")

    print("[D] frozen-calib ...", flush=True)
    rows, _ = run_frozen_calib()
    _save(rows, "exp12_frozen.csv")

    print("[D] random-gate ...", flush=True)
    rows, _ = run_random_gate()
    _save(rows, "exp12_randomgate.csv")

    print("[D] OOM boundary probe (subprocess, isolated) ...", flush=True)
    oom = [{"N": 32768, "status": "skipped-by-analysis",
            "detail": "4.3GB score matrix exceeds 3.7GB RAM by construction",
            "bytes_theory": 32768 * 32768 * 4}]
    code = ("import numpy as np, time; N=16384; rng=np.random.default_rng(0); "
            "K=rng.standard_normal((N,16)).astype(np.float32); "
            "V=rng.standard_normal((N,16)).astype(np.float32); "
            "t0=time.perf_counter(); A=K@K.T/np.sqrt(16.); "
            "A=np.exp(A-A.max(axis=1,keepdims=True)); "
            "A/=A.sum(axis=1,keepdims=True); O=A@V; "
            "print('OK', round(time.perf_counter()-t0,1), float(O.sum()))")
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=420, cwd=ROOT)
        ok = proc.returncode == 0 and "OK" in (proc.stdout or "")
        oom.append({"N": 16384,
                    "status": "ok" if ok else "failed",
                    "detail": ((proc.stdout or "") + (proc.stderr or ""))[-300:],
                    "bytes_theory": 16384 * 16384 * 4})
        print("  N=16384:", oom[-1]["status"], flush=True)
    except subprocess.TimeoutExpired:
        oom.append({"N": 16384, "status": "timeout-hang",
                    "detail": "no result in 420s (swap death counts as boundary)",
                    "bytes_theory": 16384 * 16384 * 4})
        print("  N=16384: timeout", flush=True)
    _save(oom, "exp12_oom.csv")
    print("[D] done.", flush=True)


# ================================================================ FINALIZE
def finalize():
    from statistics import NormalDist
    import math
    print("[finalize] fits, tests, CIs, figures ...", flush=True)

    # ---- A7: horizon 50%-crossing fit on existing horizon curves ----
    dfh = pd.read_csv(OUT / "exp11_horizon.csv")
    cross = {}
    for g, sub in dfh.groupby("gamma"):
        x = sub.sort_values("N")
        Ns, acc = x["N"].tolist(), x["accuracy"].tolist()
        cN = None
        for i in range(1, len(Ns)):
            if acc[i - 1] >= 0.5 >= acc[i] and acc[i - 1] > acc[i]:
                f = (acc[i - 1] - 0.5) / max(acc[i - 1] - acc[i], 1e-12)
                cN = Ns[i - 1] + f * (Ns[i] - Ns[i - 1])
                break
        elif_hi = None
        cross[str(g)] = {"N50": cN, "inv_1mg": 1.0 / (1.0 - g),
                         "fp32_bound": 16.6355 / (1.0 - g)}
    print("  horizon crossings:", cross, flush=True)

    # ---- A10: Welch t-tests from summary stats (normal approx p) ----
    def welch(m1, s1, n1, m2, s2, n2):
        se = math.sqrt(s1 * s1 / n1 + s2 * s2 / n2)
        if se == 0:
            return 0.0, 1.0
        t = (m1 - m2) / se
        p = 2 * (1 - NormalDist().cdf(abs(t)))
        return round(t, 3), round(p, 4)

    def cell(csv, filt, mean_c, std_c, n_c="trials"):
        df = pd.read_csv(OUT / csv)
        for k, v in filt.items():
            df = df[df[k] == v]
        r = df.iloc[0]
        return float(r[mean_c]), float(r[std_c]), int(r[n_c])

    tests = []
    pairs = [
        ("relu>elu+1@m8", "exp6_featuremaps.csv", {"feature_map": "relu", "m": 8, "dim": 16},
         "exp6_featuremaps.csv", {"feature_map": "elu+1", "m": 8, "dim": 16},
         "linear_mean", "linear_std"),
        ("centered>elu+1@m16", "exp11_centered.csv", {"m": 16},
         "exp1_dimensionality.csv", {"dim": 16, "m": 16},
         "linear_mean", "linear_std"),
        ("centered-orth>elu-orth@m16", "exp11_centered_orth.csv", {"m": 16},
         "exp7_orthkeys.csv", {"m": 16}, "linear_mean", "linear_std"),
        ("perchannel>scalar.999", "exp11_perchannel.csv", {"mode": "perchannel"},
         "exp11_perchannel.csv", {"mode": "scalar-0.999"},
         "accuracy", "std"),
        ("oracle>fixed.99", "exp8_selective.csv", {"mode": "oracle"},
         "exp8_selective.csv", {"mode": "fixed-0.99"}, "accuracy", "std"),
        ("A8>random", "exp11_compress.csv", {"alphabet": 8},
         "exp11_compress.csv", {"alphabet": -1}, "accuracy", "std"),
        ("hybrid-w128>w0-recent", "exp8_hybrid.csv",
         {"window": 128, "position_frac": 0.9},
         "exp8_hybrid.csv", {"window": 0, "position_frac": 0.9},
         "accuracy", "std"),
        ("multihead-h1~h4@m16", "exp11_multihead.csv",
         {"heads": 1, "m": 16}, "exp11_multihead.csv",
         {"heads": 4, "m": 16}, "accuracy", "std"),
        ("hardgate~off@m8", "exp11_novelty.csv", {"mode": "hard-16-0.8", "m": 8},
         "exp11_novelty.csv", {"mode": "off", "m": 8}, "accuracy", "std"),
    ]
    for name, f1, fl1, f2, fl2, mc, sc in pairs:
        m1, s1, n1 = cell(f1, fl1, mc, sc)
        m2, s2, n2 = cell(f2, fl2, mc, sc)
        t, p = welch(m1, s1, n1, m2, s2, n2)
        tests.append({"comparison": name, "t": t, "p_normal_approx": p,
                      "n1": n1, "n2": n2})
        print(f"  {name}: t={t} p~{p}", flush=True)
    _save(tests, "exp12_ttests.csv")

    # ---- B15: CI table over every aggregate row with (mean,std,trials) ----
    ci_rows = []
    for csv in sorted(OUT.glob("exp*.csv")):
        if "trials" in csv.name or csv.name == "exp12_CIs.csv":
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if "trials" not in df.columns:
            continue
        pairs = [(c[:-4], c) for c in df.columns if c.endswith("_std")]
        if "std" in df.columns and "accuracy" in df.columns:
            pairs.append(("accuracy", "std"))  # newer modules use bare "std"
        id_cols = [c for c in ("m", "N", "dim", "gamma", "mode", "precision",
                               "sigma", "window", "chunk", "heads", "head_dim",
                               "alphabet", "shift", "position_frac", "seed",
                               "probe", "feature_map") if c in df.columns]
        for _, r in df.iterrows():
            for base, sc in pairs:
                if base not in df.columns:
                    continue
                mean = float(r[base])
                sd = float(r[sc])
                if not (np.isfinite(mean) and np.isfinite(sd)):
                    continue
                n = int(r["trials"])
                ci = 1.96 * sd / math.sqrt(max(n, 1))
                label = ",".join(f"{c}={r[c]}" for c in id_cols)
                ci_rows.append({"file": csv.name, "cell": label,
                                "metric": base, "mean": round(mean, 4),
                                "ci95": round(ci, 4)})
    _save(ci_rows, "exp12_CIs.csv")
    print(f"  CI rows: {len(ci_rows)}", flush=True)

    # ---- figures ----
    figs = []

    dfc = pd.read_csv(OUT / "exp12_centered_d.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, sub in dfc.groupby("dim"):
        x = sub.sort_values("m")
        ax.errorbar(x["m"], x["linear_mean"], yerr=x["linear_std"],
                    marker="o", capsize=3, label=f"centered d={d}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp12a: centered m* scaling with d")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12a_centered_d.png", fig))

    dfd = pd.read_csv(OUT / "exp12_centered_orth_d.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, sub in dfd.groupby("dim"):
        x = sub.sort_values("m")
        ax.errorbar(x["m"], x["linear_mean"], yerr=x["linear_std"],
                    marker="o", capsize=3, label=f"orth+centered d={d}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp12a2: orth+centered knee vs d")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12a2_centered_orth_d.png", fig))

    for csv, fname, title, grp, xl in [
            ("exp12_gatedm.csv", "exp12b_gatedm.png",
             "Exp12b: recall density under decay (recent queries)", "gamma", "m"),
            ("exp12_gatem.csv", "exp12c_gatem.png",
             "Exp12c: m-sweep with novelty gate", "mode", "m")]:
        df = pd.read_csv(OUT / csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, sub in df.groupby(grp):
            x = sub.sort_values("m")
            ax.errorbar(x["m"], x["accuracy"], yerr=x["std"], marker="o",
                        capsize=3, label=f"{grp}={key}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xl)
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs.append((fname, fig))

    dfl = pd.read_csv(OUT / "exp12_latchN.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in dfl.groupby("mode"):
        x = sub.sort_values("N")
        ax.plot(x["N"], x["accuracy"], marker="o", label=mode)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Early-marker accuracy")
    ax.set_title("Exp12d: latch retention vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12d_latchN.png", fig))

    dfs = pd.read_csv(OUT / "exp12_survival.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"Q{q}" for q in dfs["quintile"]], dfs["accuracy"],
           yerr=dfs["std"], capsize=4, color="teal")
    ax.set_xlabel("Stored-position quintile (Q0=first)")
    ax.set_ylabel("Per-binding accuracy (m=32)")
    ax.set_title("Exp12e: who survives? (catastrophic pattern test)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp12e_survival.png", fig))

    dfco = pd.read_csv(OUT / "exp12_cosine.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dfco["N"], dfco["cosine_mean"], yerr=dfco["cosine_std"],
                marker="o", capsize=4, label="cosine(out, marker)")
    ax.errorbar(dfco["N"], dfco["accuracy"], marker="s", label="binary accuracy")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Retrieval quality")
    ax.set_title("Exp12f: cosine drift (unsaturated metric)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12f_cosine.png", fig))

    dfb = pd.read_csv(OUT / "exp12_bf16.csv")
    dfb = dfb[dfb.get("status", 0).isna() if "status" in dfb.columns
              else [True] * len(dfb)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dfb["acc_dtype"], dfb["rel_error_vs_fp64"], color="purple")
    ax.set_ylabel("Relative error vs fp64")
    ax.set_title("Exp12g: bf16/fp16/fp32 accumulation")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(dfb["rel_error_vs_fp64"]):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    figs.append(("exp12g_bf16.png", fig))

    dfcs = pd.read_csv(OUT / "exp12_chunksummary.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = dfcs.pivot(index="mode", columns="position_frac", values="accuracy")
    piv.plot(kind="bar", ax=ax, color=["steelblue", "orange"])
    ax.set_ylabel("Accuracy (N=4096)")
    ax.set_title("Exp12h: chunk zero-reset vs summary vs none")
    ax.legend(["early", "recent"])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp12h_chunksummary.png", fig))

    dfhn = pd.read_csv(OUT / "exp12_hybridN.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for pf, sub in dfhn.groupby("position_frac"):
        x = sub.sort_values("window")
        ax.errorbar(x["window"], x["accuracy"], yerr=x["std"], marker="o",
                    capsize=4, label=f"pos={pf}")
    ax.set_xlabel("Window w (N=4096)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp12i: hybrid window scaling at length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12i_hybridN.png", fig))

    dfst = pd.read_csv(OUT / "exp12_stacked.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = dfst.pivot(index="mode", columns="position_frac", values="accuracy")
    piv.plot(kind="bar", ax=ax, color=["steelblue", "orange"])
    ax.set_ylabel("Accuracy (N=4096, decayed+hybrid)")
    ax.set_title("Exp12j: stacked mitigations")
    ax.legend(["early", "recent"])
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp12j_stacked.png", fig))

    dffr = pd.read_csv(OUT / "exp12_frozen.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(dffr["mode"], dffr["accuracy"], yerr=dffr["std"], capsize=5,
           color=["green", "red"])
    ax.set_ylabel("Post-shift marker accuracy")
    ax.set_title("Exp12k: frozen calibration (training-boundary toy)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp12k_frozen.png", fig))

    dfrg = pd.read_csv(OUT / "exp12_randomgate.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in dfrg.groupby("mode"):
        x = sub.sort_values("N")
        ax.plot(x["N"], x["accuracy"], marker="o", label=mode)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Early-marker accuracy")
    ax.set_title("Exp12l: gating must be informed (random vs oracle)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12l_randomgate.png", fig))

    dfoo = pd.read_csv(OUT / "exp12_oom.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax.plot(Ns, Ns * Ns * 4.0, color="darkred", label="score-matrix bytes (theory)")
    ax.axhline(3.7e9, color="black", linestyle="--", label="3.7GB machine RAM")
    for _, r in dfoo.iterrows():
        ax.scatter([r["N"]], [r["bytes_theory"]],
                   s=120, c="green" if r["status"] == "ok" else "red",
                   zorder=5, label=f"N={int(r['N'])} {r['status']}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Bytes (log)")
    ax.set_title("Exp12m: softmax scoring hits the hardware wall")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp12m_oom.png", fig))

    # regen compress figure with topped-up data
    dfcc = pd.read_csv(OUT / "exp11_compress.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dfcc["alphabet"].astype(str), dfcc["accuracy"], yerr=dfcc["std"],
           capsize=4, color="steelblue")
    ax.set_xlabel("Alphabet size (-1 = fully random)")
    ax.set_ylabel("Accuracy (N=2048, 40 trials)")
    ax.set_title("Exp11a: compressible streams survive (top-up)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp11a_compress.png", fig))

    for name, fig in figs:
        save_high_res(fig, FIG / name, dpi=300, dark_mode=False)
        plt.close(fig)
    print(f"  saved {len(figs)} figures.", flush=True)

    # ---- bundle updates ----
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["batch4_horizon_crossings"] = cross
    summary["batch4_ttests"] = tests
    summary["batch4_ci_rows"] = len(ci_rows)
    summary["batch4_unrunnable"] = {
        "real_weights": "no torch/GPU/RAM — documented, not attempted",
        "learned_gating": "no training loop — random-gate control bounds it instead",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    config["batch4"] = {"seed": SEED, "note": "final paper-gap audit",
                        "new_module": "experiments/finalbatch.py"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2),
                                     encoding="utf-8")
    readme = (OUT / "README.md").read_text(encoding="utf-8")
    if "## Batch-4 final audit" in readme:
        readme = readme.split("## Batch-4 final audit")[0].rstrip() + "\n"
    with (OUT / "README.md").open("w", encoding="utf-8") as f:
        f.write(readme)
        f.write("\n## Batch-4 final audit (seed 42)\n\n")
        f.write("- Centered m* scaling, gated/gate m-sweeps, latch-N, survival "
                "quintiles, cosine drift: exp12_centered_d(.csv), exp12_gatedm, "
                "exp12_gatem, exp12_latchN, exp12_survival, exp12_cosine.\n")
        f.write("- Compress top-up x40, multi-seed batch2/3, bf16, chunk-summary, "
                "old-exp trial logs (exp01-04_trials), hybrid-N4096, stacked, "
                "frozen-calib, random-gate, OOM probe: exp12_*.csv.\n")
        f.write("- Horizon 50%-crossings vs 1/(1-g) theory, Welch t-tests "
                "(exp12_ttests.csv), CI table over all aggregates (exp12_CIs.csv).\n")
        f.write("- Deliberately unrunnable here: real model weights (no torch/GPU/RAM), "
                "learned gating (no training loop) — bounded by oracle/random controls instead.\n")
    print("[finalize] bundle updated.", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for stage in ([which] if which != "all" else
                  ["stageC", "stageD", "finalize"]):
        {"stageC": stageC, "stageD": stageD,
         "finalize": finalize}[stage]()
