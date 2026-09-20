"""Batch-3 runs: paper-gap audit probes. Staged for timeouts.

  stageA : per-channel decay, novelty gate, multi-head, compressibility
  stageB : shift, noisy queries, centered map, top-ups, horizon curves,
           distractor sensitivity, full-sequence timing
  finalize : figures + bundle summary/README/config updates

Usage: .\\.venv\\Scripts\\python run_batch3.py [stageA|stageB|finalize|all]
"""

import json
import sys
import time
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


def _p(frac):
    print(f"    ...{frac * 100:.0f}%", flush=True)


# ================================================================ STAGE A
def stageA():
    from experiments.variants import (run_perchannel_sweep,
                                      run_novelty_sweep, run_multihead_sweep,
                                      run_compress_sweep)
    print("[A] per-channel decay ...", flush=True)
    rows, tr = run_perchannel_sweep()
    _save(rows, "exp11_perchannel.csv")
    _save(tr, "exp11_perchannel_trials.csv")
    print("[A] novelty gate ...", flush=True)
    rows, tr = run_novelty_sweep()
    _save(rows, "exp11_novelty.csv")
    _save(tr, "exp11_novelty_trials.csv")
    print("[A] multi-head banks ...", flush=True)
    rows, tr = run_multihead_sweep()
    _save(rows, "exp11_multihead.csv")
    _save(tr, "exp11_multihead_trials.csv")
    print("[A] compressibility ...", flush=True)
    rows, tr = run_compress_sweep()
    _save(rows, "exp11_compress.csv")
    _save(tr, "exp11_compress_trials.csv")
    print("[A] done.", flush=True)


# ================================================================ STAGE B
def stageB():
    from experiments.variants import (run_shift_sweep, run_noisy_sweep)
    from experiments.associative_recall import run_dimensionality_sweep
    from experiments.dissipation import run_orth_sweep
    from experiments.gated_recall import run_gated_trial, run_gamma_sweep
    from experiments.mitigations import run_selective_sweep
    from experiments.context_length import run_single_length

    print("[B] shift probe ...", flush=True)
    rows, tr = run_shift_sweep()
    _save(rows, "exp11_shift.csv")
    _save(tr, "exp11_shift_trials.csv")

    print("[B] noisy queries ...", flush=True)
    rows, tr = run_noisy_sweep()
    _save(rows, "exp11_noisy.csv")
    _save(tr, "exp11_noisy_trials.csv")

    print("[B] centered map (random + orth keys) ...", flush=True)
    rows = run_dimensionality_sweep([2, 4, 8, 16, 32, 64, 128], dim_values=(16,),
                                    trials=15, seed=SEED,
                                    feature_map="centered_elu")
    for r in rows:
        r["experiment"] = "centered_random"
    _save(rows, "exp11_centered.csv")
    rows = run_orth_sweep(feature_map="centered_elu")
    _save(rows, "exp11_centered_orth.csv")

    print("[B] top-ups: selective x20 ...", flush=True)
    _save(run_selective_sweep(trials=20), "exp8_selective.csv")
    print("[B] top-ups: gated gamma x20 ...", flush=True)
    _save(run_gamma_sweep(trials=20), "exp5_gated_decay.csv")

    print("[B] horizon curves ...", flush=True)
    hrows = []
    for g in (0.99, 0.999, 0.9999):
        for N in (512, 1024, 2048, 4096, 8192):
            accs = [1.0 if run_gated_trial(N, 16, SEED + t, gamma=g,
                                           position_frac=0.1)["recovered"]
                    else 0.0 for t in range(6)]
            hrows.append({"gamma": g, "N": N, "trials": 6,
                          "accuracy": float(np.mean(accs)),
                          "experiment": "horizon_curve"})
        print(f"  gamma={g} done", flush=True)
    _save(hrows, "exp11_horizon.csv")

    print("[B] distractor sensitivity ...", flush=True)
    drows = []
    for sc in (0.001, 0.01, 0.1):
        for N in (1024, 4096):
            for pf in (0.1, 0.9):
                accs = [1.0 if run_single_length(
                    N, 16, SEED + t, position_frac=pf,
                    distractor_scale=sc)["recovered"] else 0.0
                    for t in range(6)]
                drows.append({"distractor_scale": sc, "N": N,
                              "position_frac": pf, "trials": 6,
                              "accuracy": float(np.mean(accs)),
                              "experiment": "distractor_sensitivity"})
        print(f"  scale={sc} done", flush=True)
    _save(drows, "exp11_distractor.csv")

    print("[B] full-sequence timing ...", flush=True)
    from models.linear_attention import elu_plus_one
    trows = []
    for N in (256, 512, 1024, 2048, 4096, 8192):
        tr = 2 if N >= 8192 else 3
        lt, st_ = [], []
        for t in range(tr):
            rng = np.random.default_rng(1000 + t)
            K = rng.standard_normal((N, 16)).astype(np.float32)
            V = (rng.standard_normal((N, 16)) * 0.01).astype(np.float32)
            t0 = time.perf_counter()
            S = np.zeros((16, 16), dtype=np.float32)
            Z = np.zeros((16,), dtype=np.float32)
            chk = 0.0
            for k, v in zip(K, V):
                fk = elu_plus_one(k.astype(np.float64)).astype(np.float32)
                S += np.outer(fk, v)
                Z += fk
                o = (S.T @ fk) / float(Z @ fk)
                chk += float(o.sum())
            lt.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            A = K @ K.T / np.sqrt(16.0)
            A = np.exp(A - A.max(axis=1, keepdims=True))
            A = A / A.sum(axis=1, keepdims=True)
            O = A @ V
            chk += float(O.sum())
            st_.append(time.perf_counter() - t0)
            del A, O
        trows.append({"N": N, "trials": tr,
                      "linear_time_s": float(np.mean(lt)),
                      "softmax_time_s": float(np.mean(st_)),
                      "linear_bytes_theory": 16 * 16 * 4 + 16 * 4,
                      "softmax_bytes_theory": N * N * 4,
                      "experiment": "full_timing"})
        print(f"  N={N}: lin={np.mean(lt):.3f}s sm={np.mean(st_):.3f}s",
              flush=True)
    _save(trows, "exp11_timing.csv")
    print("[B] done.", flush=True)


# ================================================================ FINALIZE
def finalize():
    print("[finalize] figures + bundle update ...", flush=True)
    figs = []

    def bars(csv, x, y, title, xlabel, fname, color="steelblue"):
        df = pd.read_csv(OUT / csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        err = df["std"] if "std" in df.columns else None
        ax.bar(df[x].astype(str), df[y], yerr=err, capsize=4, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        figs.append((fname, fig))
        return df

    bars("exp11_compress.csv", "alphabet", "accuracy",
         "Exp11a: compressible streams survive (A=-1: random)",
         "alphabet size (-1 = fully random)", "exp11a_compress.png")
    bars("exp11_perchannel.csv", "mode", "accuracy",
         "Exp11b: per-channel decay vs scalar (early marker, N=4096)",
         "decay mode", "exp11b_perchannel.png", color="seagreen")

    dfn = pd.read_csv(OUT / "exp11_novelty.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, sub in dfn.groupby("mode"):
        x = sub.sort_values("m")
        ax.errorbar(x["m"], x["accuracy"], yerr=x["std"], marker="o",
                    capsize=3, label=f"gate={mode}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Accuracy (d=16)")
    ax.set_title("Exp11c: novelty gating vs recall density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11c_novelty.png", fig))

    dfmh = pd.read_csv(OUT / "exp11_multihead.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (h, dh), sub in dfmh.groupby(["heads", "head_dim"]):
        x = sub.sort_values("m")
        ax.errorbar(x["m"], x["accuracy"], yerr=x["std"], marker="o",
                    capsize=3, label=f"{h}x{dh} (=4096 params)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp11d: multi-head banks at equal budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11d_multihead.png", fig))

    dfsh = pd.read_csv(OUT / "exp11_shift.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dfsh["shift"].astype(str), dfsh["accuracy"], yerr=dfsh["std"],
                marker="o", capsize=4, color="darkred")
    ax.set_xlabel("Second-half key mean shift")
    ax.set_ylabel("Pre-shift marker accuracy (N=2048)")
    ax.set_title("Exp11e: kernel distribution-shift probe")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11e_shift.png", fig))

    dfnz = pd.read_csv(OUT / "exp11_noisy.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(dfnz["sigma"], dfnz["linear_accuracy"], yerr=dfnz["linear_std"],
                marker="o", capsize=4, label="linear")
    ax.errorbar(dfnz["sigma"], dfnz["softmax_accuracy"],
                yerr=dfnz["softmax_std"], marker="s", capsize=4,
                label="softmax")
    ax.set_xlabel("Query noise sigma")
    ax.set_ylabel("Accuracy (N=1024)")
    ax.set_title("Exp11f: noisy-query robustness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11f_noisy.png", fig))

    dfc = pd.read_csv(OUT / "exp11_centered.csv")
    dfco = pd.read_csv(OUT / "exp11_centered_orth.csv")
    df7o = pd.read_csv(OUT / "exp7_orthkeys.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for fm, sub in dfc.groupby("feature_map"):
        x = sub.sort_values("m")
        a1.errorbar(x["m"], x["linear_mean"], yerr=x["linear_std"],
                    marker="o", capsize=3, label=fm)
    a1.set_xscale("log", base=2)
    a1.set_xlabel("m")
    a1.set_ylabel("Accuracy (random keys, d=16)")
    a1.set_title("Centered vs elu+1")
    a1.legend(fontsize=8)
    a1.grid(True, alpha=0.3)
    a2.errorbar(dfco["m"], dfco["linear_mean"], yerr=dfco["linear_std"],
                marker="o", capsize=3, label="orth keys + centered_elu")
    a2.errorbar(df7o["m"], df7o["linear_mean"], yerr=df7o["linear_std"],
                marker="x", capsize=3, label="orth keys + elu+1 (Exp7c)")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("m")
    a2.set_title("Positive-half test: centered features?")
    a2.legend(fontsize=8)
    a2.grid(True, alpha=0.3)
    fig.suptitle("Exp11g: centered feature map (seed=42)")
    fig.tight_layout()
    figs.append(("exp11g_centered.png", fig))

    dfh = pd.read_csv(OUT / "exp11_horizon.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, sub in dfh.groupby("gamma"):
        x = sub.sort_values("N")
        ax.plot(x["N"], x["accuracy"], marker="o", label=f"gamma={g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Early-marker accuracy")
    ax.set_title("Exp11h: forgetting curves per decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11h_horizon.png", fig))

    dfd = pd.read_csv(OUT / "exp11_distractor.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (N, pf), sub in dfd.groupby(["N", "position_frac"]):
        x = sub.sort_values("distractor_scale")
        ax.plot(x["distractor_scale"], x["accuracy"], marker="o",
                label=f"N={N} pos={pf}")
    ax.set_xscale("log")
    ax.set_xlabel("Distractor scale")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp11i: distractor-scale sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp11i_distractor.png", fig))

    dft = pd.read_csv(OUT / "exp11_timing.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(dft["N"], dft["linear_time_s"], marker="o", label="linear O(N)")
    a1.plot(dft["N"], dft["softmax_time_s"], marker="s",
            label="softmax full scoring")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.set_xlabel("N")
    a1.set_ylabel("Seconds (log)")
    a1.set_title("Exp11j: full-sequence scoring time")
    a1.legend()
    a1.grid(True, alpha=0.3)
    a2.plot(dft["N"], dft["linear_bytes_theory"], marker="o", label="linear")
    a2.plot(dft["N"], dft["softmax_bytes_theory"], marker="s",
            label="softmax N^2")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlabel("N")
    a2.set_ylabel("Score-state bytes, theory (log)")
    a2.set_title("Memory scaling")
    a2.legend()
    a2.grid(True, alpha=0.3)
    fig.suptitle("Exp11j: asymptotic scaling (d=16, fp32)")
    fig.tight_layout()
    figs.append(("exp11j_timing.png", fig))

    # regenerate stale figures from topped-up CSVs
    df8s = pd.read_csv(OUT / "exp8_selective.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df8s["mode"], df8s["accuracy"], yerr=df8s["std"], capsize=5,
           color=["steelblue", "orange", "green"])
    ax.set_ylabel("Early-marker accuracy (N=4096, 20 trials)")
    ax.set_title("Exp8c: oracle selective latch vs fixed gamma")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp8c_selective.png", fig))

    df5 = pd.read_csv(OUT / "exp5_gated_decay.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df5["gamma"].astype(str), df5["accuracy"], yerr=df5["std"],
                marker="o", capsize=4)
    ax.set_xlabel("gamma")
    ax.set_ylabel("Early-marker accuracy (N=4096, 20 trials)")
    ax.set_title("Exp5: forgetting vs decay")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp5_gamma_horizon.png", fig))

    for name, fig in figs:
        save_high_res(fig, FIG / name, dpi=300, dark_mode=False)
        plt.close(fig)
    print(f"  saved {len(figs)} figures.", flush=True)

    # ---- bundle updates ----
    dfc1 = pd.read_csv(OUT / "exp11_compress.csv")
    dfpc = pd.read_csv(OUT / "exp11_perchannel.csv")
    dfmh1 = pd.read_csv(OUT / "exp11_multihead.csv")
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["batch3_compress_by_alphabet"] = {
        str(int(r["alphabet"])): round(float(r["accuracy"]), 3)
        for _, r in dfc1.iterrows()}
    summary["batch3_perchannel_by_mode"] = {
        str(r["mode"]): round(float(r["accuracy"]), 3)
        for _, r in dfpc.iterrows()}
    mh4 = dfmh1[(dfmh1["heads"] == 4) & (dfmh1["m"] == 16)]["accuracy"].iloc[0]
    mh1 = dfmh1[(dfmh1["heads"] == 1) & (dfmh1["m"] == 16)]["accuracy"].iloc[0]
    summary["batch3_multihead_m16"] = {"h1_d64": round(float(mh1), 3),
                                       "h4_d32": round(float(mh4), 3)}
    summary["batch3_topups"] = {"selective_trials": 20, "gated_trials": 20}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    config["batch3"] = {"seed": SEED, "note": "paper-gap audit probes",
                        "new_modules": ["experiments/variants.py",
                                        "centered_elu map"]}
    (OUT / "config.json").write_text(json.dumps(config, indent=2),
                                     encoding="utf-8")
    readme = (OUT / "README.md").read_text(encoding="utf-8")
    if "## Batch-3 audit probes" in readme:
        readme = readme.split("## Batch-3 audit probes")[0].rstrip() + "\n"
    new_files = ["exp11_compress(.csv,_trials.csv)",
                 "exp11_perchannel(.csv,_trials.csv)",
                 "exp11_novelty(.csv,_trials.csv)",
                 "exp11_multihead(.csv,_trials.csv)",
                 "exp11_shift(.csv,_trials.csv)",
                 "exp11_noisy(.csv,_trials.csv)",
                 "exp11_centered.csv",
                 "exp11_centered_orth.csv", "exp11_horizon.csv",
                 "exp11_timing.csv", "exp11_distractor.csv",
                 "exp8_selective.csv + exp5_gated_decay.csv topped to 20 trials"]
    with (OUT / "README.md").open("w", encoding="utf-8") as f:
        f.write(readme)
        f.write("\n## Batch-3 audit probes (seed 42)\n\n")
        f.write(f"- Compressibility (A=alphabet): "
                f"{ {str(int(r['alphabet'])): round(float(r['accuracy']), 2) for _, r in dfc1.iterrows()} }.\n")
        f.write(f"- Per-channel vs scalar: "
                f"{ {str(r['mode']): round(float(r['accuracy']), 2) for _, r in dfpc.iterrows()} }.\n")
        f.write(f"- Multi-head m=16: 1x64={float(mh1):.3f} vs 4x32={float(mh4):.3f}.\n")
        f.write("- Shift/noisy/centered/horizon/timing/distractor: see exp11_*.csv + figures.\n")
        f.write("- Per-trial records: exp11_*_trials.csv enable significance testing for new probes "
                "(older experiments log aggregates only — documented limitation).\n")
        f.write("\nNew files: " + ", ".join(f"`{n}`" for n in new_files) + "\n")
    print("[finalize] bundle updated.", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for stage in ([which] if which != "all" else
                  ["stageA", "stageB", "finalize"]):
        {"stageA": stageA, "stageB": stageB,
         "finalize": finalize}[stage]()
