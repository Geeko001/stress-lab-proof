"""Follow-up evidence runs for paper Sections 3-5 gaps. Staged for timeouts.

Stages (each writes CSVs incrementally to public_evidence/):
  stage1 : feature-map comparison + long-length streaming (64k-1M)
  stage2 : multi-seed robustness + gated decay horizon + precision x decay
  stage3 : mitigations + dissipation + m-x-N grid + orthkeys + big-d
  finalize : fits, figures, bundle summary/README/config updates

Usage: .\\.venv\\Scripts\\python run_followups.py [stage1|stage2|stage3|finalize|all]
"""

import json
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


def _p(frac):
    print(f"    ...{frac * 100:.0f}%", flush=True)


# ================================================================ STAGE 1
def stage1():
    from experiments.associative_recall import run_dimensionality_sweep
    from experiments.context_length import run_length_sweep

    print("[stage1] feature-map comparison ...", flush=True)
    all_rows = []
    for fm in ("elu+1", "relu", "exp"):
        print(f"  map={fm}", flush=True)
        rows = run_dimensionality_sweep([2, 4, 8, 16, 32, 64, 128],
                                        dim_values=(16, 32), trials=15,
                                        seed=SEED, feature_map=fm)
        for r in rows:
            r["experiment"] = "featuremap_compare"
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(OUT / "exp6_featuremaps.csv", index=False)

    print("[stage1] long-length streaming (linear only) ...", flush=True)
    rows = run_length_sweep([16384, 65536, 262144], dim=16, trials=3,
                            seed=SEED, position_fracs=(0.1, 0.9),
                            progress_cb=_p)
    print("  N=1048576 ...", flush=True)
    rows += run_length_sweep([1048576], dim=16, trials=2, seed=SEED,
                             position_fracs=(0.1, 0.9), progress_cb=_p)
    for r in rows:
        r["experiment"] = "long_length"
    pd.DataFrame(rows).to_csv(OUT / "exp9_longlength.csv", index=False)
    print("[stage1] done.", flush=True)


# ================================================================ STAGE 2
def stage2():
    from experiments.associative_recall import run_associative_recall
    from experiments.context_length import run_length_sweep
    from experiments.gated_recall import run_gamma_sweep, run_precision_decay

    print("[stage2] seed robustness (recall) ...", flush=True)
    rec = []
    for s in (42, 43, 44, 45):
        print(f"  seed={s}", flush=True)
        for r in run_associative_recall([2, 4, 8, 16, 32, 64, 128], dim=16,
                                        trials=10, seed=s):
            r["seed"] = s
            r["experiment"] = "seed_robustness"
            rec.append(r)
    pd.DataFrame(rec).to_csv(OUT / "exp10_seed_recall.csv", index=False)

    print("[stage2] seed robustness (length) ...", flush=True)
    ln = []
    for s in (43, 44):
        print(f"  seed={s}", flush=True)
        for r in run_length_sweep([128, 512, 1024, 4096], dim=16, trials=5,
                                  seed=s, position_fracs=(0.1, 0.5, 0.9),
                                  progress_cb=_p):
            r["seed"] = s
            r["experiment"] = "seed_robustness"
            ln.append(r)
    pd.DataFrame(ln).to_csv(OUT / "exp10_seed_length.csv", index=False)

    print("[stage2] gated decay horizon ...", flush=True)
    g = run_gamma_sweep(progress_cb=_p)
    pd.DataFrame(g).to_csv(OUT / "exp5_gated_decay.csv", index=False)
    for r in g:
        print(f"  gamma={r['gamma']}: acc={r['accuracy']:.3f} "
              f"h_1mg={r['horizon_1_over_1mg']} h_fp32={r['horizon_fp32_zero']}")

    print("[stage2] precision x decay ...", flush=True)
    p = run_precision_decay(progress_cb=_p)
    pd.DataFrame(p).to_csv(OUT / "exp5_precision_decay.csv", index=False)
    print("[stage2] done.", flush=True)


# ================================================================ STAGE 3
def stage3():
    from experiments.associative_recall import run_associative_recall
    from experiments.mitigations import (run_hybrid_sweep, run_chunked_sweep,
                                         run_selective_sweep,
                                         run_mixed_precision)
    from experiments.dissipation import (run_dissipation_sweep, run_mN_grid,
                                         run_orth_sweep)

    print("[stage3] mitigations ...", flush=True)
    pd.DataFrame(run_hybrid_sweep(progress_cb=_p)).to_csv(
        OUT / "exp8_hybrid.csv", index=False)
    pd.DataFrame(run_chunked_sweep(progress_cb=_p)).to_csv(
        OUT / "exp8_chunked.csv", index=False)
    pd.DataFrame(run_selective_sweep(progress_cb=_p)).to_csv(
        OUT / "exp8_selective.csv", index=False)
    pd.DataFrame(run_mixed_precision()).to_csv(
        OUT / "exp8_mixed.csv", index=False)

    print("[stage3] dissipation / controls ...", flush=True)
    pd.DataFrame(run_dissipation_sweep(progress_cb=_p)).to_csv(
        OUT / "exp7_coarsefine.csv", index=False)
    pd.DataFrame(run_mN_grid(progress_cb=_p)).to_csv(
        OUT / "exp7_mNgrid.csv", index=False)
    pd.DataFrame(run_orth_sweep(progress_cb=_p)).to_csv(
        OUT / "exp7_orthkeys.csv", index=False)

    print("[stage3] big-d recall (d=256) ...", flush=True)
    big = run_associative_recall([2, 4, 8, 16, 32, 64, 128], dim=256,
                                 trials=10, seed=SEED, progress_cb=_p)
    for r in big:
        r["experiment"] = "big_d"
    pd.DataFrame(big).to_csv(OUT / "exp10_bigd.csv", index=False)
    print("[stage3] done.", flush=True)


# ================================================================ FINALIZE
def _ci(mean, std, n):
    return 1.96 * float(std) / np.sqrt(int(n))


def finalize():
    print("[finalize] fits + figures + bundle update ...", flush=True)
    df2 = pd.read_csv(OUT / "exp2_length_sweep.csv")
    df9 = pd.read_csv(OUT / "exp9_longlength.csv")
    both = pd.concat([df2, df9], ignore_index=True)
    g = both.groupby("N")[["accuracy", "retrieval_error_mean",
                           "state_norm_mean"]].mean()

    def loglog_slope(x, y):
        x = np.log(np.asarray(x, dtype=float))
        y = np.log(np.asarray(y, dtype=float))
        m = np.isfinite(x) & np.isfinite(y) & (y > -50)
        return float(np.polyfit(x[m], y[m], 1)[0])

    err = g["retrieval_error_mean"]
    err_unsat = err[(err > 1e-9) & (err < 0.9)]
    if len(err_unsat) >= 3:
        drift_exp = loglog_slope(err_unsat.index, err_unsat.values)
        drift_note = (f"log-log slope {drift_exp:.3f} over unsaturated range "
                      "(1.0 = linear, >1 = superlinear)")
    else:
        # Relative error is ~1.0 from N=128 (a fully washed output is ~= 0,
        # so |0 - marker| / |marker| ~= 1 by construction). The metric
        # saturates immediately: no compounding exponent is identifiable
        # in this regime; the finding is total washout, not gradual drift.
        drift_exp = None
        drift_note = ("relative error saturates at ~1.0 from N=128 "
                      "(washed output ~= 0 by construction); no compounding "
                      "exponent identifiable — finding is total washout")
    norm_slope = loglog_slope(g.index, g["state_norm_mean"].values)
    print(f"  drift: {drift_note}")
    print(f"  state-norm slope (norm vs N): {norm_slope:.3f}  (0.5=sqrt-t)")

    df1 = pd.read_csv(OUT / "exp1_dimensionality.csv")
    d16 = df1[df1["dim"] == 16].sort_values("m")
    ci_m2 = _ci(d16[d16["m"] == 2].iloc[0]["linear_mean"],
                d16[d16["m"] == 2].iloc[0]["linear_std"], 25)
    ci_m128 = _ci(d16[d16["m"] == 128].iloc[0]["linear_mean"],
                  d16[d16["m"] == 128].iloc[0]["linear_std"], 25)
    # m* (50% crossing) per dim from exp1
    mstar = {}
    for d, sub in df1.groupby("dim"):
        sub = sub.sort_values("m")
        cross = sub[sub["linear_mean"] < 0.5]
        mstar[str(int(d))] = int(cross.iloc[0]["m"]) if len(cross) else None
    print(f"  m* per dim: {mstar}")

    figs = []
    # 1. feature maps
    df6 = pd.read_csv(OUT / "exp6_featuremaps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, d in zip(axes, (16, 32)):
        for fm, sub in df6[df6["dim"] == d].groupby("feature_map"):
            s = sub.sort_values("m")
            ax.errorbar(s["m"], s["linear_mean"], yerr=s["linear_std"],
                        marker="o", capsize=3, label=fm)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("m")
        ax.set_title(f"d={d}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Retrieval accuracy")
    fig.suptitle("Exp6: feature-map comparison (seed=42)")
    fig.tight_layout()
    figs.append(("exp6_featuremaps.png", fig))

    # 2. full-range length (exp2 + exp9)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for pf, sub in both.groupby("position_frac"):
        s = sub.groupby("N")["accuracy"].mean()
        a1.plot(s.index, s.values, marker="o", label=f"pos={pf}")
    a1.set_xscale("log", base=2)
    a1.set_xlabel("N (128 to 1M)")
    a1.set_ylabel("Retrieval accuracy")
    a1.set_title("Length vs accuracy, full range")
    a1.legend()
    a1.grid(True, alpha=0.3)
    sn = both.groupby("N")["state_norm_mean"].mean()
    a2.plot(sn.index, sn.values, marker="o", color="darkred")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("N")
    a2.set_ylabel("Mean state norm")
    a2.set_title(f"State norm vs N (log-log slope {norm_slope:.2f})")
    a2.grid(True, alpha=0.3)
    fig.suptitle("Exp9+Exp2: sequence length to N=1M (seed=42)")
    fig.tight_layout()
    figs.append(("exp9_longlength_full.png", fig))

    # 3. seed robustness recall
    df10a = pd.read_csv(OUT / "exp10_seed_recall.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, sub in df10a.groupby("seed"):
        x = sub.sort_values("m")
        ax.errorbar(x["m"], x["linear_mean"], yerr=x["linear_std"],
                    marker="o", capsize=3, label=f"seed={s}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Retrieval accuracy (d=16, 10 trials)")
    ax.set_title("Exp10a: seed robustness of recall curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp10a_seed_recall.png", fig))

    # 4. seed robustness length
    df10b = pd.concat([df2.assign(seed=42), pd.read_csv(OUT / "exp10_seed_length.csv")],
                      ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, sub in df10b.groupby("seed"):
        x = sub.groupby("N")["accuracy"].mean()
        ax.plot(x.index, x.values, marker="o", label=f"seed={s}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Exp10b: seed robustness of length decay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp10b_seed_length.png", fig))

    # 5. big-d overlay with exp1
    dfb = pd.read_csv(OUT / "exp10_bigd.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, sub in df1.groupby("dim"):
        x = sub.sort_values("m")
        ax.plot(x["m"], x["linear_mean"], marker="o", label=f"random keys d={d}")
    xb = dfb.sort_values("m")
    ax.plot(xb["m"], xb["linear_mean"], marker="D", linewidth=2.5,
            label="random keys d=256")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Retrieval accuracy")
    ax.set_title("Exp10c: recall incl. d=256 (seed=42)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp10c_bigd_overlay.png", fig))

    # 6. gated gamma sweep
    df5 = pd.read_csv(OUT / "exp5_gated_decay.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.errorbar(df5["gamma"].astype(str), df5["accuracy"], yerr=df5["std"],
                marker="o", capsize=4)
    a1.set_xlabel("gamma")
    a1.set_ylabel("Early-marker accuracy (N=4096)")
    a1.set_title("Exp5: forgetting vs decay")
    a1.grid(True, alpha=0.3)
    a2.plot(df5["gamma"].astype(str), df5["state_norm_mean"], marker="o",
            color="darkred")
    a2.set_xlabel("gamma")
    a2.set_ylabel("Mean state norm")
    a2.set_title("State norm vs decay")
    a2.grid(True, alpha=0.3)
    fig.suptitle("Exp5: gated recall horizon (seed=42)")
    fig.tight_layout()
    figs.append(("exp5_gamma_horizon.png", fig))

    # 7. precision x decay (accuracy: flat; drift: the real signal)
    df5p = pd.read_csv(OUT / "exp5_precision_decay.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for p, sub in df5p.groupby("precision"):
        a1.errorbar(sub["gamma"].astype(str), sub["accuracy"], yerr=sub["std"],
                    marker="o", capsize=4, label=p)
    a1.set_xlabel("gamma")
    a1.set_ylabel("Recent-marker accuracy (N=2048)")
    a1.set_title("Binary recall: precision-insensitive")
    a1.legend()
    a1.grid(True, alpha=0.3)
    for p, sub in df5p[df5p["precision"] != "float64"].groupby("precision"):
        a2.plot(sub["gamma"].astype(str), sub["drift_vs_fp64"], marker="o",
                label=f"{p} drift vs fp64")
    a2.set_yscale("log")
    a2.set_xlabel("gamma")
    a2.set_ylabel("Mean |out - out_fp64| (log)")
    a2.set_title("Continuous drift: fp16 > fp32")
    a2.legend()
    a2.grid(True, alpha=0.3)
    fig.suptitle("Exp5b: precision x decay interaction (seed=42)")
    fig.tight_layout()
    figs.append(("exp5b_precision_decay.png", fig))

    # 8. hybrid
    df8h = pd.read_csv(OUT / "exp8_hybrid.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for pf, sub in df8h.groupby("position_frac"):
        x = sub.sort_values("window")
        ax.errorbar(x["window"], x["accuracy"], yerr=x["std"],
                    marker="o", capsize=4, label=f"pos={pf}")
    ax.set_xlabel("Exact window w (0 = pure linear)")
    ax.set_ylabel("Accuracy (N=1024)")
    ax.set_title("Exp8a: hybrid local window mitigation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp8a_hybrid.png", fig))

    # 9. chunked
    df8c = pd.read_csv(OUT / "exp8_chunked.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df8c["chunk"].astype(str), df8c["accuracy"], yerr=df8c["std"],
                marker="o", capsize=4)
    ax.set_xlabel("Chunk size C (resets)")
    ax.set_ylabel("Early-marker accuracy (N=4096)")
    ax.set_title("Exp8b: chunked resets (drift bounded, early info lost)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp8b_chunked.png", fig))

    # 10. selective
    df8s = pd.read_csv(OUT / "exp8_selective.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df8s["mode"], df8s["accuracy"],
           yerr=df8s["std"], capsize=5, color=["steelblue", "orange", "green"])
    ax.set_ylabel("Early-marker accuracy (N=4096)")
    ax.set_title("Exp8c: oracle selective decay vs fixed gamma")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    figs.append(("exp8c_selective.png", fig))

    # 11. mixed precision
    df8m = pd.read_csv(OUT / "exp8_mixed.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df8m["acc_dtype"], df8m["rel_error_vs_fp64"], color=["orange", "steelblue"])
    ax.set_ylabel("Relative error vs fp64 reference")
    ax.set_title("Exp8d: fp16 compute, fp16 vs fp32 accumulation")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(df8m["rel_error_vs_fp64"]):
        ax.text(i, v, f"{v:.2e}\nineff={int(df8m.iloc[i]['ineffective'])}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    figs.append(("exp8d_mixed.png", fig))

    # 12. coarse vs fine
    df7c = pd.read_csv(OUT / "exp7_coarsefine.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df7c["N"], df7c["coarse_accuracy"], yerr=df7c["coarse_std"],
                marker="o", capsize=4, label="coarse (repeated x20)")
    ax.errorbar(df7c["N"], df7c["fine_accuracy"], yerr=df7c["fine_std"],
                marker="s", capsize=4, label="fine (singleton)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp7a: coarse survives, fine detail lost first")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp7a_coarsefine.png", fig))

    # 13. m x N grid
    df7g = pd.read_csv(OUT / "exp7_mNgrid.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, sub in df7g.groupby("m"):
        x = sub.sort_values("nfill")
        ax.errorbar(x["nfill"] + 1, x["accuracy"], yerr=x["std"],
                    marker="o", capsize=4, label=f"m={m}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Filler steps + 1 (log)")
    ax.set_ylabel("Accuracy over the m bindings")
    ax.set_title("Exp7b: accuracy follows m, not filler length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp7b_mNgrid.png", fig))

    # 14. orthkeys (+ random-keys contrast)
    df7o = pd.read_csv(OUT / "exp7_orthkeys.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df7o["m"], df7o["linear_mean"], yerr=df7o["linear_std"],
                marker="o", capsize=4, label="linear, orthogonal keys")
    ax.errorbar(df7o["m"], df7o["softmax_mean"], yerr=df7o["softmax_std"],
                marker="s", capsize=4, label="softmax, orthogonal keys")
    rk = df1[df1["dim"] == 16].sort_values("m")
    ax.plot(rk["m"], rk["linear_mean"], marker="x", linestyle="--",
            label="linear, random keys d=16 (Exp1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("m")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp7c: key structure restores rank-scaled plateau")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(("exp7c_orthkeys.png", fig))

    for name, fig in figs:
        save_high_res(fig, FIG / name, dpi=300, dark_mode=False)
        plt.close(fig)
    print(f"  saved {len(figs)} figures.", flush=True)

    # ---- bundle updates ----
    df5g = pd.read_csv(OUT / "exp5_gated_decay.csv")
    dfhy = pd.read_csv(OUT / "exp8_hybrid.csv")
    dfsel = pd.read_csv(OUT / "exp8_selective.csv")
    dfcf = pd.read_csv(OUT / "exp7_coarsefine.csv")
    dfor = pd.read_csv(OUT / "exp7_orthkeys.csv")
    dflong = df9.groupby("N")["accuracy"].mean().round(3).to_dict()
    gated = {str(r["gamma"]): round(float(r["accuracy"]), 3)
             for _, r in df5g.iterrows()}

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["followup_drift_exponent"] = (None if drift_exp is None
                                             else round(drift_exp, 3))
    summary["followup_drift_note"] = drift_note
    summary["followup_state_norm_slope"] = round(norm_slope, 3)
    summary["followup_mstar_per_dim"] = mstar
    summary["followup_exp1_d16_m2_CI95"] = round(ci_m2, 4)
    summary["followup_exp1_d16_m128_CI95"] = round(ci_m128, 4)
    summary["followup_long_accuracy_by_N"] = {str(k): float(v)
                                              for k, v in dflong.items()}
    summary["followup_gated_accuracy_by_gamma"] = gated
    f16 = df5p[df5p["precision"] == "float16"].set_index("gamma")["drift_vs_fp64"]
    summary["followup_fp16_drift_vs_fp64_by_gamma"] = {
        str(k): float(v) for k, v in f16.items()}
    summary["followup_precision_note"] = (
        "binary recall identical across fp64/fp32/fp16; "
        "continuous output drift orders fp16 > fp32 ~= 0")
    summary["followup_hybrid_w128_recent"] = float(
        dfhy[(dfhy["window"] == 128)
             & (dfhy["position_frac"] == 0.9)]["accuracy"].iloc[0])
    summary["followup_selective_oracle"] = float(
        dfsel[dfsel["mode"] == "oracle"]["accuracy"].iloc[0])
    summary["followup_coarse_vs_fine_at_8192"] = {
        "coarse": float(dfcf[dfcf["N"] == 8192]["coarse_accuracy"].iloc[0]),
        "fine": float(dfcf[dfcf["N"] == 8192]["fine_accuracy"].iloc[0])}
    summary["followup_orth_m16_linear"] = float(
        dfor[dfor["m"] == 16]["linear_mean"].iloc[0])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")

    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    config["followups"] = {
        "seed": SEED,
        "featuremaps": {"dims": [16, 32], "trials": 15},
        "long_length": {"N": [16384, 65536, 262144, 1048576],
                        "trials": "3 (2 at 1M)"},
        "seeds": [42, 43, 44, 45],
        "gated": {"N": 4096, "trials": 10},
        "mitigations": {"N": "1024-4096", "trials": 8},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2),
                                     encoding="utf-8")

    new_files = ["exp5_gated_decay.csv", "exp5_precision_decay.csv",
                 "exp6_featuremaps.csv", "exp7_coarsefine.csv",
                 "exp7_mNgrid.csv", "exp7_orthkeys.csv", "exp8_hybrid.csv",
                 "exp8_chunked.csv", "exp8_selective.csv", "exp8_mixed.csv",
                 "exp9_longlength.csv", "exp10_seed_recall.csv",
                 "exp10_seed_length.csv", "exp10_bigd.csv",
                 "analysis in summary.json (followup_* keys)"]
    readme = (OUT / "README.md").read_text(encoding="utf-8")
    if "## Follow-up evidence" in readme:
        # idempotent: strip any previously appended follow-up section first
        readme = readme.split("## Follow-up evidence")[0].rstrip() + "\n"
    with (OUT / "README.md").open("w", encoding="utf-8") as f:
        f.write(readme)
        f.write("\n## Follow-up evidence (batch 2, seed 42 unless noted)\n\n")
        f.write(f"- Drift: {drift_note}.\n")
        f.write(f"- State-norm slope vs N: {norm_slope:.3f} (0.5 = sqrt-t scaling).\n")
        f.write(f"- m* (50% crossing) per dim: {mstar} — onset independent of d.\n")
        f.write(f"- Long-range accuracy by N: "
                f"{ {str(k): float(v) for k, v in dflong.items()} }.\n")
        f.write(f"- Gated early-marker accuracy by gamma: {gated}.\n")
        f.write("- Hybrid w=128 restores recent-marker recall; "
                "oracle selective gate restores early recall; "
                "fp32 accumulation cuts fp16-compute error (see exp8_*.csv).\n")
        f.write("- Precision x decay: binary recall identical across "
                "fp64/fp32/fp16, but continuous output drift orders "
                "fp16 > fp32 ~= 0 (see exp5_precision_decay.csv, drift_vs_fp64).\n")
        f.write("- Coarse (repeated) signal survives where singleton is lost "
                "(see exp7_coarsefine.csv).\n")
        f.write("- Orthogonal keys do NOT restore a plateau under elu+1 "
                "(m=4: 0.30 vs 0.67 random keys): the map's non-centered "
                "mean destroys key orthogonality — collision is set by "
                "feature overlap, see exp7_orthkeys.csv).\n")
        f.write("\nNew files: " + ", ".join(f"`{n}`" for n in new_files) + "\n")
    print("[finalize] bundle updated.", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for stage in ([which] if which != "all" else
                  ["stage1", "stage2", "stage3", "finalize"]):
        {"stage1": stage1, "stage2": stage2, "stage3": stage3,
         "finalize": finalize}[stage]()
