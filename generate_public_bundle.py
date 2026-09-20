"""Generate the public evidence bundle for the paper.

Reruns all three experiments with fixed, documented settings and saves
raw data, configs, figures, and the unit-test report to public_evidence/.

Everything reported is an observed measurement, not proof of any claim.

Usage (from the project root):
    .\\.venv\\Scripts\\python generate_public_bundle.py
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.associative_recall import run_dimensionality_sweep
from experiments.context_length import run_length_sweep
from experiments.numerical_saturation import (
    shrinking_update_experiment,
    decay_experiment,
)
from utils.plotting import (
    plot_accuracy_vs_bindings,
    plot_multi_curve,
    plot_simple,
    save_high_res,
)

ROOT = Path(__file__).parent
OUT = ROOT / "public_evidence"
FIG = OUT / "figures"

# ---------------------------------------------------------------- settings
# Fixed public-validation settings. Change only with a new bundle version.
SEED = 42
EXP1_DIMS = (8, 16, 32, 64)
EXP1_M = [2, 4, 8, 16, 32, 64, 128]
EXP1_TRIALS = 25
EXP1_FEATURE_MAP = "elu+1"
EXP2_N = [128, 512, 1024, 4096]
EXP2_POSITIONS = (0.1, 0.5, 0.9)
EXP2_TRIALS = 10
EXP2_DIM = 16
EXP3_PRECISIONS = ("float64", "float32", "float16")
EXP3_STEPS = 60
EXP3_DECAY = 0.5
EXP3_GAMMAS = (0.99, 0.999, 0.9999)


def env_info():
    import numpy, pandas, matplotlib as mpl, streamlit
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": mpl.__version__,
        "streamlit": streamlit.__version__,
        "platform": platform.platform(),
    }


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    env = env_info()

    print(f"[1/5] Experiment 1: dimensionality sweep (seed={SEED}) ...")
    exp1_rows = run_dimensionality_sweep(
        EXP1_M, dim_values=EXP1_DIMS, trials=EXP1_TRIALS, seed=SEED,
        feature_map=EXP1_FEATURE_MAP, dtype=np.float64)
    for r in exp1_rows:
        r["experiment"] = "associative_recall"
    df1 = pd.DataFrame(exp1_rows)
    df1.to_csv(OUT / "exp1_dimensionality.csv", index=False)

    print("[2/5] Experiment 2: sequence-length sweep ...")
    exp2_rows = run_length_sweep(
        EXP2_N, dim=EXP2_DIM, trials=EXP2_TRIALS, seed=SEED,
        position_fracs=EXP2_POSITIONS, feature_map=EXP1_FEATURE_MAP,
        dtype=np.float64)
    for r in exp2_rows:
        r["experiment"] = "context_length"
    df2 = pd.DataFrame(exp2_rows)
    df2.to_csv(OUT / "exp2_length_sweep.csv", index=False)

    print("[3/5] Experiment 3: numerical saturation ...")
    exp3_rows = shrinking_update_experiment(
        n_steps=EXP3_STEPS, start_magnitude=1.0, decay=EXP3_DECAY,
        dim=8, precisions=EXP3_PRECISIONS, seed=SEED)
    df3 = pd.DataFrame(exp3_rows)
    df3.to_csv(OUT / "exp3_shrinking.csv", index=False)
    decay_rows = decay_experiment(gamma_values=EXP3_GAMMAS, dim=8,
                                  precision="float64", seed=SEED)
    dfd = pd.DataFrame(decay_rows)
    dfd.to_csv(OUT / "exp3_decay.csv", index=False)

    print("[4/5] Figures (300 DPI) ...")
    figs = []
    curves = {f"d={d}": [r["linear_mean"] for r in exp1_rows if r["dim"] == d]
              for d in EXP1_DIMS}
    figs.append(plot_multi_curve(EXP1_M, curves,
                                 "Number of stored bindings (m)",
                                 "Retrieval accuracy (linear)",
                                 "Exp1: accuracy vs bindings by state dim",
                                 xscale="log"))
    sub16 = [r for r in exp1_rows if r["dim"] == 16]
    figs.append(plot_accuracy_vs_bindings(
        [r["m"] for r in sub16],
        [r["linear_mean"] for r in sub16],
        [r["softmax_mean"] for r in sub16],
        title="Exp1: linear vs softmax reference (d=16, seed=42)"))
    acc_curves = {}
    norm_curves = {}
    for pf in EXP2_POSITIONS:
        sub = df2[df2["position_frac"] == pf].sort_values("N")
        acc_curves[f"pos={pf}"] = sub["accuracy"].tolist()
        norm_curves[f"pos={pf}"] = sub["state_norm_mean"].tolist()
    figs.append(plot_multi_curve(EXP2_N, acc_curves, "Sequence length (N)",
                                 "Retrieval accuracy",
                                 "Exp2: length vs accuracy (seed=42)",
                                 xscale="log"))
    figs.append(plot_multi_curve(EXP2_N, norm_curves, "Sequence length (N)",
                                 "Mean state norm",
                                 "Exp2: length vs state norm (seed=42)",
                                 xscale="log"))
    eff_curves = {}
    for prec in df3["precision"].unique():
        sub = df3[df3["precision"] == prec].sort_values("timestep")
        eff_curves[prec] = sub["effective_change"].tolist()
    figs.append(plot_multi_curve(df3["timestep"].unique().tolist(), eff_curves,
                                 "Timestep", "Effective state change",
                                 "Exp3: update magnitude vs effective change"))
    figs.append(plot_simple(dfd["gamma"].tolist(),
                            dfd["final_state_magnitude"].tolist(),
                            "Gamma", "Final state magnitude",
                            "Exp3: gamma vs state retention"))
    fig_names = ["exp1_overlay_by_dim.png", "exp1_linear_vs_softmax_d16.png",
                 "exp2_accuracy_vs_length.png", "exp2_statenorm_vs_length.png",
                 "exp3_effective_change.png", "exp3_gamma_retention.png"]
    for fig, name in zip(figs, fig_names):
        save_high_res(fig, FIG / name, dpi=300, dark_mode=False)
        plt.close(fig)

    print("[5/5] Unit-test report ...")
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"],
                          capture_output=True, text=True, cwd=ROOT)
    report = proc.stdout + "\n" + proc.stderr
    # Scrub local machine paths so the committed bundle carries no
    # username / absolute-path info. Test results themselves are untouched.
    report = report.replace(sys.executable, "<venv>/Scripts/python.exe")
    report = report.replace(str(ROOT), "<repo>")
    (OUT / "pytest_report.txt").write_text(report, encoding="utf-8")
    tests_ok = proc.returncode == 0
    print(f"      tests passed: {tests_ok}")

    # ---- summary numbers (observed measurements) ----
    d16_first = sub16[0]["linear_mean"]
    d16_last = sub16[-1]["linear_mean"]
    len_by_n = df2.groupby("N")["accuracy"].mean().round(3).to_dict()
    ineff = df3.groupby("precision")["ineffective"].sum().to_dict()
    summary = {
        "bundle_generated_utc": timestamp,
        "seed": SEED,
        "env": env,
        "exp1_d16_m2_accuracy": round(float(d16_first), 4),
        "exp1_d16_m128_accuracy": round(float(d16_last), 4),
        "exp2_mean_accuracy_by_N": {str(k): float(v) for k, v in len_by_n.items()},
        "exp3_ineffective_updates_by_precision": {k: int(v) for k, v in ineff.items()},
        "unit_tests_passed": tests_ok,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config = {
        "seed": SEED,
        "exp1": {"dims": list(EXP1_DIMS), "m_values": EXP1_M,
                 "trials": EXP1_TRIALS, "feature_map": EXP1_FEATURE_MAP,
                 "precision": "float64"},
        "exp2": {"N_values": EXP2_N, "positions": list(EXP2_POSITIONS),
                 "dim": EXP2_DIM, "trials": EXP2_TRIALS,
                 "feature_map": EXP1_FEATURE_MAP, "precision": "float64"},
        "exp3": {"precisions": list(EXP3_PRECISIONS), "n_steps": EXP3_STEPS,
                 "decay": EXP3_DECAY, "gammas": list(EXP3_GAMMAS)},
        "generated_utc": timestamp,
        "env": env,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    readme = f"""# Public Evidence Bundle — Linear Attention Stress-Test Lab

Validation data for: *Stress-Testing Linear Attention: Architectural
Breakpoints and Context Failure Modes* (Aashirwad Sharma).

Generated (UTC): {timestamp} · Random seed: {SEED}

## What this is

Raw, unedited outputs of the three lab experiments, plus the unit-test
report. Everything here is an **observed measurement** from a small
synthetic CPU-only setup — not proof of any theoretical claim. The lab
distinguishes theory / prediction / observation / interpretation; this
bundle contains observations.

## Key observed numbers

- Exp1 (d=16): retrieval accuracy {d16_first:.3f} at m=2 → {d16_last:.3f} at m=128.
- Exp2 mean accuracy by N: { {str(k): float(v) for k, v in len_by_n.items()} }.
- Exp3 numerically ineffective updates: { {k: int(v) for k, v in ineff.items()} }.
- Unit tests: {"ALL PASSED" if tests_ok else "FAILURES PRESENT — see pytest_report.txt"}.

## Files

| File | Contents |
|---|---|
| `config.json` | Full run configuration + environment versions |
| `summary.json` | Key numbers above, machine-readable |
| `exp1_dimensionality.csv` | Accuracy vs binding count, d ∈ {{8,16,32,64}}, {EXP1_TRIALS} trials |
| `exp2_length_sweep.csv` | Marker recall vs sequence length × position, {EXP2_TRIALS} trials |
| `exp3_shrinking.csv` | Per-step effective state change per precision |
| `exp3_decay.csv` | Gamma retention summary |
| `figures/*.png` | 300 DPI plots of the above |
| `pytest_report.txt` | Full unit-test log |

## Reproduce it

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate   # Windows
pip install -r requirements.txt pytest
python generate_public_bundle.py
```

Same seed ({SEED}) replays identical curves. Environment: {env['platform']},
Python {env['python']}, NumPy {env['numpy']}.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Done. Bundle in {OUT} ({len(list(OUT.rglob('*')))} files).")


if __name__ == "__main__":
    main()
