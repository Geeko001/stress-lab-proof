"""Run Exp 4 (Softmax vs Linear baseline) and update the evidence bundle.

Outputs (matching the existing bundle layout):
  public_evidence/exp4_softmax_baseline.csv
  public_evidence/figures/exp4_softmax_comparison.png   (300 DPI)
  public_evidence/summary.json  (exp4 block added)
  public_evidence/config.json   (exp4 block added)
  public_evidence/README.md     (exp4 numbers + file row added)

Usage (from the project root):
    .\\.venv\\Scripts\\python run_exp4.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.softmax_baseline import run_softmax_baseline
from utils.plotting import save_high_res

ROOT = Path(__file__).parent
OUT = ROOT / "public_evidence"

SEED = 42
N_VALUES = [128, 512, 1024, 4096]
DIM = 16
TRIALS = 10
POSITIONS = (0.1, 0.5, 0.9)


def main():
    print(f"Exp 4: softmax vs linear baseline (seed={SEED}) ...")
    rows = run_softmax_baseline(N_VALUES, dim=DIM, trials=TRIALS, seed=SEED,
                                position_fracs=POSITIONS)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp4_softmax_baseline.csv", index=False)
    for r in rows:
        print(f"  N={r['N']:>5}: linear={r['linear_accuracy']:.3f} "
              f"softmax={r['softmax_accuracy']:.3f} "
              f"t_lin={r['linear_time_mean_s']*1e3:.1f}ms "
              f"t_sm={r['softmax_time_mean_s']*1e3:.1f}ms "
              f"bytes lin/sm={r['linear_bytes']}/{r['softmax_bytes']}")

    # ---- 300 DPI comparison figure (accuracy + runtime panels) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
    ax1.errorbar(df["N"], df["linear_accuracy"], yerr=df["linear_std"],
                 marker="o", capsize=4, label="Linear (elu+1)")
    ax1.errorbar(df["N"], df["softmax_accuracy"], yerr=df["softmax_std"],
                 marker="s", capsize=4, label="Softmax reference")
    ax1.set_xlabel("Sequence length (N)")
    ax1.set_ylabel("Retrieval accuracy")
    ax1.set_title("Exp4: accuracy — softmax vs linear (seed=42)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax2.plot(df["N"], np.array(df["linear_time_mean_s"]) * 1e3,
             marker="o", label="Linear (elu+1)")
    ax2.plot(df["N"], np.array(df["softmax_time_mean_s"]) * 1e3,
             marker="s", label="Softmax reference")
    ax2.set_xlabel("Sequence length (N)")
    ax2.set_ylabel("Mean ingest+query time (ms)")
    ax2.set_title("Exp4: runtime — softmax vs linear")
    ax2.legend()
    fig.tight_layout()
    save_high_res(fig, OUT / "figures" / "exp4_softmax_comparison.png",
                  dpi=300, dark_mode=False)
    plt.close(fig)
    print("  figure saved.")

    # ---- bundle summary / config / README update ----
    by_n = {str(r["N"]): {"linear": round(r["linear_accuracy"], 4),
                          "softmax": round(r["softmax_accuracy"], 4)}
            for r in rows}
    n_max = rows[-1]
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["exp4_accuracy_by_N"] = by_n
    summary["exp4_N4096_time_ms"] = {
        "linear": round(n_max["linear_time_mean_s"] * 1e3, 3),
        "softmax": round(n_max["softmax_time_mean_s"] * 1e3, 3),
    }
    summary["exp4_N4096_bytes"] = {
        "linear": n_max["linear_bytes"],
        "softmax": n_max["softmax_bytes"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_path = OUT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["exp4"] = {"N_values": N_VALUES, "dim": DIM, "trials": TRIALS,
                      "positions": list(POSITIONS), "seed": SEED,
                      "feature_map": "elu+1", "precision": "float64",
                      "added_utc": datetime.now(timezone.utc).strftime(
                          "%Y-%m-%dT%H:%M:%SZ")}
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    readme_path = OUT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    exp4_line = (f"- Exp4 softmax vs linear accuracy by N: "
                 f"{ {k: (v['linear'], v['softmax']) for k, v in by_n.items()} }.\n")
    anchor = "- Unit tests: ALL PASSED."
    if "Exp4 softmax vs linear" not in readme and anchor in readme:
        readme = readme.replace(anchor, exp4_line + anchor)
    files_anchor = "| `pytest_report.txt` | Full unit-test log |"
    exp4_files = ("| `pytest_report.txt` | Full unit-test log |\n"
                  "| `exp4_softmax_baseline.csv` | Softmax vs linear accuracy/time/storage per N |\n"
                  "| `figures/exp4_softmax_comparison.png` | 300 DPI Exp4 comparison plot |")
    if "exp4_softmax_baseline.csv" not in readme and files_anchor in readme:
        readme = readme.replace(files_anchor, exp4_files)
    readme_path.write_text(readme, encoding="utf-8")
    print("  bundle summary/config/README updated.")


if __name__ == "__main__":
    main()
