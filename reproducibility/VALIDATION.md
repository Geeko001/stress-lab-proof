# VALIDATION.md

How to verify a reproduction matches the frozen results. No new statistics are created here; all thresholds below come from the existing evidence.

## Commands (from the project root)

```powershell
.\.venv\Scripts\python -m pytest tests/     # unit-test suite (must pass; see public_evidence/pytest_report.txt)
```

Then re-run any single stage (back up `public_evidence/` first — runners overwrite it):

```powershell
.\.venv\Scripts\python run_followups.py stage1
```

## What to compare

- Regenerated CSVs vs frozen `public_evidence/*.csv`, cell by cell, at the precision quoted in the manuscript (e.g. accuracies to ~3 decimals, slope 0.448, counts exact).
- Key reproducibility anchors: d=16/m=2 → 0.96; m=128 → 0.008; length 1.0/1.0/0.8/0.2; shrinking cumulative 49/36/7; bf16 52 ineffective; gatedm 0.102/0.094/0.109; latch 0.05 → 0.40; bytes 2176 vs 1048576.
- Statistical files: 9 rows in `exp12_ttests.csv` with identical t/n; 193 rows in `exp12_CIs.csv`.

## Tolerances and caveats

- Small floating-point display differences (last-digit rounding in means/stds) can occur across NumPy versions or OS BLAS builds; integer counts, 0/1 trial outcomes, and accuracies at quoted precision should match.
- Runtimes and timing comparisons must be re-baselined on your hardware — never compare absolute seconds against the manuscript's i3-machine timings.
- The OOM entry is an analysis record (theory bytes vs machine RAM), not a crash log; reproducing it means recomputing the byte arithmetic, not re-crashing a machine.
- Precision probes (`exp3_shrinking`, bf16, mixed) are single probe runs with no trial structure — exact repetition is expected only under the same library versions (ml_dtypes 0.6.0 for bf16).

## Known indexing / re-run issues (documented, not hidden)

- `exp10_bigd.csv` d=128 row is a later rerun (`run_dimensionality_sweep(m_values=[4], dim_values=[128], trials=25, seed=42)`); re-running the full big-d stage as filed will NOT reproduce that row — run the quoted call to reproduce it.
- `exp5_gated_decay.csv` γ labels: row 4 is γ=0.9999 (identified by its horizon columns 10000/166355), not 1.0; the sweep signature is `(1.0, 0.9, 0.99, 0.999, 0.9999)` in `experiments/gated_recall.py::run_gamma_sweep`.
- Horizon N50 ≈ 3072 for γ=0.9999 is an interpolation between the measured 2048/4096 bracket (recorded as such in `summary.json: batch4_horizon_crossings`), not a measured grid point.
- `summary.json` compressibility (0.75/0.625) and followup-oracle (0.25) values are stale relative to the CSVs; validate against the CSVs, not the snapshot.
