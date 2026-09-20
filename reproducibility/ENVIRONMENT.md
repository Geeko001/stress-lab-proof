# ENVIRONMENT.md

Verified environment facts for the experiments behind the paper. Every fact below comes from a repository or evidence file; anything unavailable is marked "not recorded".

## Language and platform

- Python 3.14.7 (per `public_evidence/summary.json` env block and `config.json` env block).
- Windows 11 (`Windows-11-10.0.26200-SP0` in both files).
- CPU-only execution throughout. No GPU code paths exist in the repo; no torch dependency.

## Package versions (per `summary.json`; confirmed installed in `.venv` at audit time)

| Package | Version |
|---|---|
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| matplotlib | 3.11.2 |
| streamlit | 1.64.0 |
| fpdf2 (`fpdf`) | 2.8.8 (installed; not pinned in requirements.txt) |
| ml_dtypes | 0.6.0 (installed; used by the bf16 probe; not pinned in requirements.txt) |
| pytest | 9.1.1 (installed; test suite in `tests/`) |

`requirements.txt` pins: streamlit, numpy, pandas, matplotlib (unversioned). A reproducer should additionally install `fpdf2`, `ml_dtypes`, and `pytest`.

## Random seeds

- Base seed **42** in every runner (`SEED = 42` in `run_exp4.py`, `run_followups.py`, `run_batch3.py`, `run_batch4.py`).
- One trial = one full task run on `default_rng(base_seed + trial)` (manuscript §4.2; per-trial files record seeds 42–71 across probes).
- Cross-seed checks: recall curves at seeds 42–45; length decay and five follow-up probes at 43–44.

## Default experimental settings

- State dimension **d = 16** default (square setup k = d_v = d); sweeps cover d ∈ {8, 16, 32, 64} plus d = 128/256 spot points.
- Feature map **elu+1** unless noted (relu/exp comparisons; centered map with constant 1.1605 calibrated under Gaussian keys).
- Synthetic Gaussian keys/values; distractor scale 0.01 (absolute accuracies depend on it; rankings do not — manuscript §4.2).
- Precisions tested: float64 (default), float32, float16, and real bf16 via ml_dtypes (probe-specific).

## Hardware of record

- Intel i3-class CPU, ~3.7 GB usable RAM (manuscript §4.2/§5.3; OOM analysis record). Exact CPU model: not recorded. Timing results are hardware-specific and should be re-baselined, not compared absolutely.

## Not recorded

- Exact CPU model string; wall-clock expectations per stage; matplotlib backend beyond `Agg` (hard-coded in runners).
