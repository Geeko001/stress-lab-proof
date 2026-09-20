# Reproducibility — Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes

Main entry point for reproducing and verifying this research.

## Paper

**"Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes"** — manuscript: `Paper_1_Draft_v3.md`, frozen PDF: `Paper_1_Draft_v3.pdf`.

## What the research actually evaluates

The experimental object is **only** the synthetic fixed-state additive linear-attention recurrence

`S_t = S_{t-1} + φ(k_t)v_tᵀ`

and its tested synthetic variants (scalar/per-channel gated decay, novelty gates, oracle salience latch, hybrid exact windows, chunked resets/summaries, multi-head banks, mixed-precision accumulation), all implemented in CPU-only NumPy.

## Scope statement (read first)

**Mamba, RWKV, RetNet, Linear Transformers, and trained language models were NOT experimentally tested.** They appear in the paper solely as architectural context and motivation (manuscript §2.1, §2.5 categories `[A]`/`[Arch]`). No checkpoints, weights, or production-model runs exist in this repository. Any reproduction claiming to evaluate those architectures from this repo would be misrepresenting its contents.

## Repository structure relevant to reproduction

```text
experiments/          task implementations (associative_recall, context_length,
                      gated_recall, mitigations, variants, dissipation,
                      numerical_saturation, softmax_baseline, finalbatch)
models/               synthetic NumPy models ONLY (linear_attention.py,
                      softmax_attention.py) — not production architectures
utils/                generators, metrics, plotting helpers
run_exp4.py           Exp 4 (softmax baseline) + bundle update
run_followups.py      follow-up probes, stages: stage1|stage2|stage3|finalize|all
run_batch3.py         gap-audit probes, stages: stageA|stageB|finalize|all
run_batch4.py         final probes/stats/figures, stages: stageC|stageD|finalize|all
generate_public_bundle.py  reruns Exps 1–3 with fixed settings into public_evidence/
build_paper_pdf.py    builds the manuscript PDF from Paper_1_Draft_v3.md
gen_schematics.py     generates content-only schematic figures (no measured data)
public_evidence/      frozen evidence bundle: 61 CSVs, config.json, summary.json,
                      pytest_report.txt, 48 figures/  (DO NOT overwrite lightly — see below)
tests/                unit tests (pytest)
```

## Prerequisites

- Windows, macOS, or Linux with Python 3.14 (original env: Python 3.14.7; see ENVIRONMENT.md).
- Install dependencies: `pip install -r requirements.txt` (streamlit, numpy, pandas, matplotlib) plus `fpdf2`, `ml_dtypes`, `pytest` for the PDF build, bf16 probe, and test suite.
- CPU only. No GPU, no torch, no datasets, no downloads.
- **No API keys, `.env` files, credentials, or secrets are required.** None exist in this repo; none are needed.

## Exact reproduction commands (from the project root)

Each script documents its own usage in its header docstring; the commands below are quoted from those headers. Runners write incrementally into `public_evidence/`:

```powershell
.\.venv\Scripts\python generate_public_bundle.py        # Exps 1-3, fixed settings
.\.venv\Scripts\python run_exp4.py                      # Exp 4 softmax baseline
.\.venv\Scripts\python run_followups.py [stage1|stage2|stage3|finalize|all]
.\.venv\Scripts\python run_batch3.py [stageA|stageB|finalize|all]
.\.venv\Scripts\python run_batch4.py [stageC|stageD|finalize|all]
.\.venv\Scripts\python -m pytest tests/                 # unit-test validation
.\.venv\Scripts\python build_paper_pdf.py Paper_1_Draft_v3.md Paper_1_Draft_v3.pdf
.\.venv\Scripts\python gen_schematics.py                # schematic figures only
```

Single-condition checks can also be run directly, e.g. `experiments/associative_recall.py::run_dimensionality_sweep` (this is how the d=128/m=4 point now stored in `exp10_bigd.csv` was produced: `m_values=[4], dim_values=[128], trials=25, seed=42`).

WARNING: re-running the full stages **overwrites** files in `public_evidence/`. Back up that folder first if you want to compare against the frozen bundle. Long sweeps (N=1M streaming, 8k timing) take a long time on CPU.

## Validation

See VALIDATION.md. In short: run the unit tests, re-run a stage, and compare the regenerated CSVs against the frozen bundle cell-by-cell; small floating-point display differences can occur, integer counts and accuracies at the reported precision should match. Do not expect bit-identical runtimes.

## Expected workflow

setup → run experiment stage → run validation/tests → compare outputs against `public_evidence/` → consult EVIDENCE_INDEX.md / CLAIM_EVIDENCE_MAP.md to link results to manuscript claims.

## Evidence pointers (links, not copies)

- Per-file inventory: `EVIDENCE_INDEX.md`
- Claim → file mapping (C001–C025): `CLAIM_EVIDENCE_MAP.md`
- Experiment families + scripts: `EXPERIMENT_REGISTRY.md`
- Known gaps/contradictions: `EVIDENCE_GAPS.md`
- Overview + Claude reading order: `RESEARCH_EVIDENCE_SUMMARY.md`
