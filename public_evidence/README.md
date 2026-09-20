# Public Evidence Bundle — Linear Attention Stress-Test Lab

Validation data for: *Stress-Testing Linear Attention: Architectural
Breakpoints and Context Failure Modes* (Aashirwad Sharma).

Generated (UTC): 2026-09-17T12:38:23Z · Random seed: 42

## What this is

Raw, unedited outputs of the three lab experiments, plus the unit-test
report. Everything here is an **observed measurement** from a small
synthetic CPU-only setup — not proof of any theoretical claim. The lab
distinguishes theory / prediction / observation / interpretation; this
bundle contains observations.

## Key observed numbers

- Exp1 (d=16): retrieval accuracy 0.960 at m=2 → 0.008 at m=128.
- Exp2 mean accuracy by N: {'128': 1.0, '512': 1.0, '1024': 0.8, '4096': 0.2}.
- Exp3 numerically ineffective updates: {'float16': 49, 'float32': 36, 'float64': 7}.
- Exp4 softmax vs linear accuracy by N: {'128': (1.0, 1.0), '512': (1.0, 1.0), '1024': (0.9, 1.0), '4096': (0.3667, 1.0)}.
- Unit tests: ALL PASSED.

## Files

| File | Contents |
|---|---|
| `config.json` | Full run configuration + environment versions |
| `summary.json` | Key numbers above, machine-readable |
| `exp1_dimensionality.csv` | Accuracy vs binding count, d ∈ {8,16,32,64}, 25 trials |
| `exp2_length_sweep.csv` | Marker recall vs sequence length × position, 10 trials |
| `exp3_shrinking.csv` | Per-step effective state change per precision |
| `exp3_decay.csv` | Gamma retention summary |
| `figures/*.png` | 300 DPI plots of the above |
| `pytest_report.txt` | Full unit-test log |
| `exp4_softmax_baseline.csv` | Softmax vs linear accuracy/time/storage per N |
| `figures/exp4_softmax_comparison.png` | 300 DPI Exp4 comparison plot |

## Reproduce it

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt pytest
python generate_public_bundle.py
```

Same seed (42) replays identical curves. Environment: Windows-11-10.0.26200-SP0,
Python 3.14.7, NumPy 2.5.3.

## Follow-up evidence (batch 2, seed 42 unless noted)

- Drift: relative error saturates at ~1.0 from N=128 (washed output ~= 0 by construction); no compounding exponent identifiable — finding is total washout.
- State-norm slope vs N: 0.448 (0.5 = sqrt-t scaling).
- m* (50% crossing) per dim: {'8': 8, '16': 8, '32': 8, '64': 8} — onset independent of d.
- Long-range accuracy by N: {'16384': 0.0, '65536': 0.0, '262144': 0.1, '1048576': 0.0}.
- Gated early-marker accuracy by gamma: {'1.0': 0.4, '0.9': 0.0, '0.99': 0.1, '0.999': 0.1, '0.9999': 0.3}.
- Hybrid w=128 restores recent-marker recall; oracle selective gate restores early recall; fp32 accumulation cuts fp16-compute error (see exp8_*.csv).
- Precision x decay: binary recall identical across fp64/fp32/fp16, but continuous output drift orders fp16 > fp32 ~= 0 (see exp5_precision_decay.csv, drift_vs_fp64).
- Coarse (repeated) signal survives where singleton is lost (see exp7_coarsefine.csv).
- Orthogonal keys do NOT restore a plateau under elu+1 (m=4: 0.30 vs 0.67 random keys): the map's non-centered mean destroys key orthogonality — collision is set by feature overlap (see exp7_orthkeys.csv).

New files: `exp5_gated_decay.csv`, `exp5_precision_decay.csv`, `exp6_featuremaps.csv`, `exp7_coarsefine.csv`, `exp7_mNgrid.csv`, `exp7_orthkeys.csv`, `exp8_hybrid.csv`, `exp8_chunked.csv`, `exp8_selective.csv`, `exp8_mixed.csv`, `exp9_longlength.csv`, `exp10_seed_recall.csv`, `exp10_seed_length.csv`, `exp10_bigd.csv`, `analysis in summary.json (followup_* keys)`

## Batch-3 audit probes (seed 42)

- Compressibility (A=alphabet): {'8': 0.75, '64': 0.62, '-1': 0.62}.
- Per-channel vs scalar: {'scalar-0.99': 0.0, 'scalar-0.999': 0.5, 'perchannel': 0.62}.
- Multi-head m=16: 1x64=0.094 vs 4x32=0.083.
- Shift/noisy/centered/horizon/timing/distractor: see exp11_*.csv + figures.
- Per-trial records: exp11_*_trials.csv enable significance testing for new probes (older experiments log aggregates only — documented limitation).

New files: `exp11_compress(.csv,_trials.csv)`, `exp11_perchannel(.csv,_trials.csv)`, `exp11_novelty(.csv,_trials.csv)`, `exp11_multihead(.csv,_trials.csv)`, `exp11_shift(.csv,_trials.csv)`, `exp11_noisy(.csv,_trials.csv)`, `exp11_centered.csv`, `exp11_centered_orth.csv`, `exp11_horizon.csv`, `exp11_timing.csv`, `exp11_distractor.csv`, `exp8_selective.csv + exp5_gated_decay.csv topped to 20 trials`

## Batch-4 final audit (seed 42)

- Centered m* scaling, gated/gate m-sweeps, latch-N, survival quintiles, cosine drift: exp12_centered_d(.csv), exp12_gatedm, exp12_gatem, exp12_latchN, exp12_survival, exp12_cosine.
- Compress top-up x40, multi-seed batch2/3, bf16, chunk-summary, old-exp trial logs (exp01-04_trials), hybrid-N4096, stacked, frozen-calib, random-gate, OOM probe: exp12_*.csv.
- Horizon 50%-crossings vs 1/(1-g) theory, Welch t-tests (exp12_ttests.csv), CI table over all aggregates (exp12_CIs.csv).
- Deliberately unrunnable here: real model weights (no torch/GPU/RAM), learned gating (no training loop) — bounded by oracle/random controls instead.
