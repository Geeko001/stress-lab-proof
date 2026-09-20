# CLAIM_EVIDENCE_MAP.md

Source manuscript: `Paper_1_Draft_v3.md` (frozen). Each claim is mapped to the exact evidence file(s) that support it. Status scale: SUPPORTED (direct file evidence) / PARTIALLY SUPPORTED (some aspects unmeasured or n small) / NOT LOCATED / AMBIGUOUS (conflicting artifacts).

## Claim ID: C001

Claim:
Under standard elu+1 features, associative recall falls steeply from low density with 50%-crossing at m* = 8 across tested dims (d = 8–256).

Paper location:
Abstract; §3.3; §4.2 Density; Table 1; Conclusion.

Evidence:
- `public_evidence/exp1_dimensionality.csv`
- `public_evidence/exp01_trials.csv`
- `public_evidence/exp10_bigd.csv`
- `public_evidence/exp6_featuremaps.csv`

Evidence type:
processed + raw.

What the evidence actually demonstrates:
d=16: 0.96 → 0.008 across m = 2 → 128; m=4 values > 0.5 and m=8 values < 0.5 for every tested dim including d=256.

Conditions:
elu+1, float64, Gaussian keys, seed base 42; Exp 1 n=25; bigd n=10 (d=256), n=25 rerun row (d=128).

Limitations:
Grid resolution 2× (crossing localized to the m=4/m=8 bracket, not interpolated). Distractor scale 0.01 conditions absolute values.

Status:
SUPPORTED.

## Claim ID: C002

Claim:
Feature map moves the recall curve where dimension cannot (relu 0.73 / exp 0.56 / elu+1 0.32 at m=8, d=16; p ≈ 9×10⁻¹⁵).

Paper location:
§3.3; §4.2 Density; §4.2 Robustness.

Evidence:
- `public_evidence/exp6_featuremaps.csv`
- `public_evidence/exp12_ttests.csv` (row relu>elu+1@m8: t=7.748, n=15+15)

Evidence type:
processed + statistical (p recomputed from stored t via normal approximation; stored p column reads 0.0).

What the evidence actually demonstrates:
d=16, m=8 means: relu 0.733, exp 0.558, elu+1 0.325.

Conditions:
float64, n=15 per group.

Limitations:
p-value is a normal-approximation recomputation, secondary evidence.

Status:
SUPPORTED.

## Claim ID: C003

Claim:
Orthogonal keys do not help under elu+1 (m=4: 0.30 vs 0.67 random).

Paper location:
§3.3; §4.2 Density.

Evidence:
- `public_evidence/exp7_orthkeys.csv` (0.30, n=10)
- `public_evidence/exp6_featuremaps.csv` (0.667 random-key reference, n=15)

Evidence type:
processed.

What the evidence actually demonstrates:
Exactly orthogonal keys collide under non-centered features (shared mean direction dominates overlaps).

Conditions:
d=16, elu+1.

Limitations:
Random-key reference comes from a different file/run (n=15 vs n=10).

Status:
SUPPORTED.

## Claim ID: C004

Claim:
Centered features restore a plateau with dimension-scaled breakpoint m* ≈ 2d (16/64/128 for d = 8/32/64) — conditional empirical result.

Paper location:
§3.3; Fig 3 caption; §4.2 Density; Abstract; Conclusion.

Evidence:
- `public_evidence/exp11_centered.csv`
- `public_evidence/exp12_centered_d.csv`
- `public_evidence/exp12_ttests.csv` (centered>elu+1@m16: t=31.741)

Evidence type:
processed + statistical.

What the evidence actually demonstrates:
First grid point at/below 0.5: m=16 (d=8), m=64 (d=32), m=128 (d=64).

Conditions:
Centered map (constant 1.1605 calibrated under Gaussian keys), n=15 (centered), n=12 (centered_d).

Limitations:
2× grid resolution; constant mis-centers non-Gaussian keys; NOT a universal law (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C005

Claim:
Centered-orth d=16 holds flat 1.0 to m=16, then 0.5 at m=32, with complete separation across all 10 trials.

Paper location:
§4.2 Density; §4.2 Robustness.

Evidence:
- `public_evidence/exp11_centered_orth.csv`

Evidence type:
processed (all std = 0.0).

What the evidence actually demonstrates:
Means exactly as claimed with zero variance (degenerate case for normal-approx p).

Conditions:
d=16, n=10.

Limitations:
Zero-variance case noted in manuscript; t=149 stored in t-tests file.

Status:
SUPPORTED.

## Claim ID: C006

Claim:
Accuracy follows binding count, not background length (filler sweeps saturate); washed states collapse to a constant predictor (≈1/m).

Paper location:
§4.2 Density.

Evidence:
- `public_evidence/exp7_mNgrid.csv`

Evidence type:
processed.

What the evidence actually demonstrates:
m=4 accuracy 0.656 at nfill=0 vs 0.25 at nfill=256/2048 (saturation, not continued fall).

Conditions:
d=16, n=8.

Limitations:
Fine — "collapse to constant predictor" is an interpretation of saturation, stated as such.

Status:
SUPPORTED.

## Claim ID: C007

Claim:
Marker recall 1.0/1.0/0.8/0.2 at N = 128/512/1024/4096, flat across marker positions; streaming floor ≈0 for N ≥ 16k to 1M.

Paper location:
§4.2 Length; Abstract; Conclusion.

Evidence:
- `public_evidence/exp2_length_sweep.csv`
- `public_evidence/exp02_trials.csv`
- `public_evidence/exp9_longlength.csv` (0.0/0.0/0.1/0.0 at 16k/65k/262k/1M)

Evidence type:
processed + raw.

What the evidence actually demonstrates:
Values exactly as claimed; positions 0.1/0.5/0.9 identical per N.

Conditions:
d=16, elu+1; Exp 2 n=10; Exp 9 n=10 (≤65k), 3 (262k), 2 (1M).

Limitations:
Streaming-floor points rest on small n (2–3 at the largest N); single 0.1 reading at N=262144.

Status:
SUPPORTED (with small-n caveat at the far end, disclosed in gaps).

## Claim ID: C008

Claim:
State-norm log-log slope 0.448 over N = 128 → 1M, consistent with √t growth for zero-mean streams.

Paper location:
§3.5; Table 1; Abstract; Conclusion.

Evidence:
- `public_evidence/exp2_length_sweep.csv` + `public_evidence/exp9_longlength.csv` (combined dedup fit = 0.448)
- `public_evidence/summary.json` (followup_state_norm_slope: 0.448)

Evidence type:
processed + supporting snapshot.

What the evidence actually demonstrates:
Recomputed combined fit reproduces 0.448 exactly.

Conditions:
Zero-mean update streams; elu+1.

Limitations:
Conditional on zero-mean/weakly-correlated updates (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C009

Claim:
Gated 50%-recall crossings ≈1024 (γ=0.999) and ≈3072 interpolated (γ=0.9999), horizon sweep n=6; task forgetting precedes numerical zeroing.

Paper location:
§3.4 decay bullet; Table 1.

Evidence:
- `public_evidence/exp11_horizon.csv` (γ=0.999: 1.0@512, 0.5@1024; γ=0.9999: 0.667@2048, 0.333@4096)
- `public_evidence/summary.json` (batch4_horizon_crossings N50: 1024.0 / 3072.0)

Evidence type:
processed + supporting snapshot.

What the evidence actually demonstrates:
1024 is a grid point at exactly 0.50; 3072 is the interpolation midpoint recorded in summary.json.

Conditions:
n=6.

Limitations:
3072 is interpolated, labeled as such in manuscript.

Status:
SUPPORTED.

## Claim ID: C010

Claim:
Distribution-shift probe: 0.75 → 0.25 over 0–4σ shift; frozen calibration 0.375 vs 0.875.

Paper location:
§3.4 shift bullet; §4.2 Length pointer.

Evidence:
- `public_evidence/exp11_shift.csv` (+ `exp11_shift_trials.csv`)
- `public_evidence/exp12_frozen.csv`

Evidence type:
processed (+ raw for shift).

What the evidence actually demonstrates:
Shift means 0.75/0.625/0.375/0.25; frozen 0.375 vs oracle-c 0.875.

Conditions:
n=8; N=2048; d=16. Synthetic analogue, not trained positional extrapolation (manuscript states this).

Limitations:
Small n; monotone decline observed on 4 grid points.

Status:
SUPPORTED.

## Claim ID: C011

Claim:
Cosine similarity declines 0.95 (N=128) → 0.18 (N=4096), 0.23 at N=8192, while relative error saturates.

Paper location:
§3.4 drift bullet; §4.2 Length pointer.

Evidence:
- `public_evidence/exp12_cosine.csv` (+ `exp12_cosine_trials.csv`)

Evidence type:
processed (+ raw).

What the evidence actually demonstrates:
cosine_mean 0.951 → … → 0.180 @4096 → 0.230 @8192.

Conditions:
n=8.

Limitations:
Non-monotone last step (0.18 → 0.23); manuscript reports both values.

Status:
SUPPORTED.

## Claim ID: C012

Claim:
Shrinking updates go ineffective after ≈11 steps (fp16), ≈24 (fp32), effectively never in fp64 (49/36/7 of 60); bf16 worse than fp16 here (52; 2.9e−03 vs 5.2e−04); fp32 accumulation cuts fp16 error ≈8×.

Paper location:
§3.5; §4.2 Numerics; Table 1/2; Abstract; Conclusion.

Evidence:
- `public_evidence/exp3_shrinking.csv`
- `public_evidence/exp12_bf16.csv`
- `public_evidence/exp8_mixed.csv`

Evidence type:
raw probe logs.

What the evidence actually demonstrates:
Cumulative ineffective counts 49/36/7; first-fail steps 11 (fp16) / 24 (fp32); bf16 52 with 2.9e−03 error; mixed 6.7e−05 vs 5.2e−04 (≈7.7×).

Conditions:
60-step probes, d=8 (shrinking); no trial structure (single probe runs).

Limitations:
Probe logs, not multi-trial experiments; bf16 result is probe-specific (manuscript scopes it).

Status:
SUPPORTED.

## Claim ID: C013

Claim:
Decayed steady-state magnitudes saturate ≈10–100× below naive linear-theory scale (0.324/1.41/2.52 vs 2.83/28.3/282.8).

Paper location:
§3.5.

Evidence:
- `public_evidence/exp3_decay.csv`

Evidence type:
processed.

What the evidence actually demonstrates:
Values exactly as claimed (0.3239/1.4135/2.5224 vs theory).

Conditions:
d=8, float64, single measurement per gamma.

Limitations:
Single measurement per gamma (no dispersion).

Status:
SUPPORTED.

## Claim ID: C014

Claim:
fp16 outputs drift 1–3e−05 from fp64 under decay while fp32 drift is ≈0; binary recall identical across precisions at this scale.

Paper location:
§3.5.

Evidence:
- `public_evidence/exp5_precision_decay.csv`
- `public_evidence/summary.json` (fp16 drifts 1.08e−05 / 2.72e−05; precision note)

Evidence type:
processed + supporting snapshot.

What the evidence actually demonstrates:
drift_vs_fp64 exactly as claimed; accuracies identical across precisions per gamma.

Conditions:
n=8, N=2048.

Limitations:
Magnitude-level effect only at this scale (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C015

Claim:
Softmax reference holds 1.0 at tested N with 2,176 B vs 1 MB persistent storage (N=4096); timing crosses over (≈10× linear faster at 8k: 0.39 vs 3.89 s); naive N×N fails at 32768 (4.3 GB > RAM), succeeds at 16384.

Paper location:
§3.1; §4.2 Length; Tables; Abstract (implicitly).

Evidence:
- `public_evidence/exp4_softmax_baseline.csv` (2176 / 1048576 bytes; 1.0 accuracies)
- `public_evidence/exp11_timing.csv` (0.389 / 3.890 s at 8192; n=3, n=2 at top)
- `public_evidence/exp12_oom.csv` (ok@16384; skipped-by-analysis@32768)

Evidence type:
processed + analysis record.

What the evidence actually demonstrates:
All values exactly as claimed; OOM is a by-analysis skip, not a crash log.

Conditions:
NumPy CPU (i3, 3.7 GB RAM per manuscript); fp32 full scoring; persistent-state accounting only.

Limitations:
Implementation-specific by design (manuscript states this); exp4 `trials` col reads 30 (see gaps).

Status:
SUPPORTED.

## Claim ID: C016

Claim:
Oracle latch restores decayed recall 0.05 → 0.4 (p=0.004, n=20) and holds 1.0 → 0.3 over N = 512 → 8192 (n=10); random gate ≈ fixed within noise.

Paper location:
§4.2 Mitigations; Tables; §4.2 Robustness (t=2.85).

Evidence:
- `public_evidence/exp8_selective.csv`
- `public_evidence/exp12_latchN.csv`
- `public_evidence/exp12_randomgate.csv`
- `public_evidence/exp12_ttests.csv` (oracle>fixed.99: t=2.845, p=0.0044, n=20+20)

Evidence type:
processed + statistical.

What the evidence actually demonstrates:
Fixed-0.99 0.05 → oracle 0.40 @4096; oracle 1.0/0.9/0.6/0.3/0.3 across N; random tracks fixed within noise.

Conditions:
As stated per file.

Limitations:
Oracle is an upper-bound control, not a learned mechanism (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C017

Claim:
Hybrid windows restore recent recall (w=128@1024, w=512@4096) without helping early markers (directional, p≈0.32).

Paper location:
§4.2 Mitigations; Table 2; Robustness.

Evidence:
- `public_evidence/exp8_hybrid.csv` (w=128: early 0.875, recent 1.0)
- `public_evidence/exp12_hybridN.csv` (w=512: early 0.333, recent 1.0)
- `public_evidence/exp12_ttests.csv` (hybrid-w128>w0-recent: t=1.0, p=0.3173, n=8)

Evidence type:
processed + statistical.

What the evidence actually demonstrates:
Recent-position restoration at both scales; early markers unhelped.

Conditions:
n=8 (@1024), n=6 (@4096).

Limitations:
Directional/non-significant (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C018

Claim:
Chunk resets cap state norm (14.81 → 5.17) but wipe early markers; mean-summary carry-forward is inert; stacked helps only under window coverage.

Paper location:
§4.2 Mitigations.

Evidence:
- `public_evidence/exp8_chunked.csv` (14.808 → 5.174 at chunk 256)
- `public_evidence/exp12_chunksummary.csv` (zero/summary wipe early 0.125; none keeps 0.375)
- `public_evidence/exp12_stacked.csv` (linear/hybrid 0.375 vs decayed/stacked 0.125)

Evidence type:
processed.

What the evidence actually demonstrates:
Values exactly as claimed at N=4096, n=8.

Conditions:
chunk sizes as filed; w=128 hybrid in stacked.

Limitations:
Single N tested for chunk norms.

Status:
SUPPORTED.

## Claim ID: C019

Claim:
Equal-budget multi-head banks show no effect (0.094/0.083/0.094 at m=16; t=0.35, p≈0.73); gated/novelty sweeps show no onset movement (0.102/0.094/0.109; off==hard 0.375@m8, t=0, p≈1.0).

Paper location:
§4.2 Mitigations; Table 1/2; Robustness.

Evidence:
- `public_evidence/exp11_multihead.csv` (+ trials)
- `public_evidence/exp12_gatedm.csv`
- `public_evidence/exp12_gatem.csv` (off==hard 0.375@m8, n=8)
- `public_evidence/exp12_ttests.csv` (multihead t=0.349 p=0.7269 n=6; hardgate t=0 p=1.0 n=6)

Evidence type:
processed + statistical.

What the evidence actually demonstrates:
Null results exactly as reported; note t-tests ran on n=6 (novelty/multihead) files while 0.375 equality also holds in n=8 gatem file.

Conditions:
Equal total budget (heads × head_dim constant per file layout).

Limitations:
Correct nulls, narrowly scoped (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C020

Claim:
Per-binding survival uniform across positions (0.28–0.38, n=30) — position-blind interference.

Paper location:
§4.2 Density; §5.1(ii).

Evidence:
- `public_evidence/exp12_survival.csv` (+ trials)

Evidence type:
processed (+ raw).

What the evidence actually demonstrates:
Quintile means 0.283/0.367/0.283/0.300/0.383.

Conditions:
m=8, d=16, n=30.

Limitations:
Single (m, d) configuration.

Status:
SUPPORTED.

## Claim ID: C021

Claim:
Dissipation: ×20-repeated coarse signal survives (1.0) where singletons are lost (0.0) at every tested N including N=128.

Paper location:
§4.2 Dissipation; §4.3 Interactions.

Evidence:
- `public_evidence/exp7_coarsefine.csv` (n=8)
- `public_evidence/exp12_multiseed.csv` (dissipation probe at seeds 43/44, n=10)

Evidence type:
processed.

What the evidence actually demonstrates:
coarse 1.0 / fine 0.0 at N = 128/512/2048/8192 in both files.

Conditions:
As filed.

Limitations:
None noted beyond setup.

Status:
SUPPORTED.

## Claim ID: C022

Claim:
Per-channel decay gradient 0.0 → 0.5 → 0.625 (directional, p≈0.63, n=8); compressibility directional (0.675 vs 0.55; A8>random p≈0.82, n=40).

Paper location:
Table 1; §5.1(i); Robustness.

Evidence:
- `public_evidence/exp11_perchannel.csv` (+ trials)
- `public_evidence/exp11_compress.csv` (+ trials)
- `public_evidence/exp12_ttests.csv` (perchannel t=0.475 p=0.6347; A8>random t=0.234 p=0.8153)

Evidence type:
processed + statistical.

What the evidence actually demonstrates:
Values exactly as claimed.

Conditions:
n=8 (perchannel); n=40 (compress).

Limitations:
Both directional/non-significant (manuscript states this). summary.json holds contradictory older compress values (see EVIDENCE_GAPS).

Status:
SUPPORTED.

## Claim ID: C023

Claim:
Recall overlays tightly across seeds 42–45; length/probe repeats at 43–44; distractor scale conditions absolute accuracies.

Paper location:
§4.2 Robustness; §4.2 Density/Length.

Evidence:
- `public_evidence/exp10_seed_recall.csv`
- `public_evidence/exp10_seed_length.csv`
- `public_evidence/exp12_multiseed.csv`
- `public_evidence/exp11_distractor.csv`

Evidence type:
processed.

What the evidence actually demonstrates:
Seed curves within noise of each other; distractor 1.0 / 0.667–0.167 / 0.167–0.0 pattern.

Conditions:
As filed per probe.

Limitations:
Four seeds = consistency check, not broad robustness (manuscript states this).

Status:
SUPPORTED.

## Claim ID: C024

Claim:
Theoretical statements: γ^N expansion, zeroing condition N > ln(ε)/ln(γ), e-folding −1/ln γ (≈1/(1−γ)), fp32-zero ≈16.6/(1−γ).

Paper location:
§3.4 decay bullet; Table 1.

Evidence:
Derivations (no data file required); cross-checked arithmetically during audit. `public_evidence/exp5_gated_decay.csv` stores matching theory-horizon columns (10/100/1000/10000; 166.4/1663.6/16635.5/166355.3).

Evidence type:
derived (+ stored theory columns as consistency check).

What the evidence actually demonstrates:
Algebra verified; stored horizon columns match the formulas.

Conditions:
γ ≈ 1 for the approximation; ε = 2^−24 for fp32.

Limitations:
Theory of the synthetic recurrence only.

Status:
SUPPORTED.

## Claim ID: C025

Claim (architectural context, NOT experimental):
Linear Transformers / RetNet / RWKV / Mamba use bounded recurrent/state representations per their cited specifications.

Paper location:
§2.1 bullets; §2.5 [A]/[Arch]; Table 1 [Arch] cells.

Evidence:
Cited specifications [1]–[4] (literature, not data files). Code docstrings reference the same mapping with disclaimers (e.g. `models/linear_attention.py`, `experiments/gated_recall.py`, `experiments/variants.py`).

Evidence type:
architectural/literature.

What the evidence actually demonstrates:
Descriptions match the cited works at the level stated; no measurements of these systems exist anywhere in the repo (Phase 5 verdict).

Conditions:
Literature-level only.

Limitations:
Must never be read as experimental validation (manuscript states this repeatedly).

Status:
SUPPORTED (as scoped architectural context).
