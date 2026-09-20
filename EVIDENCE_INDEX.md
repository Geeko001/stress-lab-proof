# EVIDENCE_INDEX.md

Audit of empirical evidence for **“Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes”** (manuscript: `Paper_1_Draft_v3.md`, frozen).

Rules followed: no metadata invented. Where a field cannot be determined from the file, it says **“Not determinable from available file.”** All paths are repo-relative. Environment for all runs (per `public_evidence/summary.json`, `public_evidence/config.json`): CPU-only NumPy lab, base seed 42, Windows 11, Python 3.14.7 / NumPy 2.5.3.

Conventions: `trials` column = number of independent trials per condition (one trial = one full task run on an independent `default_rng(base_seed + trial)` stream, per manuscript §4.2). `raw` = per-trial records; `processed` = per-condition aggregates (mean/std).

## A. Density / associative-capacity family

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp1_dimensionality.csv` | Recall accuracy vs bindings m × dim, elu+1, float64 | 25 | Not determinable from available file (config/run protocol: base 42) | m ∈ {2…128}, d ∈ {8,16,32,64} | linear_mean accuracy | processed | §4.2 Density (0.96→0.008; m* = 8; m4 ceiling) | Direct |
| `public_evidence/exp01_trials.csv` | Per-trial records for Exp 1 | 25 (seeds 42–66) | 42–66 | m × d grid | linear/softmax/state_norm per trial | raw | Same as above; CI recomputation | Direct |
| `public_evidence/exp10_bigd.csv` | Recall at large d (rerun-added d=128 row: m=4, 25 trials, seed 42) | 10 (d=256); 25 (d=128 row) | 42 (d=128 row only; others not determinable) | d ∈ {128, 256}, elu+1 | linear_mean | processed (+1 rerun row, see EVIDENCE_GAPS) | §4.2 Density (m* = 8 incl. d=256; 0.86 at d=128) | Direct |
| `public_evidence/exp10_seed_recall.csv` | Recall curves repeated across seeds | 10 per seed | 42, 43, 44, 45 | d=16, elu+1 | linear_mean | processed | §4.2 Robustness (tight overlays) | Direct |
| `public_evidence/exp6_featuremaps.csv` | Feature-map comparison (elu+1/relu/exp × d 16/32) | 15 | Not determinable from available file | m ∈ {2…128} | linear_mean | processed | §4.2 Density (relu 0.73 etc.; t-test input) | Direct |
| `public_evidence/exp7_orthkeys.csv` | Orthogonal vs random keys under elu+1 | 10 | Not determinable from available file | d=16, m ∈ {2…128} | linear_mean | processed | §3.3/§4.2 (0.30 orth vs random) | Direct |
| `public_evidence/exp7_mNgrid.csv` | Bindings m × filler length (background-length control) | 8 | Not determinable from available file | d=16, m=4, nfill ∈ {0,256,2048} | accuracy | processed | §4.2 (filler saturation; binding-count not background) | Direct |
| `public_evidence/exp11_centered.csv` | Centered-feature recall, d=16 | 15 | Not determinable from available file | m ∈ {2…128} | linear_mean | processed | §3.3/§4.2 plateau (n=15 part) | Direct |
| `public_evidence/exp12_centered_d.csv` | Centered-feature recall × dim | 12 | Not determinable from available file | d ∈ {8,32,64} | linear_mean | processed | m* ≈ 16/64/128, m* ≈ 2d (n=12 part) | Direct |
| `public_evidence/exp11_centered_orth.csv` | Centered features + orthogonal keys, d=16 | 10 | Not determinable from available file | m ∈ {2…128} | linear_mean | processed | centered-orth flat 1.0 → 0.5@m32 | Direct |
| `public_evidence/exp12_centered_orth_d.csv` | Centered-orth × dim (d=8/32/64 only) | 10 | Not determinable from available file | d ∈ {8,32,64} | linear_mean | processed | Non-monotonicity across dims (note in brief R5) | Supporting |
| `public_evidence/exp12_multiseed.csv` | Five follow-up probes repeated at seeds 43, 44 | 10 per seed | 43, 44 | d=16; probes: gated_gamma, selective, perchannel, centered_d16, dissipation | mixed | processed | §4.2 Robustness | Direct |
| `public_evidence/exp12_survival.csv` | Per-binding-position survival, m=8 | 30 | Not determinable from available file | d=16, quintiles 0–4 | accuracy | processed | Position-blind interference (0.28–0.38) | Direct |
| `public_evidence/exp12_survival_trials.csv` | Per-trial survival records | 30 (seeds 42–71) | 42–71 | m=8, d=16 | recovered per trial | raw | Same as above | Direct |

## B. Length / long-context family

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp2_length_sweep.csv` | Marker recall × N × position | 10 | Not determinable from available file (protocol: base 42) | N ∈ {128,512,1024,4096}, pos {0.1,0.5,0.9}, d=16 | accuracy, retrieval_error, state_norm | processed | §4.2 Length (1.0/1.0/0.8/0.2; flat positions) | Direct |
| `public_evidence/exp02_trials.csv` | Per-trial records for Exp 2 | 10 per cell (trial col present; seeds not recorded) | Not determinable from available file | Same grid | recovered/error/state_norm | raw | Same as above | Direct |
| `public_evidence/exp9_longlength.csv` | Streaming recall to N=1M | 10 at N≤65536; 3 at 262144; 2 at 1M (per `trials` col) | Not determinable from available file | N ∈ {16384…1048576}, d=16 | accuracy, state_norm | processed | Floor ≈0 for N≥16k; norm slope 0.448 (with Exp 2) | Direct |
| `public_evidence/exp10_seed_length.csv` | Length decay repeated at seeds 43, 44 | 5 per seed | 43, 44 | d=16 | linear_mean | processed | §4.2 Robustness (decay shape repeats) | Direct |
| `public_evidence/exp11_horizon.csv` | Recall × N × gamma (horizon sweep) | 6 | Not determinable from available file | γ ∈ {0.99, 0.999, 0.9999}, N ∈ {512…8192} | accuracy | processed | 50%-crossings ≈1024 / ≈3072-interp. | Direct |
| `public_evidence/exp11_shift.csv` | Key-statistics shift probe | 8 | Not determinable from available file | shift ∈ {0,1,2,4}σ, N=2048, d=16 | accuracy | processed | Shift 0.75→0.25 | Direct |
| `public_evidence/exp11_shift_trials.csv` | Per-trial shift records | 8 (seeds 42–49) | 42–49 | Same grid | recovered per trial | raw | Same as above | Direct |
| `public_evidence/exp12_frozen.csv` | Frozen vs oracle centering constant | 8 | Not determinable from available file | N=2048, d=16 | accuracy | processed | Frozen 0.375 vs oracle 0.875 | Direct |
| `public_evidence/exp12_cosine.csv` | Cosine-similarity drift × N | 8 | Not determinable from available file | N ∈ {128…8192}, d=16 | cosine_mean, accuracy | processed | Cosine 0.95→0.18@4096, 0.23@8192 | Direct |
| `public_evidence/exp12_cosine_trials.csv` | Per-trial cosine records | 8 (seeds 42–49) | 42–49 | Same grid | per-trial values | raw | Same as above | Direct |
| `public_evidence/exp11_distractor.csv` | Distractor-scale sensitivity | 6 | Not determinable from available file | scale ∈ {0.001,0.01,0.1}, N ∈ {1024,4096} | accuracy | processed | Absolute-accuracy conditioning | Direct |
| `public_evidence/exp11_compress.csv` | Alphabet-size (compressibility) probe | 40 | Not determinable from available file | alphabet ∈ {8,64,−1}, N=2048, d=16 | accuracy | processed | §5.1(i) (0.675 vs 0.55; p≈0.82 on A8>random) | Direct |
| `public_evidence/exp11_compress_trials.csv` | Per-trial compressibility (recovered 0/1) | 40 per alphabet | Seeds present per row (col `seed`) | Same grid | recovered | raw | Same as above | Direct |
| `public_evidence/exp11_noisy.csv` | Query-noise robustness | 10 | Not determinable from available file | σ ∈ {0,0.1,0.3,1.0}, N=1024 | linear/softmax accuracy | processed | NOT cited in manuscript (see EVIDENCE_GAPS) | Unused |
| `public_evidence/exp11_noisy_trials.csv` | Per-trial noise records | 10 (seeds 42–51) | 42–51 | Same grid | linear/softmax per trial | raw | NOT cited in manuscript | Unused |

## C. Decay / gating family

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp5_gated_decay.csv` | Accuracy at N=4096 × gamma | 20 | Not determinable from available file (protocol: base 42) | γ ∈ {1.0,0.9,0.99,0.999,0.9999}, d=16 | accuracy, state_norm, theory horizons | processed | Latch bounds context; γ=0.9999 label fixed (see gaps) | Direct |
| `public_evidence/exp12_gatedm.csv` | Recall density under decay (m-sweep) | 8 | Not determinable from available file | γ ∈ {1.0,0.99,0.9}, m ∈ {2…128} | accuracy | processed | No onset movement (0.102/0.094/0.109@m16) | Direct |
| `public_evidence/exp12_gatedm_trials.csv` | Per-trial gatedm records | 8 (seeds 42–49) | 42–49 | Same grid | recovered per trial | raw | Same as above | Direct |
| `public_evidence/exp12_gatem.csv` | Novelty-gate m-sweep (off/soft/hard) | 8 | Not determinable from available file | d=16, m ∈ {2…128} | accuracy | processed | off==hard 0.375@m8 | Direct |
| `public_evidence/exp12_gatem_trials.csv` | Per-trial gatem records | 8 (seeds 42–49) | 42–49 | Same grid | recovered per trial | raw | Same as above | Direct |
| `public_evidence/exp11_novelty.csv` | Novelty-gate comparison (older run, n=6) | 6 | Not determinable from available file | d=16 | accuracy | processed | Superseded by gatem for quoted numbers; t-test n=6 source | Supporting |
| `public_evidence/exp11_novelty_trials.csv` | Per-trial novelty records | 6 (seeds 42–47) | 42–47 | Same grid | recovered per trial | raw | t-test hardgate~off (t=0) input | Supporting |
| `public_evidence/exp11_perchannel.csv` | Per-channel vs scalar decay | 8 | Not determinable from available file | modes scalar-0.99/scalar-0.999/perchannel, N=1024 | accuracy | processed | Gradient 0→0.5→0.625, p≈0.63 | Direct |
| `public_evidence/exp11_perchannel_trials.csv` | Per-trial perchannel records | 8 (seeds 42–49) | 42–49 | Same grid | recovered per trial | raw | Same as above | Direct |
| `public_evidence/exp12_randomgate.csv` | Fixed vs oracle vs random gating × N | 8 | Not determinable from available file | N ∈ {1024…8192}, d=16 | accuracy | processed | Random ≈ fixed within noise | Direct |

## D. Mitigations family

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp8_selective.csv` | Fixed vs oracle salience latch | 20 | Not determinable from available file (protocol: base 42) | N=4096, d=16 | accuracy | processed | Latch 0.05→0.4, p=0.004 | Direct |
| `public_evidence/exp12_latchN.csv` | Latch × N (fixed-1.0/fixed-0.99/oracle) | 10 | Not determinable from available file | N ∈ {512…8192} | accuracy | processed | Latch holds 1.0→0.3 | Direct |
| `public_evidence/exp8_hybrid.csv` | Hybrid exact window × position | 8 | Not determinable from available file | w ∈ {0…1024}, N=1024, d=16 | accuracy | processed | w=128 restores recent @1024 | Direct |
| `public_evidence/exp12_hybridN.csv` | Hybrid windows at N=4096 | 6 | Not determinable from available file | w ∈ {0…512}, d=16 | accuracy | processed | w=512 restores recent @4096 | Direct |
| `public_evidence/exp8_chunked.csv` | Chunk-reset sweep (chunk 256/1024/4096) | 8 | Not determinable from available file | N=4096, d=16 | accuracy, max_state_norm | processed | Norm cap 14.81→5.17 | Direct |
| `public_evidence/exp12_chunksummary.csv` | Zero/summary/no-op chunk modes | 8 | Not determinable from available file | chunk=1024, N=4096 | accuracy, max_state_norm | processed | Resets wipe early; mean-summary inert | Direct |
| `public_evidence/exp11_multihead.csv` | Multi-head banks (1/4/16 heads) | 6 | Not determinable from available file | m ∈ {2…128} | accuracy | processed | No effect at equal budget (0.094/0.083/0.094@m16) | Direct |
| `public_evidence/exp11_multihead_trials.csv` | Per-trial multihead records | 6 (seeds 42–47) | 42–47 | Same grid | recovered per trial | raw | Same as above; t-test input | Direct |
| `public_evidence/exp12_stacked.csv` | Stacked decayed+hybrid × position | 8 | Not determinable from available file | N=4096, d=16 | accuracy | processed | Helps only under window coverage | Direct |
| `public_evidence/exp7_coarsefine.csv` | Coarse (×20 repeat) vs fine singleton | 8 | Not determinable from available file | N ∈ {128…8192}, d=16 | coarse/fine accuracy | processed | Dissipation (1.0 vs 0.0 at every scale) | Direct |

## E. Precision family

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp3_shrinking.csv` | Shrinking-update effectiveness × precision | Per-step probe (60 steps; no trial structure) | Not applicable | fp64/fp32/fp16, d=8 | ineffective flags/cumulative | raw probe log | 49/36/7 ineffective of 60; first-fail ≈11/≈24 steps | Direct |
| `public_evidence/exp03_trials.csv` | Per-step trial records for Exp 3 | 5 (seeds 42–46) | 42–46 | d=8 | per-step values | raw | Same as above | Direct |
| `public_evidence/exp3_decay.csv` | Decayed steady-state magnitudes × gamma | Single measurement per gamma | Not determinable from available file | γ ∈ {0.99,0.999,0.9999}, d=8 | final_state_magnitude | processed | 0.324/1.41/2.52 vs naive theory | Direct |
| `public_evidence/exp12_bf16.csv` | Real bf16 vs fp16/fp32 accumulation | Per-run probe (60 steps; no trial structure) | Not applicable | ml_dtypes bf16 | ineffective, rel_error | raw probe log | bf16 52 ineffective, 2.9e−03 error | Direct |
| `public_evidence/exp8_mixed.csv` | fp32 vs fp16 state accumulation | Per-run probe (60 steps; no trial structure) | Not applicable | 60 steps | ineffective, rel_error | raw probe log | 8× error cut (6.7e−05 vs 5.2e−04) | Direct |
| `public_evidence/exp5_precision_decay.csv` | Recall + drift vs precision × gamma | 8 | Not determinable from available file | precisions × γ ∈ {0.99,0.999}, N=2048 | accuracy, drift_vs_fp64 | processed | fp16 drift 1–3e−05; recall identical | Direct |

## F. Baseline / timing / OOM

| File | Contents | Trials | Seeds | Conditions | Metric | Type | Supports (manuscript) | Mapping confidence |
|---|---|---|---|---|---|---|---|---|
| `public_evidence/exp4_softmax_baseline.csv` | Softmax vs linear reference | 30 (per `trials` col; likely 10 trials × 3 positions) | 42 | N ∈ {128…4096}, d=16 | accuracies, times, bytes | processed | Softmax flat 1.0; 2176 B vs 1 MB | Direct |
| `public_evidence/exp04_trials.csv` | Per-trial records for Exp 4 | trial col 0–9 × positions (seeds not recorded) | Not determinable from available file | Same grid | linear/softmax per trial | raw | Same as above | Direct |
| `public_evidence/exp11_timing.csv` | Full-sequence timing sweep | 3 (2 at N=8192) | Not applicable (timing) | N ∈ {256…8192} | linear/softmax seconds, theory bytes | processed | Crossover; 0.39 vs 3.89 s at 8k | Direct |
| `public_evidence/exp12_oom.csv` | OOM probe outcomes | Not applicable (2 rows: ok / skipped-by-analysis) | Not applicable | N ∈ {16384, 32768} | status + theory bytes | analysis record | Naive N×N fails at 32768 (4.3 GB > RAM) | Direct |

## G. Statistics / bundle metadata

| File | Contents | Type | Supports (manuscript) |
|---|---|---|---|
| `public_evidence/exp12_ttests.csv` | 9 Welch t-tests (comparison, t, normal-approx p, n1, n2) | statistical | All reported t/p/n (Robustness §) |
| `public_evidence/exp12_CIs.csv` | 193-row normal 95% CI table (file, cell, metric, mean, ci95) | statistical | CI ±0.05@m2 etc.; CI method |
| `public_evidence/summary.json` | Bundle-level key results + env + batch notes (generated 2026-09-17) | supporting (computed snapshot) | Cross-check only — NOTE: compress values here (0.75/0.625) contradict CSVs; see EVIDENCE_GAPS |
| `public_evidence/config.json` | Experiment configuration (grids, trials, seeds, env) | supporting (protocol record) | Trial/seed/dim/N grids for Exps 1–4, followups, batch3/4 |
| `public_evidence/pytest_report.txt` | Unit-test report (per summary.json: tests passed) | supporting | Reproducibility; not cited in manuscript |

## H. Figures (48 PNGs in `public_evidence/figures/`)

15 measured figures embedded in the PDF (exp → PNG mapping in `build_paper_pdf.py` FIGURES list); 3 content-only schematics (`schematic_conflict.png`, `schematic_storage.png`, `schematic_taxonomy.png`, generated by `gen_schematics.py`, no measured data); remaining ~30 PNGs are alternate/superseded views (e.g. `exp10a/b/c`, `exp3_gamma_retention`, `exp5b_precision_decay`, `exp8a/b/d`, `exp11a–j`, `exp12a2/b/c/d/e/h/i/k/l`) — figure sources traceable to the CSVs above by name. Full PNG listing omitted for brevity; every embedded figure's source CSV is identified in its row above.

## I. Code provenance (scripts that generated the evidence; not evidence themselves)

`run_exp4.py` (Exps 1–4), `run_followups.py` (Exps 5–10 probes), `run_batch3.py` (Exp 11 probes + top-ups), `run_batch4.py` (Exp 12 probes + stats + figures), `generate_public_bundle.py` (bundle assembly), `experiments/*.py` (task implementations), `models/linear_attention.py` + `models/softmax_attention.py` (synthetic NumPy models only — no production architectures).
