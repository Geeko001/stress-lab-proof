# EXPERIMENTS.md

Reproduction index for the experiments already performed. Groups follow EXPERIMENT_REGISTRY.md. Commands are quoted from each script's own header docstring (`[stage|…|all]` selects a stage; default `all`). All runners write incrementally into `public_evidence/`. Claim IDs refer to CLAIM_EVIDENCE_MAP.md.

## Associative interference / density-driven capacity

| Experiment | Purpose | Script / command | Key parameters | Output → evidence | Claims |
|---|---|---|---|---|---|
| Exp 1 dimensionality sweep | Recall vs bindings × dim | `generate_public_bundle.py` (Exps 1–3) | m ∈ {2…128}, d ∈ {8,16,32,64}, n=25, elu+1 | `exp1_dimensionality.csv`, `exp01_trials.csv` | C001 |
| Big-d spot points | m* at d=256; d=128 rerun point | `run_followups.py stage3` (+ direct `run_dimensionality_sweep(m_values=[4], dim_values=[128], trials=25, seed=42)` for the d=128 row) | d ∈ {128, 256} | `exp10_bigd.csv` | C001 |
| Feature-map comparison | relu/exp/elu+1 × d | `run_followups.py stage1` | d ∈ {16,32}, n=15 | `exp6_featuremaps.csv` | C002 |
| Orthogonal keys | Key geometry vs feature geometry | `run_followups.py stage3` | d=16, n=10 | `exp7_orthkeys.csv` | C003 |
| Filler (m×N) grid | Binding count vs background length | `run_followups.py stage3` | m=4, nfill ∈ {0,256,2048}, n=8 | `exp7_mNgrid.csv` | C006 |
| Centered features (+orth, +dims) | Plateau + rank-scaled knee | `run_batch3.py stageB` (centered), `run_batch4.py stageC` (centered_d) | n=15 / n=12 | `exp11_centered.csv`, `exp12_centered_d.csv`, `exp11_centered_orth.csv`, `exp12_centered_orth_d.csv` | C004, C005 |
| Survival pattern | Per-position survival, m=8 | `run_batch4.py stageC` | n=30, quintiles | `exp12_survival.csv`, `exp12_survival_trials.csv` | C020 |
| Seed robustness (recall) | Cross-seed overlays | `run_followups.py stage2` | seeds 42–45, n=10 | `exp10_seed_recall.csv` | C023 |

## Sequence-length / long-horizon behavior

| Experiment | Purpose | Script / command | Key parameters | Output → evidence | Claims |
|---|---|---|---|---|---|
| Length sweep | Recall × N × position | `generate_public_bundle.py` | N ∈ {128…4096}, pos {0.1,0.5,0.9}, n=10 | `exp2_length_sweep.csv`, `exp02_trials.csv` | C007 |
| Long streaming | Recall to N=1M, norm slope | `run_followups.py stage1` | N ∈ {16384…1048576}, n=10/3/2 | `exp9_longlength.csv` | C007, C008 |
| Seed length | Decay-shape repeat | `run_followups.py stage2` | seeds 43–44, n=5 | `exp10_seed_length.csv` | C023 |
| Shift / frozen / cosine / distractor / compress | Distribution-shift analogues + sensitivity | `run_batch3.py stageB` (shift, noisy, centered, distractor, compress), `run_batch4.py stageD` (frozen) | n=8 (shift/frozen/cosine), n=6 (distractor), n=40 (compress) | `exp11_shift.csv` (+trials), `exp12_frozen.csv`, `exp12_cosine.csv` (+trials), `exp11_distractor.csv`, `exp11_compress.csv` (+trials) | C010, C011, C022, C023 |

## Explicit temporal decay / attenuation

| Experiment | Purpose | Script / command | Key parameters | Output → evidence | Claims |
|---|---|---|---|---|---|
| Gated decay @4096 | Accuracy × gamma + theory horizons | `run_followups.py` (`run_gamma_sweep`) / `run_batch3.py` top-up (n=20) | γ ∈ {1.0,0.9,0.99,0.999,0.9999}, n=20 | `exp5_gated_decay.csv` | C009 (context), C016 (bounds) |
| Horizon curves | 50%-crossings × N × gamma | `run_batch3.py stageB` | γ ∈ {0.99,0.999,0.9999}, n=6 | `exp11_horizon.csv` | C009 |
| Gated m-sweep / gate m-sweep | Density under decay; novelty gates | `run_batch4.py stageC` | n=8 | `exp12_gatedm.csv` (+trials), `exp12_gatem.csv` (+trials), `exp11_novelty.csv` (+trials, older n=6 run) | C019 |
| Per-channel decay | Scalar vs per-channel gates | `run_batch3.py stageA` | n=8 | `exp11_perchannel.csv` (+trials) | C022 |
| Random gate | Random-gating lower bound | `run_batch4.py stageD` | n=8 | `exp12_randomgate.csv` | C016 |

## Numerical precision loss

| Experiment | Purpose | Script / command | Key parameters | Output → evidence | Claims |
|---|---|---|---|---|---|
| Shrinking updates | First-ineffective step × precision | `generate_public_bundle.py` | 60 steps, d=8, fp64/32/16 | `exp3_shrinking.csv`, `exp03_trials.csv` | C012 |
| Decay magnitudes | Steady-state vs naive theory | `generate_public_bundle.py` | γ set, d=8 | `exp3_decay.csv` | C013 |
| Mixed precision | fp32 vs fp16 accumulation | `run_followups.py` | 60 steps | `exp8_mixed.csv` | C012 |
| bf16 probe | Real bf16 (ml_dtypes) behavior | `run_batch4.py stageD` | 60 steps | `exp12_bf16.csv` | C012 |
| Precision × decay | Recall + drift vs precision | `run_followups.py stage2` | n=8, N=2048 | `exp5_precision_decay.csv` | C014 |

## Mitigation probes

| Experiment | Purpose | Script / command | Key parameters | Output → evidence | Claims |
|---|---|---|---|---|---|
| Oracle latch | Upper-bound selective control | `run_followups.py` / `run_batch3.py` top-up | N=4096, n=20 | `exp8_selective.csv` | C016 |
| Latch × N | Latch retention curve | `run_batch4.py stageC` | N ∈ {512…8192}, n=10 | `exp12_latchN.csv` | C016 |
| Hybrid windows | Exact window + compressed state | `run_followups.py` (N=1024, n=8); `run_batch4.py stageD` (N=4096, n=6) | w ∈ {0…1024} / {0…512} | `exp8_hybrid.csv`, `exp12_hybridN.csv` | C017 |
| Stacked | Decayed + hybrid combined | `run_batch4.py stageD` | N=4096, n=8 | `exp12_stacked.csv` | C018 |
| Chunked / chunk-summary | Resets and carry-forward | `run_followups.py` (chunked); `run_batch4.py stageD` (summary) | n=8 | `exp8_chunked.csv`, `exp12_chunksummary.csv` | C018 |
| Multi-head banks | Equal-budget head comparison | `run_batch3.py stageA` | n=6 | `exp11_multihead.csv` (+trials) | C019 |
| Coarse vs fine | Dissipation probe | `run_followups.py stage3`; `run_batch4.py stageD` (multiseed) | n=8 / n=10 | `exp7_coarsefine.csv`, `exp12_multiseed.csv` | C021 |
| Softmax baseline / timing / OOM | Reference, crossover, naive bound | `run_exp4.py`; `run_batch3.py stageB` (timing); `run_batch4.py stageD` (OOM) | n=10 / n=3 (2@8192) / analysis record | `exp4_softmax_baseline.csv` (+trials), `exp11_timing.csv`, `exp12_oom.csv` | C015 |

## Statistics / bundle outputs (not experiments)

Produced by `run_batch4.py finalize`: `exp12_ttests.csv` (9 Welch tests), `exp12_CIs.csv` (193 rows), plus `summary.json` / `config.json` updates. Figures rendered by each runner's `finalize` stage into `public_evidence/figures/`.
