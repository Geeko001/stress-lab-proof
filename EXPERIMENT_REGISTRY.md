# EXPERIMENT_REGISTRY.md

Experiment families actually present in the repository, with purpose, scripts, data, stats, figures, and manuscript claims. No experiment is forced into a category its files do not support. Generation scripts: `run_exp4.py`, `run_followups.py`, `run_batch3.py`, `run_batch4.py`, `generate_public_bundle.py`; task code in `experiments/*.py`; models in `models/*.py` (synthetic NumPy only).

## 1. Associative capacity / density

Purpose: measure recall vs binding count, dimension, feature map, key geometry.
Scripts: `run_exp4.py` (Exp 1), `run_followups.py` (featuremaps/orthkeys), `run_batch3.py` (centered), `run_batch4.py` (centered_d, survival, multiseed), `experiments/associative_recall.py`, `experiments/finalbatch.py`.
Raw: `exp01_trials.csv`, `exp12_survival_trials.csv`.
Processed: `exp1_dimensionality.csv`, `exp10_bigd.csv`, `exp10_seed_recall.csv`, `exp6_featuremaps.csv`, `exp7_orthkeys.csv`, `exp7_mNgrid.csv`, `exp11_centered.csv`, `exp12_centered_d.csv`, `exp11_centered_orth.csv`, `exp12_centered_orth_d.csv`, `exp12_survival.csv`, `exp12_multiseed.csv` (centered_d16 probe).
Stats: `exp12_ttests.csv` (relu, centered, centered-orth rows), `exp12_CIs.csv` (193 rows incl. density cells).
Figures: `exp1_overlay_by_dim.png`, `exp1_linear_vs_softmax_d16.png`, `exp6_featuremaps.png`, `exp7c_orthkeys.png`, `exp11g_centered.png`, `exp12a_centered_d.png`, `exp12a2_centered_orth_d.png`, `exp12e_survival.png`, `exp10a_seed_recall.png`, `exp10c_bigd_overlay.png`.
Claims: C001–C006, C020, C023.

## 2. Sequence-length / long-context behavior

Purpose: recall vs N, positions, streaming to 1M, cross-seed decay shape.
Scripts: `run_exp4.py` (Exp 2), `run_followups.py` (longlength, seeds), `run_batch4.py`.
Raw: `exp02_trials.csv`.
Processed: `exp2_length_sweep.csv`, `exp9_longlength.csv`, `exp10_seed_length.csv`, `exp12_multiseed.csv` (shape repeats).
Stats: `exp12_CIs.csv` (length cells).
Figures: `exp2_accuracy_vs_length.png`, `exp2_statenorm_vs_length.png`, `exp9_longlength_full.png`, `exp10b_seed_length.png`.
Claims: C007, C008, C023.

## 3. Decay / gated recurrence

Purpose: scalar-γ and per-channel-γ decay, novelty gating, random gating, horizon crossings.
Scripts: `run_followups.py`, `run_batch3.py` (horizon sweep, gated top-ups), `run_batch4.py` (gatedm, multiseed), `experiments/gated_recall.py`, `experiments/variants.py`, `experiments/finalbatch.py`.
Raw: `exp12_gatedm_trials.csv`, `exp12_gatem_trials.csv`, `exp11_novelty_trials.csv`, `exp11_perchannel_trials.csv`.
Processed: `exp5_gated_decay.csv`, `exp11_horizon.csv`, `exp12_gatedm.csv`, `exp12_gatem.csv`, `exp11_novelty.csv`, `exp11_perchannel.csv`, `exp12_randomgate.csv`, `exp12_multiseed.csv` (gated_gamma/selective/perchannel probes).
Stats: `exp12_ttests.csv` (perchannel, hardgate rows).
Figures: `exp5_gamma_horizon.png`, `exp11h_horizon.png`, `exp11b_perchannel.png`, `exp11c_novelty.png`, `exp12b_gatedm.png`, `exp12c_gatem.png`, `exp12l_randomgate.png`.
Claims: C009, C016 (bounds), C019, C022 (per-channel), C023.

## 4. Numerical precision

Purpose: shrinking-update effectiveness, steady-state magnitudes, bf16/fp16/fp32/fp64 behavior, precision×decay drift.
Scripts: `run_exp4.py` (Exp 3), `run_followups.py` (mixed, precision_decay, bf16 in batch4), `experiments/numerical_saturation.py`.
Raw: `exp03_trials.csv`, `exp3_shrinking.csv`, `exp12_bf16.csv`, `exp8_mixed.csv`.
Processed: `exp3_decay.csv`, `exp5_precision_decay.csv`.
Stats: none (probe-level; dispersions not applicable).
Figures: `exp3_effective_change.png`, `exp3_gamma_retention.png`, `exp5b_precision_decay.png`, `exp12g_bf16.png`, `exp8d_mixed.png`.
Claims: C012–C014.

## 5. Distribution / channel shift

Purpose: key-statistics shift, frozen calibration, distractor scale, compressibility, query noise (unused).
Scripts: `run_batch3.py` (shift, frozen, distractor, compress, noisy), `experiments/variants.py`.
Raw: `exp11_shift_trials.csv`, `exp11_compress_trials.csv`, `exp11_noisy_trials.csv`.
Processed: `exp11_shift.csv`, `exp12_frozen.csv`, `exp11_distractor.csv`, `exp11_compress.csv`, `exp11_noisy.csv`.
Stats: `exp12_ttests.csv` (A8>random row), `exp12_CIs.csv` (compress cells).
Figures: `exp11e_shift.png`, `exp12k_frozen.png`, `exp11i_distractor.png`, `exp11a_compress.png`, `exp11f_noisy.png`.
Claims: C010, C022 (compress), C023 (distractor). Noisy files support NO manuscript claim (see EVIDENCE_GAPS).

## 6. Timing / computational comparison

Purpose: full-sequence crossover timing; storage accounting; naive OOM bound.
Scripts: `run_exp4.py` (Exp 4 bytes/times), `run_batch3.py` (timing sweep), `run_batch4.py` (OOM analysis record).
Raw: `exp04_trials.csv`.
Processed: `exp4_softmax_baseline.csv`, `exp11_timing.csv`, `exp12_oom.csv`.
Stats: none.
Figures: `exp4_softmax_comparison.png`, `exp11j_timing.png`, `exp12m_oom.png`.
Claims: C015.

## 7. Chunk summarization

Purpose: chunk resets, zero/summary carry-forward, norm capping vs retention trade-off.
Scripts: `run_followups.py` (chunked), `run_batch4.py` (chunksummary), `experiments/mitigations.py`, `experiments/finalbatch.py`.
Raw: none (aggregates only).
Processed: `exp8_chunked.csv`, `exp12_chunksummary.csv`.
Stats: none.
Figures: `exp8b_chunked.png`, `exp12h_chunksummary.png`.
Claims: C018.

## 8. Hybrid windows

Purpose: exact local window + compressed state, at N=1024 and N=4096, stacked combination.
Scripts: `run_followups.py` (hybrid), `run_batch4.py` (hybridN, stacked), `experiments/mitigations.py`, `experiments/finalbatch.py`.
Raw: none (aggregates only).
Processed: `exp8_hybrid.csv`, `exp12_hybridN.csv`, `exp12_stacked.csv`.
Stats: `exp12_ttests.csv` (hybrid-w128 row).
Figures: `exp8a_hybrid.png`, `exp12i_hybridN.png`, `exp12j_stacked.png`.
Claims: C017, C018 (stacked).

## 9. Oracle latch (selective upper bound)

Purpose: oracle salience latch vs fixed decay, across N.
Scripts: `run_followups.py`/`run_batch3.py` (selective top-up to n=20), `run_batch4.py` (latchN, multiseed selective), `experiments/mitigations.py`.
Raw: none (aggregates only).
Processed: `exp8_selective.csv`, `exp12_latchN.csv`, `exp12_multiseed.csv` (selective probe).
Stats: `exp12_ttests.csv` (oracle>fixed.99 row).
Figures: `exp8c_selective.png`, `exp12d_latchN.png`.
Claims: C016.

## 10. Random / staged gating

Purpose: lower-bound controls (random gate), novelty hard/soft gates.
Covered under family 3 files (`exp12_randomgate.csv`, `exp11_novelty*.csv`, `exp12_gatem*.csv`); no separate scripts. Claims: C016 (random≈fixed), C019 (hard-gate null).

## 11. Multi-head / state-bank experiments

Purpose: equal-budget head-count comparison.
Scripts: `run_batch3.py`, `experiments/variants.py` (layout: heads × head_dim in file).
Raw: `exp11_multihead_trials.csv`.
Processed: `exp11_multihead.csv`.
Stats: `exp12_ttests.csv` (multihead row).
Figures: `exp11d_multihead.png`.
Claims: C019.

## 12. Robustness / statistical validation

Purpose: cross-seed repeats, CI table, Welch tests, bundle metadata.
Files: `exp10_seed_recall.csv`, `exp10_seed_length.csv`, `exp12_multiseed.csv`, all `*_trials.csv`, `exp12_ttests.csv`, `exp12_CIs.csv`, `summary.json`, `config.json`, `pytest_report.txt`.
Claims: C023 + every reported t/p/n/CI in §4.2 Robustness.

## 13. Direct architecture experiments (Mamba / RWKV / RetNet / trained LMs)

NOT PRESENT. No checkpoints, no torch, no production-model runs exist in the repository (Phase 5). All architecture-named code paths are synthetic NumPy controls with methodology disclaimers (`models/linear_attention.py:11`, `experiments/gated_recall.py:1`, `experiments/variants.py:3,52`, `experiments/mitigations.py:12`, `experiments/finalbatch.py:84`). Corresponding manuscript content is category C025 (architectural context only).

## 14. Dissipation / coarse-vs-fine (cross-cutting probe)

Purpose: repeated-signal survival vs singleton loss at every scale.
Scripts: `run_followups.py`, `run_batch4.py` (multiseed dissipation).
Raw: none (aggregates only).
Processed: `exp7_coarsefine.csv`, `exp12_multiseed.csv` (dissipation probe).
Stats: none.
Figures: `exp7a_coarsefine.png`.
Claims: C021.
