# Paper Revision Brief — Linear Attention Stress-Test Lab → Paper
Feed this file + the draft PDF to Claude. Everything below is observed measurement (seed 42, CPU-only NumPy lab, synthetic Gaussian keys/values, tiny exact-softmax reference). Never upgrade "observed/consistent" to "proves".

## 0. Setup context (put in Methods)
- d=16 default; elu+1 feature map unless noted; distractor values ~ N(0, 0.01) — absolute accuracies depend on this scale (rankings don't; see R14).
- Trials: Exp1 25, Exp2/4 10, probes 6–20 (per-file `trials` column). Per-trial records: exp01–04_trials.csv (old experiments) + exp11_*_trials.csv (compress, perchannel, novelty, multihead, shift, noisy) + exp12 gatedm/gatem/survival/cosine trials. 193-row CI table + 9 Welch t-tests exist.
- Language rule: THEORY / PREDICTION / OBSERVATION / INTERPRETATION kept separate. No "proof", only "candidate region".

## 1. Results dossier (quote these numbers exactly)
- R1 recall density (d=16, elu+1): m=2→128 gives 0.96→0.008. Steep fall from m=4, NO plateau, NO knee. Supports collisions; contradicts Fig 3 shape.
- R2 m* (first grid point <50%): m*=8 for EVERY d in {8,16,32,64}; d=256 overlays same collapse. Onset is d-independent. d only lifts low-density ceiling (m=4: 0.54 at d=8 → 0.86 at d=128).
- R3 feature maps @m8 d16: relu 0.73, exp 0.56, elu+1 0.32 (t=7.7, p≈0). φ moves the curve where d cannot.
- R4 orthogonal keys under elu+1: m=4: 0.30 vs 0.67 random — NO plateau. elu+1's mean (~1.16) destroys key orthogonality; hash condition must hold at FEATURE level.
- R5 centered_elu (elu+1 minus 1.1605): random keys plateau to m=16 (1.0,0.98,0.96,0.85 then 0.3@32); m* scales with d: d=8→16, d=32→64, d=64→128 (≈2d). Centered-orth d=16: flat 1.0 to m=16, 0.5@32 (rank knee). Other dims @m=16 non-monotonic (d=8: 0.42, d=32: 0.49, d=64: 0.16) — centering constant must match key distribution (Gaussian-tuned constant mis-centers one-hot keys); quote d=16 only.
- R6 length: N=128/512/1024/4096 → 1.0/1.0/0.8/0.2; flat across 10/50/90% positions (no recency → density-driven).
- R7 long range: 16k:0.0, 65k:0.0, 262k:0.1 (n=20, noise), 1M:0.0. Floor ≈0 for N≥16k. State-norm log-log slope 0.448 (≈√t CONFIRMED, N=128→1M).
- R8 drift: relative error saturates ~1.0 from N=128 (washed output ≈0 BY CONSTRUCTION) — no exponent identifiable; finding is total washout. Cosine(out,marker) works instead: 0.95→0.18 across N=128→8192.
- R9 gated early-marker N=4096 (n=20): γ=1.0:0.4, 0.9:0.0, 0.99:0.05, 0.999:0.1, 0.9999:0.35. Directional only.
- R10 horizon 50%-crossings: γ=0.99 never ≥0.5; γ=0.999 →N≈1024; γ=0.9999 →N≈3072. Compare 1/(1−γ): 100/1000/10000; fp32-zero bound 16.64/(1−γ): 1664/16636/166355. Observed sits near 1/(1−γ)×1–3.
- R11 precision×decay (recent marker, N=2048): accuracy 0.375 (γ.99) / 0.875 (γ.999) IDENTICAL across fp64/32/16; continuous drift fp16 1–3e-05, fp32≈0. Recall robust; damage is magnitude-level.
- R12 shrinking stream (60 steps): ineffective updates fp16:49, fp32:36, fp64:7. bf16 (ml_dtypes, real): 52 ineffective, rel-err 2.9e-03 — WORSE than fp16. Never lump bf16/fp16.
- R12b decayed steady-state magnitudes (2000 steps, update 0.01): γ=0.99: 0.324 (naive theory 2.83); γ=0.999: 1.41 (theory 28.3); γ=0.9999: 2.52 (theory 282.8). Measured saturation sits ~10–100× BELOW naive linear-theory scale — decay bounds accumulation hard. Use for §3.5 + Table 1 horizon cell.
- R13 mixed precision: fp32 accumulation rel-err 6.7e-05 vs fp16 5.2e-04 (~8× cut).
- R14 distractor sensitivity: scale 0.001→1.0 everywhere; 0.01→0.667/0.167; 0.1→0.167/0.0. Rankings stable; absolutes are scale-conditional. DISCLOSE scale beside every number.
- R15 dissipation: coarse (×20) 1.0 vs singleton 0.0 at EVERY N incl. 128 (where lone marker =1.0). Strong §4.2(c) backing.
- R16 m×N grid: accuracy follows m (m=4: 0.656→0.25 with ANY filler; 256 vs 2048 filler identical); washed states collapse to constant predictor (accuracy exactly 1/m, std 0).
- R17 survival (m=8, n=30): quintiles 0.28/0.37/0.28/0.30/0.38 — UNIFORM, no primacy/recency. Position-blind catastrophic interference.
- R18 hybrid: N=1024 w=128 recent→1.0 (early flat); N=4096 needs w=512 for recent→1.0 (coverage mechanics). t=1.0, p≈0.32 — NOT significant at n=8; present as directional.
- R19 selective latch (n=20): fixed-1.0:0.4, fixed-0.99:0.05, oracle:0.4 (t=2.8, p≈0.004 ✓). Latch×N: oracle 1.0→0.9→0.6→0.3→0.3 vs fixed-1.0 1.0→0.9→0.6→0.4→0.2 vs fixed-0.99 floored ≤0.3 (512→8192). A pre-fix one-step-gate probe scored 0.125 = fixed (superseded run, do NOT quote as data): retention must latch, not fire once — keep as mechanism reasoning only. Random gate = fixed (0.0@8192): gating must be INFORMED.
- R20 chunked: zero-reset max norm 14.81→8.26 (C=1024)→5.17 (C=256) but early wiped (0.375→0.125); mean-summary carry-forward is INERT (zero==summary exactly: early 0.12, recent 0.75, norms identical) — needs real content selection.
- R21 gated m-sweep @m=16: γ=1.0:0.102, 0.99:0.094, 0.9:0.109 — decay does NOT mitigate dense collisions (Table 1's "partial mitigation" unsupported; revise cell).
- R22 gate m-sweep @m=8: off 0.375 == hard-gate 0.375. Selectivity doesn't move m* here.
- R23 per-channel (N=1024): 0.0/0.5/0.625 scalar.99/scalar.999/perchannel — gradient as predicted BUT p≈0.63, n=8: directional only.
- R24 multi-head equal-budget @m=16: 0.094/0.083/0.094 — identical (t=0.35, p≈0.73, correctly null); not even a delay. Revise Table 2 row to "no effect at equal budget in this setup".
- R25 novelty: hard-threshold == off (0.417; t=0, p≈1.0, correctly null); soft gate HURTS (0.333) — naive cosine gates attenuate everything at d=16 (chance overlap ~0.5); gates need a margin.
- R26 compressibility x40: A=8:0.75 vs random 0.625, p≈0.82 — directional only, §5.1(i) stays weakly supported.
- R27 shift probe: 0.75→0.625→0.375→0.25 over shift 0→4. Monotonic kernel-shift degradation ✓.
- R28 noisy queries: linear 0.9 / softmax 1.0 flat across σ=0→1. Ranking never flips.
- R29 softmax baseline: linear 1.0/1.0/0.9/0.367 vs softmax flat 1.0; 2,176 B vs 1 MB @4096. Timing: softmax faster ≤512, linear 10× faster @8k full scoring (0.39s vs 3.89s). Never present lab timing as "linear slower" — crossover is the story.
- R30 frozen-calibration toy: distribution-matched centering 0.875 vs stale-frozen 0.375. Miniature train/test-mismatch demo.
- R31 OOM boundary: full N×N fp32 scoring OK @16384 (1 GB); N=32768 impossible by construction (4.3 GB > 3.7 GB RAM).
- R32 robustness: seeds 42–45 recall overlays tight; length seeds 43/44 reproduce shape; exp12_multiseed.csv (44 rows: gated/selective/perchannel/centered/dissipation × seeds 43/44). All bundle CSVs carry seed/trials/precision/versions.

## 2. Mandatory paper fixes (anchor = section + exact current text)
- F1 Abstract claim "systematically maps these breakpoints" + "We close this gap" + conclusion "We show three independent breakpoints" vs §4 containing zero measurements ("not yet measured" cells). FIX: rename §4 to predicted/taxonomy scope OR insert R1–R32 as measured subsection; downgrade verbs to derive/measure/initial evidence.
- F2 Fig 3 "hold near-exact until m*… then fall sharply — a knee, not a gradual slope" contradicted by R1. FIX: revise to steep early fall; knee exists ONLY iff features centered (R5: m*≈2d) — make Fig 3 conditional, add measured curve.
- F3 §4.2(a) "fixed critical m* near the effective rank" + Table 2 "raise the collision threshold m*": contradicted by R2/R22/R24. FIX: onset set by kernel overlap, d-independent; rank governs low-density ceiling; multi-head no effect at equal budget.
- F4 §3.2 equations omit normalizer z_t (lab implements it; Katharopoulos has it). ADD z update + normalized output.
- F5 §3.5 "‖S_t‖ ~ √t" stated unconditionally — requires zero-mean; elu+1 drifts linearly. QUALIFY + quote measured 0.448 (R7).
- F6 §3.4 "N ≫ 1/(1−γ) ⟹ γ^N ≈ 0": wrong ~5–15× (γ=0.99 → γ^100=0.37). REPLACE with N > ln(1/ε)/(1−γ) (≈16.6/(1−γ) fp32) + R10 crossings.
- F7 Notation garbling (4 places): "S ^{d×d} ∈ ℝ" (Abstract), "S ∈ ℝ^{d×d} (or ℝ^{d}…)" + "on ∈ ℝ ℝ" (§2.2), "V ^{N×d} … ∈ ℝ" (§3.1), "φ(K)ᵀV ^{d×d} ∈ ℝ" (§3.2). All must read X ∈ ℝ^{dims}.
- F8 §3.4 dangling "For" before display equation; §4.1 garbled "states, explicitly, which quantities"; title "Scaling Beyond N > 64k" redundant; Table 1 "(m→d)" → "m ≳ d".
- F9 "breakpoints cluster near N > 64k" UNCITED and unmeasured (lab floor is ~16k in d=16 synthetic setup). Rephrase as outer stress bound + cite, or scope to measured regime.
- F10 ADD citations: Kamradt 2023 NIAH (blog + gkamradt/LLMTest_NeedleInAHaystack) positioning Exp-2 as mechanism-isolating complement; Krotov & Hopfield 2016 (NeurIPS) + Demircigil et al. 2017 (J. Stat. Phys. 168, 288–299, doi:10.1007/s10955-017-1806-y) scoping Hopfield claim to bilinear outer-product form.
- F11 Table 1 "RetNet: decay adds partial mitigation" contradicted by R21 → revise to "no mitigation observed; hurts-or-neutral on dense recall". "Governed by kernel conditioning" (Linear horizon cell) → replace with R10 numbers. "≈ 1/(1−γ)" → corrected formula (F6).
- F12 §3.5 lumps "fp16/bf16": R12 shows bf16 WORSE (52 vs 49 ineffective). Separate them.
- F13 §5.1(ii) "failures present as breakpoints rather than smooth curves": lab curves are mostly gradual — soften to "can present as narrow rapid transitions (R1 falls 7× over 4× density)" + R17 symmetry finding.

## 3. Table 2: add measured "lab probe" values
Dynamic decay → R19 latch 0.05→0.4 (p≈0.004); hybrid → R18 w=128/512 restores recent only (p≈0.32, directional); chunked → R20 norm capped, early wiped, summary inert; multi-head → R24 no effect; mixed-precision → R13 8× error cut. Stacked (N=4096): early — linear 0.375/hybrid 0.375/decayed 0.125/stacked 0.125; recent — identical pattern (hybrid window can't rescue what decay erased outside it). Mitigations compose only where coverage overlaps.

## 4. Methods paragraph Claude must include
Seed 42 (robustness 42–45); CPU-only NumPy; distractor_scale=0.01 with R14 sensitivity disclosure; n per result; ns-results reported as directional (R23/R26/R18); noise cells flagged (262k 0.1 n=20; gated γ cells n≤20); timing crossover R29 with implementation-artifact note.

## 5. Label as future work (do NOT invent)
Real Mamba/RWKV/RetNet weights (no torch/GPU/RAM); learned gating (no training loop — oracle/random bound it instead); trained-position knee-at-boundary (synthetic setup has no training boundary; R30 toy is the closest probe); recall-level precision effects at longer horizons (binary metric too coarse); bf16→ use measured R12 instead.

## 6. Deliverable from Claude
Full replacement text for Abstract, §3.3–§3.5, §4 (with measured-results subsection + updated Tables 1–2), §6, all affected captions; a change log mapped to F1–F13 and R1–R32 plus R12b (decay steady-state); and a "TESTS LEFT" section containing ONLY item 5's list. Measured 300-DPI figures for the new curves live in public_evidence/figures/ (exp1_overlay_by_dim, exp6_featuremaps, exp11g_centered, exp7c_orthkeys, exp9_longlength_full, exp4_softmax_comparison, exp12* batch-4 set) — reference them in captions; upload any of them alongside if needed.
