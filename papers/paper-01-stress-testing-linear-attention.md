# Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes

**Paper 1 — Independent Research in Sequence Model Architectures**

| Slot | Value |
|---|---|
| Author | Aashirwad Sharma |
| Affiliation | Independent Research — Sequence Model Architectures |
| Contact | [SLOT — author email for correspondence] |
| Date | [SLOT — publication / last-revised date] |
| Venue / version | [SLOT — e.g. arXiv preprint, workshop submission, or personal preprint v1] |
| Working draft | `Paper_1_Draft_v3.md` (repo root; technical content source for this slot) |
| Evidence bundle | `public_evidence/` (61 CSVs + `config.json` + `summary.json` + 48 figures) |
| Code laboratory | `experiments/`, `models/`, `utils/`, `app.py` (Streamlit lab) |
| License | [SLOT — e.g. CC-BY 4.0 for the paper; MIT/Apache-2.0 for code] |

> **How to use this slot.** This file is the canonical Paper 1 record: title, author, abstract,
> and full section structure, with every empirical number traceable to the evidence bundle
> (claim IDs C001–C025 in `CLAIM_EVIDENCE_MAP.md`). Anything marked **[SLOT]** is a personal
> decision for the author to fill before publishing. Nothing empirical here is invented:
> all measurements are CPU-only NumPy observations at base seed 42, and scope limits are
> stated explicitly (§2.4, §5.3).

---

## Abstract

Fixed-size recurrent state trades explicit sequence-growing key–value storage for bounded
memory: the entire key–value history is compressed into S ∈ ℝ^{k×d_v} (square d×d in our
laboratory) instead of an explicitly stored per-token cache. This paper studies that
trade-off in the synthetic additive recurrence S_t = S_{t-1} + φ(k_t)v_tᵀ and its gated
synthetic controls, using controlled CPU-only NumPy experiments (base seed 42, d = 16 default).

We report three measurable stress regimes in this synthetic system: (1) **associative
interference** — retrieval accuracy falls with binding density, with a 50%-crossing at
m\* = 8 across tested dimensions under standard elu+1 features and a dimension-scaled
crossing (≈2d) under centered Gaussian-key conditions; (2) **temporal attenuation** —
recall degrades over long horizons, reaching ≈0 from N ≈ 16k undecayed, with gated
50%-crossings near the e-folding scale; (3) **numerical precision loss** — shrinking
updates go numerically ineffective first in fp16, then fp32 (≈11 and ≈24 steps), with
bf16 trailing fp16 on this probe. Controlled mitigation probes (oracle latch, hybrid
windows, chunk resets, multi-head banks, fp32 accumulation) move different breakpoints
differently.

These findings apply directly to the studied synthetic recurrence and motivate — but do
not experimentally establish — claims about production linear-attention architectures
such as Mamba, RWKV, or RetNet, which were not measured here. The contribution is
diagnostic: failure regimes, a breakpoint-oriented framework, and measurements separating
associative interference, temporal attenuation/decay, and numerical precision loss —
validated only within the synthetic conditions reported.

---

## 1. Introduction

### 1.1 From softmax to linear recurrence

Softmax attention computes, for every query, a full comparison against every key,
producing an explicit per-token reference store: the KV cache grows as O(N) in memory
and attention computation as O(N²) in time [5]. Architectures avoiding the N×N score
matrix use recurrent or kernelized formulations with bounded state:

- **Linear Transformers** [1] replace softmax(QKᵀ) with a kernel feature map φ(·).
- **RetNet** [2] reframes attention as retention with explicit exponential decay.
- **RWKV** [3] uses linear time-mixing recurrent updates.
- **Mamba / SSMs** [4] use structured, input-dependent linear recurrences with selective gating.

This paper studies one specific additive fixed-state synthetic recurrence (§3) as a
mechanism-isolating model. Production architectures are context and motivation, not
experimental instances — nothing here measures them (§2.4).

### 1.2 The core conflict

A state S ∈ ℝ^{k×d_v} holds a fixed number of degrees of freedom while every token
writes into the same budget — the efficiency gain is a bet that most sequence
information is not needed at fine grain, long range, or high density. When that bet is
wrong, the recurrence fails within identifiable regimes rather than degrading uniformly.

### 1.3 Contributions

- Capacity-constraint analysis of the synthetic update rule: why fixed-size S implies
  density-growing interference (§3).
- Three constructed stress regimes — recall density, length scaling, numerical
  saturation — each isolating a distinct mechanism (§3–4).
- Controlled measurements with trial counts and seeds, reported as observations (§4).
- A failure-mode taxonomy (interference / attenuation / precision loss) with probed
  mitigations, giving a diagnostic vocabulary (§4–5).

---

## 2. Related Work and Scope of Evidence

### 2.1 Positioning

The superposition-memory framing is classical Hopfield theory [6]: additive outer-product
storage with characterized capacity limits of the bilinear form (higher-order memories
[11, 12] are explicitly out of scope). Ramsauer et al. [7] connect modern Hopfield
updates to Transformer attention — the recurrence studied here inherits Hopfield-style
interference of the bilinear form. Long-context benchmarks (Lost in the Middle [8],
RULER [9], Needle-in-a-Haystack [10]) motivate breakpoint framing; this paper's
marker-recall tasks are their mechanism-isolating complement — synthetic, seeded, run
against the update rule itself.

### 2.2 Scope of evidence [SLOT — keep; edit only with new experiments]

Every claim belongs to exactly one category (Tables 1–2 tag each cell):

- **[D] Derived theory** — mathematics of S_t = S_{t-1} + φ(k_t)v_tᵀ (and scalar-γ gated form).
- **[A] Architectural description** — RetNet/RWKV/Mamba per cited specs [2–4]; unmeasured here.
- **[M] Measured in our synthetic lab** — CPU-only NumPy, seed 42 base, with trial counts.
- **[O] Open / future work** — explicitly unmeasured.

Scalar-γ / per-channel-γ decay, novelty gates, oracle latches, hybrid windows, chunked
resets, multi-head banks, and mixed-precision accumulation are synthetic controls
*inspired by* [2–4], not implementations of those architectures.

---

## 3. Theory and Stress-Test Scenarios

### 3.1 Softmax reference

A = Softmax(QKᵀ / √d_k)V. Memory O(N), compute O(N²), no compression. Lab baseline:
exact-marker accuracy 1.0 at every tested N (128–4096); float64 K+V storage 32 KB → 1 MB
at d = 16 (`public_evidence/exp4_softmax_baseline.csv`).

### 3.2 Linear recurrence under test

A_linear = φ(Q)(φ(K)ᵀV), with φ(K)ᵀV ∈ ℝ^{k×d_v} fixed-size in N. Recurrent form
(Katharopoulos et al. [1] with normalizer):

S_t = S_{t-1} + φ(k_t)v_tᵀ, z_t = z_{t-1} + φ(k_t), o_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t

Gated control: S_t = γS_{t-1} + φ(k_t)v_tᵀ (0 < γ < 1), scalar-γ and per-channel-γ as
synthetic controls. S_t accumulates every token's contribution in one superposition —
the Hopfield-structured memory whose bilinear capacity limits it inherits.

### 3.3 Scenario 1 — Associative recall density

State after m bindings: S_m = Σᵢ φ(k_i)v_iᵀ; retrieval needs φ(q)ᵀφ(k_i) ≈ δ_ij.
As m grows against ≤ k usable feature directions, overlaps accumulate as cross-term
interference (**state collisions**). Rank bounds directions (≤ k); retrieval quality is
set by overlap statistics + map geometry + key/value structure + precision + gating.

**Measured refinement.** Under elu+1 (feature mean ≈ 1.16), even orthogonal keys collide
(m = 4: 0.30 vs 0.67 random keys). The plateau-then-knee appears only for centered
features φ_c(x) = elu(x) + 1 − 1.1605: 50%-crossing scales m\* ≈ 2d (m\* ≈ 16/64/128
for d = 8/32/64, Gaussian keys, n = 12–15) — conditional, not universal. Under standard
maps, m\* = 8 for all tested d ∈ {8, 16, 32, 64, 256}; dimension lifts only the
low-density ceiling. (Source: `exp1_dimensionality.csv`, `exp10_bigd.csv`,
`exp6_featuremaps.csv`, `exp12_centered_d.csv`.)

### 3.4 Scenario 2 — Sequence-length scaling

Three distinct effects, not conflated: (i) **task-level forgetting** (measured);
(ii) **numerical zeroing** (derived: token 0 weight γ^N < ε exactly when
N > ln(ε)/ln(γ)); (iii) **e-folding horizon** N = −1/ln γ (≈1/(1−γ) for γ ≈ 1).

Measured 50%-recall crossings (N ≈ 1024 for γ = 0.999; N ≈ 3072 for γ = 0.9999 by
interpolation, horizon sweep n = 6) sit near e-folding scale, far below the fp32-zero
bound — forgetting precedes zeroing. Distribution-shift analogue: 0–4σ second-half
shift degrades pre-shift recall 0.75 → 0.25 (n = 8); frozen calibration halves accuracy
vs matched constant (0.375 vs 0.875, n = 8). Streaming: recall ≈ 0 for all N ≥ 16k to
N = 1M undecayed — the 64k stress region is an outer bound, not onset.
(Sources: `exp2_length_sweep.csv`, `exp9_longlength.csv`, `exp11_horizon.csv`,
`exp11_shift.csv`, `exp12_frozen.csv`.)

### 3.5 Scenario 3 — Numerical saturation

For zero-mean streams, ‖S_t‖ ~ √t (measured log-log slope 0.448, N = 128 → 1M);
shrinking updates go ineffective after ≈11 steps (fp16), ≈24 (fp32), effectively never
(fp64): 49/36/7 of 60. Real bf16 (ml_dtypes) trails fp16 (52 ineffective; rel. err
2.9e−03 vs 5.2e−04). Decayed steady states saturate ≈10–100× below naive theory
(0.324/1.41/2.52 vs 2.83/28.3/282.8 for γ = 0.99/0.999/0.9999). Under decay, fp16
magnitude drift is 1–3e−05 vs fp64 while fp32 ≈ 0 — damage lives at magnitude level
before recall level. (Sources: `exp3_shrinking.csv`, `exp3_decay.csv`,
`exp12_bf16.csv`, `exp5_precision_decay.csv`.)

---

## 4. Measurements and Failure-Mode Taxonomy

### 4.1 Structural comparison (per-cell tags: [M] measured, [A] analytic, [Arch] from specs, [O] open)

| Property | Softmax | Linear (Katharopoulos) | RetNet | RWKV | Mamba / SSM |
|---|---|---|---|---|---|
| State [Arch] | Full KV cache O(N) | Fixed S ∈ ℝ^{k×d_v} | S + decay γ | Time-mixed linear | Input-selective |
| Recall | Direct lookup; 1.0, N = 128–4096 [M, n = 10] | Steep fall from m = 4; m\* = 8 [M, n = 25] | No mitigation at m = 16 [M, n = 8] | Gate variants 0.0 → 0.625 [M-dir, n = 8] | Oracle latch 0.05 → 0.4 [M, p = 0.004, n = 20] |
| Long N | No horizon [A]; flat 1.0 [M] | ≈0 for N ≥ 16k; slope 0.448 [M] | Crossings ≈1024/≈3072 [M, n = 6] | Knee [O] | Latch 1.0 → 0.3, N = 512 → 8192 [M, n = 10] |
| Numerics | None [A] | Saturation; bf16 trails fp16 [M] | Cancellation [A]; fp16 drift 1–3e−05 [M] | Cancellation [A] | Selective reset [Arch] |
| Compute | O(N²); ≈10× slower at N = 8k here [M-impl, n = 2–3] | O(N); streams to 1M [M] | O(N), chunk-parallel [Arch] | O(N) [Arch] | O(N) [Arch] |

### 4.2 Key measurements (seed 42; d = 16, elu+1 unless noted; conventions: m\*/N\* = first grid point at mean recall ≤ 0.5)

- **Density (n = 25):** d = 16 accuracy 0.96 → 0.008 (m = 2 → 128; 95% CI ±0.05 at m = 2);
  relu 0.73 / exp 0.56 / elu+1 0.32 at m = 8 (t = 7.75, p ≈ 9×10⁻¹⁵). Centered-orth
  d = 16: flat 1.0 to m = 16, 0.5 at m = 32 (complete separation, p < 0.001 noted
  degenerate). Survival at m = 8 uniform across positions (0.28–0.38, n = 30).
- **Length (n = 10):** N = 128/512/1024/4096 → 1.0/1.0/0.8/0.2, position-flat.
  Storage: linear constant 2,176 B vs softmax 32 KB → 1 MB (persistent float64 only).
  Timing (NumPy CPU-only, i3/3.7 GB — implementation-specific): linear ≈10× faster at
  N = 8k (0.39 s vs 3.89 s, n = 3); naive fp32 scoring OOMs at N = 32768 (4.3 GB matrix).
- **Mitigations:** hybrid windows restore recent recall only (directional, p ≈ 0.32);
  oracle latch restores decayed early recall to 0.4 (n = 20); random gating ≈ fixed decay;
  chunk resets cap norm 14.8 → 5.2 but wipe early markers; multi-head at equal budget
  null (p ≈ 0.73); per-channel vs scalar null (p ≈ 0.63).
- **Robustness:** seeds 42–45 overlays tight; 193-row CI table + 9 Welch t-tests in bundle.

### 4.3 Taxonomy

**(a) Associative interference / state collisions** — density-driven, overlap-set onset.
**(b) Temporal attenuation / decay** — γ^{N−t} horizons; compare against e-folding/zeroing scales.
**(c) Numerical precision loss** — implementation-level saturation/cancellation; vary precision at fixed task.
**Interactions:** (a)+(b) degrade alone and converge with (c) on dissipation (×20-repeated
signals survive where singletons are lost at every N, n = 8–10); decay does not mitigate
collisions; windows bypass (a) only inside coverage.

---

## 5. Mitigations, Limitations, Future Work

| Mitigation | Targets | Trade-off | Lab probe [M] |
|---|---|---|---|
| Input-dependent decay | Attenuation | +params/compute; still bounded | Latch 0.05 → 0.4 (p = 0.004); learned gating untested |
| Local-softmax hybrid windows | Interference, dissipation | O(w²) window cost | Recent restored (directional, p ≈ 0.32) |
| Chunked resets / hierarchy | Attenuation, precision | Boundary loss | Norm 14.8 → 5.2; early wiped; mean-summary inert |
| Multi-head banks | Interference | Linear memory/compute | No effect at equal budget (n = 6) |
| Mixed precision (fp32 state) | Precision loss | Memory/bandwidth | 8× error cut; bf16 reported separately |

**Limitations (binding):** no production-architecture runs (no torch/GPU/RAM); no learned
gating (oracle/random bounds instead); no trained-positional effects; binary metrics
blind to recall-level precision drift at scale; small-n probes (n = 6–40) with normal
approximations — effect sizes and CIs lead, p-values secondary. Generalization beyond the
synthetic system requires direct evaluation. **Left open:** real weights, learned gates,
trained-position knees, longer-horizon precision effects, non-mean summarization, window
scaling laws, stacked-mitigation interactions.

---

## 6. Conclusion

The studied additive recurrence exhibits mechanistically distinguishable failure regimes
under density, horizon, and precision stress, measurable via breakpoint diagnostics
(m\*, crossing length, first-ineffective-update step). Signatures: m\* = 8 under standard
maps (≈2d centered, Gaussian keys); ≈0 recall from N ≈ 16k with √t norm growth;
precision-ordered decay with bf16 trailing fp16. Capacity, retention, and precision are
separate design constraints for budgeting exactness vs acceptable lossy compression.

---

## References

[1] Katharopoulos et al. (2020). Transformers are RNNs. ICML, PMLR 119, 5156–5165.
[2] Sun et al. (2023). Retentive Network. arXiv:2307.08621.
[3] Peng et al. (2023). RWKV. EMNLP Findings 2023, 14048–14077.
[4] Gu & Dao (2023). Mamba. arXiv:2312.00752.
[5] Vaswani et al. (2017). Attention Is All You Need. NeurIPS 30.
[6] Hopfield (1982). Neural Networks and Physical Systems. PNAS 79(8), 2554–2558.
[7] Ramsauer et al. (2020). Hopfield Networks is All You Need. arXiv:2008.02217 (ICLR 2021).
[8] Liu et al. (2023). Lost in the Middle. arXiv:2307.03172.
[9] Hsieh et al. (2024). RULER. arXiv:2404.06654.
[10] Kamradt (2023). Needle in a Haystack. Evaluation suite (open source).
[11] Krotov & Hopfield (2016). Dense Associative Memory. NeurIPS 29.
[12] Demircigil et al. (2017). Associative Memory with Huge Storage Capacity. J. Stat. Phys. 168.

---

## Appendix A. Evidence slot map [SLOT — re-verify paths before each release]

Full claim map: `CLAIM_EVIDENCE_MAP.md` (C001–C025). Per-file inventory: `EVIDENCE_INDEX.md`.
Known caveats: `EVIDENCE_GAPS.md` (e.g. `summary.json` compress values vs
`exp11_compress*.csv` — manuscript follows the CSVs; far-end streaming n = 2–3).

| Paper section | Evidence files |
|---|---|
| §3.3 density | `exp1_dimensionality.csv`, `exp10_bigd.csv`, `exp10_seed_recall.csv`, `exp6_featuremaps.csv`, `exp7_orthkeys.csv`, `exp12_centered_d.csv`, `exp11_centered_orth.csv`, `exp12_survival.csv`, `exp01_trials.csv` |
| §3.4 length | `exp2_length_sweep.csv`, `exp9_longlength.csv`, `exp10_seed_length.csv`, `exp11_horizon.csv`, `exp11_shift.csv`, `exp12_frozen.csv`, `exp02_trials.csv` |
| §3.5 numerics | `exp3_shrinking.csv`, `exp3_decay.csv`, `exp12_bf16.csv`, `exp5_precision_decay.csv`, `exp8_mixed.csv`, `exp03_trials.csv` |
| §4.1 comparison | `exp4_softmax_baseline.csv`, `exp11_timing.csv`, `exp12_oom.csv`, `exp04_trials.csv` |
| §4.2 mitigations | `exp8_hybrid.csv`, `exp8_chunked.csv`, `exp8_selective.csv`, `exp12_latchN.csv`, `exp12_randomgate.csv`, `exp12_stacked.csv`, `exp11_perchannel.csv`, `exp11_multihead.csv`, `exp11_novelty.csv` |
| Statistics | `exp12_ttests.csv`, `exp12_CIs.csv`, `*_trials.csv` companions |
| Figures | `public_evidence/figures/` (measured PNGs; schematics via `gen_schematics.py`, no measured data) |

## Appendix B. Reproducibility slot

Environment: Python 3.14.7, Windows 11, CPU-only, base seed 42 (see `reproducibility/ENVIRONMENT.md`).
`pip install -r requirements.txt` plus `fpdf2`, `ml_dtypes`, `pytest` for PDF build, bf16 probe, tests.
Staged runners (each documents usage in its header; all write into `public_evidence/` — back it up first):

```powershell
.\.venv\Scripts\python generate_public_bundle.py        # Exps 1-3, fixed settings
.\.venv\Scripts\python run_exp4.py                      # Exp 4 softmax baseline
.\.venv\Scripts\python run_followups.py [stage1|stage2|stage3|finalize|all]
.\.venv\Scripts\python run_batch3.py [stageA|stageB|finalize|all]
.\.venv\Scripts\python run_batch4.py [stageC|stageD|finalize|all]
.\.venv\Scripts\python -m pytest tests/                 # 17 tests
.\.venv\Scripts\python build_paper_pdf.py Paper_1_Draft_v3.md Paper_1_Draft_v3.pdf
```

## Appendix C. Author slots [SLOT — fill before publishing]

- **Author bio:** [SLOT — 2–3 lines: independent researcher, focus areas, background]
- **Contact:** [SLOT — correspondence email]
- **Acknowledgments:** [SLOT — anyone to thank, or "None"]
- **Competing interests:** [SLOT — e.g. "The author declares none"]
- **Paper license:** [SLOT — recommended CC-BY 4.0]
- **Citation slot:** [SLOT — venue identifier / DOI once assigned]

*Drafted from theory by the author; measurements from the accompanying CPU-only laboratory
(base seed 42). Every strong statement traceable to a derivation, a measurement, or a cited
architectural fact; every reported number checked against the evidence bundle.*
