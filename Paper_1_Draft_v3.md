# Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes

**Aashirwad Sharma**
*Independent Research — Sequence Model Architectures*

*A diagnostic study of fixed-capacity recurrent state: breakpoint diagnostics for a synthetic linear-attention recurrence, motivated by Linear Transformers, RetNet, RWKV, and Mamba / SSMs. Theory for the synthetic recurrence plus controlled measurements from a CPU-only NumPy laboratory (seed 42). All empirical statements below are observed measurements of the synthetic system, not proof of the theoretical claims and not measurements of production architectures.*

---

## 1. Abstract

Fixed-size recurrent state trades explicit sequence-growing KV storage for bounded memory: the entire key–value history is compressed into S ∈ ℝ^{k×d_v} (square d×d in our laboratory) instead of an explicitly stored per-token cache. This paper studies that trade-off in the synthetic additive recurrence S_t = S_{t-1} + φ(k_t)v_tᵀ and its gated synthetic controls, using controlled CPU-only NumPy experiments (base seed 42, d = 16 default).

We report three measurable stress regimes in this synthetic system: (1) associative interference — retrieval accuracy falls with binding density, with a 50%-crossing at m* = 8 across tested dimensions under standard elu+1 features and a dimension-scaled crossing (≈2d) under centered Gaussian-key conditions; (2) temporal attenuation — recall degrades over long horizons, reaching ≈0 from N ≈ 16k undecayed, with gated 50%-crossings near the e-folding scale; (3) numerical precision loss — shrinking updates go numerically ineffective first in fp16, then fp32 (≈11 and ≈24 steps), with bf16 trailing fp16 on this probe. Density, length, and precision thresholds provide breakpoint diagnostics that distinguish these mechanisms, and controlled mitigation probes (oracle latch, hybrid windows, chunk resets, multi-head banks, fp32 accumulation) move different breakpoints differently.

These findings apply directly to the studied synthetic recurrence and motivate — but do not experimentally establish — claims about production linear-attention architectures such as Mamba, RWKV, or RetNet, which were not measured here.

The contribution is diagnostic: a controlled analysis of failure regimes in a fixed-size additive recurrent state, a breakpoint-oriented diagnostic framework, and measurements separating associative interference, temporal attenuation/decay, and numerical precision loss — validated only within the synthetic conditions reported here.

---

## 2. Introduction

### 2.1 From Softmax to Linear Recurrence

Softmax attention computes, for every query, a full comparison against every key in the sequence, producing an explicit per-token reference store: the KV cache grows as O(N) in memory and the attention computation as O(N²) in time [5]. This is the accuracy reference for sequence modeling, but it does not scale to long contexts.

The response has included architectures that avoid materializing the explicit N×N score matrix, using recurrent or kernelized formulations with a bounded state representation:

- **Linear Transformers** (Katharopoulos et al. [1]) replace softmax(QKᵀ) with a kernel feature map φ(·), reassociating the computation so it never materializes the N×N matrix.
- **RetNet** (Sun et al. [2]) reframes attention as a retention mechanism with an explicit exponential decay term, recovering a chunkwise-parallel recurrent form.
- **RWKV** (Peng et al. [3]) uses a linear, time-mixing recurrent update inspired by both RNNs and attention.
- **Mamba / State-Space Models** (Gu & Dao [4]) use a structured, input-dependent linear recurrence (S4-style) with selective gating over what enters and leaves the state.

This paper studies a specific additive fixed-state synthetic recurrence (§3.2) as a mechanism-isolating model. Some modern sequence architectures also use bounded recurrent or state-space representations, but they differ substantially in parameterization, gating, state dynamics, and training; Mamba, RWKV, and RetNet are therefore architectural context and motivation here, not experimental instances of the recurrence studied. Nothing in this paper measures these production architectures themselves; all measurements concern the synthetic recurrence of §3.2 and small controls inspired by the above mechanisms (§2.5).

### 2.2 The Core Architectural Conflict

This is the central tension this paper interrogates: *fixed-size state capacity vs. unbounded history length.*

A state S ∈ ℝ^{k×d_v} (vector states ℝ^{d} in scalar-gated variants; our lab uses the square case k = d_v = d) has a fixed number of degrees of freedom — on the order of k·d_v real numbers (d² in our square setup). Every token processed writes an update into this same fixed budget. As an intuitive capacity argument under this fixed-state representation, the state budget available per token shrinks toward zero as N grows without bound, unless the underlying task itself has a compressible, low-rank structure. Softmax attention has no such conflict: its "state" is the KV cache itself, growing with N.

The consequence is that the studied fixed-state recurrence's efficiency gain is not architecture-neutral — it is a targeted bet that most of the sequence's information is not needed at fine grain, at long range, at high density. When that bet is wrong, the model can fail within identifiable regimes (see §4.2) rather than degrading uniformly.

*Figure 1. Conceptual schematic; not measured data: as N grows, softmax storage grows slot-per-token while the linear state stays one fixed-size bank — the structural conflict analyzed in §3.2.*

### 2.3 Contributions

- An analysis of the recurrent-state capacity constraint directly from the synthetic linear-attention update rule, showing why a fixed-size S implies interference that grows with recall density (§3).
- Three constructed stress-test regimes — associative recall density, sequence-length scaling, and numerical / floating-point saturation of the recurrent state — each isolating a distinct failure mechanism rather than conflating them into a single aggregate metric (§3–4).
- Initial controlled measurements of each regime from a synthetic laboratory, reported as observations with trial counts and seeds (§4.2).
- A failure-mode taxonomy (associative interference, temporal attenuation, numerical precision loss) paired with concrete architectural mitigations, each with an initial probe measurement where feasible, providing a diagnostic vocabulary for characterizing failure regimes rather than post-hoc benchmark surprise (§4–5).

### 2.4 Related Work and Positioning

The superposition-memory framing used throughout this paper is not new to associative-memory theory: it is structurally identical to the classical Hopfield network [6], whose weight matrix stores patterns additively and suffers a well-characterized capacity limit. This claim is scoped to the bilinear outer-product form: dense associative memories with higher-order or exponential interaction functions achieve far larger capacities [11, 12], so the limitation derived here attaches to the additive φ(k)vᵀ update specifically, not to superposition memories in general. Ramsauer et al. [7] later showed that the modern, continuous-state Hopfield update is mathematically equivalent to the attention mechanism used in Transformers — which is why the additive outer-product recurrence studied here inherits Hopfield-style interference of the bilinear form, rather than exhibiting some new, unrelated failure pattern. This paper's contribution is to use that connection to motivate analysis of the corresponding synthetic recurrent update, compare the mechanism to contemporary architectures at the conceptual level, and turn it into concrete, falsifiable stress tests — without claiming experimental analysis of production implementations.

On the empirical side, long-context evaluation motivates the breakpoint framing adopted here without directly measuring the linear-attention-specific mechanisms this paper isolates. Liu et al. [8] ("Lost in the Middle") showed that even full-attention Transformers use long contexts non-uniformly, with accuracy degrading for information placed away from the context boundaries — a positional effect distinct from, but easily confounded with, the density- and length-driven breakpoints analyzed here. Hsieh et al. [9] (RULER) subsequently demonstrated that models' claimed context windows substantially overstate their effective context length once task difficulty is controlled for. Kamradt [10] ("Needle in a Haystack") pressure-tested production long-context models on single-fact retrieval across context lengths and depths. None of these studies isolates the recurrent-state mechanism directly: NIAH tests production models end to end, while this paper's marker-recall tasks are its mechanism-isolating complement — synthetic, seeded, and run against the update rule itself. The stress tests here are offered as a complementary, architecture-level diagnostic alongside these benchmark-level findings.

### 2.5 Scope of Evidence

Every claim in this paper belongs to exactly one of four categories, and Tables 1–2 tag each cell accordingly:

- **[D] Derived theory** — mathematics of the synthetic recurrence S_t = S_{t-1} + φ(k_t)v_tᵀ (and its scalar-γ gated form). Applies to that recurrence only.
- **[A] Architectural description** — how RetNet, RWKV, and Mamba/SSMs work according to their cited specifications [2–4]. No claim is made to have measured these systems.
- **[M] Measured in our synthetic laboratory** — CPU-only NumPy, seed 42 base, reported with trial counts; [M-dir] marks directional results that do not reach significance.
- **[O] Open / future work** — explicitly unmeasured here.

In particular: scalar-γ decay, per-channel-γ decay, novelty-gated writes, oracle salience latches, hybrid exact windows, chunked resets, multi-head banks, and mixed-precision accumulation are small synthetic controls *inspired by* the mechanisms of [2–4], not implementations or measurements of RetNet, RWKV, or Mamba themselves.

---

## 3. Theoretical Formulation & Stress-Test Architecture

### 3.1 Softmax Attention as the Reference Point

Standard Softmax attention over queries Q, keys K, values V ∈ ℝ^{N×d} [5]:

A = Softmax(QKᵀ / √d_k) V

Every output token i attends to an explicit score over all N keys. Memory is O(N) (the cache), compute is O(N²). There is no compression — token j's contribution to the output at step i is retrievable without interference from any other token, up to floating-point precision. Our lab baseline shows the flat profile: exact-marker accuracy 1.0 at every tested N (128–4096), with persistent float64 K+V storage growing 32 KB → 1 MB at d = 16 (2·N·d·8 bytes; transient score buffers, queries, and outputs excluded).

### 3.2 Generic Linear Attention

Linear attention replaces the softmax kernel with a feature map φ(·) applied to Q, K, exploiting associativity to avoid the N×N matrix:

A_linear = φ(Q)(φ(K)ᵀV)

Critically, φ(K)ᵀV ∈ ℝ^{k×d_v} is a fixed-size matrix regardless of N (d×d in our square lab setup). In recurrent form, following Katharopoulos et al. [1] including the normalizer, this is built incrementally:

S_t = S_{t-1} + φ(k_t) v_tᵀ,    z_t = z_{t-1} + φ(k_t),    o_t = φ(q_t)ᵀS_t / φ(q_t)ᵀz_t

In the literature, RetNet [2] and gated SSMs [4] add a decay/gate term to this structure,

S_t = γS_{t-1} + φ(k_t) v_tᵀ    (0 < γ < 1),

with z decayed identically. Our laboratory implements scalar-γ decay and per-channel-γ decay as small synthetic controls inspired by these mechanisms — not the architectures themselves (see §2.5).

S_t has a fixed dimensionality k×d_v but accumulates a contribution from every one of the t tokens seen so far: an **additive superposition memory** with the same outer-product update structure as a classical Hopfield weight matrix [6] — and it inherits the corresponding capacity limits of the bilinear form. This connection is specific to the recurrence studied here.

### 3.3 Breakpoint Scenario 1: High-Density Associative Recall

*Figure 2. Conceptual schematic; not measured data: direct lookup over per-token slots versus similarity-based retrieval over one superposed bank — the mechanism behind state collisions (§3.3).*

Consider a synthetic multi-query associative-recall task: the model must store m key–value bindings (k_i, v_i) within a context and later retrieve v_i given k_i. Under the additive update, the state after m bindings is S_m = Σᵢ φ(k_i)v_iᵀ, and retrieval computes φ(q)ᵀS_m = Σᵢ(φ(q)ᵀφ(k_i))v_iᵀ. Exact recall of v_j requires a near-orthogonal, near-delta kernel response, φ(q)ᵀφ(k_i) ≈ δ_ij — i.e., the feature map must behave like a near-perfect hash. As m grows relative to the number of usable feature directions (at most k), pairwise overlaps φ(k_i)ᵀφ(k_j) ≠ 0 accumulate as cross-term interference in every retrieval. This is a **state collision**: distinct bindings are not stored in separate slots but superimposed in the same k-dimensional feature subspace. Softmax attention has no such collision in the tested reference implementation because each stored (k_i, v_i) pair occupies a distinct cache entry rather than a shared superposition.

State rank alone does not determine associative-memory capacity: state dimensionality ≠ exact associative capacity. Rank bounds the number of linearly independent feature directions (≤ k), but retrieval quality is set by the pairwise kernel-overlap statistics together with feature-map geometry, key overlap, value structure, normalization, query distribution, signal/noise structure, numerical precision, gating/decay, and the retrieval criterion. Two systems with equal rank can have different interference onsets if their overlap distributions differ — exactly what our elu+1 vs centered-map comparison shows (§4.2).

**Refinement from measurement (§4.2).** The hash condition must hold at the *feature* level, and standard non-centered maps violate it by construction: under elu+1 (feature mean ≈ 1.16 for Gaussian keys), even exactly orthogonal keys collide (m = 4 accuracy 0.30 vs 0.67 for random keys), because the shared mean direction dominates every pairwise overlap. The predicted knee appears only once features are centered: with φ_c(x) = elu(x) + 1 − 1.1605, random-key recall holds a plateau to m = 16 and the 50%-crossing scales with dimension (m* ≈ 16/64/128 for d = 8/32/64). Under the centered feature construction and synthetic Gaussian-key conditions tested here, the measured breakpoint scaled approximately as m* ≈ 2d (centered runs n = 15) — an empirical conditional result, not a universal capacity law, and not generalizable to arbitrary feature maps, distributions, architectures, or trained models. Under standard maps, onset is set by kernel-overlap statistics and is independent of d across all tested dimensions (m* = 8 for d ∈ {8, 16, 32, 64, 256}). Dimension raises only the low-density ceiling. The centering constant here is calibrated under Gaussian keys; it mis-centers one-hot keys, so the rank-scaled knee is reported only where measured.

*Figure 3. Associative-recall breakpoint, conditional on the feature map. Under standard non-centered maps (elu+1), measured retrieval accuracy falls steeply from low density with no plateau (d = 16: 0.96 → 0.008 across m = 2 → 128). The plateau-then-knee shape appears only for centered features, with the knee near m* ≈ 2d under these centered-feature, Gaussian-key conditions. Curves after measured data (§4.2); the unconditional knee of the previous draft is withdrawn.*

### 3.4 Breakpoint Scenario 2: Sequence-Length Scaling

A second, independent failure axis is length extrapolation. Three distinct length-related effects should not be conflated — task-level forgetting (recall accuracy vs N, measured), numerical zeroing (contributions below a precision floor, derived), and the effective decay horizon (analytic scale):

- **Decay-term underflow.** Gated variants (S_t = γS_{t-1} + …, γ < 1) exponentially discount early tokens. Exactly: token 0 carries weight γ^N after N steps (no approximation — direct expansion of the recurrence). It falls below a precision floor ε exactly when γ^N < ε, i.e. N > ln(ε)/ln(γ). The effective decay (e-folding) horizon, defined by γ^N = 1/e, is exactly N = −1/ln γ, which for γ ≈ 1 is well approximated by 1/(1−γ) — so 1/(1−γ) is presented here as the e-folding approximation, not the zeroing condition (for fp32 zeroing, ε = 2^−24, the exact bound is ≈ 16.6/(1−γ); note γ = 0.99 gives γ^100 = 0.37, still far from zero). Below the zeroing horizon, early-token contributions are numerically indistinguishable from zero. Measured 50%-recall crossings (N ≈ 1024 for γ = 0.999; N ≈ 3072 for γ = 0.9999 by interpolation between the measured 2048/4096 bracket; horizon sweep n = 6) sit within a small factor of the e-folding scale and far below the fp32-zero bound, consistent with task-level forgetting preceding numerical zeroing.
- **Key-statistics distribution shift (synthetic analogue).** A deployed trained system would operate its feature map under a bounded training regime; presenting longer or shifted sequences pushes features out-of-distribution for the feature map itself. Our probe is a miniature distribution-shift analogue, not trained positional extrapolation: there is no trained positional encoding being extrapolated and no pretrained model — it is a controlled sequence-length stress test. Measured: shifting second-half key statistics by 0–4σ degrades pre-shift marker recall monotonically (0.75 → 0.25, n = 8), and a centering constant frozen on pre-shift calibration data halves post-shift accuracy relative to a distribution-matched constant (0.375 vs 0.875, n = 8). True trained-position extrapolation is beyond a synthetic setup and is left as future work.
- **Long-horizon state/output drift.** Because S_t is produced by a long product/sum of per-step updates, small per-step approximation error (from the linear kernel not exactly matching softmax) accumulates over t steps. A direct accumulation-law fit proved unidentifiable here: relative retrieval error saturates near 1.0 from N = 128 (a fully washed output is ≈ 0, so the metric saturates by construction). The finding is total washout, not gradual drift; a scale-invariant cosine metric instead declines gradually (0.95 at N = 128 → 0.18 at N = 4096, 0.23 at N = 8192, n = 8). No superlinear-compounding claim is made; establishing an accumulation law would require a finer metric and is left open.

Streaming runs show marker recall at ≈ 0 for all N ≥ 16k out to N = 1M (d = 16, undecayed). The widely cited 64k stress region is therefore treated here as an outer bound, not a predicted onset: in this synthetic configuration, failure arrives far earlier.

### 3.5 Breakpoint Scenario 3: Recurrent State Saturation and Floating-Point Instability

The additive update S_t = S_{t-1} + φ(k_t)v_tᵀ (or its gated form) is executed in finite-precision arithmetic. Two numerical failure modes follow directly:

- **Magnitude saturation:** without a decay term and for zero-mean update streams, ‖S_t‖ ~ √t, eventually exceeding the dynamic range where low-precision mantissas can represent small updates — new tokens' contributions become exactly zero in floating point, not merely small. (The √t scaling requires zero-mean updates; biased feature maps or value streams add a linear drift component that reaches saturation sooner.) Measured: state-norm log-log slope 0.448 over N = 128 → 1M, confirming √t growth for zero-mean streams; shrinking-update probes go numerically ineffective after ≈ 11 steps in fp16, ≈ 24 in fp32, effectively never in fp64 (49/36/7 ineffective updates out of 60). Measured separately rather than lumped: real bf16 arithmetic (ml_dtypes) trails fp16 here (52 ineffective, relative error 2.9e−03 vs 5.2e−04), since bf16 trades mantissa for range. Decayed steady-state magnitudes likewise saturate far below naive linear-theory scale (≈ 10–100×: 0.324/1.41/2.52 measured vs 2.83/28.3/282.8 naive for γ = 0.99/0.999/0.9999).
- **Cancellation under decay:** with γ < 1, the accumulated old state and the new low-magnitude update can differ by orders of magnitude; finite-precision scaling and addition can therefore introduce rounding error, producing state drift unrelated to true information loss. Measured at the magnitude level: fp16 outputs drift 1–3e−05 from fp64 references under decay while fp32 drift is ≈ 0 — though binary recall outcomes are identical across fp64/fp32/fp16 at this scale, so numerical damage lives at magnitude level before it reaches recall level.

Both mechanisms mean the failure is not purely statistical (a property of the task) but partly an artifact of the numerical implementation — a distinction the taxonomy in Section 4 keeps explicit.

---

## 4. Controlled Measurements & Failure-Mode Taxonomy

### 4.1 Comparative Structural Table

Table 1 summarizes the structural differences across architectures. Every cell carries an evidence tag — [M] measured in our synthetic lab (trial counts in §4.2), [M-dir] directional but not significant, [A] analytical/derived, [Arch] architectural description from the cited specifications, [O] open/unmeasured — per the scope definitions of §2.5. No cell reports a measurement of a production architecture.

| Property | Softmax Attention | Linear Attn. (Katharopoulos) | RetNet | RWKV | Mamba / SSM |
|---|---|---|---|---|---|
| State representation [Arch] | Full KV cache, O(N) | Fixed matrix S ∈ ℝ^{k×d_v} (square d×d in our lab) | S with exponential decay γ | Time-mixed linear state | Input-selective linear state |
| Recall mechanism [Arch; lab analogues §4.2] | Direct lookup over stored pairs | Superposition (kernel similarity) | Superposition + decay | Superposition + decay | Selective gating mitigates, does not eliminate (claim [4]; lab analogue: oracle latch) |
| Recall accuracy at high density | Lookup; 1.0 across N = 128–4096 [M, n = 10] | Steep fall from m = 4; 50%-crossing at m = 8 for tested d [M, n = 25] | Decay gives no mitigation at m = 16 (0.102/0.094/0.109 for γ = 1.0/0.99/0.9) [M, n = 8] | 0.0 → 0.5 → 0.625 across gate variants [M-dir, p ≈ 0.63, n = 8] | Oracle latch 0.05 → 0.4 [M, p = 0.004, n = 20]; onset unmoved [M] |
| State drift at long N | N/A — no compression [A] | ≈0 for N ≥ 16k; norm slope 0.448 [M] | 50%-crossings ≈1024 (γ = 0.999), ≈3072 interpolated (γ = 0.9999) [M, n = 6] | Decay-horizon knee [O] | Latch holds 1.0 → 0.3 over N = 512 → 8192 [M, n = 10] |
| Context decay horizon | No decay horizon [A]; flat 1.0 over tested N [M] | Crossings ≈1–3× e-folding scale, below fp32-zero bound [M+A] | E-folding −1/ln γ (≈1/(1−γ)); fp32-zero ≈16.6/(1−γ) [A] | Learned per-channel decay [Arch] | Input-dependent, selective [Arch] |
| Numerical failure mode | None, no accumulation [A] | Saturation [M]; bf16 trails fp16 here (52 vs 49 ineffective) [M] | Cancellation under decay [A]; fp16 drift 1–3e−05 [M] | Cancellation under decay [A] | Selective reset gates [Arch] |
| Asymptotic compute | O(N²); ≈10× slower at N = 8k here [M-impl, n = 2–3] | O(N); streams to 1M [M] | O(N), chunk-parallel [Arch] | O(N) [Arch] | O(N) [Arch] |

### 4.2 Controlled Measurements (synthetic lab, base seed 42, d = 16, elu+1 unless noted)

Scope, trials, seeds, statistics. One trial is one complete run of the task on an independent pseudorandom stream (NumPy default_rng seeded with base seed + trial index); n denotes the trial count. Dispersions are sample standard deviations. The bundle's 193-row CI table uses normal 95% intervals (1.96·SD/√n); the 9 Welch t-tests use group summary statistics with normal-approximation p-values. Base seed is 42 for all sweeps. Cross-seed checks use seeds 42–45 for recall curves and 43–44 for length decay and for five follow-up probes (gated decay, selective latch, per-channel decay, centered d = 16, dissipation); all other probes run the single base seed. Four seeds is a consistency check across nearby streams, not broad robustness. Breakpoint convention used throughout: m* is the first tested density with mean recall at or below 0.5; the length breakpoint is the first tested N with mean recall at or below 0.5 (interpolated where the crossing falls between grid points, stated as such); the precision breakpoint is the first timestep whose updates are numerically ineffective. These thresholds are measurement conventions for comparing conditions, not theoretical constants. Storage figures count persistent float64 state only — linear S(16×16) + z(16) = 2,176 B; softmax full K+V at 2·N·d·8 B — with transient score buffers, queries, and outputs excluded on both sides (score-buffer scaling is reported separately via the OOM probe below).

**Density.** Associative recall (25 trials): d = 16 accuracy 0.96 → 0.008 across m = 2 → 128 (95% CI ±0.05 at m = 2), falling steeply from m = 4 with no plateau. The 50%-crossing sits at m = 8 for every tested dimension including d = 256; dimension lifts only the low-density ceiling (m = 4: 0.54 at d = 8 → 0.86 at d = 128). Feature map moves the curve where dimension cannot (m = 8: relu 0.73, exp 0.56, elu+1 0.32; Welch t = 7.75, p ≈ 9×10⁻¹⁵ normal approx, n = 15+15). Orthogonal keys under elu+1 do not help (m = 4: 0.30 vs 0.67 random). Centered features restore the plateau with dimension-scaled onset (m* ≈ 2d in the square setup with Gaussian keys, n = 12–15; centered-orth d = 16 holds flat 1.0 to m = 16 then 0.5 at m = 32 — complete separation across all 10 trials, p < 0.001 by normal approximation, noting the degenerate zero-variance case). Accuracy follows binding count, not background length: filler sweeps saturate (256 vs 2048 identical), and washed states collapse to a constant predictor (accuracy exactly 1/m). Per-binding survival at m = 8 (n = 30) is uniform across stored positions (0.28–0.38, no primacy or recency) — position-blind interference.

**Length.** Marker recall (10 trials): N = 128/512/1024/4096 → 1.0/1.0/0.8/0.2, flat across marker positions (no recency in the undecayed state). Streaming to N = 1M: accuracy ≈ 0 throughout N ≥ 16k. Distribution-shift (0.75 → 0.25 over shift 0–4σ, n = 8), frozen-calibration (0.375 vs 0.875, n = 8), and cosine-drift (0.95 at N = 128 → 0.18 at N = 4096) probes are reported with their mechanisms in §3.4. Softmax reference holds 1.0 at all tested N with persistent storage 32 KB → 1 MB vs linear's constant 2,176 B under the accounting above. Full-sequence timing crosses over in our NumPy CPU-only build (i3, 3.7 GB RAM, fp32 scoring protocol — implementation-specific, not an architectural constant): softmax faster at N ≤ 512, linear approximately 10× faster at N = 8k (0.39 s vs 3.89 s, n = 3; n = 2 at N = 8192); full N×N fp32 scoring succeeds at N = 16384 (1 GB) and fails for our naive implementation at N = 32768, whose 4.3 GB score matrix exceeds machine RAM — a bound on our implementation (production implementations avoid materializing the score matrix via chunked/FlashAttention-style kernels and O(N) KV caches), not on Softmax attention itself.

**Numerics.** See §3.5 for magnitudes; additionally, fp32 accumulation cuts fp16-compute error ≈ 8× (6.7e−05 vs 5.2e−04).

**Dissipation.** A ×20-repeated coarse signal survives (1.0) where a singleton is lost (0.0) at every N including N = 128 (n = 8–10), where a lone marker is perfectly recoverable.

**Mitigations probed.** Exact local windows restore recent recall (w = 128 at N = 1024, n = 8; w = 512 at N = 4096, n = 6) without helping early markers (hybrid restoration p ≈ 0.32, n = 8, directional). Oracle salience latching restores decayed early recall to undecayed level (0.4, n = 20) and holds 1.0 → 0.3 across N = 512 → 8192 (n = 10); random gating matches fixed decay within noise — gating must be informed. Chunk resets bound state norm (14.81 → 5.17) at the cost of wiping early markers; mean-summary carry-forward is inert. Gated and novelty-gated m-sweeps show no onset movement (decay: 0.102/0.094/0.109; gate off == hard-gate 0.375 at m = 8, t = 0, p ≈ 1.0, correctly null). Multi-head banks at equal budget are identical (0.094/0.083/0.094 at m = 16, t = 0.35, p ≈ 0.73, correctly null). Stacked decayed + hybrid helps only where window coverage overlaps decayed survival (N = 4096, n = 8: linear-only and hybrid-only 0.375, decayed-only and stacked 0.125, at both early and recent markers). Absolute accuracies are conditional on distractor scale 0.01 (1.0/0.667–0.167/0.167–0.0 across scales 0.001/0.01/0.1); rankings are stable.

**Robustness.** Recall overlays tightly across seeds 42–45; length seeds 43–44 reproduce the decay shape; five follow-up probes repeat at 43–44. Per-trial records accompany all batch-2/3 probes and re-logged Exps 1–4; a 193-row CI table and 9 Welch t-tests (normal-approximation p) accompany the bundle. Non-significant contrasts (per-channel p ≈ 0.63, compressibility p ≈ 0.82, hybrid p ≈ 0.32, multi-head, hard-gate) are reported as directional only. Significant: relu over elu+1 (t = 7.75, p ≈ 9×10⁻¹⁵), centered over elu+1 (t = 31.7, p < 0.001, normal-approx floor), centered-orth complete separation (p < 0.001, degenerate case noted), oracle latch (t = 2.85, p = 0.004).

### 4.3 Failure Mode Taxonomy

Three distinct mechanisms, which can combine:

**(a) Associative interference / state collisions.** Density-driven retrieval degradation from kernel overlaps in a fixed-capacity superposition memory (§3.3). Diagnostic: hold N fixed, vary m; under standard maps onset is density-set and dimension-independent, with dimension-scaled onset recovered only for centered features (§4.2).

**(b) Temporal attenuation / decay.** Gated forgetting over time: early contributions decay as γ^{N−t}, producing effective horizons (§3.4). Diagnostic: hold task difficulty fixed, vary N and γ; compare measured crossings against the e-folding and zeroing scales rather than assuming a sharp knee, since measured curves are predominantly gradual.

**(c) Numerical precision loss.** Saturation and cancellation from finite-precision accumulation (§3.5), a property of the numerical implementation rather than task content. Diagnostic: vary precision at fixed task; compare continuous drift, since binary recall can be insensitive at small scale.

**Interactions.** High recall density (a) and long sequences (b) each degrade retrieval alone; compounded by precision loss (c), they converge on fine-grained information dissipation: repeated, redundant signals survive where singletons are lost at every scale (§4.2 dissipation). Decay (b) does not mitigate density-driven collisions (a) — it erases indiscriminately — while exact local windows bypass (a) only inside their coverage. Dissipation is thus the joint symptom, not a fourth independent root cause.

*Figure 4. Conceptual schematic; not measured data: the failure-mode taxonomy — associative interference, temporal attenuation, and numerical precision loss are distinct mechanisms converging on information dissipation (§4.3). Narrow regions of rapid decline can appear at critical m* or N*; measured curves are otherwise gradual (see §4.2).*

---

## 5. Discussion & Mitigation Strategies

### 5.1 Linear States as Lossy Learned Summaries

The unifying frame for all three breakpoints is that S_t is a lossy learned summary of the past, shaped implicitly (never by an explicit reconstruction objective, as no reconstruction loss is applied) to preserve whatever a downstream training loss rewards — a framing about deployed trained systems, studied here through untrained synthetic probes. Two consequences follow directly from this framing:

- (i) the compression is task-distribution-dependent, so a model's effective context length is not a fixed architectural number but a function of how compressible the deployment-time inputs are relative to the training distribution — directional lab support: alphabet-8 streams recall above alphabet-64 streams (0.675 vs 0.55), though the alphabet-8 vs random contrast is non-significant (p ≈ 0.82, weak at n = 40);
- (ii) because there is no explicit reconstruction objective, graceful degradation should not be assumed — and measured forgetting is position-blind and all-or-nothing per binding (§4.2 survival), consistent with narrow regions of rapid decline (accuracy falls ≈ 7× over a 4× density increase) rather than gentle slopes.

### 5.2 Proposed Architectural Interventions

Table 2 summarizes five concrete mitigation directions, the failure mode each targets, its mechanism, its trade-off, and the initial lab probe. None of these eliminate the fixed-capacity conflict identified in §2.2 — they each trade compute, memory, or architectural complexity for a higher or later breakpoint. This reframes the design space: reporting breakpoint location (critical m*, critical N*, critical dynamic-range budget) alongside aggregate perplexity/throughput would make such trade-offs explicit.

| Mitigation | Targets | Mechanism | Trade-off | Lab probe [M] |
|---|---|---|---|---|
| Dynamic (input-dependent) state decay | Temporal attenuation | Per-token learned gate (analogue only); lab bounds: oracle latch (upper), random gate (lower) | Adds parameters/compute; still bounded by fixed capacity | Latch 0.05 → 0.4 (p = 0.004, n = 20); random ≈ fixed within noise; learned gating untested |
| Local-softmax hybrid windows | Interference, dissipation | Exact sliding window + compressed state | O(w²) cost for window w; hybrid complexity | Recent restored at both scales (directional, p ≈ 0.32) |
| Chunked state resets / hierarchical states | Attenuation, precision loss | Reset or summarize S at chunk boundaries | Boundary information loss | Norm capped 14.8 → 5.2 (n = 8); early wiped; mean-summary inert |
| Higher-rank / multi-head state banks | Interference | Parallel banks at equal budget | Linear memory/compute cost | No effect at equal budget (n = 6) |
| Mixed-precision state accumulation (fp32 state, fp16 compute) | Precision loss | Higher-precision state than compute graph | Memory/bandwidth cost | 8× error cut; bf16 trails fp16 here, reported separately |

### 5.3 Limitations and Open Experiments

Deliberately unrunnable on the 3.7 GB CPU-only setup used here: real Mamba/RWKV/RetNet weights (no torch/GPU/RAM), learned gating (no training loop — bounded above by oracle and below by random controls instead), and trained-position knee-at-boundary effects (the synthetic setup has no training boundary; the frozen-calibration toy is the closest probe). Recall-level precision effects at longer horizons exceed what binary metrics can resolve. Small synthetic trial counts (n = 6–40 across probes) and normal approximations to summary statistics mean reported p-values are secondary evidence; effect sizes and confidence intervals should be emphasized where available. Per-trial records accompany all batch-2/3 probes and re-logged Exps 1–4 (exp01–04_trials.csv); remaining aggregate-only rows are documented in the bundle. Distractor-scale sensitivity conditions every absolute accuracy reported. The centering constant is calibrated under Gaussian keys and mis-centers other key distributions. These limitations constrain generalization from the measured synthetic system to production architectures.

---

## 6. Conclusion

The studied fixed-state additive recurrence exhibits identifiable failure regimes under associative density, long temporal horizons, and finite numerical precision. These regimes are mechanistically distinguishable — associative interference from superposition, temporal attenuation from decay, numerical precision loss from finite arithmetic — and can be measured using breakpoint-style diagnostics: the 50%-crossing density m*, the 50%-crossing length, and the numerically-ineffective-update timestep. The experiments demonstrate these effects under controlled synthetic conditions; extending the conclusions to trained production architectures requires direct evaluation.

Measured signatures within the synthetic laboratory: collision onset at m* = 8 across tested dimensions under standard elu+1 features, with a dimension-scaled breakpoint (≈2d) restored by centering under the tested Gaussian-key conditions; recall at ≈0 from N ≈ 16k with √t state-norm growth beneath it in zero-mean streams; precision-ordered numerical decay with bf16 trailing fp16 on the shrinking-update probe.

The resulting taxonomy (associative interference, temporal attenuation, numerical precision loss) gives a vocabulary for diagnosing why the studied recurrence fails on a given synthetic input, rather than only that it failed.

For future hybrid sequence-model design, the practical implication is architectural: state capacity, temporal retention, and numerical precision should be treated as separate design constraints when budgeting where exactness is preserved (recent tokens, high-salience bindings, chunk boundaries) versus where lossy compression is acceptable.

---

## References

[1] Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, pp. 5156–5165.

[2] Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621.

[3] Peng, B., Alcaide, E., Anthony, Q., et al. (2023). RWKV: Reinventing RNNs for the Transformer Era. Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14048–14077.

[4] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS) 30.

[6] Hopfield, J. J. (1982). Neural Networks and Physical Systems with Emergent Collective Computational Abilities. Proceedings of the National Academy of Sciences, 79(8), 2554–2558.

[7] Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2020). Hopfield Networks is All You Need. arXiv:2008.02217 (published at ICLR 2021).

[8] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172; Transactions of the Association for Computational Linguistics, 12, 157–173 (2024).

[9] Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., & Ginsburg, B. (2024). RULER: What's the Real Context Size of Your Long-Context Language Models? arXiv:2404.06654.

[10] Kamradt, G. (2023). Needle in a Haystack: Pressure Testing Long-Context LLMs. Blog and open-source evaluation suite (gkamradt/LLMTest_NeedleInAHaystack). Our marker-recall tasks are its mechanism-isolating complement: synthetic, seeded, and run against the update rule itself rather than production models end to end.

[11] Krotov, D., & Hopfield, J. J. (2016). Dense Associative Memory for Pattern Recognition. Advances in Neural Information Processing Systems, 29.

[12] Demircigil, M., Heusel, J., Löwe, M., Upgang, S., & Vermet, F. (2017). On a Model of Associative Memory with Huge Storage Capacity. Journal of Statistical Physics, 168, 288–299. https://doi.org/10.1007/s10955-017-1806-y

---

## Appendix A. Change Log (draft v2 → v3 → final)

- Scope: new §2.5 defines Derived / Architectural / Measured / Open evidence; synthetic controls (scalar-γ, per-channel-γ, novelty gates, oracle latches, hybrid windows, chunk resets, multi-head banks, mixed precision) explicitly distinguished from RetNet/RWKV/Mamba implementations.
- Notation: general state S ∈ ℝ^{k×d_v} throughout; d×d identified as the square lab setup.
- Claims: "unbounded exact memory" → exact-match reference store; "accuracy ceiling" → accuracy reference; "hard interference floor" → density-growing interference; "must fail … independent of training data or scale" → measured-regimes framing validated only in synthetic conditions; §2.2 consequence softened; §4.3(b) steepness qualified as predominantly gradual.
- m* ≈ 2d reported as empirical and conditional (square setup, Gaussian keys, centered map, n = 12–15).
- Rank argument reworked: rank bounds directions (≤ k); retrieval set by overlap statistics, with elu+1 vs centered evidence.
- Length: exact decay expansion γ^N; zeroing condition N > ln(ε)/ln(γ); e-folding N = −1/ln γ with 1/(1−γ) as the γ≈1 approximation; three named concepts (task forgetting, numerical zeroing, e-folding horizon).
- Shift probe renamed as synthetic distribution-shift analogue; trained positional extrapolation stays future work.
- "Compounding state drift" → "long-horizon state/output drift"; no accumulation law claimed.
- Taxonomy split into interference / attenuation / precision loss plus an interactions paragraph; dissipation reframed as joint symptom; Fig 4 caption softened.
- "Lossy autoencoder" → "lossy learned summary"; "rarely stated explicitly" removed.
- OOM result described as naive-implementation bound (chunked/FlashAttention-style kernels noted).
- Timing qualified as NumPy CPU-only, protocol-specific (0.39 s vs 3.89 s at N = 8k).
- Statistics: exact p-values/bounds (relu p ≈ 9×10⁻¹⁵; centered and centered-orth p < 0.001 with degeneracy noted); trial defined as one independent stream; Welch-t/normal-approx and CI methods stated.
- Seeds: base 42 map with 42–45 and 43–44 checks enumerated; four seeds described as consistency check, not broad robustness.
- Storage: equivalent persistent-float64 accounting stated on both sides; transient buffers excluded, reported separately.
- Table 1 redesigned with per-cell [M]/[M-dir]/[A]/[Arch]/[O] tags; Mamba/RWKV/RetNet rows reference lab analogues without claiming production measurements.
- Terminology: "Mamba latch"/"RWKV result" phrasing replaced with oracle-latch / per-channel-control terms plus inspiration notes.
- §5.3 updated (old-experiment trial logs now exist); Appendix B unchanged.

Final correction pass (v3 → final, verified against `public_evidence/*.csv`):
- Scope: "exact-match store" → explicitly stored per-token KV cache (abstract, §2.1, Fig 2); Table 1 Softmax cells → "Direct lookup over stored pairs" / "Lookup; 1.0 across N = 128–4096 [M]" and "No decay horizon [A]; flat 1.0 over tested N [M]"; conclusion scoped to the studied additive recurrence and synthetic inputs; "highest-value design axis" → neutral "important design axis" wording (later "separate design constraints").
- Capacity argument reframed as intuitive (not a formal theorem); Hopfield connection restricted to the studied outer-product recurrence; "confirms" → "shows"; "must not be conflated" → "should not be conflated"; "no guarantee of graceful degradation" → "should not be assumed"; "catastrophic interference" → "interference"; "(c)" no longer claims independence from task statistics.
- Numbers corrected to logs: horizon crossings re-tagged to horizon sweep (n = 6, γ = 0.9999 value by interpolation); cosine endpoint (0.18 at N = 4096, 0.23 at N = 8192); compressibility (0.675 vs 0.55 with p ≈ 0.82 on alphabet-8 vs random); hybrid trial counts split (n = 8 at N = 1024, n = 6 at N = 4096); timing trial counts (n = 3, n = 2 at N = 8192); random-gate phrasing softened to within-noise match.
- Data fixes in bundle: `exp10_bigd.csv` gained the rerun d = 128/m = 4 point (0.86, 25 trials, seed 42; same protocol as Exp 1); `exp5_gated_decay.csv` γ label 1.0 → 0.9999 on the row whose horizons (10000/166355) identify it. No other reruns; all other reported numbers verified verbatim against logs.
- Statistics: added the small-n/normal-approximation limitation to §5.3; p-values kept as reported (relu p ≈ 9×10⁻¹⁵ recomputed deterministically from stored t = 7.748 via normal approximation; all other p-values verbatim from `exp12_ttests.csv`).

Scope-hardening pass (final brief): subtitle/abstract/conclusion rewritten to scope all claims to the synthetic recurrence; related-work contribution sentence narrowed to conceptual comparison; rank≠capacity factor list added; m* ≈ 2d conditionalized everywhere; operational breakpoint convention defined (50%-crossing / interpolated / first-ineffective-step); distribution-shift probe explicitly not trained positional extrapolation; √t restricted to tested zero-mean streams; softmax/OOM/timing/storage wording scoped to the tested implementation; compressibility, cosine endpoint, and horizon-n numbers corrected to logs; limitations gained the generalization-constraint sentence; Tables 1–2 trimmed to concise tagged cells; Figs 1/2/4 replaced from placeholders with content-only schematics (`gen_schematics.py`, no measured data).

## Appendix B. Tests Left (future work — nothing invented)

Real model weights; learned gating; trained-position knee-at-boundary; recall-level precision effects at longer horizons; hierarchical (non-mean) chunk summarization; hybrid window scaling laws; stacked-mitigation interactions beyond the initial probe.

---

*Author: Aashirwad Sharma. Drafted from theory by the author; measurements from the accompanying CPU-only laboratory (base seed 42). v3 technical-correction pass plus final verification pass: every strong statement traceable to a derivation, a measurement, or a cited architectural fact; every reported number checked against the evidence bundle.*
