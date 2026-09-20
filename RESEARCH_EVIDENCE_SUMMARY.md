# RESEARCH_EVIDENCE_SUMMARY.md

Evidence audit for **“Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes”** (`Paper_1_Draft_v3.md`, frozen). No experiments run, no data modified, no results invented. Prior agent's handoff brief (`claude_paper_brief.md`) exists but contains stale numbers — CSVs override it everywhere (see EVIDENCE_GAPS §8).

## Totals

- **63 data/config files** in `public_evidence/` (61 CSVs + `config.json` + `summary.json`), plus `pytest_report.txt`.
- **48 figure PNGs** in `public_evidence/figures/` (15 measured embedded in PDF + 3 generated schematics + ~30 superseded alternates).
- **14 experiment families** (registry §1–12 + dissipation + the null architecture family §13).
- **25/25 manuscript claims SUPPORTED** (CLAIM_EVIDENCE_MAP.md C001–C025). Zero NOT LOCATED.
- **Direct production-architecture experiments found: NONE.** No Mamba/RWKV/RetNet/trained-LM runs exist. `models/` holds only synthetic NumPy (`linear_attention.py`, `softmax_attention.py`). All architecture-named code paths are "-style" synthetic controls with disclaimers. Recorded verdict: **“Direct experimental evidence not located in repository.”** Manuscript categories hold: (A) synthetic recurrence experiments, (B) no production-architecture experiments, (C) conceptual/literature comparisons only.
- **Partially supported / caveated:** streaming floor at N≥16k rests on n=2–3 per far-end N; decay magnitudes single-measurement; exp4 `trials`=30 vs cited n=10 (likely 10×3 positions, undocumented in-file); most aggregate CSVs lack seed columns (base-42 protocol in config).
- **Key inconsistency:** `summary.json` compress values (0.75/0.625) contradict `exp11_compress*.csv` (0.675/0.55/0.65); manuscript follows CSVs. Same for followup oracle 0.25 vs filed 0.40.
- **Unused evidence:** `exp11_noisy*.csv` (query-noise probe) has no manuscript claim; pytest report; superseded PNGs.
- **Files Claude should read first:** `Paper_1_Draft_v3.md`, then the registry → index → claim map → gaps (order below).

## CLAUDE_READING_ORDER

1. `Paper_1_Draft_v3.md` — the frozen manuscript (all claims).
2. `EXPERIMENT_REGISTRY.md` — the 14 experiment families, scripts, and figure mapping.
3. `EVIDENCE_INDEX.md` — per-file contents, trials, seeds, conditions, mapping confidence.
4. `CLAIM_EVIDENCE_MAP.md` — claim IDs C001–C025 with exact supporting paths.
5. `EVIDENCE_GAPS.md` — contradictions (summary.json vs CSVs), small-n caveats, unused files, provenance notes.
6. Primary raw result files as needed: `public_evidence/exp1_dimensionality.csv`, `exp2_length_sweep.csv`, `exp9_longlength.csv`, `exp6_featuremaps.csv`, `exp12_centered_d.csv`, `exp5_gated_decay.csv`, `exp11_horizon.csv`, `exp8_selective.csv`, `exp3_shrinking.csv`, `exp12_bf16.csv`, `exp4_softmax_baseline.csv`, `exp11_timing.csv`.
7. Statistical outputs: `public_evidence/exp12_ttests.csv`, `public_evidence/exp12_CIs.csv` (+ `*_trials.csv` companions for raw pools).
8. Figure/table source data: same CSVs as (6); embedded-figure ↔ file mapping in `build_paper_pdf.py` FIGURES list; schematics from `gen_schematics.py` (no data).
