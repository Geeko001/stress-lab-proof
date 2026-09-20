# DATA_AND_EVIDENCE.md

Where the existing evidence lives and what each part is. Nothing is copied into `reproducibility/` — all paths below are the live repository locations.

## Layout

- Aggregates + raw trials + stats: `public_evidence/*.csv` (61 files).
- Bundle metadata: `public_evidence/config.json` (protocol grids/seeds), `public_evidence/summary.json` (computed snapshot + env, generated 2026-09-17).
- Figures: `public_evidence/figures/` (48 PNGs: 15 measured embedded in the PDF, 3 generated schematics, ~30 superseded alternates).
- Unit-test record: `public_evidence/pytest_report.txt`.
- Manuscript: `Paper_1_Draft_v3.md` + frozen `Paper_1_Draft_v3.pdf`.

## What supports what

- Per-file contents, trials, seeds, conditions, and manuscript section: **EVIDENCE_INDEX.md** (authoritative table).
- Claim (C001–C025) → exact file paths: **CLAIM_EVIDENCE_MAP.md**.
- Experiment families → scripts → files → figures: **EXPERIMENT_REGISTRY.md**.
- Contradictions, small-n caveats, unused files, provenance notes: **EVIDENCE_GAPS.md**.
- Overview + reading order: **RESEARCH_EVIDENCE_SUMMARY.md**.

## Data kinds (do not confuse them)

- **Raw data:** `*_trials.csv` companions (per-trial 0/1 or per-step records with seeds where recorded).
- **Derived results:** per-condition aggregates (`exp*.csv` with mean/std columns), probe logs (`exp3_shrinking.csv`, `exp12_bf16.csv`, `exp8_mixed.csv`), the OOM analysis record (`exp12_oom.csv`).
- **Validation outputs:** `tests/` suite results, `pytest_report.txt`, `exp12_ttests.csv`, `exp12_CIs.csv`.
- **Manuscript claims:** `Paper_1_Draft_v3.md` only. Claims are interpretations of the above, tagged `[D]/[A]/[M]/[O]` per §2.5.

## Known artifact quirks (reported, not hidden)

- `summary.json` compressibility values (0.75/0.625) contradict `exp11_compress*.csv` (0.675/0.55); the manuscript follows the CSVs. Same for its followup oracle value (0.25) vs `exp8_selective.csv` (0.40).
- `exp10_bigd.csv` contains one later rerun row (d=128, n=25, seed 42); `exp5_gated_decay.csv` row 4 carries a corrected γ label (1.0 → 0.9999). Both documented in EVIDENCE_GAPS.md and the manuscript change log.
- `exp11_noisy*.csv` (query-noise probe) has no manuscript claim; `claude_paper_brief.md` holds stale numbers and is superseded by the CSVs.
