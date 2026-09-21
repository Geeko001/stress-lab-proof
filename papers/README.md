# Papers — Independent Research Series

Slot index for the paper series. Paper 1 is slotted; later slots are reserved.

| # | Title | Status | File |
|---|---|---|---|
| 1 | Stress-Testing Linear Attention: Architectural Breakpoints and Context Failure Modes | Draft slot created — author slots open (contact, date, venue, bio, license) | `paper-01-stress-testing-linear-attention.md` |
| 2 | [SLOT — reserved] | Empty | — |
| 3 | [SLOT — reserved] | Empty | — |

## Paper 1 sources

- Working draft: `../Paper_1_Draft_v3.md` (285 lines; technical content source)
- Canonical manuscript PDF (GitHub, public): `../Sharma_Stress-Testing_Linear_Attention_Manuscript.pdf`
  (raw: `https://raw.githubusercontent.com/Geeko001/stress-lab-proof/main/Sharma_Stress-Testing_Linear_Attention_Manuscript.pdf` — this is the URL the app's Paper page fetches)
- PDF builder: `../build_paper_pdf.py` (usage: `python build_paper_pdf.py Paper_1_Draft_v3.md Paper_1_Draft_v3.pdf`)
- Evidence bundle: `../public_evidence/` (61 CSVs + config + summary + 48 figures)
- Claim map: `../CLAIM_EVIDENCE_MAP.md` (C001–C025) · Gaps: `../EVIDENCE_GAPS.md`
- Lab: `../experiments/`, `../models/`, `../utils/`, `../app.py`

## Before publishing Paper 1

Fill every `[SLOT]` in the paper file: contact email, date, venue/version, author bio,
acknowledgments, competing interests, paper license. Re-verify Appendix A paths against
the frozen bundle. Do not paste API keys or secrets anywhere in `papers/`.
