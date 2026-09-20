"""Build the shareable portrait academic PDF from the revised markdown.

- Portrait A4, navy headings, DejaVu fonts (full math-symbol coverage).
- Full Tables 1 & 2 rendered on landscape pages (data hard-coded below,
  matching the revised manuscript).
- Measured PNGs embedded at anchored paragraphs; original schematics
  (Figs 1, 2, 4) as labeled placeholders for the author to swap in.
- Standalone equation lines rendered as centered blocks.
"""

import html
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Stress-Testing_Linear_Attention_Sharma_REVISED.md"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "Stress-Testing_Linear_Attention_Sharma.pdf"
FIGDIR = ROOT / "public_evidence" / "figures"
FONT_DIR = (Path(__import__("matplotlib").__file__).parent
            / "mpl-data" / "fonts" / "ttf")

from fpdf import FPDF  # noqa: E402
from fpdf.fonts import FontFace  # noqa: E402

NAVY = (31, 56, 100)
GRAY = (90, 90, 90)

MATH_CHARS = set("φγϵεδ√‖²ᵀ_^Σ≈→×∈ℝσ=")

FIGURES = [
    # (anchor substring in paragraph, filename, caption)
    ("unconditional knee of the previous draft is withdrawn",
     "exp11g_centered.png",
     "Figure 3 (measured). Conditional knee: plateau with rank scaling under centered features; steep early fall under elu+1."),
    ("0.96 → 0.008",
     "exp1_overlay_by_dim.png",
     "Figure 5 (measured). Associative-recall collapse is steep from low density and identical across d = 8–64 (seed 42)."),
    ("relu 0.73",
     "exp6_featuremaps.png",
     "Figure 6 (measured). Feature map moves the curve where dimension cannot (d = 16)."),
    ("m* ≈ 16/64/128",
     "exp12a_centered_d.png",
     "Figure 6b (measured). Centered features restore rank scaling: m* ≈ 2d."),
    ("0.30 vs 0.67",
     "exp7c_orthkeys.png",
     "Figure 6c (measured). Orthogonal keys do not help under elu+1; centered-orth (d = 16) restores the knee."),
    ("flat across marker positions",
     "exp9_longlength_full.png",
     "Figure 7 (measured). Length floor near zero from N ≈ 16k to 1M; state norm grows as √t (slope 0.448)."),
    ("32 KB → 1 MB",
     "exp4_softmax_comparison.png",
     "Figure 8 (measured). Softmax holds 1.0 at all N; linear decays past N ≈ 1024 while storage stays constant."),
    ("49/36/7 ineffective",
     "exp3_effective_change.png",
     "Figure 9a (measured). Shrinking updates go numerically ineffective first in fp16, then fp32, effectively never in fp64."),
    ("52 ineffective",
     "exp12g_bf16.png",
     "Figure 9b (measured). Real bf16 performs worse than fp16 on shrinking magnitudes."),
    ("restores decayed early recall to undecayed level",
     "exp8c_selective.png",
     "Figure 10a (measured). Oracle salience latching restores decayed early recall (p ≈ 0.004)."),
    ("helps only where window coverage overlaps decayed survival",
     "exp12j_stacked.png",
     "Figure 10b (measured). Stacked decayed + hybrid helps only under window coverage (N = 4096: linear/hybrid 0.375, decayed/stacked 0.125)."),
    ("where a lone marker is perfectly recoverable",
     "exp7a_coarsefine.png",
     "Figure 11 (measured). Repeated coarse signals survive where singletons die, at every scale."),
    ("far below the fp32-zero bound",
     "exp5_gamma_horizon.png",
     "Figure 12 (measured). Forgetting vs decay: task-level forgetting precedes numerical zeroing."),
    ("declines gradually (0.95 at N",
     "exp12f_cosine.png",
     "Figure 12b (measured). Cosine similarity declines gradually where relative error saturates (0.95 at N = 128 to 0.18 at N = 4096)."),
    ("4.3 GB score matrix exceeds machine RAM",
     "exp12m_oom.png",
     "Figure 13 (measured). Full N×N scoring fails at N = 32768 for our naive implementation (4.3 GB > 3.7 GB RAM); linear streams to 1M."),
]

TABLE1_HEADERS = ["Property", "Softmax Attention", "Linear Attn. (Katharopoulos)",
                  "RetNet", "RWKV", "Mamba / SSM"]
TABLE1_ROWS = [
    ["State representation [Arch]", "Full KV cache, O(N)",
     "Fixed matrix S ∈ ℝ^{k×d_v} (square d×d in our lab)",
     "S with exponential decay γ", "Time-mixed linear state",
     "Input-selective linear state"],
    ["Recall mechanism [Arch; lab analogues §4.2]", "Direct lookup over stored pairs",
     "Superposition (kernel similarity)",
     "Superposition + decay", "Superposition + decay",
     "Selective gating mitigates, does not eliminate (claim [4]; lab analogue: oracle latch)"],
    ["Recall accuracy at high density",
     "Lookup; 1.0 across N = 128–4096 [M, n = 10]",
     "Steep fall from m = 4; 50%-crossing at m = 8 for tested d [M, n = 25]",
     "Decay gives no mitigation at m = 16 (0.102/0.094/0.109 for γ = 1.0/0.99/0.9) [M, n = 8]",
     "0.0 → 0.5 → 0.625 across gate variants [M-dir, p ≈ 0.63, n = 8]",
     "Oracle latch 0.05 → 0.4 [M, p = 0.004, n = 20]; onset unmoved [M]"],
    ["State drift at long N", "N/A — no compression [A]",
     "≈0 for N ≥ 16k; norm slope 0.448 [M]",
     "50%-crossings ≈1024 (γ = 0.999), ≈3072 interpolated (γ = 0.9999) [M, n = 6]",
     "Decay-horizon knee [O]",
     "Latch holds 1.0 → 0.3 over N = 512 → 8192 [M, n = 10]"],
    ["Context decay horizon", "No decay horizon [A]; flat 1.0 over tested N [M]",
     "Crossings ≈1–3× e-folding scale, below fp32-zero bound [M+A]",
     "E-folding −1/ln γ (≈1/(1−γ)); fp32-zero ≈16.6/(1−γ) [A]",
     "Learned per-channel decay [Arch]", "Input-dependent, selective [Arch]"],
    ["Numerical failure mode", "None, no accumulation [A]",
     "Saturation [M]; bf16 trails fp16 here (52 vs 49 ineffective) [M]",
     "Cancellation under decay [A]; fp16 drift 1–3e−05 [M]",
     "Cancellation under decay [A]",
     "Selective reset gates [Arch]"],
    ["Asymptotic compute",
     "O(N²); ≈10× slower at N = 8k here [M-impl, n = 2–3]",
     "O(N); streams to 1M [M]", "O(N), chunk-parallel [Arch]", "O(N) [Arch]",
     "O(N) [Arch]"],
]
TABLE1_WIDTHS = (30, 42, 52, 42, 42, 51)

TABLE2_HEADERS = ["Mitigation", "Targets", "Mechanism", "Trade-off", "Lab probe [M]"]
TABLE2_ROWS = [
    ["Dynamic (input-dependent) state decay", "Temporal attenuation",
     "Per-token learned gate (analogue only); lab bounds: oracle latch (upper), random gate (lower)",
     "Adds parameters/compute; still bounded by fixed capacity",
     "Latch 0.05 → 0.4 (p = 0.004, n = 20); random ≈ fixed within noise; learned gating untested"],
    ["Local-softmax hybrid windows", "Interference, dissipation",
     "Exact sliding window + compressed state",
     "O(w²) cost for window w; hybrid complexity",
     "Recent restored at both scales (directional, p ≈ 0.32)"],
    ["Chunked state resets / hierarchical states", "Attenuation, precision loss",
     "Reset or summarize S at chunk boundaries",
     "Boundary information loss",
     "Norm capped 14.8 → 5.2 (n = 8); early wiped; mean-summary inert"],
    ["Higher-rank / multi-head state banks", "Interference",
     "Parallel banks at equal budget",
     "Linear memory/compute cost",
     "No effect at equal budget (n = 6)"],
    ["Mixed-precision state accumulation (fp32 state, fp16 compute)", "Precision loss",
     "Higher-precision state than compute graph",
     "Memory/bandwidth cost",
     "8× error cut; bf16 trails fp16 here, reported separately"],
]
TABLE2_WIDTHS = (34, 34, 72, 62, 55)


def md_inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*([A-Za-z][A-Za-z0-9 ,;:'\"().\-–—]*?)\*(?![\w*])",
               r"<i>\1</i>", s)
    return s


def is_equation(line):
    s = line.strip()
    if not s or s[0] in "*->#|>":
        return False
    return "=" in s and any(c in MATH_CHARS for c in s)


class Paper(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"A. Sharma — Stress-Testing Linear Attention — p. {self.page_no()}/{{nb}}",
                  align="C")


def render_table(pdf, headers, rows, widths):
    pdf.add_page(orientation="L")
    pdf.set_font("DejaVu", "", 8)
    with pdf.table(col_widths=list(widths), width=sum(widths),
                   text_align="LEFT", first_row_as_headings=True,
                   headings_style=FontFace(emphasis="BOLD", size_pt=8),
                   line_height=4.6) as table:
        hdr = table.row()
        for h in headers:
            hdr.cell(h)
        for r in rows:
            row = table.row()
            for c in r:
                row.cell(c)
    pdf.add_page(orientation="P")


def place_image(pdf, path, caption, w=150):
    im = Image.open(path)
    h = w * im.size[1] / im.size[0]
    if pdf.get_y() + h + 16 > pdf.page_break_trigger:
        pdf.add_page()
    x = (pdf.w - w) / 2
    pdf.image(str(path), x=x, w=w)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 5, caption, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def placeholder(pdf, text):
    if pdf.get_y() + 58 > pdf.page_break_trigger:
        pdf.add_page()
    x0, ww = pdf.l_margin, pdf.epw
    y0 = pdf.get_y()
    pdf.set_fill_color(235, 241, 250)
    pdf.set_draw_color(*NAVY)
    pdf.rect(x0, y0, ww, 38, style="DF")
    pdf.set_xy(x0, y0 + 13)
    pdf.set_font("DejaVu", "I", 10)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(ww, 6, text, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(y0 + 42)


def main():
    lines = SRC.read_text(encoding="utf-8").split("\n")
    pdf = Paper(orientation="P", format="A4")
    pdf.alias_nb_pages("{nb}")
    pdf.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", FONT_DIR / "DejaVuSans-Oblique.ttf")
    pdf.add_font("DejaVu", "BI", FONT_DIR / "DejaVuSans-BoldOblique.ttf")
    pdf.set_auto_page_break(True, margin=16)
    pdf.set_margins(18, 15, 18)
    pdf.add_page()

    def body_font():
        pdf.set_font("DejaVu", "", 10.5)
        pdf.set_text_color(0, 0, 0)

    def para(text, indent=False):
        body_font()
        if indent:
            pdf.set_x(pdf.l_margin + 8)
            pdf.multi_cell(pdf.epw - 8, 5.4, md_inline(text))
        else:
            pdf.write_html(f"<p>{md_inline(text)}</p>")
        pdf.ln(1.5)

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("# "):
            pdf.set_font("DejaVu", "B", 21)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(0, 9, line[2:].strip(), align="C")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("DejaVu", "B", 14.5)
            pdf.set_text_color(*NAVY)
            pdf.ln(2)
            pdf.multi_cell(0, 7.5, line[3:].strip())
            pdf.set_draw_color(*NAVY)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin,
                     pdf.get_y())
            pdf.ln(3)
        elif line.startswith("### "):
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(*NAVY)
            pdf.ln(1)
            pdf.multi_cell(0, 6.5, line[4:].strip())
            pdf.ln(1)
        elif line.strip() == "---":
            pdf.ln(2)
            pdf.set_draw_color(160, 160, 160)
            pdf.line(pdf.l_margin + 40, pdf.get_y(),
                     pdf.w - pdf.r_margin - 40, pdf.get_y())
            pdf.ln(4)
        elif line.startswith("|"):
            # no pipe tables in manuscript; skip defensively
            i += 1
            continue
        elif line.startswith("- "):
            buf = [line[2:]]
            i += 1
            while i < n and lines[i].startswith("- "):
                buf.append(lines[i][2:])
                i += 1
            for b in buf:
                body_font()
                pdf.set_x(pdf.l_margin + 6)
                pdf.multi_cell(pdf.epw - 6, 5.4, "•  " + md_inline(b).strip())
                pdf.set_x(pdf.l_margin)
                for anchor, fname, caption in FIGURES:
                    if anchor in b:
                        place_image(pdf, FIGDIR / fname, caption)
            pdf.ln(1.5)
            continue
        elif (line.startswith("**") and line.endswith("**")
              and len(line) < 60):
            pdf.set_font("DejaVu", "B", 13)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 7, line.strip("*").strip(), align="C")
            pdf.ln(1)
        elif line.startswith("*Figure 1."):
            place_image(pdf, FIGDIR / "schematic_conflict.png",
                        "Figure 1 (conceptual schematic; not measured data).")
            para(line.strip("*").strip())
        elif line.startswith("*Figure 2."):
            place_image(pdf, FIGDIR / "schematic_storage.png",
                        "Figure 2 (conceptual schematic; not measured data).")
            para(line.strip("*").strip())
        elif line.startswith("*Figure 4."):
            place_image(pdf, FIGDIR / "schematic_taxonomy.png",
                        "Figure 4 (conceptual schematic; not measured data).")
            para(line.strip("*").strip())
        elif line.startswith("*Figure 3."):
            if "unconditional knee of the previous draft is withdrawn" in line:
                place_image(pdf, FIGDIR / "exp11g_centered.png",
                            "Figure 3 (measured). Conditional knee: plateau with rank scaling under centered features; steep early fall under elu+1.")
            para(line.strip("*").strip())
        elif line.startswith("*") and line.endswith("*"):
            pdf.set_font("DejaVu", "I", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6, line.strip("*").strip(), align="C")
            pdf.ln(1)
        elif is_equation(line):
            if pdf.get_y() + 14 > pdf.page_break_trigger:
                pdf.add_page()
            pdf.set_font("DejaVu", "", 11)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6.5, line.strip(), align="C")
            pdf.ln(2)
        else:
            # join wrapped paragraph lines
            buf = [line]
            i += 1
            while (i < n and lines[i].strip()
                   and not lines[i].startswith(("#", "- ", "|", "*"))
                   and lines[i].strip() != "---"
                   and not is_equation(lines[i])):
                buf.append(lines[i].strip())
                i += 1
            text = " ".join(buf)
            para(text)
            # injections anchored on paragraph text
            if "Table 1 summarizes the structural differences" in text:
                render_table(pdf, TABLE1_HEADERS, TABLE1_ROWS, TABLE1_WIDTHS)
                pdf.set_font("DejaVu", "I", 9)
                pdf.set_text_color(*GRAY)
                pdf.multi_cell(0, 5, "Table 1. Comparative structural table with measured cell statuses (§4.2).",
                               align="C")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
            if "Table 2 summarizes five concrete mitigation" in text:
                render_table(pdf, TABLE2_HEADERS, TABLE2_ROWS, TABLE2_WIDTHS)
                pdf.set_font("DejaVu", "I", 9)
                pdf.set_text_color(*GRAY)
                pdf.multi_cell(0, 5, "Table 2. Mitigation strategies with initial lab probes (§4.2).",
                               align="C")
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
            for anchor, fname, caption in FIGURES:
                if anchor in text:
                    place_image(pdf, FIGDIR / fname, caption)
            continue
        i += 1

    pdf.output(str(DST))
    print(f"Wrote {DST} ({DST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
