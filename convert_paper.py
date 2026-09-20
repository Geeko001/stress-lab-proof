"""Convert the revised paper markdown to a readable PDF (reading copy only).

Uses DejaVu Sans bundled with matplotlib so math symbols
(phi, gamma, in, R, arrows) render correctly.
"""

from pathlib import Path

import markdown
from fpdf import FPDF

ROOT = Path(__file__).parent
SRC = ROOT / "Stress-Testing_Linear_Attention_Sharma_REVISED.md"
DST = ROOT / "Stress-Testing_Linear_Attention_Sharma_REVISED.pdf"
FONT_DIR = (Path(__import__("matplotlib").__file__).parent
            / "mpl-data" / "fonts" / "ttf")


class PaperPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"Reading copy — p. {self.page_no()}/{{nb}}",
                  align="C")


def main():
    md = SRC.read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables"])
    # landscape A4 so the wide comparison tables fit on one page
    pdf = PaperPDF(orientation="L", format="A4")
    pdf.alias_nb_pages("{nb}")
    pdf.add_font("DejaVu", "", FONT_DIR / "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "B", FONT_DIR / "DejaVuSans-Bold.ttf")
    pdf.add_font("DejaVu", "I", FONT_DIR / "DejaVuSans-Oblique.ttf")
    pdf.add_font("DejaVu", "BI", FONT_DIR / "DejaVuSans-BoldOblique.ttf")
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    pdf.set_font("DejaVu", "", 11)
    pdf.write_html(html)
    pdf.output(str(DST))
    print(f"Wrote {DST} ({DST.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
