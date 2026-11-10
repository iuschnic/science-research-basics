"""Простой конвертер REVIEW.md -> REVIEW.docx (заголовки, абзацы, списки, таблицы, код, **жирный**, `код`)."""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

# Жирный текст может содержать одиночную звёздочку (ν*), поэтому ищем ближайшую пару **.
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_runs(par, text):
    text = text.replace("\\*", "*")
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0x80, 0x20, 0x20)
        else:
            par.add_run(part)


lines = src.read_text(encoding="utf-8").splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    s = line.strip()
    if s.startswith("```"):
        i += 1
        code = []
        while not lines[i].strip().startswith("```"):
            code.append(lines[i])
            i += 1
        p = doc.add_paragraph()
        r = p.add_run("\n".join(code))
        r.font.name = "Consolas"
        r.font.size = Pt(9)
    elif s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
            i += 1
        i -= 1
        t = doc.add_table(rows=len(rows), cols=len(rows[0]))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                par = t.cell(ri, ci).paragraphs[0]
                add_runs(par, cell)
                for run in par.runs:
                    run.font.size = Pt(9.5)
                    if ri == 0:
                        run.bold = True
        doc.add_paragraph()
    elif re.fullmatch(r"!\[(.*)\]\((.+)\)", s):
        caption, image = re.fullmatch(r"!\[(.*)\]\((.+)\)", s).groups()
        doc.add_picture(str(src.parent / image), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.add_run(caption).italic = True
    elif s.startswith("#"):
        level = len(s) - len(s.lstrip("#"))
        h = doc.add_heading(level=min(level, 3))
        add_runs(h, s.lstrip("#").strip())
    elif s == "---" or not s:
        pass
    elif re.match(r"^\s*[-*] ", line):
        indent = (len(line) - len(line.lstrip())) // 2
        p = doc.add_paragraph(style="List Bullet 2" if indent else "List Bullet")
        add_runs(p, re.sub(r"^\s*[-*] ", "", line))
    elif re.match(r"^\d+\. ", s):
        p = doc.add_paragraph(style="List Number")
        add_runs(p, re.sub(r"^\d+\. ", "", s))
    else:
        add_runs(doc.add_paragraph(), s)
    i += 1

doc.save(dst)
print("saved", dst)
