# -*- coding: utf-8 -*-
"""把 uploads 下的 PDF 抽成带页码标记的 txt，便于人工阅读出题。"""
import pathlib

from pypdf import PdfReader

UP = pathlib.Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_\data\uploads")
OUT = pathlib.Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\_smoke_corpus\corpus_text")
OUT.mkdir(parents=True, exist_ok=True)

for p in sorted(UP.glob("*.pdf")):
    reader = PdfReader(p)
    parts = []
    for i, pg in enumerate(reader.pages, 1):
        parts.append("[[P%d]]\n%s" % (i, (pg.extract_text() or "")))
    text = "\n".join(parts)
    (OUT / (p.stem + ".txt")).write_text(text, encoding="utf-8")
    print("%-46s %3d页 %7d字符" % (p.stem, len(reader.pages), len(text)))
