# -*- coding: utf-8 -*-
"""整理 DocMind_ 的语料台：归档错配语料，把制度类 PDF 提到 uploads 根目录。

只做 move，不做任何删除 —— 归档物全部落在 data/uploads/_attic/ 可随时取回。
"""
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_")
UP = ROOT / "data" / "uploads"
RAW = UP / "corpus_raw"
ATTIC = UP / "_attic"

ATTIC.mkdir(parents=True, exist_ok=True)

ARCHIVE = [
    "美的集团2025年年度报告.pdf",
    "比亚迪2025年年度报告.pdf",
    "贵州茅台2025年年度报告.pdf",
    "公司规章制度一万字测试用.txt",
]

print("=== 1. 归档（move 进 _attic，不删除） ===")
for name in ARCHIVE:
    src = UP / name
    if src.exists():
        shutil.move(str(src), str(ATTIC / name))
        print("   ->_attic/  " + name)
    else:
        print("   [skip] 不存在 " + name)

print()
print("=== 2. 制度语料提到 uploads 根目录（rebuild 才扫得到） ===")
pdfs = sorted(RAW.glob("*.pdf"))
if not pdfs:
    print("   [warn] corpus_raw 里没有 pdf（可能已移动过）")
for p in pdfs:
    shutil.move(str(p), str(UP / p.name))
    print("   ->uploads/ " + p.name)

if RAW.exists():
    leftover = list(RAW.iterdir())
    if not leftover:
        RAW.rmdir()
        print("   已删除空目录 corpus_raw")
    else:
        print("   corpus_raw 仍有内容，保留：", [x.name for x in leftover])

print()
print("=== 3. 最终 uploads 根目录（rebuild 会入库的集合） ===")
for p in sorted(UP.iterdir()):
    size = p.stat().st_size if p.is_file() else 0
    print("   %s %10d  %s" % ("DIR " if p.is_dir() else "FILE", size, p.name))

print()
print("=== 4. _attic 归档区 ===")
for p in sorted(ATTIC.iterdir()):
    print("   %10d  %s" % (p.stat().st_size if p.is_file() else 0, p.name))
