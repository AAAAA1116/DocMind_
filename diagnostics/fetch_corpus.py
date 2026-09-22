# -*- coding: utf-8 -*-
"""下载巨潮上的企业制度类 PDF（公司章程 / 内部控制 / 信息披露管理制度）并验证抽取质量。"""
import os
import sys
import time
import urllib.request
from collections import Counter

DEST = r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_\data\uploads\corpus_raw"
STATIC = "http://static.cninfo.com.cn/"
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "http://www.cninfo.com.cn/",
    "Accept": "application/pdf,*/*",
}

TARGETS = [
    ("601588_北辰实业_公司章程.pdf", "finalpage/2026-09-19/1225572679.PDF"),
    ("603341_龙旗科技_公司章程.pdf", "finalpage/2026-09-19/1225572384.PDF"),
    ("002705_新宝股份_公司章程.pdf", "finalpage/2026-09-19/1225573194.PDF"),
    ("300755_华致酒行_内部控制制度.pdf", "finalpage/2026-08-29/1225524093.PDF"),
    ("603979_金诚信_内部控制管理制度.pdf", "finalpage/2026-09-08/1225552324.PDF"),
    ("301172_君逸数码_信息披露管理制度.pdf", "finalpage/2026-09-11/1225561430.PDF"),
    ("300604_长川科技_信息披露管理制度.pdf", "finalpage/2026-09-15/1225564420.PDF"),
]

os.makedirs(DEST, exist_ok=True)


def download(name, rel):
    path = os.path.join(DEST, name)
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        print("  [skip] 已存在 %6.1f KB  %s" % (os.path.getsize(path) / 1024, name))
        return path
    url = STATIC + rel
    req = urllib.request.Request(url, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        print("  [ok]   %6.1f KB  %s" % (len(data) / 1024, name))
        time.sleep(0.5)
        return path
    except Exception as e:
        print("  [FAIL] %s -> %s" % (name, e))
        return None


print("=" * 90)
print("STEP 1  下载")
print("=" * 90)
paths = []
for name, rel in TARGETS:
    p = download(name, rel)
    if p:
        paths.append(p)

print()
print("=" * 90)
print("STEP 2  抽取质量验证（pypdf）")
print("=" * 90)

from pypdf import PdfReader  # noqa: E402

rows = []
for p in paths:
    try:
        reader = PdfReader(p)
        npages = len(reader.pages)
        pages = [(pg.extract_text() or "") for pg in reader.pages]
    except Exception as e:
        print("[ERR] %s -> %s" % (os.path.basename(p), e))
        continue

    text = "\n".join(pages)
    total_chars = len(text.replace("\n", "").replace(" ", ""))

    head, tail = Counter(), Counter()
    for t in pages:
        lines = [l.strip() for l in t.split("\n") if l.strip()]
        if not lines:
            continue
        for l in lines[:2]:
            head[l] += 1
        for l in lines[-2:]:
            tail[l] += 1

    top_head = head.most_common(1)[0] if head else ("", 0)
    top_tail = tail.most_common(1)[0] if tail else ("", 0)
    hdr_ratio = 100.0 * top_head[1] / max(npages, 1)

    empty = sum(1 for t in pages if len(t.strip()) < 20)

    rows.append((os.path.basename(p), npages, total_chars, hdr_ratio, empty,
                 top_head[0][:38], top_tail[0][:38]))
    print()
    print("-" * 90)
    print("%s" % os.path.basename(p))
    print("  页数 %d | 去空白字数 %d | 页眉重复率 %.0f%% | 近空页 %d" %
          (npages, total_chars, hdr_ratio, empty))
    print("  最高频页首行: [%d次] %s" % (top_head[1], top_head[0][:60]))
    print("  最高频页尾行: [%d次] %s" % (top_tail[1], top_tail[0][:60]))
    sample = "\n".join(l for l in text.split("\n") if l.strip())[:420]
    print("  ---- 正文样本 ----")
    for line in sample.split("\n")[:12]:
        print("    " + line[:78])

print()
print("=" * 90)
print("STEP 3  汇总")
print("=" * 90)
tot = 0
for n, np_, c, h, e, th, tt in rows:
    tot += c
    print("  %-46s %4d页 %8d字  页眉%.0f%%" % (n, np_, c, h))
print("  TOTAL 去空白字数 = %d  (约 %.1f 万字)" % (tot, tot / 10000))
