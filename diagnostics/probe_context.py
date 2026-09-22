# -*- coding: utf-8 -*-
"""验证「块内缺少文档级上下文」是不是跨文档检索失败的主因。

做法：从正式集合取出 458 块，建两个临时集合做对照——
  A. 原文块（现状）
  B. 每块内容前加「《来源文件名》」前缀（contextual retrieval 最简版）
只测检索，不调大模型。测完删掉临时集合，不碰正式库、不碰源码。
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_")
sys.path.insert(0, str(ROOT))

import chromadb  # noqa: E402
import config  # noqa: E402
from core import embed_store  # noqa: E402

client = chromadb.PersistentClient(path=str(config.INDEX_DIR))
src = client.get_collection("docmind")
got = src.get(include=["documents", "metadatas"])
docs, metas = got["documents"], got["metadatas"]
print("源集合 docmind 块数 =", len(docs))
print()

# (标签, 查询, 期望来源关键词, 期望内容关键词)
CASES = [
    ("q01 北辰董事会（用全名）", "北京北辰实业股份有限公司的董事会由几名董事组成？其中独立董事几名？", "北辰", "9人组成"),
    ("q01b 北辰董事会（用简称）", "北辰实业的董事会由几名董事组成？", "北辰", "9人组成"),
    ("q02 北辰董事任期", "北辰实业的董事任期是几年？届满后可以连任吗？", "北辰", "任期三年"),
    ("q04 龙旗董事会", "上海龙旗科技股份有限公司的董事会由几名董事组成？", "龙旗", "7 名董事"),
    ("q12 金诚信资料保存", "金诚信的内部控制管理制度对内部控制检查工作资料的保存期限是怎么规定的？", "金诚信", "不少于十年"),
    ("q19 君逸文件保存", "君逸数码的信息披露文件及公告需要保存多久？", "君逸", "保存期限为 10 年"),
]


def build(name, with_prefix):
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001
        pass
    store = embed_store.EmbedStore(collection_name=name)
    payload = []
    for d, m in zip(docs, metas):
        m = m or {}
        source = m.get("source", "")
        text = ("《%s》\n%s" % (source, d)) if with_prefix else d
        payload.append({
            "content": text,
            "metadata": {"source": source, "chunk_index": m.get("chunk_index")},
        })
    store.add_documents(payload)
    return store


def run(store, tag):
    print("#" * 90)
    print("## %s" % tag)
    print("#" * 90)
    hit = 0
    for label, q, want_src, want_kw in CASES:
        hits = store.search(q, k=3)
        ok = False
        lines = []
        for i, (doc, s) in enumerate(hits, 1):
            md = doc.get("metadata") or {}
            sname = md.get("source", "?")
            body = (doc.get("content") or "").replace("\n", " ")
            good = (want_src in sname) and (want_kw in body)
            if good:
                ok = True
            lines.append("     [%d] %-42s cos=%.4f %s %s" %
                         (i, sname, s, "✅" if good else "  ", body[:70]))
        hit += 1 if ok else 0
        print("  %-28s %s" % (label, "命中 ✅" if ok else "未命中 ❌"))
        for line in lines:
            print(line)
    print("  ==> 命中 %d / %d" % (hit, len(CASES)))
    print()
    return hit


print("正在建临时集合 A（原文块）...")
store_a = build("tmp_probe_plain", with_prefix=False)
print("正在建临时集合 B（加文件名前缀）...")
store_b = build("tmp_probe_prefixed", with_prefix=True)
print()

a = run(store_a, "对照组 A：现状（块内没有公司名）")
b = run(store_b, "实验组 B：每块前缀《来源文件名》")

print("=" * 90)
print("结论：现状 %d/%d  →  加前缀 %d/%d" % (a, len(CASES), b, len(CASES)))
print("=" * 90)

for name in ("tmp_probe_plain", "tmp_probe_prefixed"):
    try:
        client.delete_collection(name)
        print("已删除临时集合", name)
    except Exception as e:  # noqa: BLE001
        print("删除失败", name, e)
