# -*- coding: utf-8 -*-
"""诊断：新语料上为什么大面积误拒。打印检索内容 + 模型原始回答。"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\A\WorkBuddy\2026-09-18-14-19-51\DocMind_")
sys.path.insert(0, str(ROOT))

from core import embed_store, rag_chain  # noqa: E402

print("索引条数 =", embed_store.count())
print()

QUESTIONS = [
    "北京北辰实业股份有限公司的董事会由几名董事组成？其中独立董事几名？",
    "北辰实业的董事任期是几年？届满后可以连任吗？",
]

for q in QUESTIONS:
    print("=" * 90)
    print("Q:", q)
    out = rag_chain.answer(q, use_hybrid=False, use_rerank=False)
    print("refused = %s   原因 = %s" % (out["refused"], out["refuse_reason"]))
    print("--- 模型回答 ---")
    print(out["answer"])
    print("--- 检索到的 %d 个块 ---" % len(out["sources"]))
    for i, s in enumerate(out["sources"], 1):
        print("  [%d] %s  chunk=%s  cos=%.4f" % (i, s["source"], s["chunk_index"], s["score"]))
        body = s["content"].replace("\n", " ")
        print("      %s" % body[:260])
    print("--- system_prompt（前 900 字）---")
    print(out["system_prompt"][:900])
    print()
