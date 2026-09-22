# -*- coding: utf-8 -*-
"""阶段二（预演）：用真实问答验证 TOP_K 的收益与代价

不跑 ragas、不改仓库代码 —— 直接调 ``rag_chain.answer(question, top_k=K)``，
把同一道题在 K=3 和 K=8 下的**答案全文**摆在一起看。

顺带验证一个从代码里读出来的成本结论：

    ``_retrieve_hybrid`` 里 ``pool = max(RERANK_CANDIDATES, k)``，
    精排会对**整池**候选做前向，最后只取 ``k`` 个。
    RERANK_CANDIDATES = 10、TOP_K = 3 时 —— 精排算了 10 个，扔掉 7 个。

    推论：**K 从 3 调到 ≤10 应该是免费的**（重排耗时不变，只是少扔几个）。
    这里实测 K=3 与 K=8 的耗时差。

用法
----
    ./.venv/Scripts/python.exe -u ../_smoke_corpus/verify_topk_cases.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "DocMind_"
sys.path.insert(0, str(ROOT))

import config                        # noqa: E402
from core import rag_chain           # noqa: E402

#: 阶段一判定「K 调大能救回来」的题（need_k=5 / need_k=7）
GAIN_CASES = ["q19", "q08"]
#: 阶段一判定「K 再大也救不回来」的题（用来确认归因没错）
FAIL_CASES = ["q05"]
#: 验证「K 增大是否零成本」的题（开重排）
COST_CASE = "q19"

KS = [3, 8]


def load() -> dict:
    rows = [json.loads(l) for l in (ROOT / "eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def run(row: dict, k: int, *, use_rerank: bool) -> None:
    tag = f"K={k}　重排{'开' if use_rerank else '关'}"
    t0 = time.perf_counter()
    out = rag_chain.answer(row["question"], top_k=k, use_rerank=use_rerank, use_hybrid=True)
    ms = (time.perf_counter() - t0) * 1000
    print(f"  ── {tag}　检索 {len(out['sources'])} 块　{ms/1000:.1f}s")
    print(f"     拒答 = {out['refused']}　{out['refuse_reason']}")
    print(f"     来源 = {', '.join(s['source'][:16] for s in out['sources'])}")
    print(f"     答案 = {out['answer'].strip()[:640]}")
    print()


def main() -> None:
    ds = load()

    print("=" * 100)
    print("阶段二预演：同一道题在 K=3 与 K=8 下的真实答案对比（混合检索开、重排关）")
    print("=" * 100)
    for qid in GAIN_CASES:
        row = ds[qid]
        print(f"【{qid}】{row['question']}")
        print(f"  参考答案：{row['ground_truth'][:300]}")
        print()
        for k in KS:
            run(row, k, use_rerank=False)

    print("=" * 100)
    print(f"反例（阶段一判定 K 救不回来）：{FAIL_CASES[0]}")
    print("=" * 100)
    row = ds[FAIL_CASES[0]]
    print(f"【{FAIL_CASES[0]}】{row['question']}")
    print(f"  参考答案：{row['ground_truth'][:340]}")
    print()
    for k in KS:
        run(row, k, use_rerank=False)

    print("=" * 100)
    print(f"成本验证：开重排时 K 从 3 调到 8，耗时是否变化（{COST_CASE}）")
    print("=" * 100)
    print(f"  RERANK_CANDIDATES = {config.RERANK_CANDIDATES}，TOP_K = {config.TOP_K}")
    print(f"  预期：pool = max(10, k)，k ≤ 10 时精排前向次数不变 → 耗时应当基本相同")
    print()
    row = ds[COST_CASE]
    for k in KS:
        run(row, k, use_rerank=True)


if __name__ == "__main__":
    main()
