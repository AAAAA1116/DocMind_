# -*- coding: utf-8 -*-
"""身份维度：B 方案之后还差多少？C 方案能补多少？A 方案值不值得做？

背景
----
块内容里已经嵌了 `《文件名》`（方案 B，1 行改动，已生效）。本脚本要回答：
**B 之后，身份这一维还剩多少缺口？补它值不值？**

三个口径（全部从**同一次检索**的结果里离线重排，零额外检索、零 LLM 调用）：

    any    —— 不看来源，top-K 里有块含该原子。= 现状 B 的内容锚能力
    right  —— 只保留「期望文档」的块，再取 top-K。= 方案 A（硬过滤）的上限
    idfir  —— 把「期望文档」的块整体提前（stable partition），再取 top-K。
              = 方案 C（身份优先重排）。不删任何候选，只改顺序

为什么 idfir 是 A 的软化版
--------------------------
硬过滤（A）的风险是「把正确答案筛掉」；而 idfir 不删候选，只是在顺序上优待
匹配块 —— 匹配块不足 k 时自动用其余块补齐，**最坏情况等于没加**。
所以先看 idfir：如果它能逼近 right，就没必要上 A。

同时统计「前置步骤可不可靠」：方案 A/C 都要从**查询文本**里抽出实体，
这一层的准确率决定它们的天花板。

用法
----
    ./.venv/Scripts/python.exe -u ../_smoke_corpus/probe_identity.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent / "DocMind_"
sys.path.insert(0, str(ROOT))

import config                        # noqa: E402
from core import embed_store, hybrid  # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"
K_MAX = 8
KEY_MIN = 6
KEY_WIN = 12

COMPANIES = ["北辰实业", "新宝股份", "龙旗科技", "华致酒行", "金诚信", "长川科技", "君逸数码"]

#: ⚠️ 别名表 —— 不修这个，right 口径会退化成 any 口径。
#: 题面里的公司名**写法/简称和正式名对不上**：q05 写的是「广东新宝电器股份有限公司」，
#: 用词表里的「新宝股份」去匹配**匹配不到** → expected 变成空集 →
#: ``is_right()`` 对空集恒返回 True → 该题的 right 口径 == any 口径，
#: **把这一条的身份问题整体吞掉**，看板上却显示「身份缺口 0」。
_ALIASES: Dict[str, List[str]] = {
    "北辰实业": ["北辰实业", "北辰"],
    "新宝股份": ["新宝股份", "新宝电器", "新宝"],
    "龙旗科技": ["龙旗科技", "龙旗"],
    "华致酒行": ["华致酒行", "华致"],
    "金诚信": ["金诚信"],
    "长川科技": ["长川科技", "长川"],
    "君逸数码": ["君逸数码", "君逸"],
}


def match_companies(text: str) -> List[str]:
    """从文本里认出提到了哪几家公司（全称和简称都认）。"""
    return [c for c, aliases in _ALIASES.items() if any(a in text for a in aliases)]


EXPECTED_OVERRIDE: Dict[str, List[str]] = {
    "q07": ["北辰实业", "新宝股份", "龙旗科技"],
    "q26": ["长川科技", "君逸数码"],
    "q27": ["长川科技", "君逸数码"],
}

_SPLIT_RE = re.compile(r"[；。！？\n]|（[一二三四五六七八九十]+）|\([0-9]+\)|[①-⑩]")
_WS_RE = re.compile(r"[\s\u3000\u200b]+")


def split_keys(gt: str) -> List[str]:
    parts = [p.strip(" ，、;；") for p in _SPLIT_RE.split(gt)]
    return [p for p in parts if len(p) >= 4]


def norm(text: str) -> str:
    return _WS_RE.sub("", text)


def key_hit(key: str, text: str) -> bool:
    key = norm(key)
    text = norm(text)
    if len(key) < KEY_MIN:
        return key in text
    if key in text:
        return True
    if len(key) <= KEY_WIN:
        return False
    return any(key[i:i + KEY_WIN] in text for i in range(len(key) - KEY_WIN + 1))


_CORPUS = ""


def corpus_text() -> str:
    global _CORPUS
    if not _CORPUS:
        _CORPUS = norm("\n".join(d.get("content", "") for d in embed_store.all_documents()))
    return _CORPUS


_reach: Dict[str, bool] = {}


def reachable(key: str) -> bool:
    if key not in _reach:
        _reach[key] = key_hit(key, corpus_text())
    return _reach[key]


def expected_companies(qid: str, text: str) -> List[str]:
    if qid in EXPECTED_OVERRIDE:
        return EXPECTED_OVERRIDE[qid]
    return match_companies(text)


def is_right(source: str, expected: List[str]) -> bool:
    return True if not expected else any(c in source for c in expected)


def bar(v: float, width: int = 26) -> str:
    return "█" * int(round(v * width))


def main() -> None:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [r for r in rows if not r.get("should_refuse")]

    print("=" * 100)
    print("身份维度：B（块内嵌文件名）之后还剩多少缺口，C（身份优先重排）能补多少")
    print("=" * 100)
    print(f"  语料      {embed_store.count()} 块 / {len({(d.get('metadata') or {}).get('source') for d in embed_store.all_documents()})} 个文件")
    print(f"  评测集    {len(rows)} 条（正例 {len(pos)}）")
    print(f"  当前 TOP_K = {config.TOP_K}　（方案 A=硬过滤 / C=身份优先重排，全部离线模拟）")
    print("-" * 100)
    sys.stdout.flush()

    t0 = time.time()
    hybrid.get_index()
    print(f"  BM25 就绪：{hybrid.index_stats()}　（{time.time() - t0:.1f}s）")
    print()
    sys.stdout.flush()

    per_row: List[Dict[str, Any]] = []
    for r in pos:
        q = r["question"]
        raw_keys = split_keys(r["ground_truth"])
        keys = [k for k in raw_keys if reachable(k)]
        expected = expected_companies(r["id"], q + r["ground_truth"])

        records = hybrid.hybrid_search(q, k=K_MAX)
        order = [{"source": (rec["doc"].get("metadata") or {}).get("source") or "?",
                  "content": rec["doc"].get("content", "")} for rec in records]

        if not keys:
            per_row.append({"id": r["id"], "type": r.get("type", ""), "question": q,
                            "keys": [], "expected": expected, "order": order,
                            "mentioned": match_companies(q), "na": True})
            print(f"  [{r['id']}] n/a（参考答案无可达要点）")
            sys.stdout.flush()
            continue

        per_row.append({"id": r["id"], "type": r.get("type", ""), "question": q,
                        "keys": keys, "expected": expected, "order": order,
                        "mentioned": match_companies(q), "na": False})
        print(f"  [{r['id']}] rrf top-{K_MAX}：{' '.join(str(i + 1) + ('★' if is_right(c['source'], expected) else '·') for i, c in enumerate(order))}")
        sys.stdout.flush()

    print()
    evaluable = [p for p in per_row if not p["na"]]
    print(f"  可评估 {len(evaluable)} 条，n/a {len(per_row) - len(evaluable)} 条")
    print()
    sys.stdout.flush()

    # ---- 三个口径的覆盖率曲线 ---------------------------------------------
    def cover(order: List[Dict[str, Any]], keys: List[str], K: int,
              *, right_only: bool, expected: List[str]) -> float:
        chunks = []
        for c in order[:K]:
            if right_only and not is_right(c["source"], expected):
                continue
            chunks.append(c["content"])
        return sum(1 for k in keys if any(key_hit(k, ch) for ch in chunks)) / len(keys)

    def idfir_order(order: List[Dict[str, Any]], expected: List[str]) -> List[Dict[str, Any]]:
        """期望文档的块整体提前，其余保持原序（stable partition）。"""
        return ([c for c in order if is_right(c["source"], expected)]
                + [c for c in order if not is_right(c["source"], expected)])

    curves = {"any": [], "right": [], "idfir": []}
    for K in range(1, K_MAX + 1):
        for name in curves:
            vals = []
            for p in evaluable:
                if name == "any":
                    vals.append(cover(p["order"], p["keys"], K, right_only=False, expected=p["expected"]))
                elif name == "right":
                    vals.append(cover(p["order"], p["keys"], K, right_only=True, expected=p["expected"]))
                else:
                    vals.append(cover(idfir_order(p["order"], p["expected"]), p["keys"], K,
                                      right_only=False, expected=p["expected"]))
            curves[name].append(sum(vals) / len(vals) if vals else 0.0)

    print("=" * 100)
    print("三个口径的 top-K 要点覆盖率（27 条正例均值）")
    print("=" * 100)
    print("  any   = 现状（B 之后，不看来源）       right = 方案 A 硬过滤的上限")
    print("  idfir = 方案 C 身份优先重排（不删候选）  ★ = 当前 TOP_K")
    print()
    print(f"  {'K':>3s}  {'any(B现状)':>10s}  {'right(A上限)':>12s}  {'idfir(C)':>9s}   身份缺口(any-right)")
    print("  " + "-" * 90)
    for i, K in enumerate(range(1, K_MAX + 1)):
        a, rr, d = curves["any"][i], curves["right"][i], curves["idfir"][i]
        mark = "  ★ 当前" if K == config.TOP_K else ""
        print(f"  {K:>3d}  {a:>10.4f}  {rr:>12.4f}  {d:>9.4f}   {a - rr:>+8.4f}{mark}")
    print("  " + "-" * 90)
    kk = config.TOP_K
    print(f"  ⇒ K={kk}：现状 {curves['any'][kk-1]:.4f}　C 方案 {curves['idfir'][kk-1]:.4f}　"
          f"A 上限 {curves['right'][kk-1]:.4f}")
    print(f"  ⇒ C 相对现状 {(curves['idfir'][kk-1] - curves['any'][kk-1]):+.4f}；"
          f"A 相对现状 {(curves['right'][kk-1] - curves['any'][kk-1]):+.4f}")
    gap = curves['right'][kk - 1] - curves['any'][kk - 1]
    if gap > 1e-9:
        eaten = (curves['idfir'][kk - 1] - curves['any'][kk - 1]) / gap
        print(f"  ⇒ C 吃掉了 A 上限的 {eaten:.0%}")
    else:
        print("  ⇒ A 的上限不高于现状：硬过滤没有可补的空间")

    # ---- 直接看「混进别家的块」有多少：这是 A/C 能作用的最大空间 -----------
    tot_chunks = tot_foreign = 0
    for p in evaluable:
        for c in p["order"]:
            tot_chunks += 1
            if not is_right(c["source"], p["expected"]):
                tot_foreign += 1
    print(f"  ⇒ 检索 top-{K_MAX} 的 {tot_chunks} 个候选块里，来自**别家文档**的有 {tot_foreign} 个"
          f"（{tot_foreign / max(tot_chunks, 1):.1%}）")
    print("     这就是 A/C 能作用的最大空间 —— 过滤和重排都只能动这一部分。")
    print()
    sys.stdout.flush()

    # ---- C 的副作用 -------------------------------------------------------
    print("=" * 100)
    print("方案 C 的副作用：有没有哪条题**变差**了（任何改动都必须先看这个）")
    print("=" * 100)
    worse, better, same = [], [], []
    for p in evaluable:
        a = cover(p["order"], p["keys"], K_MAX, right_only=False, expected=p["expected"])
        d = cover(idfir_order(p["order"], p["expected"]), p["keys"], K_MAX,
                  right_only=False, expected=p["expected"])
        (better if d > a + 1e-9 else (worse if d < a - 1e-9 else same)).append((p, a, d))
    print(f"  K={K_MAX}（看全候选池，排除「只差名次」的干扰）")
    print(f"    变好 {len(better)} 条　不变 {len(same)} 条　**变差 {len(worse)} 条**")
    for p, a, d in worse:
        print(f"      {p['id']} {a:.2f} → {d:.2f}　{p['question'][:52]}")
    if better:
        print("    变好的：")
        for p, a, d in better:
            print(f"      {p['id']} {a:.2f} → {d:.2f}　{p['question'][:52]}")
    print()
    sys.stdout.flush()

    # ---- 前置步骤可不可靠 -------------------------------------------------
    print("=" * 100)
    print("前置步骤：方案 A/C 都要「从查询文本里抽出实体」，这一层有多可靠")
    print("=" * 100)
    exact = partial = none_ = 0
    none_list: List[str] = []
    for p in per_row:
        m, e = set(p["mentioned"]), set(p["expected"])
        if not m:
            none_ += 1
            none_list.append(f"{p['id']}（期望 {','.join(e) or '未标注'}）")
        elif m == e:
            exact += 1
        else:
            partial += 1
    total = len(per_row)
    print(f"  题面公司名 == 期望来源（可精确过滤）  {exact:>2d} / {total} = {exact/total:.0%}")
    print(f"  题面公司名 ⊂ 期望来源（要放宽）      {partial:>2d} / {total} = {partial/total:.0%}")
    print(f"  **题面没提公司名（不能过滤）**        {none_:>2d} / {total} = {none_/total:.0%}")
    if none_list:
        print(f"      {('、'.join(none_list))[:88]}")
    print()
    print("  ⇒ 这一层的不可靠性是 A/C 的**共同上限**：题面没提实体时，两者都退化成现状。")
    print()
    sys.stdout.flush()

    # ---- 逐条明细 ---------------------------------------------------------
    print("=" * 100)
    print(f"逐条明细（K={config.TOP_K}；★ = 期望文档的块）")
    print("=" * 100)
    print(f"  {'id':<5s} {'type':<5s} {'原子':>4s} {'any':>6s} {'right':>6s} {'idfir':>6s} {'别家块':>7s}   题面")
    print("  " + "-" * 92)
    for p in per_row:
        if p["na"]:
            print(f"  {p['id']:<5s} {p['type']:<5s}   n/a      —      —      —       —   {p['question'][:40]}")
            continue
        kk = config.TOP_K
        a = cover(p["order"], p["keys"], kk, right_only=False, expected=p["expected"])
        rr = cover(p["order"], p["keys"], kk, right_only=True, expected=p["expected"])
        d = cover(idfir_order(p["order"], p["expected"]), p["keys"], kk,
                  right_only=False, expected=p["expected"])
        foreign = sum(1 for c in p["order"] if not is_right(c["source"], p["expected"]))
        flag = "  ← 身份缺口" if a - rr > 0.3 else ""
        print(f"  {p['id']:<5s} {p['type']:<5s} {len(p['keys']):>4d} {a:>6.2f} {rr:>6.2f} {d:>6.2f} "
              f"{foreign:>3d}/{K_MAX:<3d}   {p['question'][:36]}{flag}")
    print()

    out = ROOT.parent / "_smoke_corpus" / "probe_identity_result.json"
    out.write_text(json.dumps({
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": config.TOP_K, "k_max": K_MAX,
        "curves": curves,
        "side_effect": {"better": len(better), "same": len(same), "worse": len(worse)},
        "entity_extraction": {"exact": exact, "partial": partial, "none": none_, "total": total},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已落盘：{out}")
    print(f"总用时 {time.time() - t0:.1f}s（零 LLM 调用）")


if __name__ == "__main__":
    main()
