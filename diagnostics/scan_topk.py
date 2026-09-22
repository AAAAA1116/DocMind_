# -*- coding: utf-8 -*-
"""阶段一：TOP_K 召回上限扫描（零 LLM 成本）

目的
----
回答一个问题：**答案块最早出现在第几名？** 扫出所有题的分布，就知道 K 该设多少。
全程不调大模型、不跑 ragas —— 一次检索十几毫秒，20 个 K 档一起算完。

判据为什么必须带「可达性过滤」
------------------------------
参考答案里有相当一部分字串**在语料里根本不存在**：
出题时为了跨文档对比能自解释，把原文的「公司」写成了「北辰实业」；
还写了一些概括语（「共五项：」「三个条件：」）——原文里没有这种句子。

这些 key 永远匹配不上，会把覆盖率的天花板压到 0.67 左右，
让你误以为「检索连 20 块都盖不住答案」。所以先做一遍全局可达性检查，
把不可达的 key 剔出分母 —— 顺便也量化了「评测集有多少水分」。

两条口径
--------
* ``any``   —— 不看来源，只要 top-K 里有块含答案
* ``right`` —— 只算**期望文档**的块（严口径）。两者差距 = 「召回到的是别家的同构条款」

用法
----
    ./.venv/Scripts/python.exe -u ../_smoke_corpus/scan_topk.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent / "DocMind_"
sys.path.insert(0, str(ROOT))

import config                        # noqa: E402
from core import embed_store         # noqa: E402
from core import hybrid              # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"
K_MAX = 20
KEY_MIN = 6
KEY_WIN = 12

COMPANIES = ["北辰实业", "新宝股份", "龙旗科技", "华致酒行", "金诚信", "长川科技", "君逸数码"]

#: 题面没点名公司、但实际答案只在特定文件里的题（q07 三家章程均有；q26/q27 是信披制度章节）
EXPECTED_OVERRIDE: Dict[str, List[str]] = {
    "q07": ["北辰实业", "新宝股份", "龙旗科技"],
    "q26": ["长川科技", "君逸数码"],
    "q27": ["长川科技", "君逸数码"],
}


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"[；。！？\n]|（[一二三四五六七八九十]+）|\([0-9]+\)|[①-⑩]")


def split_keys(gt: str) -> List[str]:
    parts = [p.strip(" ，、;；") for p in _SPLIT_RE.split(gt)]
    return [p for p in parts if len(p) >= 4]


#: PDF 抽取会往正文里塞空格和硬换行（``（一） 内部环境。`` / ``包括 3 名董事``），
#: 不做归一化，任何「连续 N 字」的判据都会被这些空白切断，把命中误判成未命中。
_WS_RE = re.compile(r"[\s\u3000\u200b]+")


def norm(text: str) -> str:
    """去掉所有空白，让跨行/带空格的原文与参考答案能对上。"""
    return _WS_RE.sub("", text)


def key_hit(key: str, text: str) -> bool:
    """key 是否出现在 text 里（两边都已归一化空白）。长 key 允许 12 字滑窗。"""
    key = norm(key)
    text = norm(text)
    if len(key) < KEY_MIN:
        return key in text
    if key in text:
        return True
    if len(key) <= KEY_WIN:
        return False
    return any(key[i:i + KEY_WIN] in text for i in range(len(key) - KEY_WIN + 1))


def lcs_len(a: str, b: str) -> int:
    """最长公共子串长度。作为不依赖「要点切分」的独立命中质量信号。"""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best = v
        prev = cur
    return best


# ---------------------------------------------------------------------------
# 全语料拼接（可达性检查用；比逐块扫快得多）
# ---------------------------------------------------------------------------
_CORPUS_TEXT: str = ""


def corpus_text() -> str:
    global _CORPUS_TEXT
    if not _CORPUS_TEXT:
        _CORPUS_TEXT = norm("\n".join(s.get("content", "") for s in embed_store.all_documents()))
    return _CORPUS_TEXT


_reach_cache: Dict[str, bool] = {}


def reachable(key: str) -> bool:
    """这个要点在**全语料里**能不能找到。找不到 = 出题时写的字面不是原文，剔出分母。"""
    if key not in _reach_cache:
        _reach_cache[key] = key_hit(key, corpus_text())
    return _reach_cache[key]


def coverage(keys: List[str], chunks: List[str]) -> float:
    if not keys:
        return 0.0
    return sum(1 for k in keys if any(key_hit(k, c) for c in chunks)) / len(keys)


def expected_companies(qid: str, text: str) -> List[str]:
    if qid in EXPECTED_OVERRIDE:
        return EXPECTED_OVERRIDE[qid]
    return [c for c in COMPANIES if c in text]


def is_right_source(source: str, expected: List[str]) -> bool:
    return True if not expected else any(c in source for c in expected)


def company_of(source: str) -> str:
    for c in COMPANIES:
        if c in source:
            return c
    return "?"


# ---------------------------------------------------------------------------
def main() -> None:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [r for r in rows if not r.get("should_refuse")]
    neg = [r for r in rows if r.get("should_refuse")]

    print("=" * 96)
    print("阶段一：TOP_K 召回上限扫描（不调大模型，零成本）")
    print("=" * 96)
    print(f"  语料      {embed_store.count()} 块")
    print(f"  评测集    {len(rows)} 条（正例 {len(pos)}，负例 {len(neg)} 不参与）")
    print(f"  候选深度  K_MAX = {K_MAX}")
    print("-" * 96)
    sys.stdout.flush()

    t0 = time.time()
    hybrid.get_index()
    print(f"  BM25 就绪：{hybrid.index_stats()}　预热 {time.time() - t0:.1f}s")
    print("-" * 96)
    sys.stdout.flush()

    # ---- 可达性过滤（先做，才知道题目有多少水分）--------------------------
    print("  可达性检查：参考答案的要点能否在语料里找到")
    print(f"  {'id':<5s} {'要点数':>5s} {'不可达':>5s}   被剔掉的要点（语料里找不到 = 出题时加了原文没有的字）")
    print("  " + "-" * 90)
    total_keys = total_unreach = 0
    per_row: List[Dict[str, Any]] = []

    for i, r in enumerate(pos, 1):
        q = r["question"]
        raw_keys = split_keys(r["ground_truth"])
        keys = [k for k in raw_keys if reachable(k)]
        dropped = [k for k in raw_keys if not reachable(k)]
        total_keys += len(raw_keys)
        total_unreach += len(dropped)
        expected = expected_companies(r["id"], q + r["ground_truth"])
        per_row.append({"id": r["id"], "type": r.get("type", ""), "question": q,
                        "ground_truth": r["ground_truth"], "keys": keys, "expected": expected})

        show = "；".join(d[:26] for d in dropped)
        print(f"  {r['id']:<5s} {len(raw_keys):>5d} {len(dropped):>5d}   {show[:80]}")
        sys.stdout.flush()

    print("  " + "-" * 90)
    print(f"  合计 {total_keys} 个要点，其中 {total_unreach} 个（{total_unreach/max(total_keys,1):.0%}）"
          f"语料里不存在，已剔出分母")
    print(f"  ⇒ 这 {total_unreach/max(total_keys,1):.0%} 就是参考答案相对语料的「不忠实度」。"
          f"ragas 的 context_recall 拿同一份 reference 打分，也会被这部分拉低。")
    print()
    sys.stdout.flush()

    # ---- 检索 ------------------------------------------------------------
    for i, row in enumerate(per_row, 1):
        q = row["question"]
        t = time.perf_counter(); vec = embed_store.search(q, k=K_MAX); t_vec = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); bm = hybrid.bm25_search(q, k=K_MAX); t_bm = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); fused = hybrid.hybrid_search(q, k=K_MAX); t_hy = (time.perf_counter() - t) * 1000

        def pack(hits):
            out = []
            for doc, score in hits:
                md = doc.get("metadata") or {}
                out.append({"source": md.get("source") or "?", "content": doc.get("content", ""),
                            "score": float(score)})
            return out

        row["vec"] = pack(vec)
        row["bm"] = pack(bm)
        row["fused"] = [{"source": rec["doc"].get("metadata", {}).get("source") or "?",
                         "content": rec["doc"].get("content", "")} for rec in fused]
        row["ms"] = {"vec": round(t_vec, 1), "bm25": round(t_bm, 1), "hybrid": round(t_hy, 1)}
        print(f"  [{i}/{len(per_row)}] {row['id']} [{row['type']}] rrf={t_hy:.0f}ms")
        sys.stdout.flush()

    print("-" * 96)
    print(f"  检索完成，总用时 {time.time() - t0:.1f}s")
    print()
    sys.stdout.flush()

    # ---- 逐条明细 --------------------------------------------------------
    print("=" * 96)
    print("逐条明细（★ = 期望文档的块；序号 = 融合名次）")
    print("=" * 96)
    for row in per_row:
        flags = []
        for j, c in enumerate(row["fused"], 1):
            star = "★" if is_right_source(c["source"], row["expected"]) else "·"
            flags.append(f"{j}{star}{company_of(c['source'])[:2]}")
        print(f"  {row['id']} [{row['type']}] 期望={','.join(row['expected'])}　要点 {len(row['keys'])} 个")
        print(f"      {' '.join(flags)}")
    print()

    # ---- 覆盖率曲线 -------------------------------------------------------
    def curve(field: str, right_only: bool) -> List[float]:
        out = []
        for K in range(1, K_MAX + 1):
            covs = []
            for row in per_row:
                if not row["keys"]:              # 参考答案与原文对不上，无法评估，不进分母
                    continue
                chunks = [c["content"] for c in row[field][:K]
                          if (not right_only) or is_right_source(c["source"], row["expected"])]
                covs.append(coverage(row["keys"], chunks))
            out.append(sum(covs) / len(covs) if covs else 0.0)
        return out

    print("=" * 96)
    print(f"平均要点覆盖率 @K（{len(per_row)} 条正例均值，只算语料里可达的要点）")
    print("=" * 96)
    print(f"  {'K':>3s}  {'RRF融合':>9s} {'纯向量':>9s} {'纯BM25':>9s}   {'RRF仅期望文档':>13s}   曲线")
    print("  " + "-" * 88)
    c_rrf = curve("fused", False)
    c_vec = curve("vec", False)
    c_bm = curve("bm", False)
    c_rrf_r = curve("fused", True)
    for i, K in enumerate(range(1, K_MAX + 1)):
        gain = c_rrf[i] - c_rrf[i - 1] if i else c_rrf[i]
        mark = "  ← 边际 <0.01，再往上加没用了" if 0 < gain < 0.01 and i > 2 else ""
        bar = "█" * int(c_rrf[i] * 34)
        print(f"  {K:>3d}  {c_rrf[i]:>9.4f} {c_vec[i]:>9.4f} {c_bm[i]:>9.4f}   {c_rrf_r[i]:>13.4f}   {bar}{mark}")
    print()

    # ---- 独立信号：最长公共子串（不依赖「要点切分」这个人为口径）------------
    print("=" * 96)
    print("独立信号：单个块与参考答案的最长公共子串长度（看「命中有多实」）")
    print("=" * 96)
    print(f"  {'id':<5s} {'top-1':>6s} {'top-2':>6s} {'top-3':>6s}  {'gt长度':>6s}   判定")
    print("  " + "-" * 88)
    strong = weak = miss = 0
    for row in per_row:
        gt = row["ground_truth"]
        vals = [lcs_len(c["content"], gt) for c in row["fused"][:3]]
        while len(vals) < 3:
            vals.append(0)
        verdict = "强命中" if vals[0] >= 20 else ("弱命中" if vals[0] >= 12 else "几乎没命中")
        strong += verdict == "强命中"; weak += verdict == "弱命中"; miss += verdict == "几乎没命中"
        print(f"  {row['id']:<5s} {vals[0]:>6d} {vals[1]:>6d} {vals[2]:>6d}  {len(norm(gt)):>6d}   {verdict}")
    print("  " + "-" * 88)
    print(f"  top-1 强命中 {strong} 条 / 弱命中 {weak} 条 / 几乎没命中 {miss} 条")
    print()

    # ---- 首次满分所需名次 --------------------------------------------------
    print("=" * 96)
    print("每条题「要多少块才能答全」（可达要点 100% 覆盖的名次；>20 = 20 块都不够）")
    print("=" * 96)
    need: List[Tuple[str, str, int, int, int]] = []
    for row in per_row:
        if not row["keys"]:                      # 0 = 无法评估（参考答案里没有能对上原文的要点）
            need.append((row["id"], row["type"], 0, 0, 0))
            continue

        def first_full(field: str) -> int:
            for K in range(1, K_MAX + 1):
                if coverage(row["keys"], [c["content"] for c in row[field][:K]]) >= 0.999:
                    return K
            return 999                  # 999 = 20 块都不够（真·覆盖不足）
        need.append((row["id"], row["type"], first_full("fused"), first_full("vec"), first_full("bm")))

    def show(v: int) -> str:
        return "n/a" if v == 0 else (">20" if v == 999 else str(v))

    print(f"  {'id':<5s} {'type':<5s} {'RRF':>5s} {'向量':>5s} {'BM25':>5s}   题面")
    print("  " + "-" * 88)
    for nid, ntype, a, b, c in need:
        q = next(x["question"] for x in per_row if x["id"] == nid)
        print(f"  {nid:<5s} {ntype:<5s} {show(a):>5s} {show(b):>5s} {show(c):>5s}   {q[:44]}")
    print("  " + "-" * 88)
    for label, idx in (("RRF融合", 2), ("纯向量", 3), ("纯BM25", 4)):
        vals = [n[idx] for n in need if n[idx] != 0]          # 只在可评估的题上统计
        cover = lambda K: sum(1 for v in vals if v <= K) / len(vals)
        ok = sorted(v for v in vals if v != 999)
        print(f"  {label:<8s} 中位数={ok[len(ok)//2] if ok else '—'}　"
              f"K=3:{cover(3):.0%}  K=5:{cover(5):.0%}  K=8:{cover(8):.0%}  K=12:{cover(12):.0%}  "
              f"K=20:{cover(20):.0%}　（20块还不够 {vals.count(999)} 条）")
    na = [n for n in need if n[2] == 0]
    print(f"  可评估 {len(need) - len(na)} 条；n/a {len(na)} 条（参考答案与原文对不上，需人工或 LLM 判）")
    for nid, ntype, *_ in na:
        q = next(x["question"] for x in per_row if x["id"] == nid)
        print(f"    {nid} [{ntype}] {q[:54]}")
    print()
    print()

    # ---- TOP_K=3 vs 更大 K 的差距 ----------------------------------------
    K0 = config.TOP_K
    print("=" * 96)
    print(f"把 TOP_K 从 {K0} 调大，能救回哪些题（覆盖率从 <1 变成 1）")
    print("=" * 96)
    gainers = [n for n in need if n[2] not in (0, 999) and K0 < n[2] <= 12]
    for nid, ntype, a, b, c in sorted(gainers, key=lambda x: x[2]):
        q = next(x["question"] for x in per_row if x["id"] == nid)
        print(f"  {nid} [{ntype}]  K={K0} → {a} 可答全　{q[:52]}")
    still = [n for n in need if n[2] == 999]
    print()
    print(f"  调大 K 也救不回来的（{len(still)} 条）—— 这些是**别的病**，不是 TOP_K 的问题：")
    for nid, ntype, a, b, c in still:
        q = next(x["question"] for x in per_row if x["id"] == nid)
        print(f"    {nid} [{ntype}]  {q[:56]}")
    print()

    out = ROOT.parent / "_smoke_corpus" / "scan_topk_result.json"
    out.write_text(json.dumps({
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "k_max": K_MAX, "key_win": KEY_WIN, "index_count": embed_store.count(),
        "unreachable": {"total": total_keys, "dropped": total_unreach},
        "curve": {"rrf": c_rrf, "vector": c_vec, "bm25": c_bm, "rrf_right_only": c_rrf_r},
        "need_k": [{"id": a, "type": b, "rrf": c, "vector": d, "bm25": e} for a, b, c, d, e in need],
        "rows": per_row,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已落盘：{out}")


if __name__ == "__main__":
    main()
