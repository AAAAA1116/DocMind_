# -*- coding: utf-8 -*-
"""把「判定」拆到原子级：RAG 到底在判什么？

动机
----
这几轮里出现过一堆看起来毫不相干的判定：

    TOP_K 覆盖率扫描      —— 要点是否落在 top-K 里
    context_recall        —— 参考答案是否被检索上下文支持
    context_precision     —— 检索到的块是否与问题相关
    faithfulness          —— 答案的每句是否被 context 支持
    REFUSAL_TEXT in text  —— 是否吐出了那句固定话术
    hdr_ratio             —— 某行是否跨页高频重复
    BM25 召回             —— 查询词项与块是否有交集
    页眉识别              —— 首行是否在多页重复

它们的形式是同一个：

    Judge(x, D) = ∃ y ∈ D,  y ⪰ x

x = 一个**原子断言**（不可再分的最小可验证事实），D = 材料集合，
⪰ = 「能充当 x 的出处」。判定全是「存在性」问题，差别只在 ⪰ 选得多严。
所以「这个东西」= 原子断言 + 它的锚定范围 —— 也就是**证据锚**。

本脚本做两件事，把它从抽象变成可计数的东西
--------------------------------------------
1. **判定强度谱系**：把参考答案拆成原子，逐个在**全语料**上判它需要多强的
   判定器才能成立（L0 逐字 / L1 归一化 / L2 片段 / L3 仅词汇 / L4 字面不可达）。
   落到 L0~L1 的原子，字面判定就够；落到 L4 的，只有语义/LLM 能判。

2. **锚定唯一性**（⪰ 的第二个正交维度）：同一个原子在语料里能命中**几份文档**。
   命中 1 份 = 唯一锚定，判定即完成；命中 ≥2 份 = 同构重述，
   **内容判对了也没用，必须再判「是不是同一个对象」** —— 这正是 q02 那个 bug。

全程零 LLM 调用，纯字符串分析，几秒钟跑完。

用法
----
    ./.venv/Scripts/python.exe -u ../_smoke_corpus/probe_evidence.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent / "DocMind_"
sys.path.insert(0, str(ROOT))

import config                        # noqa: E402
from core import embed_store         # noqa: E402

DATASET = ROOT / "eval" / "dataset.jsonl"

COMPANIES = ["北辰实业", "新宝股份", "龙旗科技", "华致酒行", "金诚信", "长川科技", "君逸数码"]

_SPLIT_RE = re.compile(r"[；。！？\n]|（[一二三四五六七八九十]+）|\([0-9]+\)|[①-⑩]")
_WS_RE = re.compile(r"[\s\u3000\u200b]+")
_NUM_RE = re.compile(r"\d")

WIN = 8                 # L2 片段窗口
BIGRAM_MIN = 4          # L3 最少公共 bigram 数
BIGRAM_RATIO = 0.5      # L3 公共 bigram 占 key 的比例下限

LEVEL_NAME = {0: "L0 逐字", 1: "L1 归一化", 2: "L2 片段", 3: "L3 仅词汇", 4: "L4 字面不可达"}
LEVEL_DESC = {
    0: "原样照抄，连空白都对",
    1: "去空白后连续命中（PDF 抽取噪声，非内容缺失）",
    2: "无连续整串，但有 8 字连续片段（标点/语序有变）",
    3: "连 8 字片段都没有，只剩词汇重叠",
    4: "全语料无字面依据，只有语义/LLM 可能救",
}
#: 哪些档位算「字面判定器就能搞定」
LITERAL = {0, 1, 2}


def split_keys(gt: str) -> List[str]:
    parts = [p.strip(" ，、;；") for p in _SPLIT_RE.split(gt)]
    return [p for p in parts if len(p) >= 4]


def norm(text: str) -> str:
    return _WS_RE.sub("", text)


def bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()


def classify(key_raw: str, key_norm: str, c_raw: str, c_norm: str) -> Optional[int]:
    """这个原子在这块里最强能判到哪一档；None = 这块完全不含它。"""
    if key_raw in c_raw:
        return 0
    if not key_norm:
        return None
    if key_norm in c_norm:
        return 1
    if len(key_norm) >= WIN:
        for i in range(len(key_norm) - WIN + 1):
            if key_norm[i:i + WIN] in c_norm:
                return 2
    kb = bigrams(key_norm)
    if kb:
        inter = len(kb & bigrams(c_norm))
        if inter >= BIGRAM_MIN and inter / len(kb) >= BIGRAM_RATIO:
            return 3
    return None


def bar(v: float, width: int = 30) -> str:
    return "█" * int(round(v * width))


# ---------------------------------------------------------------------------
def main() -> None:
    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    pos = [r for r in rows if not r.get("should_refuse")]
    neg = [r for r in rows if r.get("should_refuse")]

    print("=" * 100)
    print("把「判定」拆到原子级：RAG 到底在判什么")
    print("=" * 100)
    print(f"  语料集   {embed_store.count()} 块")
    print(f"  评测集   {len(rows)} 条（正例 {len(pos)}，负例 {len(neg)} 不参与）")
    print("  判定式   Judge(x, D) = ∃ y ∈ D, y ⪰ x   （x = 原子断言，⪰ = 能否充当出处）")
    print("-" * 100)
    sys.stdout.flush()

    t0 = time.time()
    docs = embed_store.all_documents()
    chunks: List[Tuple[str, str, str]] = []          # (source, raw, norm)
    for d in docs:
        md = d.get("metadata") or {}
        raw = d.get("content", "") or ""
        chunks.append((md.get("source") or "?", raw, norm(raw)))
    n_chunks = len(chunks)
    print(f"  已载入 {n_chunks} 块 / {len({c[0] for c in chunks})} 个文件　（{time.time() - t0:.1f}s）")
    print()
    sys.stdout.flush()

    # ---- 逐原子判定 -------------------------------------------------------
    atoms: List[Dict[str, Any]] = []
    for r in pos:
        for raw_key in split_keys(r["ground_truth"]):
            kn = norm(raw_key)
            levels: Dict[int, List[str]] = defaultdict(list)
            for src, craw, cnorm in chunks:
                lvl = classify(raw_key, kn, craw, cnorm)
                if lvl is not None:
                    levels[lvl].append(src)
            best = min(levels) if levels else 4
            docs_hit = set(levels.get(best, []))
            atoms.append({
                "qid": r["id"], "type": r.get("type", ""), "key": raw_key,
                "level": best, "n_docs": len(docs_hit),
                "n_chunks": len(levels.get(best, [])),
                "docs": sorted(docs_hit),
                "has_company": any(c in raw_key for c in COMPANIES),
                "has_num": bool(_NUM_RE.search(raw_key)),
            })

    n_atoms = len(atoms)
    print("=" * 100)
    print(f"【段 A】判定强度谱系 —— {n_atoms} 个原子，各需要多强的判定器才能成立")
    print("=" * 100)
    lv = Counter(a["level"] for a in atoms)
    print(f"  {'档位':<14s} {'原子数':>5s} {'占比':>7s}  {'':<30s}  说明")
    print("  " + "-" * 92)
    for k in range(5):
        v = lv.get(k, 0)
        p = v / max(n_atoms, 1)
        print(f"  {LEVEL_NAME[k]:<14s} {v:>5d} {p:>6.1%}  {bar(p):<30s}  {LEVEL_DESC[k]}")
    print("  " + "-" * 92)
    lit = sum(lv.get(k, 0) for k in LITERAL)
    print(f"  ⇒ 字面判定器（L0~L2）能搞定的原子：{lit}/{n_atoms} = {lit / max(n_atoms,1):.1%}")
    print(f"  ⇒ 必须上语义/LLM 的原子（L3+L4）：{n_atoms - lit}/{n_atoms} = {(n_atoms - lit) / max(n_atoms,1):.1%}")
    print()
    sys.stdout.flush()

    # ---- 锚定唯一性 -------------------------------------------------------
    print("=" * 100)
    print("【段 B】锚定唯一性 —— ⪰ 的第二个正交维度：这个原子在全语料里能定位到几份文档")
    print("=" * 100)
    uni = [a for a in atoms if a["n_docs"] == 1]
    multi = [a for a in atoms if a["n_docs"] >= 2]
    dead = [a for a in atoms if a["n_docs"] == 0]
    print(f"  {'情形':<18s} {'原子数':>5s} {'占比':>7s}  {'':<30s}  含义")
    print("  " + "-" * 92)
    for nm, arr, desc in (
        ("唯一锚定 (1 份)", uni, "内容判对即完成，判定是 1 维的"),
        ("多文档重述 (≥2 份)", multi, "同构条款互相冒充，必须再判「是不是同一个对象」"),
        ("不可达 (0 份)", dead, "语料里根本没有，任何判定器都判不了 = 判据水分"),
    ):
        p = len(arr) / max(n_atoms, 1)
        print(f"  {nm:<18s} {len(arr):>5d} {p:>6.1%}  {bar(p):<30s}  {desc}")
    print("  " + "-" * 92)
    if multi:
        mc = Counter(a["n_docs"] for a in multi)
        print("  多文档重述的扩散范围：" + "　".join(f"{k}份×{v}个" for k, v in sorted(mc.items())))
        print(f"  最严重的一个原子出现在 {max(a['n_docs'] for a in multi)} 份文档里")
    print()
    sys.stdout.flush()

    # ---- 交叉：身份维度到底缺不缺 -----------------------------------------
    print("=" * 100)
    print("【段 C】身份缺口 —— 多文档重述的原子中，有多少**自己不带公司名**（无法自锚定）")
    print("=" * 100)
    multi_named = [a for a in multi if a["has_company"]]
    multi_anon = [a for a in multi if not a["has_company"]]
    print(f"  多文档重述共 {len(multi)} 个。其中：")
    print(f"    自带公司名（能自锚定）    {len(multi_named):>5d}  {len(multi_named)/max(len(multi),1):>6.1%}")
    print(f"    不含公司名（必须靠 metadata）{len(multi_anon):>5d}  {len(multi_anon)/max(len(multi),1):>6.1%}  ← 身份缺口")
    print()
    if multi_anon:
        print("  举例（这些原子字面一模一样地出现在多家文档里，而它们本身不含任何公司名）：")
        for a in multi_anon[:6]:
            print(f"    {a['qid']} {a['key'][:44]:<46s} 出现在 {a['n_docs']} 份："
                  f"{'、'.join(d.split('_')[1] if '_' in d else d[:8] for d in a['docs'][:4])}")
    print()
    sys.stdout.flush()

    # ---- 交叉：原子类型 ---------------------------------------------------
    print("=" * 100)
    print("【段 D】原子类型 × 判定档 —— 「事实常量」和「叙述」是不是同一类问题")
    print("=" * 100)
    for kind, pred in (("事实常量型（含数字或公司名）", lambda a: a["has_num"] or a["has_company"]),
                       ("叙述型（纯文字描述）", lambda a: not (a["has_num"] or a["has_company"]))):
        arr = [a for a in atoms if pred(a)]
        if not arr:
            continue
        lvc = Counter(a["level"] for a in arr)
        lits = sum(lvc.get(k, 0) for k in LITERAL)
        print(f"  {kind}　共 {len(arr)} 个（{len(arr)/max(n_atoms,1):.1%}）")
        dist = "　".join(f"{LEVEL_NAME[k].split()[0]}:{lvc.get(k,0)}" for k in range(5))
        print(f"      {dist}")
        print(f"      其中字面可判 {lits}/{len(arr)} = {lits/len(arr):.1%}"
              f"　多文档重述 {sum(1 for a in arr if a['n_docs'] >= 2)/len(arr):.1%}"
              f"　不可达 {sum(1 for a in arr if a['n_docs'] == 0)/len(arr):.1%}")
    print()
    sys.stdout.flush()

    # ---- 段 E：字面唯一 ≠ 语义唯一 ----------------------------------------
    print("=" * 100)
    print("【段 E】字面判对、语义判错 —— 换个判定器，同一个原子会不会锚到别家去")
    print("=" * 100)
    print(f"  对 {len(uni)} 个**字面唯一锚定**的原子各做一次向量检索 top-5。")
    print("  这些原子字面上只存在于一份文档里，所以「唯一来源」就是标准答案。")
    print("  看语义判定器的 top-1 会不会跑偏 —— 跑偏率 = 语义判定器的天然假阳性率。")
    print("  （首次调用要加载 embedding 模型，约 1 分钟）")
    print("-" * 100)
    sys.stdout.flush()

    t1 = time.time()
    wrong: List[Dict[str, Any]] = []
    sem_docs_dist: Counter = Counter()
    for i, a in enumerate(uni, 1):
        hits = embed_store.search(a["key"], k=5)
        srcs = [(h[0].get("metadata") or {}).get("source") or "?" for h in hits]
        a["sem_top1"] = srcs[0] if srcs else "?"
        a["sem_docs"] = len(set(srcs))
        a["sem_right"] = (a["sem_top1"] == a["docs"][0])
        sem_docs_dist[a["sem_docs"]] += 1
        if not a["sem_right"]:
            wrong.append(a)
        if i % 12 == 0:
            print(f"    ...{i}/{len(uni)}")
            sys.stdout.flush()

    n_uni = len(uni)
    print("  " + "-" * 92)
    print(f"  语义 top-1 命中正确来源：{n_uni - len(wrong)}/{n_uni} = {(n_uni - len(wrong))/max(n_uni,1):.1%}")
    print(f"  语义 top-1 **跑到别家去**：{len(wrong)}/{n_uni} = {len(wrong)/max(n_uni,1):.1%}  ← 语义判定器的假阳性率")
    print(f"  语义 top-5 里混进的不同文档数分布：" + "　".join(f"{k}份×{v}个" for k, v in sorted(sem_docs_dist.items())))
    print()
    if wrong:
        print("  跑偏的原子（字面唯一，语义却锚到别家）：")
        for a in wrong[:12]:
            exp = a["docs"][0].split("_")[1] if "_" in a["docs"][0] else a["docs"][0][:10]
            got = a["sem_top1"].split("_")[1] if "_" in a["sem_top1"] else a["sem_top1"][:10]
            print(f"    {a['qid']} 期望【{exp}】→ 语义给【{got}】  {a['key'][:40]}")
    print()
    print(f"  ⇒ 同一批原子、同一份语料：字面判定器 0% 跑偏，语义判定器 {len(wrong)/max(n_uni,1):.0%} 跑偏。")
    print("    ⪰ 放宽不是「更宽容地接受正确答案」，是「开始接受错误的出处」。")
    print(f"  （检索耗时 {time.time() - t1:.1f}s）")
    print()
    sys.stdout.flush()

    # ---- 逐条明细 ---------------------------------------------------------
    print("=" * 100)
    print("【明细】每条题的原子的落档与锚定（U=唯一 M=多文档 X=不可达）")
    print("=" * 100)
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in atoms:
        by_q[a["qid"]].append(a)
    for r in pos:
        arr = by_q.get(r["id"], [])
        if not arr:
            continue
        tag = "".join(
            "U" if a["n_docs"] == 1 else ("M" if a["n_docs"] >= 2 else "X") for a in arr)
        lvls = "".join(str(a["level"]) for a in arr)
        print(f"  {r['id']:<5s} [{r.get('type',''):<4s}] 原子{len(arr):>2d}  档位{lvls:<8s} 锚定{tag:<8s} {r['question'][:44]}")
    print()
    sys.stdout.flush()

    # ---- 落盘 -------------------------------------------------------------
    out = ROOT.parent / "_smoke_corpus" / "probe_evidence_result.json"
    out.write_text(json.dumps({
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_chunks": n_chunks, "n_atoms": n_atoms,
        "levels": {str(k): lv.get(k, 0) for k in range(5)},
        "anchor": {"unique": len(uni), "multi": len(multi), "dead": len(dead)},
        "identity_gap": {"multi": len(multi), "named": len(multi_named), "anon": len(multi_anon)},
        "literal_vs_semantic": {"uni": n_uni, "sem_wrong": len(wrong),
                                "sem_wrong_rate": len(wrong) / max(n_uni, 1)},
        "atoms": atoms,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已落盘：{out}")
    print(f"总用时 {time.time() - t0:.1f}s（零 LLM 调用）")


if __name__ == "__main__":
    main()
