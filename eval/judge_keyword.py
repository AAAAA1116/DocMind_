#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""keyword 裁判：给 Golden Set 用例卡 v2 的判据算分（第八阶段）

为什么要在 ragas 之外再写一个裁判
----------------------------------
ragas 的四个指标全都要过一遍 LLM，慢、花钱、还有采样波动；而且它只回答
「像不像」，回答不了「**要求的那个要点到底有没有出现**」。

用例卡 v2 用的是 `keyword_loose` 判据：每条题带三组关键词——

    should_retrieve    召回块里**必须出现**的短语（测检索有没有捞对）
    must_include       答案里**必须出现**的短语（测答全没答全）
    must_not_include   答案里**不该出现**的短语（测有没有串到别家 / 编造）

这三组是**确定性**的：同一个答案跑一百遍，分数一样。所以它当不了唯一裁判
（它不认识同义改写、也不判顺序和对应关系），但它是**最便宜、最稳、最能定位
问题在哪一环**的那把尺子。两者一起用：ragas 看整体质量，keyword 看要点命中。

匹配前必须做的归一化（不做会大面积假失败）
------------------------------------------
索引里的块是**硬换行**切过的，例如君逸原文被切成：

    第四十七条 公司决定对特定信息作暂缓、豁免披露处理的，应当由董事会秘
    书负责登记……

「董事会秘书」在这种文本里是**不连续**的。所以匹配前要去掉所有空白
（含全角空格）、去掉 `[[Pn]]` 页码标记、去掉 markdown 痕迹。
不做这一步，判据会成片假失败——而你会以为是检索坏了。

用法
----
    # 先跑链路拿到答案与上下文
    python eval/run_eval.py --dataset eval/dataset_real.jsonl --label v1.5-kw

    # 再用判据打分（读结果文件，不重跑链路）
    python eval/judge_keyword.py --result eval/results/v1.5-kw.json

    # 只看某一桶（定位问题用）
    python eval/judge_keyword.py --result eval/results/v1.5-kw.json --bucket adv

已知局限（诚实交代）
--------------------
* **测不了对应关系和顺序**。q17「三个期限 ↔ 三种报告」、q22「业务部门→董秘→
  董事长」这种，本裁判只能看词在不在，判不出配对对不对、顺序有没有反。
  这类题在 note 里标了「需人工抽查」。
* **区分力弱的判据不算错**。同构语料里「董事会秘书」在 7 份文档里都有，
  命中它不能证明检索对了——报告里单列 `weak` 标记，不参与真假判定。
* 判据是子串匹配，**答案必须用原文措辞**才得分。模型的同义改写（「董秘」写成
  「董事会秘书」没问题，但写成「公司秘书」就丢分）会被记成未命中。
  这是 keyword 法的固有代价，也是为什么它只能和 ragas 配合、不能独当一面。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"

#: 三组判据的字段名
F_RETRIEVE = "should_retrieve"
F_INCLUDE = "must_include"
F_EXCLUDE = "must_not_include"

#: 系统固定拒答话术。负例题的 must_include 就是它——它不在语料里，是系统输出。
REFUSAL_TEXT = "知识库中未找到相关信息"


# ---------------------------------------------------------------------------
# 归一化：这一步错了，后面全错
# ---------------------------------------------------------------------------
_PAGE_MARK = re.compile(r"\[\[P\d+\]\]")
_WS = re.compile(r"[\s\u3000]+")


def norm(s: str) -> str:
    """把文本压成可做子串匹配的形式。

    依次去掉：[[Pn]] 页码标记 → 所有空白（含全角）→ markdown 痕迹。
    顺序有讲究：先删页码标记再删空白，否则 ``a [[P1]] b`` 会变成 ``a[[P1]]b``
    再被拆开处理，反而留残渣。
    """
    if not s:
        return ""
    s = _PAGE_MARK.sub("", s)
    s = _WS.sub("", s)
    for ch in ("**", "`", "\\n", "\u201c", "\u201d", '"'):
        s = s.replace(ch, "")
    return s


# ---------------------------------------------------------------------------
# 单条打分
# ---------------------------------------------------------------------------
def judge_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """给一条样本打三个分。没有判据的样本返回 None（跳过，不是 0 分）。"""
    if F_INCLUDE not in row:
        return None

    ans_raw = row.get("answer") or ""
    ctx_raw = row.get("contexts") or []
    ans = norm(ans_raw)
    ctx = norm("\n".join(ctx_raw))

    retrieve = list(row.get(F_RETRIEVE) or [])
    include = list(row.get(F_INCLUDE) or [])
    exclude = list(row.get(F_EXCLUDE) or [])

    r_hit = [kw for kw in retrieve if norm(kw) in ctx]
    r_miss = [kw for kw in retrieve if norm(kw) not in ctx]
    i_hit = [kw for kw in include if norm(kw) in ans]
    i_miss = [kw for kw in include if norm(kw) not in ans]
    e_hit = [kw for kw in exclude if norm(kw) in ans]

    return {
        "id": row.get("id"),
        "kw_id": row.get("kw_id"),
        "bucket": row.get("kw_bucket", "?"),
        "category": row.get("kw_category", "?"),
        "question": row.get("question", ""),
        "is_negative": bool(row.get("should_refuse")),
        "refused": bool(row.get("refused")),
        "pipeline_error": row.get("pipeline_error"),
        "n_contexts": row.get("n_contexts", len(ctx_raw)),
        # 检索
        "retrieve_total": len(retrieve),
        "retrieve_hit": len(r_hit),
        "retrieve_miss": r_miss,
        # 覆盖率
        "include_total": len(include),
        "include_hit": len(i_hit),
        "include_miss": i_miss,
        # 违例
        "forbidden_total": len(exclude),
        "forbidden_hit": e_hit,
        # 是否算「整条通过」：要点全中 + 无违例
        # 负例题额外要求「确实拒答了」
        "pass": (len(i_miss) == 0 and len(e_hit) == 0
                 and (not row.get("should_refuse") or bool(row.get("refused")))),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def aggregate(items: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    pos = [it for it in items if not it["is_negative"]]
    neg = [it for it in items if it["is_negative"]]

    out: Dict[str, Any] = {"label": label, "n_total": len(items),
                           "n_positive": len(pos), "n_negative": len(neg)}

    # 检索命中率：只看正例、且该题确实配了 should_retrieve
    pr = [it for it in pos if it["retrieve_total"]]
    out["retrieval_hit_rate"] = _rate(sum(it["retrieve_hit"] for it in pr),
                                      sum(it["retrieve_total"] for it in pr))
    out["retrieval_all_hit_rows"] = sum(1 for it in pr if not it["retrieve_miss"])
    out["retrieval_rows"] = len(pr)

    # 要点覆盖率
    out["must_cover_rate"] = _rate(sum(it["include_hit"] for it in items),
                                   sum(it["include_total"] for it in items))
    out["full_cover_rows"] = sum(1 for it in items if not it["include_miss"])
    out["cover_rows"] = len(items)

    # 违例
    out["forbidden_hit_count"] = sum(len(it["forbidden_hit"]) for it in items)
    out["forbidden_rows"] = sum(1 for it in items if it["forbidden_hit"])

    # 整条通过率
    out["pass_rows"] = sum(1 for it in items if it["pass"])
    out["pass_rate"] = _rate(out["pass_rows"], len(items))

    # 分桶
    out["by_bucket"] = {}
    for b in sorted({it["bucket"] for it in items}):
        sub = [it for it in items if it["bucket"] == b]
        out["by_bucket"][b] = {
            "n": len(sub),
            "pass": sum(1 for it in sub if it["pass"]),
            "pass_rate": _rate(sum(1 for it in sub if it["pass"]), len(sub)),
            "must_cover_rate": _rate(sum(it["include_hit"] for it in sub),
                                     sum(it["include_total"] for it in sub)),
            "retrieval_hit_rate": _rate(
                sum(it["retrieve_hit"] for it in sub if it["retrieve_total"]),
                sum(it["retrieve_total"] for it in sub if it["retrieve_total"])),
        }
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def print_report(res: Dict[str, Any], items: List[Dict[str, Any]], bucket: Optional[str]) -> None:
    agg = res["summary"]
    print()
    print("=" * 88)
    print(f"判据得分（keyword_loose）　{res['label']}　来源 {res['source_result']}")
    print("=" * 88)
    print(f"  样本   {agg['n_total']} 条（正例 {agg['n_positive']}，负例 {agg['n_negative']}）"
          f"　带判据 {agg['cover_rows']} 条")
    print("-" * 88)
    print("  检索层（召回块里有没有要求的短语）")
    rh = agg["retrieval_hit_rate"]
    print(f"    短语命中率        {'—' if rh is None else f'{rh:.1%}':>8s}   "
          f"（{agg['retrieval_rows']} 道题配了检索判据）")
    print(f"    整题全命中        {agg['retrieval_all_hit_rows']:>8d} / {agg['retrieval_rows']}   道题")
    print()
    print("  答案层（要求的要点答全没有）")
    mc = agg["must_cover_rate"]
    print(f"    要点覆盖率        {'—' if mc is None else f'{mc:.1%}':>8s}   "
          f"（{agg['cover_rows']} 条带判据的题）")
    print(f"    要点全中的题      {agg['full_cover_rows']:>8d} / {agg['cover_rows']}   条")
    print()
    print("  违例（不该出现的短语出现了几次）")
    print(f"    违例短语数        {agg['forbidden_hit_count']:>8d}")
    print(f"    有违例的题        {agg['forbidden_rows']:>8d} / {agg['cover_rows']}   条")
    print()
    print(f"  整条通过（要点全中 且 无违例）  {agg['pass_rows']} / {agg['n_total']}"
          f"　= {agg['pass_rate']:.1%}")
    print()
    print("  分桶")
    print(f"    {'桶':<8s}{'条数':>6s}{'通过':>6s}{'通过率':>9s}{'要点覆盖':>11s}{'检索命中':>11s}")
    for b, d in agg["by_bucket"].items():
        f = lambda v: "—" if v is None else f"{v:.0%}"
        print(f"    {b:<8s}{d['n']:>6d}{d['pass']:>6d}{f(d['pass_rate']):>9s}"
              f"{f(d['must_cover_rate']):>11s}{f(d['retrieval_hit_rate']):>11s}")
    print("=" * 88)

    show = [it for it in items if bucket is None or it["bucket"] == bucket]
    print()
    print(f"  逐条明细{'（只显示 %s 桶）' % bucket if bucket else ''}"
          f"　✔=通过　✗=要点缺　⊘=有违例")
    print("-" * 88)
    for it in show:
        flags = ("✔" if it["pass"] else "✗")
        if it["forbidden_hit"]:
            flags += "⊘"
        if it["pipeline_error"]:
            flags = "E"
        r = f"{it['retrieve_hit']}/{it['retrieve_total']}" if it["retrieve_total"] else "  —"
        c = f"{it['include_hit']}/{it['include_total']}"
        print(f"    {flags} {it['id']:<5s}{it['bucket']:<6s}召回{r:>6s}　要点{c:>5s}　"
              f"{it['question'][:34]}")
        if it["retrieve_miss"]:
            print(f"        召回缺: {' / '.join(m[:26] for m in it['retrieve_miss'])[:86]}")
        if it["include_miss"]:
            print(f"        要点缺: {' / '.join(m[:26] for m in it['include_miss'])[:86]}")
        if it["forbidden_hit"]:
            print(f"        ⚠ 违例: {' / '.join(m[:26] for m in it['forbidden_hit'])[:86]}")
    print()


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Golden Set 判据裁判（keyword_loose）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True,
                    help="run_eval.py 产出的结果文件（含 answer / contexts / 判据字段）")
    ap.add_argument("--bucket", default=None, choices=["prod", "adv", "edge"],
                    help="只显示某一桶的逐条明细")
    ap.add_argument("--out", default=None, help="判据结果另存路径（默认 results/<label>.keyword.json）")
    args = ap.parse_args()

    path = Path(args.result)
    if not path.is_file():
        sys.exit(f"结果文件不存在：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    items, skipped = [], []
    for row in raw.get("rows", []):
        got = judge_row(row)
        if got is None:
            skipped.append(row.get("id"))
        else:
            items.append(got)
    if not items:
        sys.exit("结果文件里没有任何带判据的样本——请确认用的是 dataset_real.jsonl。")

    label = raw.get("label", path.stem)
    res = {
        "label": label,
        "judged_at": raw.get("finished_at"),
        "source_result": path.name,
        "dataset": raw.get("dataset"),
        "pipeline_config": raw.get("config"),
        "summary": aggregate(items, label),
        "skipped_no_criteria": skipped,
        "rows": items,
    }

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{label}.keyword.json"
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(res, items, args.bucket)
    print(f"  跳过的无判据样本 {len(skipped)} 条（V1 独有题，还没出判据）: "
          f"{', '.join(str(s) for s in skipped) if skipped else '无'}")
    print(f"  判据结果已保存：{out_path}")
    print()


if __name__ == "__main__":
    main()
