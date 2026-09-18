#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测闭环：让每一次优化都有据可依（第七阶段）

为什么需要它
------------
你已经做过两轮检索优化（Rerank、混合检索），但那两轮的结论都是**局部实验**——
拿几条构造语料量一量，说明「在这个语料上 top-1 从 11/12 变成 12/12」。
这种实验能定「有没有用」，但定不了「整体效果提升多少」，更发现不了
「A 指标涨了、B 指标却掉了」这类互相抵消的改动。

所以这里建一个**固定不动的评测集 + 可复现的跑批**：改完代码重跑一次，
和上一次的结果并排看。这就是「有据可依」。

    评测集(eval/dataset.jsonl) → 跑真实问答链路 → 收集(答案, 检索到的上下文)
        → ragas 打分 → 落盘(eval/results/*.json) → 与历史结果对比

用法
----
    # 跑一遍（参数默认跟随 config.py），结果存为 eval/results/<label>.json
    python eval/run_eval.py --label v1.2-default

    # 对照实验：关掉重排（快得多），看关掉之后指标掉多少
    python eval/run_eval.py --label v1.2-no-rerank --no-rerank

    # 冒烟：只跑前 3 条，验证链路通不通
    python eval/run_eval.py --label smoke --limit 3 --no-rerank

    # 对比两次结果，输出逐指标 diff
    python eval/run_eval.py --compare eval/results/A.json eval/results/B.json

四个指标各测什么（ragas 官方定义）
----------------------------------
| 指标 | 测的是链路哪一环 | 需要什么 |
|---|---|---|
| faithfulness | **生成**：答案里的话是不是都有资料支撑，有没有编 | 答案 + 上下文 |
| answer_relevancy | **生成**：答案有没有正面回答问题（而不是绕开） | 问题 + 答案 |
| context_recall | **检索**：回答所需的信息，检索有没有全捞回来 | 问题 + 上下文 + 参考答案 |
| context_precision | **检索**：捞回来的资料里，有多少是真相关的、排得够不够前 | 问题 + 上下文 + 参考答案 |

faithfulness / answer_relevancy 是**无参考答案**的，所以它们能直接用在生产流量上做监控。
context_recall / context_precision 必须有参考答案，只能跑评测集。

三条设计取舍（都踩过坑）
------------------------
1. **用新式指标接口，不用 `evaluate()`**。
   ragas 0.4.3 里 `from ragas.metrics import Faithfulness` 已经打上 deprecation 警告
   （官方说 v1.0 移除），新接口在 `ragas.metrics.collections`，且只接受
   `llm_factory` 产出的 instructor LLM。旧式 `evaluate()` + `EvaluationDataset` 那条路
   走的是另一套 LLM 封装，两边不通用，混用会在运行时才炸。这里统一走新式。
   代价：没有 `result.to_pandas()` 那种现成 DataFrame，汇总表得自己写（本文件干了）。

2. **负例题不参与四指标均值**。
   负例题的参考答案是「知识库中未找到相关信息」，context_recall / context_precision
   拿它当 reference 算出来的数是**没有意义**的（分母语义就不对）。混进均值只会
   把主指标搅浑。所以负例单独看 `refusal_accuracy`（该拒答的有没有拒答），
   正例单独看 `false_refusal_rate`（不该拒答的有没有被误拒）。

3. **不改被测代码，不覆盖 config**。
   评测脚本只调 `rag_chain.answer()`，不为了「测起来方便」去改链路或临时调阈值。
   所以 `config.THRESHOLD=0.6` 这种会让所有问题都拒答的设置，会**原样暴露**在
   `false_refusal_rate=1.0` 上——那正是评测该干的事。想测别的配置就传参，
   别去改 config。

已知局限（诚实交代）
--------------------
* **裁判和被测是同一个模型**（都是 deepseek-chat）。这叫 self-evaluation bias：
  模型倾向于认为自己的风格是合理的。分数**绝对值**要打折看，
  但**同一条评测集上的相对变化**（改前 vs 改后）是可信的——偏差在同模型上是恒定的。
  要更严格，就把裁判换成更强的模型（拆开 config.MODEL_NAME 和裁判模型）。
* ragas 的判分 prompt 是英文的，用中文语料会被「翻译损耗」打折。
  这不影响对比（两次都打折），但会让绝对值偏低。
* 单次跑的分数受 LLM 采样波动影响（temperature）。评的是**趋势**，不是小数点后四位。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ragas 会在后台发匿名统计，本机外网不通，白等超时，关掉
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

import config  # noqa: E402
from core import embed_store, rag_chain  # noqa: E402

DATASET_PATH = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

#: 参与评分的四个指标（顺序即报告里的列顺序）
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

#: 指标 -> 中文名，报告里用
METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_recall": "上下文召回",
    "context_precision": "上下文精度",
}

#: 指标 -> 该指标用到的字段。返回 None 表示这条样本不适用该指标（跳过而不是记 0）
def _metric_args(metric: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = row.get("contexts") or []
    q = row["question"]
    if metric == "faithfulness":
        # 检索到空上下文时算不出来，而且「空上下文 + 有答案」本身就说明检索挂了
        return {"user_input": q, "response": row["answer"], "retrieved_contexts": ctx} if ctx else None
    if metric == "answer_relevancy":
        return {"user_input": q, "response": row["answer"]}
    if metric == "context_recall":
        return {"user_input": q, "retrieved_contexts": ctx, "reference": row["ground_truth"]} if ctx else None
    if metric == "context_precision":
        return {"user_input": q, "reference": row["ground_truth"], "retrieved_contexts": ctx} if ctx else None
    return None


# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """读 JSONL 评测集。空行和 # 开头的行忽略。"""
    if not path.is_file():
        sys.exit(f"评测集不存在：{path}")
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"评测集第 {lineno} 行不是合法 JSON：{e}")
    if not rows:
        sys.exit(f"评测集为空：{path}")
    return rows


# ---------------------------------------------------------------------------
# 第一段：跑真实问答链路，收集答案与检索上下文
# ---------------------------------------------------------------------------
def run_pipeline(
    rows: List[Dict[str, Any]],
    *,
    use_rerank: Optional[bool],
    use_hybrid: Optional[bool],
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """逐条走 ``rag_chain.answer()``。

    注意这里是**串行**的，不是不会写并发，而是：
    精排是纯 CPU 前向，本机只有 4 核，并发只会互相抢核、把单条耗时拉长，
    总时长并不会变短。串行还能让进度和日志保持线性可读。
    """
    out: List[Dict[str, Any]] = []
    total = len(rows)
    for i, row in enumerate(rows, 1):
        q = row["question"]
        kind = row.get("type", "")
        print(f"  [{i}/{total}] ({kind}) {q[:38]}…" if len(q) > 38 else f"  [{i}/{total}] ({kind}) {q}")

        t0 = time.perf_counter()
        try:
            result = rag_chain.answer(
                q, use_rerank=use_rerank, use_hybrid=use_hybrid, threshold=threshold
            )
        except Exception as e:  # 单条失败不该让整批评测挂掉
            print(f"        ✗ 链路异常：{type(e).__name__}: {str(e)[:140]}")
            item = dict(row)
            item.update({"answer": "", "contexts": [], "refused": False,
                         "n_contexts": 0, "total_ms": None,
                         "pipeline_error": f"{type(e).__name__}: {str(e)[:200]}"})
            out.append(item)
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        contexts = [s.get("content", "") for s in result.get("sources", [])]
        item = dict(row)
        item.update({
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "refused": bool(result.get("refused")),
            "n_contexts": len(contexts),
            "total_ms": round(elapsed_ms, 1),
        })
        flag = "拒答" if item["refused"] else f"{len(contexts)} 块"
        print(f"        → {flag}　{elapsed_ms / 1000:.1f}s　{str(item['answer'])[:60]}")
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# 第二段：ragas 打分
# ---------------------------------------------------------------------------
def build_metrics():
    """构造四个指标实例。

    裁判 LLM 走 DeepSeek（OpenAI 兼容协议），embedding 走本地 bge，
    两者都复用了项目已有的配置和本地模型，不额外花钱、不出网。
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    from core.embed_store import _resolve_local_model  # 复用项目里的本地模型定位逻辑

    api_key = os.environ.get(config.LLM_API_KEY_ENV)
    if not api_key:
        sys.exit(
            f"缺少 {config.LLM_API_KEY_ENV}（裁判模型要用它）。"
            f"请在项目根目录 .env 里配置。"
        )

    client = AsyncOpenAI(api_key=api_key, base_url=config.LLM_BASE_URL)
    llm = llm_factory(config.MODEL_NAME, client=client)

    local_emb = _resolve_local_model(config.EMBED_MODEL_NAME)
    if not local_emb:
        print(f"  ⚠ 本地没找到 {config.EMBED_MODEL_NAME}，将尝试联网下载")
    embeddings = HuggingFaceEmbeddings(model=local_emb or config.EMBED_MODEL_NAME)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=3),
        "context_recall": ContextRecall(llm=llm),
        "context_precision": ContextPrecisionWithReference(llm=llm),
    }


async def score_rows(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    *,
    concurrency: int = 5,
    retries: int = 1,
) -> None:
    """就地把每个指标的分数写回 rows。

    指标之间互相独立，可以并发；但裁判是被同一个 API 服务的，
    并发开太大会撞限流，所以用一个信号量卡住。
    """
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for row in rows:
        for name in METRIC_NAMES:
            args = _metric_args(name, row)
            if args is None:
                row[name] = None
                row[f"{name}_skip"] = "该样本不适用（无上下文）"
                continue
            tasks.append(_score_one(sem, row, name, metrics[name], args, retries))

    if not tasks:
        return
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"    裁判打分 {done}/{len(tasks)}", end="\r")
    print(" " * 40, end="\r")


async def _score_one(sem, row, name, metric, args, retries) -> None:
    async with sem:
        last = None
        for attempt in range(retries + 1):
            try:
                result = await metric.ascore(**args)
                row[name] = round(float(result.value), 4)
                return
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:150]}"
                if attempt < retries:
                    await asyncio.sleep(2 * (attempt + 1))
        row[name] = None
        row[f"{name}_error"] = last


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive = [r for r in rows if not r.get("should_refuse")]
    negative = [r for r in rows if r.get("should_refuse")]

    summary: Dict[str, Any] = {"n_total": len(rows), "n_positive": len(positive), "n_negative": len(negative)}

    # 四指标只在正例题上取均值（理由见模块 docstring 第 2 条）
    for name in METRIC_NAMES:
        vals = [r[name] for r in positive if isinstance(r.get(name), (int, float))]
        summary[name] = round(sum(vals) / len(vals), 4) if vals else None
        summary[f"{name}_scored"] = f"{len(vals)}/{len(positive)}"

    # 拒答相关：两个方向的错误分开看
    if negative:
        summary["refusal_accuracy"] = round(
            sum(1 for r in negative if r.get("refused")) / len(negative), 4
        )
    else:
        summary["refusal_accuracy"] = None
    if positive:
        summary["false_refusal_rate"] = round(
            sum(1 for r in positive if r.get("refused")) / len(positive), 4
        )
    else:
        summary["false_refusal_rate"] = None

    times = [r["total_ms"] for r in rows if isinstance(r.get("total_ms"), (int, float))]
    summary["avg_total_ms"] = round(sum(times) / len(times), 1) if times else None

    return summary


def config_snapshot(
    use_rerank: Optional[bool],
    use_hybrid: Optional[bool],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """把这一次跑的时候生效的关键参数记下来，否则过几天看结果文件会不知道当时是什么配置。"""
    return {
        "docmind_version": config.DOCMIND_VERSION,
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.OVERLAP,
        "top_k": config.TOP_K,
        "threshold": config.THRESHOLD if threshold is None else threshold,
        "embed_model": config.EMBED_MODEL_NAME,
        "hybrid_enabled": config.HYBRID_ENABLED if use_hybrid is None else use_hybrid,
        "hybrid_per_source": config.HYBRID_PER_SOURCE,
        "hybrid_rrf_k": config.HYBRID_RRF_K,
        "rerank_enabled": config.RERANK_ENABLED if use_rerank is None else use_rerank,
        "rerank_candidates": config.RERANK_CANDIDATES,
        "llm_model": config.MODEL_NAME,
        "index_count": embed_store.count(),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def print_report(result: Dict[str, Any]) -> None:
    s = result["summary"]
    cfg = result["config"]
    print()
    print("=" * 78)
    print(f"评测结果　{result['label']}　（{result['finished_at']}）")
    print("=" * 78)
    print(f"  配置    切分 {cfg['chunk_size']}/{cfg['overlap']}　top_k {cfg['top_k']}　"
          f"阈值 {cfg['threshold']}　混合检索 {'开' if cfg['hybrid_enabled'] else '关'}　"
          f"重排 {'开' if cfg['rerank_enabled'] else '关'}　索引 {cfg['index_count']} 块")
    print(f"  样本    {s['n_total']} 条（正例 {s['n_positive']}，负例 {s['n_negative']}）")
    print("-" * 78)
    print("  答案质量（正例题均值，越高越好）")
    for name in METRIC_NAMES:
        v = s[name]
        shown = f"{v:.4f}" if isinstance(v, float) else "—（未算出）"
        print(f"    {METRIC_LABELS[name]:<12s} {shown:>10s}   （{s[f'{name}_scored']} 条算出）")
    print()
    print("  拒答行为")
    ra = s["refusal_accuracy"]
    fr = s["false_refusal_rate"]
    print(f"    该拒答时拒答了   {'—' if ra is None else f'{ra:.0%}':>10s}   （{s['n_negative']} 条负例题）")
    print(f"    不该拒答却拒答   {'—' if fr is None else f'{fr:.0%}':>10s}   （{s['n_positive']} 条正例题）")
    print()
    print(f"  平均单条耗时     {s['avg_total_ms']} ms")
    print("=" * 78)
    print()
    print("  逐条明细（✗ 表示该项没算出）")
    for r in result["rows"]:
        mark = "拒答" if r.get("refused") else f"{r.get('n_contexts', 0)}块"
        parts = []
        for name in METRIC_NAMES:
            v = r.get(name)
            parts.append("✗" if v is None else f"{v:.2f}")
        print(f"    {r['id']:<4s} {r.get('type', ''):<4s} [{mark:>4s}] "
              f"忠{parts[0]} 关{parts[1]} 召{parts[2]} 精{parts[3]}　{r['question'][:26]}")
    print()


def cmd_compare(paths: List[str]) -> None:
    """并排对比两份（或多份）评测结果，输出每个指标的差值。"""
    results = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            sys.exit(f"结果文件不存在：{path}")
        results.append(json.loads(path.read_text(encoding="utf-8")))

    print()
    print("=" * 78)
    print("评测结果对比")
    print("=" * 78)
    header = f"{'指标':<16s}" + "".join(f"{r['label'][:14]:>16s}" for r in results)
    if len(results) == 2:
        header += f"{'变化':>16s}"
    print(header)
    print("-" * 78)

    rows_to_show = [(n, METRIC_LABELS[n]) for n in METRIC_NAMES]
    rows_to_show += [("refusal_accuracy", "拒答准确率"), ("false_refusal_rate", "误拒率"),
                     ("avg_total_ms", "平均耗时(ms)")]

    for key, label in rows_to_show:
        line = f"{label:<16s}"
        vals = []
        for r in results:
            v = r["summary"].get(key)
            vals.append(v)
            if v is None:
                line += f"{'—':>16s}"
            elif key == "avg_total_ms":
                line += f"{v:>16,.0f}"
            elif key in ("refusal_accuracy", "false_refusal_rate"):
                line += f"{v:>15.0%} "
            else:
                line += f"{v:>16.4f}"
        if len(results) == 2 and vals[0] is not None and vals[1] is not None:
            d = vals[1] - vals[0]
            if key == "avg_total_ms":
                line += f"{d:>+15,.0f} "
            elif key in ("refusal_accuracy", "false_refusal_rate"):
                line += f"{d:>+15.0%} "
            else:
                arrow = "↑" if d > 0.0005 else ("↓" if d < -0.0005 else "=")
                line += f"{d:>+15.4f}{arrow}"
        print(line)

    print("-" * 78)
    for r in results:
        c = r["config"]
        print(f"  {r['label']:<18s} 混合检索 {'开' if c['hybrid_enabled'] else '关'}　"
              f"重排 {'开' if c['rerank_enabled'] else '关'}　"
              f"阈值 {c['threshold']}　索引 {c['index_count']} 块　{r['finished_at']}")
    print()
    if len(results) == 2:
        print("  提示：单次评测有采样波动，看趋势别看小数点后四位；"
              "耗时那行单位是毫秒。")
        print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def cmd_run(args) -> None:
    rows = load_dataset(Path(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    print("=" * 78)
    print(f"DocMind_ 评测　label={args.label}")
    print("=" * 78)
    snapshot = config_snapshot(args.rerank, args.hybrid, args.threshold)
    print(f"  配置    切分 {snapshot['chunk_size']}/{snapshot['overlap']}　top_k {snapshot['top_k']}　"
          f"阈值 {snapshot['threshold']}")
    print(f"          混合检索 {'开' if snapshot['hybrid_enabled'] else '关'}　"
          f"重排 {'开' if snapshot['rerank_enabled'] else '关'}　"
          f"索引 {snapshot['index_count']} 块")
    print(f"  样本    {len(rows)} 条　数据集 {Path(args.dataset).name}")
    if snapshot["rerank_enabled"]:
        print("  ⚠ 重排开着：本机是纯 CPU，每条可能要几十秒，整批请预留十几分钟。")
    print("-" * 78)
    print("第一段：跑真实问答链路")
    t_start = time.time()
    rows = run_pipeline(
        rows, use_rerank=args.rerank, use_hybrid=args.hybrid, threshold=args.threshold
    )
    print(f"  链路跑完，用时 {time.time() - t_start:.1f}s")

    print("-" * 78)
    print("第二段：ragas 打分（裁判 = " + config.MODEL_NAME + "）")
    metrics = build_metrics()
    asyncio.run(score_rows(rows, metrics))

    result = {
        "label": args.label,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(Path(args.dataset).name),
        "config": snapshot,
        "summary": summarize(rows),
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(result)
    print(f"结果已保存：{out_path}")
    print(f"下次对比：python eval/run_eval.py --compare {out_path} <另一次>.json")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocMind_ 评测闭环（ragas）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python eval/run_eval.py --label v1.2-default\n"
               "  python eval/run_eval.py --label v1.2-no-rerank --no-rerank\n"
               "  python eval/run_eval.py --compare eval/results/a.json eval/results/b.json\n",
    )
    parser.add_argument("--label", default="latest", help="本次结果的名称（存成 results/<label>.json）")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="评测集路径（JSONL）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="覆盖余弦相似度阈值（默认跟随 config.THRESHOLD；0 表示不过滤）。"
             "只在纯向量路径生效——混合检索路径本来就不套用余弦阈值",
    )
    parser.add_argument("--compare", nargs="+", metavar="RESULT_JSON",
                        help="不进跑批，只对比若干份历史结果")

    g1 = parser.add_mutually_exclusive_group()
    g1.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="强制开启精排（默认跟随 config.RERANK_ENABLED）")
    g1.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="强制关闭精排（快很多）")

    g2 = parser.add_mutually_exclusive_group()
    g2.add_argument("--hybrid", dest="hybrid", action="store_true", default=None,
                    help="强制开启混合召回（默认跟随 config.HYBRID_ENABLED）")
    g2.add_argument("--no-hybrid", dest="hybrid", action="store_false",
                    help="强制关闭混合召回（退回纯向量）")

    args = parser.parse_args()

    if args.compare:
        cmd_compare(args.compare)
        return
    cmd_run(args)


if __name__ == "__main__":
    main()
