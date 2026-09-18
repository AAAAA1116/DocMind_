# -*- coding: utf-8 -*-
"""RAG 问答链（第四阶段 + 第五阶段重排 + 第六阶段混合检索）

把「检索」和「大模型」串起来：

    问题
     └─► 混合召回（向量 + BM25 → RRF）      ← config.HYBRID_ENABLED
          └─► 交叉编码器精排取 top-k          ← config.RERANK_ENABLED
               └─► 拼成 context 塞进 system_prompt
                    └─► DeepSeek ─► 回答

两个开关**正交**，四种组合都成立：

| HYBRID_ENABLED | RERANK_ENABLED | 行为 |
|---|---|---|
| 关 | 关 | v1.0 行为：纯向量取 top-k |
| 关 | 开 | 第五阶段行为：向量召回 N 个 → 精排取 k |
| 开 | 关 | 向量 + BM25 → RRF → top-k（快，靠融合名次排序） |
| 开 | 开 | 向量 + BM25 → RRF → 精排取 k（最准，最慢） |

各段分工：

| 段 | 模块 | 作用 | 特点 |
|---|---|---|---|
| 召回（语义） | ``core/embed_store.py`` | 捞出「意思像」的块 | 快、可建 ANN 索引，但漏精确字面 |
| 召回（字面） | ``core/hybrid.py`` 的 BM25 部分 | 捞出「字面有」的块 | 快，但漏转述/同义改写 |
| 融合 | ``core/hybrid.py`` 的 RRF 部分 | 只用名次合成两路结果 | 跨量纲，不需要归一化、不需要调参 |
| 精排 | ``core/reranker.py`` | 给候选重新打分取 top-k | 准，但要逐对前向，慢 |

为什么必须分层：交叉编码器没法预先算向量、也没法建 ANN 索引，拿它扫全库不现实；
而只用单路召回，精度又不够。所以标准做法就是「多路召回 → 融合 → 交叉编码器精排」，
本模块只负责**编排**，算法本身都在各自的模块里。

.. note::
    **混合检索与余弦阈值（``THRESHOLD``）不共存。**

    开了混合检索后，向量那一路**不再做余弦过滤**：BM25 召回的块压根没有余弦分数，
    拿单路阈值去卡会把它们连坐误伤（这是「跨量纲比较」的典型翻车）。
    排序权交给 RRF，该不该拒答交给精排阈值 + 大模型自己判断。

对外主接口 ``answer(question)``，返回::

    {
        "question": "网关监听哪个端口？",
        "answer":   "网关默认监听 8080 端口。",
        "sources":  [
            {"source": "ops_manual.md",
             "score": 0.9832,          # 参与最终排序的分数（开了精排是精排分，否则是 RRF 分）
             "rerank_score": 0.9832,   # 精排分（sigmoid 相关性概率），开了精排才有
             "rrf_score": 0.0325,      # RRF 融合分，开了混合检索才有
             "rrf_rank": 1,            # 融合后名次，从 1 开始
             "vector_score": 0.5388,   # 向量余弦相似度；向量那路没召回它则无此键
             "vector_rank": 3,         # 向量那路的名次
             "bm25_score": 8.774,      # BM25 分；BM25 那路没召回它则无此键
             "bm25_rank": 1,
             "chunk_index": 2,
             "snippet": "网关默认以单进程模式运行……", "content": "完整块文本"},
            ...
        ],
        "rerank_used": True,            # 本次是否真的走了精排
        "hybrid_used": True,            # 本次是否真的走了混合召回
        "refused":  False,              # 是否判定为「知识库中未找到」
        "refuse_reason": "",            # 拒答原因（refused=True 时有值）
        "system_prompt": "...",         # 实际发给模型的完整提示词，排查用
        "usage": {...},                 # token 用量
    }

.. warning::
    ``rerank_score`` / ``rrf_score`` / ``vector_score`` / ``bm25_score``
    **四种分数互不同量纲，不能互相比较，更不能混用同一个阈值**：

    * 余弦相似度挤在 0.6~1 的窄区间
    * 精排分是 sigmoid 概率，实测正确答案能到 0.999
    * RRF 分是 ``Σ 权重/(k+名次)``，量级只有 0.01~0.03
    * BM25 分无上界，随语料规模变化

    它们只有两个用处：**排序**，以及看某个块**是哪一路捞上来的**。
    不要拿去设阈值，也不要跨查询比大小。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    HYBRID_ENABLED,
    REFUSAL_TEXT,
    RERANK_CANDIDATES,
    RERANK_ENABLED,
    RERANK_THRESHOLD,
    THRESHOLD,
    TOP_K,
)
from core import embed_store  # noqa: E402
from core.llm_client import LLMError, chat_with_usage  # noqa: E402

#: 第四阶段提示词要求的 system_prompt 原文。{context} 会被替换成检索到的资料。
SYSTEM_PROMPT_TEMPLATE = (
    "你是一个企业内部知识助手。请根据以下资料回答问题。"
    "如果资料中没有相关信息，请说‘知识库中未找到相关信息’。"
    "资料：\n{context}"
)

#: 日志 / 界面里展示片段时的截断长度
SNIPPET_LEN = 120


def build_context(sources: Sequence[Dict[str, Any]]) -> str:
    """把检索结果拼成给模型看的资料文本。

    带上序号和来源文件名，方便模型（以及用户）对应到具体文件。
    """
    if not sources:
        return ""
    blocks: List[str] = []
    for i, src in enumerate(sources, 1):
        name = src.get("source") or "未知来源"
        blocks.append(f"[{i}] 来源：{name}\n{src.get('content', '')}")
    return "\n\n".join(blocks)


def build_system_prompt(sources: Sequence[Dict[str, Any]]) -> str:
    """按模板生成最终 system_prompt。"""
    return SYSTEM_PROMPT_TEMPLATE.format(context=build_context(sources))


def _to_source(doc: Dict[str, Any], *, score: float, **scores: Optional[Any]) -> Dict[str, Any]:
    """把检索 / 精排返回的文档 dict 整理成对外的 sources 项。

    ``score`` 是**参与最终排序**的那个分数：开了精排就是精排分，
    开了混合检索就是 RRF 分，都没有才是余弦相似度。

    其余分数用关键字参数传入（``vector_score`` / ``bm25_score`` / ``rrf_score`` /
    ``vector_rank`` / ``bm25_rank`` / ``rerank_score``），**值为 None 的直接丢掉**：
    这样「哪一路没召回它」在结果里表现为「没有这个键」，不会被误读成 0 分。

    .. note::
        本函数只做搬运，**不做任何跨分数的换算或比较**——它们是四个互不同量纲的东西，
        详见模块 docstring 的 warning。
    """
    metadata = doc.get("metadata") or {}
    content = doc.get("content", "")
    item: Dict[str, Any] = {
        "id": doc.get("id"),
        "source": metadata.get("source") or metadata.get("file") or "未知来源",
        "score": float(score),
        "chunk_index": metadata.get("chunk_index"),
        "snippet": content[:SNIPPET_LEN],
        "content": content,
    }
    for key, value in scores.items():
        if value is not None:
            item[key] = value
    return item


def _records_by_id(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按 ``record["doc"]["id"]`` 给混合检索的记录建索引。

    用途：精排会重排候选顺序，排完之后要拿回「这个块在向量/BM25 那一路排第几」，
    就得靠 id 反查——所以**索引必须建在 ``records`` 上**，不是建在 ``records[i]["doc"]`` 上
    （踩过：把文档 dict 当记录用，``rec.get("rrf_score")`` 全是 None，
    表现是诊断字段在某些路径下静默消失，而不是报错）。

    为什么不按 ``id(doc)``（对象身份）反查：``doc["id"]`` 是文档自己的字段
    （Chroma 里就是内容 sha1），**跟着数据走**——中间任何一步顺手 ``dict(doc)`` 拷一次
    都还对得上；对象身份一拷就断，且同样**不报错**。
    """
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        doc = rec.get("doc") or {}
        if doc.get("id") is not None:
            out[str(doc["id"])] = rec
    return out


def _hybrid_records_to_sources(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 ``hybrid.hybrid_search()`` 的记录整理成 sources，用 RRF 分作为 ``score``。"""
    return [
        _to_source(
            rec["doc"],
            score=rec["rrf_score"],
            rrf_score=rec.get("rrf_score"),
            rrf_rank=rec.get("rrf_rank"),
            vector_score=rec.get("vector_score"),
            vector_rank=rec.get("vector_rank"),
            bm25_score=rec.get("bm25_score"),
            bm25_rank=rec.get("bm25_rank"),
        )
        for rec in records
    ]


def _rerank_safely(question: str, docs: Sequence[Dict[str, Any]], k: int, stage: str):
    """调精排；失败返回 ``None`` 交给调用方降级。

    精排是「锦上添花」，它挂了不该让整个问答失败——但也不能悄悄降级，
    所以这里打印一行，日志里能看见。
    """
    try:
        from core.reranker import rerank as _rerank

        return _rerank(question, docs, top_n=k)
    except Exception as e:  # noqa: BLE001
        print(f"[rag_chain] {stage}精排失败，降级为召回顺序：{type(e).__name__}: {str(e)[:160]}")
        return None


def _retrieve_vector(
    question: str,
    k: int,
    *,
    do_rerank: bool,
    threshold: Optional[float],
    rerank_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    """纯向量召回（第三 / 第五阶段的行为）。

    它同时充当**混合检索不可用时的降级路径**（比如 jieba 没装）。
    """
    # ---- 单段式：纯向量取 top-k（行为与 v1.0 完全一致）----
    if not do_rerank:
        cut = THRESHOLD if threshold is None else float(threshold)
        hits = embed_store.search(question, k=k)
        return [
            _to_source(doc, score=s, vector_score=s)
            for doc, s in hits
            if not (cut > 0 and s < cut)
        ]

    # ---- 两段式：多召回一些，再交给交叉编码器精排 ----
    recall_k = max(int(RERANK_CANDIDATES), k)
    hits = embed_store.search(question, k=recall_k)
    if not hits:
        return []

    cosine = {str(doc.get("id")): s for doc, s in hits}
    rank_of = {str(doc.get("id")): i + 1 for i, (doc, _) in enumerate(hits)}

    ranked = _rerank_safely(question, [doc for doc, _ in hits], k, "向量召回")
    if ranked is None:
        return [_to_source(doc, score=s, vector_score=s) for doc, s in hits[:k]]

    cut = RERANK_THRESHOLD if rerank_threshold is None else float(rerank_threshold)
    results: List[Dict[str, Any]] = []
    for doc, rscore in ranked:
        if cut > 0 and rscore < cut:
            continue
        doc_id = str(doc.get("id"))
        results.append(
            _to_source(
                doc,
                score=rscore,
                rerank_score=rscore,
                vector_score=cosine.get(doc_id),
                vector_rank=rank_of.get(doc_id),
            )
        )
    return results


def _retrieve_hybrid(
    question: str,
    k: int,
    *,
    do_rerank: bool,
    rerank_threshold: Optional[float],
) -> List[Dict[str, Any]]:
    """混合召回（向量 + BM25 → RRF），可选再叠一层精排。

    与纯向量路径的两点差别：

    1. **不做余弦阈值过滤**。BM25 召回的块没有余弦分数，拿单路阈值去卡会连坐误伤。
    2. 候选池比 ``k`` 大（``RERANK_CANDIDATES``），精排只从池子里取前 k 个。
    """
    from core import hybrid      # 局部导入：hybrid 会拉起 jieba，不用时不必付这个开销

    pool = max(int(RERANK_CANDIDATES), k)

    try:
        records = hybrid.hybrid_search(question, k=pool)
    except Exception as e:  # noqa: BLE001
        # 建不起 BM25（最典型的是 jieba 没装）不该让问答挂掉 → 退回纯向量
        print(f"[rag_chain] 混合检索不可用，退回纯向量：{type(e).__name__}: {str(e)[:200]}")
        return _retrieve_vector(
            question, k,
            do_rerank=do_rerank,
            # 必须显式传 0.0（=不过滤），不能传 None：None 的语义是「用 config.THRESHOLD」，
            # 而它是 0.6 —— 在这个量纲上等于把候选全砍光，降级路径会静默返回空结果。
            threshold=0.0,
            rerank_threshold=rerank_threshold,
        )

    if not records:
        return []

    # 不精排：直接按 RRF 名次返回
    if not do_rerank:
        return _hybrid_records_to_sources(records[:k])

    docs = [rec["doc"] for rec in records]
    info = _records_by_id(records)

    ranked = _rerank_safely(question, docs, k, "混合召回")
    if ranked is None:
        return _hybrid_records_to_sources(records[:k])

    cut = RERANK_THRESHOLD if rerank_threshold is None else float(rerank_threshold)
    results: List[Dict[str, Any]] = []
    for doc, rscore in ranked:
        if cut > 0 and rscore < cut:
            continue
        rec = info.get(str(doc.get("id"))) or {}
        results.append(
            _to_source(
                doc,
                score=rscore,
                rerank_score=rscore,
                rrf_score=rec.get("rrf_score"),
                rrf_rank=rec.get("rrf_rank"),
                vector_score=rec.get("vector_score"),
                vector_rank=rec.get("vector_rank"),
                bm25_score=rec.get("bm25_score"),
                bm25_rank=rec.get("bm25_rank"),
            )
        )
    return results


def retrieve(
    question: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    *,
    use_rerank: Optional[bool] = None,
    use_hybrid: Optional[bool] = None,
    rerank_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """检索候选块。「混合召回」与「交叉编码器精排」是两层独立开关，可任意组合。

    Args:
        question:  用户问题
        top_k:     最终返回几块，默认 ``config.TOP_K``
        threshold: **仅在不混合、不精排时生效**的余弦相似度下限，默认 ``config.THRESHOLD``；
                   传 0 或负数等于不过滤
        use_rerank: 是否精排，默认跟随 ``config.RERANK_ENABLED``
        use_hybrid: 是否混合召回，默认跟随 ``config.HYBRID_ENABLED``
        rerank_threshold: 精排分数的下限，默认 ``config.RERANK_THRESHOLD``；0 表示不过滤

    Returns:
        按最终分数从高到低排列的候选块列表。基础字段
        ``source`` / ``score`` / ``chunk_index`` / ``snippet`` / ``content`` / ``id``；
        开了混合检索额外带 ``rrf_score`` / ``rrf_rank`` / ``vector_score`` / ``vector_rank`` /
        ``bm25_score`` / ``bm25_rank``；开了精排额外带 ``rerank_score``。

    Note:
        **``threshold``（余弦）在「混合」或「精排」任一开启时都不生效。**

        精排那层是因为量纲不同（实测余弦取常见的 0.6 会把候选全砍光，
        本项目 6 个查询无一达标）；混合这层还多一条理由：BM25 召回的块根本没有余弦分数，
        拿余弦阈值去卡等于把 BM25 的贡献原地作废。
    """
    k = TOP_K if top_k is None else int(top_k)
    if k <= 0:
        return []

    do_rerank = RERANK_ENABLED if use_rerank is None else bool(use_rerank)
    do_hybrid = HYBRID_ENABLED if use_hybrid is None else bool(use_hybrid)

    if do_hybrid:
        return _retrieve_hybrid(
            question, k,
            do_rerank=do_rerank,
            rerank_threshold=rerank_threshold,
        )

    return _retrieve_vector(
        question, k,
        do_rerank=do_rerank,
        threshold=threshold,
        rerank_threshold=rerank_threshold,
    )


def answer(
    question: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    *,
    call_llm: bool = True,
    temperature: Optional[float] = None,
    use_rerank: Optional[bool] = None,
    use_hybrid: Optional[bool] = None,
) -> Dict[str, Any]:
    """完整走一遍 RAG：检索（含混合召回与精排）→ 组装提示词 → 调模型 → 返回回答与来源。

    Args:
        question:   用户问题
        top_k:      最终取几块，默认 config.TOP_K
        threshold:  余弦相似度阈值（**开了混合召回或精排则失效**），默认 config.THRESHOLD
        call_llm:   为 False 时只做检索不调模型（调试检索质量用）
        temperature: 覆盖默认温度
        use_rerank: 是否走精排，默认跟随 config.RERANK_ENABLED
        use_hybrid: 是否走混合召回，默认跟随 config.HYBRID_ENABLED

    Returns:
        见模块 docstring 的返回结构。

    Raises:
        ValueError: question 为空
        LLMError:  大模型调用失败（重试后仍失败），由调用方决定怎么提示用户
    """
    if not question or not str(question).strip():
        raise ValueError("question 不能为空")

    question = str(question).strip()
    rerank_used = RERANK_ENABLED if use_rerank is None else bool(use_rerank)
    hybrid_used = HYBRID_ENABLED if use_hybrid is None else bool(use_hybrid)
    sources = retrieve(
        question,
        top_k=top_k,
        threshold=threshold,
        use_rerank=use_rerank,
        use_hybrid=use_hybrid,
    )

    # 没有任何可用资料：不必浪费一次模型调用，直接按话术拒答
    if not sources:
        total_indexed = embed_store.count()
        if total_indexed == 0:
            reason = "索引为空，请先上传文件"
        elif rerank_used:
            reason = f"精排后没有候选块达到阈值 {RERANK_THRESHOLD}"
        elif hybrid_used:
            # 混合检索不做余弦阈值过滤，只剩下「两路都没召回任何东西」这一种可能
            reason = "向量与 BM25 两路都没有召回候选块"
        else:
            cut = THRESHOLD if threshold is None else threshold
            reason = f"检索到的候选块相似度均低于阈值 {cut}"
        return {
            "question": question,
            "answer": REFUSAL_TEXT,
            "sources": [],
            "rerank_used": rerank_used,
            "hybrid_used": hybrid_used,
            "refused": True,
            "refuse_reason": reason,
            "system_prompt": "",
            "usage": {},
        }

    system_prompt = build_system_prompt(sources)

    if not call_llm:
        return {
            "question": question,
            "answer": "",
            "sources": sources,
            "rerank_used": rerank_used,
            "hybrid_used": hybrid_used,
            "refused": False,
            "refuse_reason": "",
            "system_prompt": system_prompt,
            "usage": {},
        }

    result = chat_with_usage(
        system_prompt,
        question,
        temperature=temperature,
    )
    text = result["text"]
    refused = REFUSAL_TEXT in text

    return {
        "question": question,
        "answer": text,
        "sources": sources,
        "rerank_used": rerank_used,
        "hybrid_used": hybrid_used,
        "refused": refused,
        "refuse_reason": "模型判定资料不足以回答" if refused else "",
        "system_prompt": system_prompt,
        "usage": result.get("usage", {}),
    }


def answer_text(question: str, top_k: Optional[int] = None) -> str:
    """只要回答文本的便捷方法。"""
    return answer(question, top_k=top_k)["answer"]


# ---------------------------------------------------------------------------
# 自测：python -m core.rag_chain "网关监听哪个端口？"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import DOCMIND_VERSION, ensure_dirs, summary

    ensure_dirs()
    print(summary())

    query = sys.argv[1] if len(sys.argv) > 1 else "网关监听哪个端口？"
    print(f"\n索引内共 {embed_store.count()} 条\n")

    try:
        out = answer(query)
    except LLMError as e:
        print(f"[失败] {e}")
        sys.exit(1)

    print(f"版本 : v{DOCMIND_VERSION}")
    print(f"问题 : {out['question']}")
    print(f"回答 : {out['answer']}")
    print(f"混合 : {'是（向量 + BM25 → RRF）' if out.get('hybrid_used') else '否（只用向量召回）'}")
    print(f"精排 : {'是' if out.get('rerank_used') else '否'}")
    print(f"拒答 : {out['refused']}  {out['refuse_reason']}")
    print(f"用量 : {out['usage']}")
    print("\n来源：")
    for i, s in enumerate(out["sources"], 1):
        # 把「是哪一路捞上来的」打出来，混合检索调起来全靠这一行
        parts = []
        if s.get("vector_score") is not None:
            parts.append(f"向量={s['vector_score']:.4f}#{s.get('vector_rank', '-')}")
        else:
            parts.append("向量=未召回")
        if s.get("bm25_score") is not None:
            parts.append(f"BM25={s['bm25_score']:.3f}#{s.get('bm25_rank', '-')}")
        else:
            parts.append("BM25=未召回")
        if s.get("rrf_score") is not None:
            parts.append(f"RRF={s['rrf_score']:.5f}#{s.get('rrf_rank', '-')}")
        if s.get("rerank_score") is not None:
            parts.append(f"精排={s['rerank_score']:.4f}")
        print(f"  [{i}] {s['source']}  score={s['score']:.4f}  chunk={s['chunk_index']}")
        print(f"      {'  '.join(parts)}")
        print(f"      {s['snippet']}...")
