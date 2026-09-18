# -*- coding: utf-8 -*-
"""RAG 问答链（第四阶段）

把「检索」和「大模型」串起来：

    问题 → 向量检索 top-k → 拼成 context → 塞进 system_prompt → DeepSeek → 回答

对外主接口 ``answer(question)``，返回::

    {
        "question": "网关监听哪个端口？",
        "answer":   "网关默认监听 8080 端口。",
        "sources":  [
            {"source": "ops_manual.md", "score": 0.5388, "chunk_index": 2,
             "snippet": "网关默认以单进程模式运行……", "content": "完整块文本"},
            ...
        ],
        "refused":  False,          # 是否判定为「知识库中未找到」
        "refuse_reason": "",        # 拒答原因（refused=True 时有值）
        "system_prompt": "...",     # 实际发给模型的完整提示词，排查用
        "usage": {...},             # token 用量
    }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import REFUSAL_TEXT, THRESHOLD, TOP_K  # noqa: E402
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


def retrieve(
    question: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """检索候选块，并套用相似度阈值过滤。

    Args:
        question:  用户问题
        top_k:     取前 k 个，默认 ``config.TOP_K``
        threshold: 相似度下限，默认 ``config.THRESHOLD``；传 0 或负数等于不过滤

    Returns:
        按相似度从高到低排列的候选块列表，每项含
        ``source`` / ``score`` / ``chunk_index`` / ``snippet`` / ``content`` / ``id``
    """
    k = TOP_K if top_k is None else int(top_k)
    cut = THRESHOLD if threshold is None else float(threshold)
    if k <= 0:
        return []

    hits: List[Tuple[Dict[str, Any], float]] = embed_store.search(question, k=k)

    results: List[Dict[str, Any]] = []
    for doc, score in hits:
        if cut > 0 and score < cut:
            continue
        metadata = doc.get("metadata") or {}
        content = doc.get("content", "")
        results.append({
            "id": doc.get("id"),
            "source": metadata.get("source") or metadata.get("file") or "未知来源",
            "score": float(score),
            "chunk_index": metadata.get("chunk_index"),
            "snippet": content[:SNIPPET_LEN],
            "content": content,
        })
    return results


def answer(
    question: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    *,
    call_llm: bool = True,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """完整走一遍 RAG：检索 → 组装提示词 → 调模型 → 返回回答与来源。

    Args:
        question:   用户问题
        top_k:      检索条数，默认 config.TOP_K
        threshold:  相似度阈值，默认 config.THRESHOLD
        call_llm:   为 False 时只做检索不调模型（调试检索质量用）
        temperature: 覆盖默认温度

    Returns:
        见模块 docstring 的返回结构。

    Raises:
        ValueError: question 为空
        LLMError:  大模型调用失败（重试后仍失败），由调用方决定怎么提示用户
    """
    if not question or not str(question).strip():
        raise ValueError("question 不能为空")

    question = str(question).strip()
    sources = retrieve(question, top_k=top_k, threshold=threshold)

    # 没有任何可用资料：不必浪费一次模型调用，直接按话术拒答
    if not sources:
        total_indexed = embed_store.count()
        if total_indexed == 0:
            reason = "索引为空，请先上传文件"
        else:
            reason = f"检索到的候选块相似度均低于阈值 {THRESHOLD if threshold is None else threshold}"
        return {
            "question": question,
            "answer": REFUSAL_TEXT,
            "sources": [],
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
    from config import ensure_dirs, summary

    ensure_dirs()
    print(summary())

    query = sys.argv[1] if len(sys.argv) > 1 else "网关监听哪个端口？"
    print(f"\n索引内共 {embed_store.count()} 条\n")

    try:
        out = answer(query)
    except LLMError as e:
        print(f"[失败] {e}")
        sys.exit(1)

    print(f"问题 : {out['question']}")
    print(f"回答 : {out['answer']}")
    print(f"拒答 : {out['refused']}  {out['refuse_reason']}")
    print(f"用量 : {out['usage']}")
    print("\n来源：")
    for i, s in enumerate(out["sources"], 1):
        print(f"  [{i}] {s['source']}  score={s['score']:.4f}  chunk={s['chunk_index']}")
        print(f"      {s['snippet']}...")
