# -*- coding: utf-8 -*-
"""文本切分

把长文档切成固定大小的块，块之间保留重叠，供向量化与检索使用。

切分优先级（由粗到细）：段落（``\\n\\n``）> 句子（``。！？`` / 换行）> 分句（``；，、``）
> 按字符数硬切。每一轮只在当前窗口（``chunk_size`` 个字符）内找**最靠右**的切点，
所以块长会尽量贴近 ``chunk_size``，不会因为某个靠左的分隔符就把块切碎。

实现要点（这三条都是踩过坑写下的，改这块代码前请先看懂）：

1. **迭代，不递归。** 原实现用尾递归处理"剩余部分"，调用深度与块数 1:1 增长，
   一份 5.4 万字的文档就会 ``RecursionError``。改成 ``while`` 后与文档长度无关。
2. **切分粒度只按"当前窗口找不找得到"下探，不随进度降级。** 原实现把 ``depth + 1``
   往下传，于是第一块按段落切、后面被永久锁死在最细一级，块越切越碎。
3. **切点必须越过重叠区。** 原实现 ``start = max(切点 - overlap, 0)``，当切点 ≤ overlap
   时 ``start`` 被钉在 0（还被强行改成 1），每轮只前进 1 个字符，会吐出成百上千个
   几乎相同的块——实测 8093 字的文档切出 449 块、产出字符数是原文的 3.4 倍。
   现在切点必须 ≥ ``overlap + 1``，否则换更细的分隔符、最后硬切，从根上保证每轮
   至少前进 1 个字符（正常情况前进 ``chunk_size - overlap``）。
"""

from typing import List

#: 分隔符三级，由粗到细。同一级里的多个分隔符取**最靠右**的命中位置。
_SEPARATOR_GROUPS = (
    ("\n\n", "\n\r\n", "\r\n\r"),          # 段落
    ("。", "！", "？", "\n", ".\n"),        # 句子 / 换行
    ("；", "，", "；\n", "，\n", "、"),      # 分句
)


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    将文本切分为固定大小的块，块之间有重叠。

    切分优先级：段落（\\n\\n）> 句子（。！？）> 逗号/分号 > 字符数硬切。

    Args:
        text:          待切分的文本
        chunk_size:    每块最大字符数
        chunk_overlap: 相邻块之间的重叠字符数（大于等于 chunk_size 时会被收紧到
                       chunk_size - 1，否则每轮无法前进）

    Returns:
        切分后的文本块列表

    Raises:
        ValueError: chunk_size 不是正整数
    """
    if not text:
        return []

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，收到 {chunk_size}")

    # 收紧重叠：上限取「块长的一半」。两个原因——
    #   1. 重叠 >= 块长时，下一轮起点会退回到本轮起点之前，每轮只前进 1 个字符。
    #      实测 chunk_size=500/overlap=500 会把 800 字的输入吐成 301 个 500 字的块。
    #   2. 重叠超过半个块时，相邻两块有一半以上是重复内容，检索时纯属挤占名额。
    # 这里收紧而不是抛错，是为了对旧调用方保持宽容（原实现靠 `if start == 0: start = 1`
    # 硬扛，代价就是产出大量近似重复块）。
    overlap = max(0, min(int(chunk_overlap), chunk_size // 2))

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    total = len(text)
    start = 0
    # 切点下限：保证 start 严格递增（见模块 docstring 第 3 条）
    min_pos = overlap + 1

    while start < total:
        end = start + chunk_size
        if end >= total:
            # 尾巴：剩余部分不超过重叠长度时，它已被上一块完整覆盖，不必重复产出
            tail = text[start:]
            if tail.strip() and len(tail) > overlap:
                chunks.append(tail)
            break

        cut = _find_split_pos(text[start:end], chunk_size, min_pos)
        chunk = text[start:start + cut]
        if chunk.strip():
            chunks.append(chunk)

        next_start = start + cut - overlap
        if next_start <= start:        # 兜底，正常不可达（overlap < chunk_size 已保证）
            next_start = start + cut
        start = next_start

    return chunks


def _find_split_pos(window: str, chunk_size: int, min_pos: int) -> int:
    """在窗口内找切点：从粗到细试分隔符，命中即用该级**最靠右**的位置。

    Args:
        window:     当前待切窗口，长度不超过 chunk_size
        chunk_size: 硬切上限
        min_pos:    切点下限。低于它的切点一律不用——要么换更细的分隔符，
                    要么走硬切，都不能让切点落进重叠区里。

    Returns:
        相对窗口起点的切点位置，范围 ``[min_pos, len(window)]``。
    """
    for separators in _SEPARATOR_GROUPS:
        best = -1
        for sep in separators:
            pos = window.rfind(sep, 0, chunk_size)
            if pos != -1:
                # 分隔符本身留在上一块末尾，所以切点要加上它的长度。
                # 取 max 而不是遇到就 break：同级里"更靠右"的才算更好的切点，
                # 早期版本 break 在第一个命中的分隔符上，会把切点拉到很靠左的位置。
                best = max(best, pos + len(sep))
        if best >= min_pos:
            return min(best, len(window))
    return len(window)


def _get_separators(depth: int) -> List[str]:
    """按粒度层级返回分隔符列表（depth 越大越细，超出范围取最细一级）。

    保留此函数仅为兼容旧调用点。新的切分逻辑自己遍历 ``_SEPARATOR_GROUPS``，
    不再用 depth 控制"与进度相关的粒度"。
    """
    return list(_SEPARATOR_GROUPS[min(depth, len(_SEPARATOR_GROUPS) - 1)])


def split_document(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    加载文档并直接切分为文本块（loader + splitter 快捷组合）。

    Args:
        file_path:     文档路径
        chunk_size:    每块最大字符数
        chunk_overlap: 相邻块重叠字符数

    Returns:
        切分后的文本块列表
    """
    from loader import load_document

    text = load_document(file_path)
    return split_text(text, chunk_size, chunk_overlap)
