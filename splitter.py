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

import re
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# 结构切分（v1.5）
# ---------------------------------------------------------------------------
# 上面 split_text 那一套是「定长 + 分隔符对齐」：它认标点，不认文档结构。
# 对条款体文档（公司章程、管理制度）这等于明知答案在哪却绕着走 ——
# 「第X条」就写在那里，切点却跟着 500 这个数字走，于是枚举型条款被拦腰切断。
#
# 实测（7 份上市公司制度 PDF，986 条可测条款）：
#     定长切分      条款完整率 88.6%   索引展开率 1.20x
#     结构切分      条款完整率 100.0%  索引展开率 1.00x
# 而且两者是同一件事的两面：切点对了，重叠就没有存在意义（不再需要拿重复内容
# 去弥补"切在错误的地方"），索引体积反而降 17%。

#: 条款/章节标记。允许 markdown 加粗包裹（``**第一条**``）。
#: 只用它取**位置**，不解析编号 —— 中文数字还是阿拉伯数字都不影响切分。
_ARTICLE_RE = re.compile(r"(?:\*\*)?第[一二三四五六七八九十百零〇0-9]+条(?:\*\*)?")
_CHAPTER_RE = re.compile(r"(?:\*\*)?第[一二三四五六七八九十百零〇0-9]+章(?:\*\*)?")

#: 走结构切分所需的最少条款数。低于它说明这不是条款体文档（散文、笔记、
#: 一页纸的通知），硬套结构只会切得更碎 —— 此时返回 None，由调用方退回定长切分。
_MIN_ARTICLES = 5

#: 超长条款二次切时补的条款头后缀。
#: 用「…」而不是原样重复整条编号，是为了让续块**保留条款身份**（BM25 能靠
#: 「第X条」召回它），又不把它伪装成一个新的条款开头。
_CONT_SUFFIX = "…"


def split_by_structure(text: str, chunk_size: int = 500) -> Optional[List[str]]:
    """按文档自身的条款 / 章节结构切分。

    是**一个流程的三个步骤**，不是三个可选项：

    1. **主逻辑** —— 在 ``第X条`` 处切出原子条款，切点永远落在条款边界上；
    2. **补丁 A** —— 单条超过 ``chunk_size`` 时二次切，续块补回 ``第X条…`` 条款头
       （不补的话，续块失去条款身份，检索「第X条」时它不会命中）；
    3. **补丁 B** —— 相邻短条款贪心合并到 ``chunk_size``，但 ``第X章`` 处强制断开，
       绝不让两个章的条款挤进同一块（那等于给块注入错误的结构上下文）。

    两条补丁都是必需的，因为条款长度极度不均：实测 p50 只有 117 字、
    42.3% 的条款不到 100 字（"本制度由董事会负责解释。"这种），
    同时又有 51 条超过 480 字（最长 2324 字）。

    Args:
        text:       待切分的文本
        chunk_size: 每块最大字符数。**这是上限，不是目标** ——
                    块长由条款边界决定，不会被主动撑到这个数。

    Returns:
        切分后的块列表；若文档不含条款结构（条款数 < ``_MIN_ARTICLES``）返回 ``None``。
    """
    if not text:
        return None
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，收到 {chunk_size}")

    # 结构标记 = 条款 + 章节。两者按出现位置合并成一条有序序列，
    # 章节标记同时充当「不合并」的硬边界（补丁 B 的墙）。
    articles = [(m.start(), m.group(), "article") for m in _ARTICLE_RE.finditer(text)]
    if len(articles) < _MIN_ARTICLES:
        return None
    marks = articles + [(m.start(), m.group(), "chapter") for m in _CHAPTER_RE.finditer(text)]
    marks.sort(key=lambda item: item[0])

    # 切成原子段：每段 = 一个标记的起点 → 下一个标记的起点
    atoms: List[tuple] = []
    for i, (pos, label, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        seg = text[pos:end].strip()
        if seg:
            atoms.append((kind, label, seg))

    chunks: List[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for kind, label, seg in atoms:
        if kind == "chapter":
            # 章边界：硬断开，后面的条款不许并进上一章（补丁 B 的墙）。
            # 章标记本身仍走下面的通用逻辑 —— 章标题后面可能跟着一大段
            # 没有条款编号的文字（"附则"之类），那段同样要受 chunk_size 约束。
            flush()
        if len(seg) > chunk_size:
            flush()                                  # 超长段：先收尾，再二次切
            chunks.extend(_split_long_article(seg, label, chunk_size))
            continue
        if not buf:
            buf = seg
        elif len(buf) + 1 + len(seg) <= chunk_size:
            buf += "\n" + seg
        else:
            flush()
            buf = seg
    flush()
    return chunks


def _split_long_article(seg: str, label: str, chunk_size: int) -> List[str]:
    """超长条款二次切（补丁 A）。复用定长逻辑，重叠取 0。

    宽度按 ``chunk_size - len(条款头)`` 算，而不是 ``chunk_size`` ——
    续块要额外顶一个 ``第一百七十一条…`` 前缀，不预先扣掉就会突破上限。
    """
    head = label + _CONT_SUFFIX
    subs = split_text(seg, max(1, chunk_size - len(head)), 0)
    if len(subs) <= 1:
        return subs
    return [subs[0]] + [head + s.lstrip() for s in subs[1:]]


def split_text_auto(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_structure: bool = True,
) -> List[str]:
    """切分总入口：有条款结构走结构切分，否则退回定长切分。

    两条路径共用 ``chunk_size`` 作为上限，但**重叠的用法不同**：

    * 结构切分路径 **不使用重叠**。切点落在条款边界上，重叠内容纯属冗余 ——
      它和正确答案抢 TOP_K 名额，还把索引撑大 20%。
    * 定长切分路径 仍用 ``chunk_overlap``，因为它正是靠重叠来弥补"切在句子中间"。

    Args:
        text:           待切分文本
        chunk_size:     每块最大字符数
        chunk_overlap:  定长路径的相邻块重叠字数
        use_structure:  设 False 可强制走定长路径（做 A/B 对照用）

    Returns:
        切分后的文本块列表
    """
    if use_structure:
        structured = split_by_structure(text, chunk_size)
        if structured:
            return structured
    return split_text(text, chunk_size, chunk_overlap)


def split_document(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_structure: bool = True,
) -> List[str]:
    """
    加载文档并直接切分为文本块（loader + splitter 快捷组合）。

    Args:
        file_path:     文档路径
        chunk_size:    每块最大字符数
        chunk_overlap: 相邻块重叠字符数（仅定长路径使用）
        use_structure: 是否优先按条款结构切分（默认开启，无结构时自动退回定长）

    Returns:
        切分后的文本块列表
    """
    from loader import load_document

    text = load_document(file_path)
    return split_text_auto(text, chunk_size, chunk_overlap, use_structure=use_structure)
