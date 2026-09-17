from typing import List


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[str]:
    """
    将文本递归切分为固定大小的块，块之间有重叠。

    切分优先级：段落（\\n\\n）> 句子（。！？）> 逗号/分号 > 字符数硬切。

    Args:
        text:          待切分的文本
        chunk_size:    每块最大字符数
        chunk_overlap: 相邻块之间的重叠字符数

    Returns:
        切分后的文本块列表
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    _recursive_split(text, chunk_size, chunk_overlap, 0, chunks)
    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    depth: int,
    chunks: List[str],
) -> None:
    """递归切分，depth 控制切分粒度层级"""
    if len(text) <= chunk_size:
        chunks.append(text)
        return

    # 按优先级选择分隔符
    separators = _get_separators(depth)

    best_split_pos = -1
    for sep in separators:
        pos = text.rfind(sep, 0, chunk_size)
        if pos != -1:
            best_split_pos = pos + len(sep)
            break

    # 没找到合适分隔符，尝试更细粒度的分隔符层级
    if best_split_pos == -1 and depth < 2:
        _recursive_split(text, chunk_size, chunk_overlap, depth + 1, chunks)
        return

    # 所有层级都找不到，按 chunk_size 硬切
    if best_split_pos == -1 or best_split_pos > chunk_size:
        best_split_pos = chunk_size

    chunks.append(text[:best_split_pos])

    # 剩余部分带重叠继续切分（防止 overlap 过大退回起点导致无限递归）
    start = max(best_split_pos - chunk_overlap, 0)
    if start >= len(text):
        return
    if start == 0:
        start = 1  # 至少推进 1 个字符
    _recursive_split(text[start:], chunk_size, chunk_overlap, depth + 1, chunks)


def _get_separators(depth: int) -> List[str]:
    """根据递归深度返回分隔符列表，越深分隔符越细"""
    separator_groups = [
        ["\n\n", "\n\r\n", "\r\n\r"],
        ["。", "！", "？", "\n", ".\n"],
        ["；", "，", "；\n", "，\n", "、"],
    ]
    return separator_groups[min(depth, len(separator_groups) - 1)]


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