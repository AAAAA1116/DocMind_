# -*- coding: utf-8 -*-
"""DocMind_ 核心模块包。

包含：
    - embed_store: 文本向量化与 ChromaDB 检索（第三阶段）
    - llm_client:  DeepSeek 调用封装（第四阶段）
    - rag_chain:   检索 + 大模型的问答链（第四阶段）
    - logger:      问答日志（第四阶段）

为方便在任意工作目录下 import，这里把项目根目录加入 ``sys.path``，
这样 ``from config import ...`` 在核心模块里始终可用。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# embed_store 是核心依赖，且它自身对 torch / chromadb 做了懒加载，
# 所以这里直接导入不会带来启动开销。
from .embed_store import (  # noqa: E402
    COLLECTION_NAME,
    INDEX_DIR,
    MODEL_NAME,
    EmbedStore,
    add_documents,
    count,
    get_store,
    reset_index,
    search,
)

__all__ = [
    "EmbedStore",
    "add_documents",
    "search",
    "get_store",
    "reset_index",
    "count",
    "MODEL_NAME",
    "COLLECTION_NAME",
    "INDEX_DIR",
    # 以下为懒加载子模块
    "llm_client",
    "rag_chain",
    "logger",
]

#: 懒加载的子模块，避免 import core 就把 streamlit / openai 全套拉起来
_LAZY = {"llm_client", "rag_chain", "logger"}


def __getattr__(name: str):
    """PEP 562 模块级懒加载：``from core import rag_chain`` 时才真正导入。"""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
