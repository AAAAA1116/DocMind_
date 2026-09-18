# -*- coding: utf-8 -*-
"""向量化与检索（第三阶段）

职责
----
1. 用 sentence-transformers 加载 ``BAAI/bge-small-zh-v1.5``，把中文文本编码成向量
2. 用 ChromaDB 的持久化客户端把向量存到 ``data/index/``
3. 对外提供 ``add_documents(docs)`` 建索引、``search(query, k)`` 检索

两层调用方式
------------
高层（推荐，模块级函数，内部复用同一个单例）::

    from core.embed_store import add_documents, search

    add_documents([
        {"content": "网关默认监听 8080 端口", "metadata": {"source": "manual.txt"}},
    ])
    results = search("网关监听哪个端口？", k=3)
    for doc, score in results:
        print(score, doc["content"])

低层（需要多集合隔离时用类）::

    from core.embed_store import EmbedStore
    store = EmbedStore(collection_name="my_docs")
    store.add_documents(docs)

模型使用要点（以下结论均来自官方模型卡原文，不是我拍的）
--------------------------------------------------------
``BAAI/bge-*-zh-v1.5`` 是检索模型，官方用法有两条硬约束和一条可选：

1. **文档（passage）一律不加前缀**。官方原文：
   "In all cases, the documents/passages do not need to add the instruction."
2. **查询（query）前缀是可选优化**，不是必须。官方原文：
   "For the ``bge-*-v1.5``, we improve its retrieval ability when not using instruction.
   No instruction only has a slight degradation in retrieval performance compared with using instruction."
   本项目是「短查询找长文档块」，落在官方建议加前缀的场景，所以默认开启；
   想关掉设 ``DOCMIND_QUERY_INSTRUCTION=0``。官方也说了，
   "The best method to decide whether to add instructions for queries is choosing
   the setting that achieves better performance on your task." —— 该实测就实测。
3. **相似度只看相对顺序，不看绝对值**。官方原文：
   "the similarity distribution of the current BGE model is about in the interval [0.6, 1].
   So a similarity score greater than 0.5 does not indicate that the two sentences are similar."
   "what matters is the relative order of the scores, not the absolute value."

bge 输出归一化向量，因此相似度统一用**余弦相似度**（Chroma 的 cosine 空间）。

关于 HuggingFace 镜像
---------------------
国内直连 ``huggingface.co`` 会失败。本模块在导入时若发现环境变量 ``HF_ENDPOINT``
为空，会自动指向 ``https://hf-mirror.com``。要改用其它源（例如私有镜像），
在运行前设置 ``HF_ENDPOINT`` 即可覆盖，或在项目根 ``.env`` 里写一行::

    HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# HuggingFace 镜像：必须在 import huggingface_hub 之前设置好，否则不生效
# ---------------------------------------------------------------------------
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
# Windows 默认不支持符号链接，huggingface_hub 会刷一大段警告，按住它
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
COLLECTION_NAME = "docmind"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "models"

#: bge 检索时给 query 加的前缀（官方 Model List 中标注为 bge-*-zh-v1.5 的 query instruction）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

#: 是否给查询加前缀。官方模型卡原文：
#:   "For the `bge-*-v1.5`, we improve its retrieval ability when not using instruction.
#:    No instruction only has a slight degradation in retrieval performance compared with
#:    using instruction. So you can generate embedding without instruction in all cases for convenience.
#:    For a retrieval task that uses short queries to find long related documents,
#:    it is recommended to add instructions for these short queries.
#:    The best method to decide whether to add instructions for queries is choosing
#:    the setting that achieves better performance on your task."
#: 本项目是「短查询 → 长文档块」，属于官方建议加前缀的场景，故默认开启。
#: 想关掉：设环境变量 DOCMIND_QUERY_INSTRUCTION=0
USE_QUERY_INSTRUCTION = os.environ.get("DOCMIND_QUERY_INSTRUCTION", "1") not in ("0", "false", "False", "")

#: 向量维度。bge-small-zh-v1.5 固定 512 维，仅用于启动自检
EMBEDDING_DIM = 512

#: Chroma 不允许的 metadata 值类型会被过滤掉
_ALLOWED_META_TYPES = (str, int, float, bool)


class EmbedStoreError(RuntimeError):
    """本模块的业务异常，便于调用方区分是依赖缺失还是用法错误。"""


# ---------------------------------------------------------------------------
# 依赖懒加载（避免 import 本模块就吃下 torch / chromadb 的启动开销）
# ---------------------------------------------------------------------------
_model = None
_client = None
_collection_cache: Dict[str, Any] = {}


def _import_deps():
    """延迟导入重依赖，缺失时给出可直接复制的安装命令。"""
    try:
        import chromadb  # noqa: F401
    except ImportError as e:
        raise EmbedStoreError(
            "缺少 chromadb，请先安装：pip install chromadb"
        ) from e
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as e:
        raise EmbedStoreError(
            "缺少 sentence-transformers，请先安装：pip install sentence-transformers"
        ) from e


def _get_model():
    """加载并缓存 embedding 模型（进程内单例）。"""
    global _model
    if _model is not None:
        return _model

    _import_deps()
    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "")
    print(f"[embed_store] 加载模型 {MODEL_NAME}（源: {endpoint or 'huggingface.co 官方'}）")

    try:
        _model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    except TypeError:
        # 老版本 sentence-transformers 参数名可能不同
        _model = SentenceTransformer(MODEL_NAME)

    dim = _get_embedding_dim(_model)
    if dim != EMBEDDING_DIM:
        print(f"[embed_store] 注意：模型维度为 {dim}，与预期的 {EMBEDDING_DIM} 不一致")
    return _model


def _get_embedding_dim(model) -> int:
    """sentence-transformers 6.x 把 get_sentence_embedding_dimension 改名为 get_embedding_dimension，
    这里两个名字都兼容，免得升级时报警告。"""
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        fn = getattr(model, name, None)
        if callable(fn):
            return fn()
    return getattr(model, "get_sentence_embedding_dimension", lambda: EMBEDDING_DIM)()


def get_client():
    """获取 ChromaDB 持久化客户端（进程内单例），索引存放在 data/index/。"""
    global _client
    if _client is not None:
        return _client

    _import_deps()
    import chromadb

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return _client


def _get_collection(name: str = COLLECTION_NAME):
    """拿到（或创建）集合，统一使用余弦距离。"""
    if name in _collection_cache:
        return _collection_cache[name]

    client = get_client()

    # chromadb 1.x 用 configuration 传 hnsw 参数，0.4.x 用 metadata["hnsw:space"]，
    # 这里先试新 API，失败再退回旧 API，以兼容两个大版本。
    collection = None
    try:
        collection = client.get_or_create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
        )
    except (TypeError, ValueError, KeyError):
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    _warn_if_not_cosine(collection, name)
    _collection_cache[name] = collection
    return collection


def _warn_if_not_cosine(collection, name: str) -> None:
    """老索引若建成 L2 空间，相似度解释会完全不同，这里明确提示。"""
    space = None
    try:
        cfg = getattr(collection, "configuration", None)
        if isinstance(cfg, dict):
            space = (cfg.get("hnsw") or {}).get("space")
        if space is None:
            meta = getattr(collection, "metadata", None) or {}
            space = meta.get("hnsw:space")
    except Exception:
        return
    if space and space != "cosine":
        print(
            f"[embed_store] 警告：集合 '{name}' 的距离空间是 '{space}' 而非 cosine。"
            f"相似度分数将与文档说明不符，建议先 reset_index() 重建。"
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _clean_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Chroma 只接受标量 metadata，这里过滤掉 None / list / dict 等非法值。"""
    if not metadata:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, _ALLOWED_META_TYPES) and value is not None:
            cleaned[str(key)] = value
    return cleaned


def _content_id(content: str) -> str:
    """用内容哈希作为默认 id，保证重复入库时是覆盖而不是新增。"""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _normalize_docs(docs: Sequence[Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """把入参统一成 (ids, contents, metadatas) 三个平行列表。"""
    ids: List[str] = []
    contents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs):
        if isinstance(doc, str):
            content, metadata, doc_id = doc, {}, None
        elif isinstance(doc, dict):
            content = doc.get("content")
            metadata = doc.get("metadata") or {}
            doc_id = doc.get("id")
            if content is None:
                raise ValueError(f"第 {i} 个文档缺少 'content' 字段：{doc!r}")
            if not isinstance(metadata, dict):
                raise ValueError(f"第 {i} 个文档的 'metadata' 必须是 dict，实际是 {type(metadata).__name__}")
        else:
            raise TypeError(f"第 {i} 个文档类型不支持：{type(doc).__name__}，应为 dict 或 str")

        content = str(content)
        if not content.strip():
            raise ValueError(f"第 {i} 个文档的 content 为空，跳过空文档以免污染索引")

        ids.append(str(doc_id) if doc_id else _content_id(content))
        contents.append(content)
        metadatas.append(_clean_metadata(metadata))

    # 同一批里若出现重复 id，Chroma 会报错，这里保留最后一条
    seen: Dict[str, int] = {}
    dedup_ids, dedup_contents, dedup_metas = [], [], []
    for _id, _c, _m in zip(ids, contents, metadatas):
        if _id in seen:
            idx = seen[_id]
            dedup_contents[idx], dedup_metas[idx] = _c, _m
        else:
            seen[_id] = len(dedup_ids)
            dedup_ids.append(_id)
            dedup_contents.append(_c)
            dedup_metas.append(_m)

    return dedup_ids, dedup_contents, dedup_metas


def _cosine_to_score(distance: float) -> float:
    """Chroma cosine 空间返回的是 1 - 余弦相似度，这里还原成相似度。"""
    return 1.0 - float(distance)


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------
class EmbedStore:
    """向量库封装：负责编码、入库、检索。

    Args:
        collection_name: 集合名，默认 ``docmind``。多个业务线可以各用一个集合。
        batch_size:      编码时的批大小，显存/内存紧张可调小。
    """

    def __init__(self, collection_name: str = COLLECTION_NAME, batch_size: int = 32) -> None:
        self.collection_name = collection_name
        self.batch_size = batch_size

    @property
    def collection(self):
        return _get_collection(self.collection_name)

    # -- 编码 ---------------------------------------------------------------
    def encode_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """把文档编码成向量（不加查询前缀）。"""
        model = _get_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_query(self, query: str) -> List[float]:
        """把查询编码成向量。

        按模块级 ``USE_QUERY_INSTRUCTION`` 决定是否加 bge 的检索前缀
        （默认加，因为本项目属于官方建议加前缀的「短查询找长文档」场景）。
        """
        model = _get_model()
        text = QUERY_INSTRUCTION + query if USE_QUERY_INSTRUCTION else query
        vector = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    # -- 写入 ---------------------------------------------------------------
    def add_documents(self, docs: Sequence[Any]) -> int:
        """把文档列表向量化后写入集合。

        Args:
            docs: 文档列表，每个元素形如::

                    {"content": "正文文本", "metadata": {"source": "a.pdf", "page": 3}}

                - ``content`` 必填，为空会报错
                - ``metadata`` 选填，只能是 str/int/float/bool（Chroma 限制），其余类型会被丢弃
                - ``id`` 选填，不给就用 content 的 sha1，**重复入库同内容会覆盖而非重复**

                也可以直接传字符串，等价于只给 content。

        Returns:
            实际写入（upsert）的文档条数。
        """
        if not docs:
            return 0

        ids, contents, metadatas = _normalize_docs(docs)
        vectors = self.encode_documents(contents)
        self.collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=vectors,
            metadatas=metadatas,
        )
        return len(ids)

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """检索与 query 最相似的 k 个文档。

        Args:
            query: 查询文本
            k:     返回条数

        Returns:
            列表，每项为 ``(文档, 相似度分数)``：

            - 文档是 dict，含 ``id`` / ``content`` / ``metadata`` 三个键
            - 相似度是**余弦相似度**，越大越相似；整体已按它从高到低排序
            - 索引为空或 ``k <= 0`` 时返回空列表

        .. warning::
            **不要把绝对分数当作「相关/不相关」的判据。** 官方模型卡明确说明：
            bge 用温度 0.01 的对比学习训练，相似度分布集中在 [0.6, 1]，
            「大于 0.5 并不代表两句相似」；
            「真正重要的是分数的相对顺序，而不是绝对值」。
            如果要设阈值过滤，请在你自己的数据上观察分布后再定（官方举例 0.8 / 0.85 / 0.9）。
            本模块的 score 只应被用来对候选块排序。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        if k <= 0:
            return []

        total = self.collection.count()
        if total == 0:
            return []

        result = self.collection.query(
            query_embeddings=[self.encode_query(query)],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: List[Tuple[Dict[str, Any], float]] = []
        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append((
                {"id": doc_id, "content": content, "metadata": metadata or {}},
                _cosine_to_score(distance),
            ))
        return hits

    # -- 运维 ---------------------------------------------------------------
    def count(self) -> int:
        """集合内当前文档条数。"""
        return self.collection.count()

    def reset(self) -> None:
        """删除并重建集合（换模型、换切分策略后必须调一次）。"""
        client = get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        _collection_cache.pop(self.collection_name, None)
        _get_collection(self.collection_name)


# ---------------------------------------------------------------------------
# 模块级便捷函数（第三阶段提示词里要求的那两个）
# ---------------------------------------------------------------------------
_default_store: Optional[EmbedStore] = None


def get_store() -> EmbedStore:
    """获取默认集合的单例封装。"""
    global _default_store
    if _default_store is None:
        _default_store = EmbedStore()
    return _default_store


def add_documents(docs: Sequence[Any]) -> int:
    """向量化 docs 并写入默认集合。文档格式见 :meth:`EmbedStore.add_documents`。"""
    return get_store().add_documents(docs)


def search(query: str, k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
    """在默认集合里检索 top-k，返回 ``[(文档, 相似度分数), ...]``。"""
    return get_store().search(query, k=k)


def count() -> int:
    """默认集合内的文档条数。"""
    return get_store().count()


def reset_index() -> None:
    """清空默认集合并重建（会丢失已建索引）。"""
    get_store().reset()


# ---------------------------------------------------------------------------
# 自测：直接 python -m core.embed_store 就能跑
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=" * 72)
    print(f"模型      : {MODEL_NAME}")
    print(f"索引目录  : {INDEX_DIR}")
    print(f"模型缓存  : {MODEL_CACHE_DIR}")
    print(f"HF 源     : {os.environ.get('HF_ENDPOINT')}")
    print("=" * 72)

    demo_docs = [
        {"content": "网关默认以单进程模式运行，监听 8080 端口，生产环境建议前置 Nginx。",
         "metadata": {"source": "gateway.txt", "section": 1}},
        {"content": "所有外部请求必须携带 X-Auth-Token 请求头，令牌有效期为 7200 秒。",
         "metadata": {"source": "gateway.txt", "section": 2}},
        {"content": "默认对单个来源 IP 限流为每秒 100 次请求，超出返回 429 状态码。",
         "metadata": {"source": "gateway.txt", "section": 3}},
        {"content": "上游请求默认超时 15 秒，失败后最多重试 2 次，采用指数退避策略。",
         "metadata": {"source": "gateway.txt", "section": 4}},
        {"content": "系统暴露 Prometheus 格式指标，端点默认路径为 /metrics。",
         "metadata": {"source": "gateway.txt", "section": 5}},
    ]

    reset_index()
    n = add_documents(demo_docs)
    print(f"\n写入 {n} 条，当前集合共 {count()} 条\n")

    for q in ("网关监听哪个端口？", "请求超时和重试是怎么规定的？", "今天天气怎么样"):
        print(f"查询: {q}")
        for rank, (doc, score) in enumerate(search(q, k=3), 1):
            snippet = doc["content"][:34]
            print(f"   {rank}. 相似度 {score:.4f}  [{doc['metadata'].get('section')}] {snippet}...")
        print()
    sys.exit(0)
