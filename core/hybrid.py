# -*- coding: utf-8 -*-
"""混合检索（第六阶段）：BM25 字面召回 + 向量语义召回 → RRF 融合

职责
----
给检索层补上**第二路召回**，再把两路结果合成一个排序：

    问题 ─┬─► embed_store.search()   向量召回（看语义）  ─┐
          └─► BM25Index.search()     词面召回（看字面）  ─┴─► rrf_fuse() ─► 统一候选列表

为什么需要它：向量检索的盲区
----------------------------
向量检索把 query 和文档**各自**压成一个稠密向量再比距离，比的是「意思像不像」。
它在下面这几类查询上会失手，而这几类在企业知识库里恰恰很常见：

* **精确字面量**：错误码 ``E1002``、型号 ``XH-2000``、端口 ``8080``、接口路径 ``/metrics``
* **专有名词 / 缩写 / 生僻词**：向量模型没见过的新词，编码出来接近随机向量
* **极短查询**：一两个词的问法，语义信息太少，向量结果不稳定

这些正是 BM25 的强项——它只统计**词有没有出现**，命中就是命中，不做语义猜测。

.. important::
    **别对 BM25 抱错期待：它解决不了「换个说法」的问题。**

    「十三薪」和「年终双薪」是**转述关系（paraphrase）**，两串词的字面交集只有「薪」
    一个字。BM25 靠的是词项匹配，对这种同义改写同样召回不出来——真正该干这活的是
    向量那一路（以及精排模型）。

    所以两路是**互补**，各管一段：向量管「语义像不像」，BM25 管「字面有没有」。
    指望 BM25 单独打通同义词，是把它用在它不擅长的地方。

RRF 为什么只用「名次」不用「分数」
----------------------------------
余弦相似度（bge 的分数挤在 0.6~1 这个窄区间）和 BM25 分数（无上界、随语料规模变化）
**不是同一个量纲**，直接加权相加毫无意义——谁的量纲大谁说了算，另一路等于白搭。

RRF（Reciprocal Rank Fusion）干脆绕开分数，只用**名次**::

    RRF(d) = Σ_通道  权重 / (k + rank_通道(d))

``k`` 默认取 60（原论文的取值），作用是压住头部名次：排第 1 和第 2 的差距被摊平，
不让任何一路的「第一名」独裁。它不需要归一化、不需要调参、对异常分数免疫，
是混合检索里最省事、最不容易出错的融合方式。

排序细节
--------
* 候选按 ``id`` 去重：同一个块被两路都召回时，两边的名次**分别计分**（这正是 RRF 的价值）
* 某一路没召回它，就在**那一路**上不加分，而不是按「最后一名」计分
* 原分数（余弦 / BM25）会被原样带出来，只用于展示和排查，**不参与融合**

与其它模块的关系
----------------
* ``core/embed_store.py``：向量库，本模块通过 ``all_documents()`` 取全量语料建 BM25 索引
* ``core/reranker.py``：精排，接在**本模块之后**，吃本模块输出的候选
* ``core/rag_chain.py``：编排，决定用不用混合、是否再叠精排

BM25 实现说明
-------------
不引 ``rank_bm25`` 之类的小包——Okapi BM25 本体只有四十来行，倒排表加打分逻辑都在
本文件里，可读可控。**唯一新增的依赖是 ``jieba``**（中文分词），见 ``tokenize()``。

分词依赖 jieba，缺了会怎样
--------------------------
``jieba`` 未安装时 ``tokenize()`` 会抛 ``HybridError``（带可直接复制的安装命令）。
``rag_chain`` 会捕获它、打印一行提示，然后**自动退回纯向量检索**——问答不会中断。
"""

from __future__ import annotations

import math
import re
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    HYBRID_BM25_WEIGHT,
    HYBRID_PER_SOURCE,
    HYBRID_RRF_K,
    HYBRID_VECTOR_WEIGHT,
)
from core import embed_store  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: BM25 词频饱和参数。控制「出现 1 次」和「出现 5 次」差多少：越大越看重重复出现。
#: 1.2~2.0 是文献里的常见区间，1.5 是 Lucene 默认值。
DEFAULT_K1 = 1.5

#: BM25 长度归一参数，0~1。文档越长，命中同一个词的「含金量」越低。
#: 0.75 是 Lucene 默认值；设 0 等于关掉长度归一（长块会靠体量刷分）。
DEFAULT_B = 0.75

#: 英文 / 数字 / 代码串：整体抓成一个 token，不交给 jieba 切
#: （jieba 会把 ``x-auth-token`` 切成 ``x`` / ``-`` / ``auth`` / ``-`` / ``token``，
#:  精确字面查询就废了）
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+#:@\-]*")

#: 判断一段文本里有没有中日韩字符（有才需要 jieba）
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")

#: 至少含一个字母/数字/汉字才算有效词项，纯标点丢掉
_KEEP_RE = re.compile(r"[0-9A-Za-z\u3400-\u9fff]")


class HybridError(RuntimeError):
    """本模块的业务异常。调用方（rag_chain）捕获后可降级为纯向量检索。"""


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------
_jieba = None
_jieba_lock = threading.Lock()


def _get_jieba():
    """懒加载 jieba（首次 import 要读词典，别放在模块顶层）。"""
    global _jieba
    if _jieba is not None:
        return _jieba
    with _jieba_lock:
        if _jieba is None:
            try:
                import jieba
            except ImportError as e:      # pragma: no cover - 依赖缺失时的人话提示
                raise HybridError(
                    "缺少 jieba（中文分词），混合检索无法启用。安装：\n"
                    "    .venv\\Scripts\\python.exe -m pip install jieba\n"
                    "装不了可以先把 config.HYBRID_ENABLED 设为 False，退回纯向量检索。"
                ) from e
            # jieba 默认会往控制台刷「Building prefix dict...」，这里按住它
            try:
                jieba.setLogLevel(60)
            except Exception:
                pass
            _jieba = jieba
    return _jieba


def _cut_mixed(segment: str) -> List[str]:
    """对不含 ASCII 代码串的片段做分词：有中日韩字符就交给 jieba，否则原样留着。"""
    if not segment.strip():
        return []
    if not _CJK_RE.search(segment):
        return [segment.strip()]
    jieba = _get_jieba()
    return [t.strip() for t in jieba.cut(segment, cut_all=False, HMM=True) if t.strip()]


def tokenize(text: str) -> List[str]:
    """把文本切成检索用的词项（BM25 的输入单位）。

    规则（顺序很重要）：

    1. **先把英文 / 数字 / 代码串整体抠出来**，不参与中文分词。
       ``X-Auth-Token`` / ``8080`` / ``/metrics`` / ``E1002`` 必须保持完整，
       切碎之后精确字面匹配就没了意义。
    2. **剩下的片段交给 jieba 精确分词**：``年终双薪`` → ``年终`` / ``双薪``。
       用精确模式（``cut_all=False``）+ HMM 新词发现。
    3. 丢掉纯标点 / 空白词项，整体转小写。

    返回的是**保留重复**的词项列表（BM25 需要词频 tf），去重交给调用方。
    """
    if not text:
        return []

    text = str(text).lower()
    tokens: List[str] = []
    pos = 0

    for match in _ASCII_TOKEN_RE.finditer(text):
        if match.start() > pos:
            tokens.extend(_cut_mixed(text[pos:match.start()]))
        # 前后可能粘着标点（"./metrics" / "8080."），剥掉
        token = match.group().strip("._-+#:@/")
        if token:
            tokens.append(token)
        pos = match.end()

    if pos < len(text):
        tokens.extend(_cut_mixed(text[pos:]))

    return [t for t in tokens if _KEEP_RE.search(t)]


# ---------------------------------------------------------------------------
# 文档字段提取（兼容 dict / str 两种形态，与 reranker 的约定保持一致）
# ---------------------------------------------------------------------------
def _content(doc: Any) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("content", "text", "page_content"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    return ""


def _doc_id(doc: Any, fallback_pos: int) -> str:
    """取文档 id。正常来自 Chroma（内容 sha1），没有就用下标兜底，保证能去重。"""
    if isinstance(doc, dict) and doc.get("id") is not None:
        return str(doc["id"])
    return f"#pos{fallback_pos}"


# ---------------------------------------------------------------------------
# BM25 倒排索引
# ---------------------------------------------------------------------------
class BM25Index:
    """Okapi BM25 倒排索引。

    打分公式（Lucene / rank_bm25 采用的写法）::

        IDF(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score(q, d) = Σ_{t ∈ q} IDF(t) · tf · (k1 + 1) / (tf + k1 · (1 - b + b · dl / avgdl))

    两个刻意的选择：

    * **IDF 用 ``ln(1 + ...)`` 这个变体**，它恒为正。经典写法里高 df 的词 IDF 会变负数，
      导致「长文档命中一堆常见词反而扣分」，在短查询上很难解释。
    * **倒排表而不是逐文档扫描**：只给「命中了查询词」的文档打分。
      query 只有几个词时，这一步能省掉绝大部分计算量。
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = float(k1)
        self.b = float(b)

        self.docs: List[Dict[str, Any]] = []                  # 文档本体（原样保存，命中原样返回）
        self.ids: List[str] = []                              # 与 docs 平行的 id
        self.doc_len: List[int] = []                          # 每篇的词项总数
        self.postings: Dict[str, List[Tuple[int, int]]] = {}  # 词项 -> [(文档下标, 词频)]
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    # -- 建索引 -------------------------------------------------------------
    def build(self, docs: Sequence[Any]) -> "BM25Index":
        """用一批文档重建索引。会覆盖上一次的内容。

        Args:
            docs: 文档列表，元素是 dict（含 ``content`` / 可选 ``id`` / ``metadata``）或纯字符串。

        Note:
            这一步会**把所有文档分词一遍**，成本与语料规模成正比。
            几千个块的语料在秒级，所以 ``rag_chain`` 里是「语料变了就整体重建」，
            没有做增量维护——增量维护要处理删除、更新、词频回退，复杂度不划算。
        """
        self.docs = list(docs)
        self.ids = [_doc_id(doc, i) for i, doc in enumerate(self.docs)]

        doc_len: List[int] = []
        raw: Dict[str, Dict[int, int]] = {}       # 词项 -> {文档下标: 词频}

        for pos, doc in enumerate(self.docs):
            tokens = tokenize(_content(doc))
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                raw.setdefault(term, {})[pos] = tf

        n = len(self.docs)
        self.doc_len = doc_len
        self.avgdl = (sum(doc_len) / n) if n else 0.0
        self.postings = {term: sorted(hits.items()) for term, hits in raw.items()}
        self.idf = {
            term: math.log(1.0 + (n - len(hits) + 0.5) / (len(hits) + 0.5))
            for term, hits in raw.items()
        }
        return self

    # -- 检索 ---------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """检索 top-k，返回 ``[(文档, BM25 分数), ...]``，按分数从高到低。

        * ``bm25_score`` 无上界、随语料变化，**只能用于排序，不要当相关性判据**
        * 查询词一个都没在语料里出现 → 返回空列表（不是返回随机结果）
        * 同分按文档下标排（确定性），保证同一查询结果可复现
        """
        if k <= 0 or not self.docs or not query or not query.strip():
            return []

        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.postings]
        if not terms:
            return []

        scores: Dict[int, float] = {}
        k1, b, avgdl = self.k1, self.b, self.avgdl

        for term in terms:
            idf = self.idf[term]
            for pos, tf in self.postings[term]:
                dl = self.doc_len[pos] if pos < len(self.doc_len) else 0
                norm = (dl / avgdl) if avgdl > 0 else 0.0
                denom = tf + k1 * (1.0 - b + b * norm)
                if denom <= 0:
                    continue
                scores[pos] = scores.get(pos, 0.0) + idf * tf * (k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.docs[pos], score) for pos, score in ranked]

    def __len__(self) -> int:
        return len(self.docs)

    def stats(self) -> Dict[str, Any]:
        """索引概况，给界面 / 日志用。"""
        return {
            "docs": len(self.docs),
            "terms": len(self.postings),
            "avg_len": round(self.avgdl, 1),
            "k1": self.k1,
            "b": self.b,
        }


# ---------------------------------------------------------------------------
# 索引缓存：语料变了才重建
# ---------------------------------------------------------------------------
_index: Optional[BM25Index] = None
_index_key: Optional[Tuple[int, int]] = None
_index_lock = threading.Lock()


def _current_key() -> Tuple[int, int]:
    """缓存键 = ``(向量库写代数, 文档数)``。两者任一变化就重建。

    为什么两个都要：代数能抓住「删了又加、条数恰好不变」的情况，
    条数能抓住「别的进程改了索引目录」的情况。
    """
    return (embed_store.revision(), embed_store.count())


def get_index(force: bool = False) -> BM25Index:
    """拿到 BM25 索引（进程内单例），语料变了自动重建。

    .. warning::
        失效判定基于 **本进程** 的写代数 + 当前文档数。
        Streamlit 是单进程，够用；但如果你另开一个进程往同一个 ``data/index``
        写数据，本进程不会察觉（条数变了才会）——那种场景下请手动调 ``invalidate()``。
    """
    global _index, _index_key

    key = _current_key()
    if not force and _index is not None and _index_key == key:
        return _index

    with _index_lock:
        key = _current_key()
        if not force and _index is not None and _index_key == key:
            return _index
        docs = embed_store.all_documents()
        _index = BM25Index().build(docs)
        _index_key = key
        return _index


def invalidate() -> None:
    """丢掉缓存的索引，下次调用重建。"""
    global _index, _index_key
    with _index_lock:
        _index = None
        _index_key = None


def index_stats() -> Dict[str, Any]:
    """当前 BM25 索引的概况（会触发一次构建）。"""
    return get_index().stats()


# ---------------------------------------------------------------------------
# 单路入口
# ---------------------------------------------------------------------------
def bm25_search(query: str, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
    """只用 BM25 检索，返回 ``[(文档, BM25 分数), ...]``。"""
    return get_index().search(query, k=k)


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Dict[str, Any]]],
    *,
    k: int = HYBRID_RRF_K,
    weights: Optional[Sequence[float]] = None,
    top_n: Optional[int] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """RRF 融合多个「已排好序」的候选列表，返回 ``[(文档, RRF 分数), ...]``。

    :param ranked_lists: 若干候选列表，每个列表内部**必须已按各自的相关度从高到低排好**；
                         元素是带 ``id`` 的文档 dict（没有 ``id`` 会被自动兜底编号）
    :param k:            平滑常数，默认 60。越小越看重头部名次
    :param weights:      每个列表的权重，长度要与 ``ranked_lists`` 一致；``None`` 表示全 1.0
    :param top_n:        截断到前 N 个，``None`` 表示全部返回

    名次从 1 开始（``1/(k+1)`` 才符合 RRF 原式，从 0 开始会让第一名多吃一份不该有的分）。
    同分时按「首次出现的先后」排，两个列表的相对顺序就是 tie-break 顺序，结果可复现。
    """
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError(f"weights 长度 {len(weights)} 与候选列表数量 {len(ranked_lists)} 不一致")

    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order: List[str] = []
    by_id: Dict[str, Dict[str, Any]] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else float(weights[list_idx])
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(doc, rank - 1)
            if doc_id not in by_id:
                by_id[doc_id] = doc
                order.append(doc_id)
                first_seen[doc_id] = len(order)
                scores[doc_id] = 0.0
            scores[doc_id] += weight / (k + rank)

    fused = sorted(order, key=lambda d: (-scores[d], first_seen[d]))
    if top_n is not None and top_n >= 0:
        fused = fused[:top_n]
    return [(by_id[d], scores[d]) for d in fused]


def hybrid_search(
    query: str,
    k: int = 10,
    *,
    per_source: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """混合检索主入口：向量 + BM25 → RRF，返回带诊断信息的候选记录。

    Args:
        query:      用户问题
        k:          融合后返回多少个候选
        per_source: 每个通道各召回多少个，默认 ``config.HYBRID_PER_SOURCE``

    Returns:
        列表，每项形如::

            {
                "doc":          {"id", "content", "metadata"},   # 原样透传给精排
                "rrf_score":    0.0325,     # 融合分（只用于排序，跨查询不可比）
                "rrf_rank":     1,          # 融合后名次，从 1 开始
                "vector_score": 0.5412,     # 向量余弦；本路没召回就是 None
                "vector_rank":  3,
                "bm25_score":   8.774,      # BM25 分；本路没召回就是 None
                "bm25_rank":    1,
            }

        两个 ``*_score`` 各自量纲不同，**只用于观察「这个块是哪一路捞上来的」**，
        不要拿去互相比较或设阈值。

    Raises:
        HybridError: jieba 缺失等导致 BM25 建不起来。调用方应捕获并降级。
    """
    if k <= 0 or not query or not query.strip():
        return []

    depth = int(HYBRID_PER_SOURCE if per_source is None else per_source)
    depth = max(depth, k)

    # 向量那一路：注意这里**不做任何余弦阈值过滤**。
    # 阈值是「单路检索」时代的产物，混进来会把 BM25 召回的块连坐误伤
    # （它们压根没有余弦分数），也会让跨量纲比较重新污染决策。排序交给 RRF 和精排。
    vector_hits = embed_store.search(query, k=depth)
    bm25_hits = bm25_search(query, k=depth)

    vector_docs = [doc for doc, _ in vector_hits]
    bm25_docs = [doc for doc, _ in bm25_hits]

    # 权重从 config 读，默认 1.0 / 1.0 —— 刻意不去「调最优」，
    # 见 config.HYBRID_VECTOR_WEIGHT 的说明。
    fused = rrf_fuse(
        [vector_docs, bm25_docs],
        k=int(HYBRID_RRF_K),
        weights=[float(HYBRID_VECTOR_WEIGHT), float(HYBRID_BM25_WEIGHT)],
        top_n=k,
    )
    if not fused:
        return []

    vector_score = {_doc_id(doc, i): s for i, (doc, s) in enumerate(vector_hits)}
    vector_rank = {_doc_id(doc, i): i + 1 for i, (doc, _) in enumerate(vector_hits)}
    bm25_score = {_doc_id(doc, i): s for i, (doc, s) in enumerate(bm25_hits)}
    bm25_rank = {_doc_id(doc, i): i + 1 for i, (doc, _) in enumerate(bm25_hits)}

    records: List[Dict[str, Any]] = []
    for rank, (doc, score) in enumerate(fused, start=1):
        doc_id = _doc_id(doc, rank - 1)
        records.append({
            "doc": doc,
            "rrf_score": float(score),
            "rrf_rank": rank,
            "vector_score": vector_score.get(doc_id),
            "vector_rank": vector_rank.get(doc_id),
            "bm25_score": bm25_score.get(doc_id),
            "bm25_rank": bm25_rank.get(doc_id),
        })
    return records


# ---------------------------------------------------------------------------
# 自测：python -m core.hybrid
#   不碰向量库，只验证分词 / BM25 打分 / RRF 融合三段逻辑
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_docs = [
        {"id": "d1", "content": "网关默认以单进程模式运行，监听 8080 端口，生产环境建议前置 Nginx。"},
        {"id": "d2", "content": "所有外部请求必须携带 X-Auth-Token 请求头，令牌有效期为 7200 秒。"},
        {"id": "d3", "content": "默认对单个来源 IP 限流为每秒 100 次请求，超出返回 429 状态码。"},
        {"id": "d4", "content": "上游请求默认超时 15 秒，失败后最多重试 2 次，采用指数退避策略。"},
        {"id": "d5", "content": "系统暴露 Prometheus 格式指标，端点默认路径为 /metrics。"},
        {"id": "d6", "content": "错误码 E1002 表示鉴权失败，需要检查令牌是否过期。"},
        {"id": "d7", "content": "年终双薪于每年 1 月随工资发放，具体比例见薪酬制度第四章。"},
    ]

    print("=" * 72)
    print("分词验证（英文/数字/代码串必须保持完整）")
    print("=" * 72)
    for text in ("请求头 x-auth-token 怎么传？", "错误码 E1002 是什么意思", "年终双薪什么时候发"):
        print(f"  {text}")
        print(f"    -> {tokenize(text)}")

    idx = BM25Index().build(demo_docs)
    print()
    print("=" * 72)
    print(f"BM25 索引：{idx.stats()}")
    print("=" * 72)
    for query in ("E1002", "X-Auth-Token 是什么", "/metrics 在哪", "十三薪"):
        hits = idx.search(query, k=3)
        print(f"\n  查询: {query}")
        if not hits:
            print("    （无命中——查询词在语料里一个都没出现）")
        for doc, score in hits:
            print(f"    {score:7.3f}  [{doc['id']}] {doc['content'][:30]}...")

    print()
    print("=" * 72)
    print("RRF 融合：d3 在向量路排第 2、BM25 路排第 1，应当被顶上来")
    print("=" * 72)
    vector_ranked = [demo_docs[0], demo_docs[2], demo_docs[6]]
    bm25_ranked = [demo_docs[2], demo_docs[5], demo_docs[6]]
    for i, (doc, score) in enumerate(rrf_fuse([vector_ranked, bm25_ranked]), start=1):
        print(f"  {i}. {score:.5f}  [{doc['id']}] {doc['content'][:26]}...")
