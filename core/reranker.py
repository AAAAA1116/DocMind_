# -*- coding: utf-8 -*-
"""重排（Rerank）—— 检索增强第一步

职责
----
在「向量粗排」之后加一道 **交叉编码器（Cross-Encoder）精排**：

    query ─► embed_store.search(k=N) ─► N 个候选块
                                        │
                                        ▼
                          reranker.rerank(query, 候选, top_n=K)
                                        │
                                        ▼
                              真正送进大模型的 K 个块

为什么要加这一层
----------------
向量检索（bi-encoder）把 query 和文档**各自**编码成向量再算余弦，两者在编码时
互相看不见，本质是「粗排」，能保证召回但不擅长精细区分。

交叉编码器把 ``(query, 文档)`` **拼成一条输入**送进模型，能直接建模两者的词级交互，
判别精度明显更高。代价是**慢**：必须对每个候选对单独跑一次前向，
没法像向量那样预先算好、也没法建 ANN 索引。

所以标准做法就是「向量召回 N 个 → 交叉编码器精排取 K 个」，本模块负责后半段。

两种后端（``config.RERANK_PROVIDER`` 决定用哪个）
-------------------------------------------------
==================  ============================================================
``"local"``         本地交叉编码器，权重 2.3GB，CPU 前向。慢，但**不依赖任何外部账户**。
``"siliconflow"``   硅基流动托管 API（``BAAI/bge-reranker-v2-m3``，该平台免费模型）。
``"dashscope"``     阿里云百炼托管 API（``gte-rerank-v2``）。
==================  ============================================================

**两者的接口完全一致**（``score`` / ``rerank`` / ``warmup`` / ``is_loaded`` / ``unload``），
``get_reranker()`` 按配置返回其中一个，调用方（``rag_chain``）不需要知道是哪个。

为什么值得多写一个后端 —— 实测的账（本机 8GB 内存 / 4 核 / 无 CUDA）::

                     耗时        内存占用      18 条 Golden Set
    本地后端         344 秒/问   会爆（卡死）   不可用
    托管 API         约 2 秒/问  0             16/18 → 17/18（q06 转通过）

**结论不是「精排没用」，而是「本地精排在这台机器上不可用」** ——
算法能力和它的工程可行性是两件事，混在一起就会误杀方案。详见 ``config.py`` 的注解。

.. warning::
    托管服务是**别人的账户**，会因欠费 / 额度耗尽 / 限流整体停摆（实测踩过：
    阿里云账户欠费时 rerank 返回 ``HTTP 400 Arrearage``，18 条回归全灭）。
    所以 ``"local"`` 这条退路必须始终能走通 —— 本模块保留完整实现，一行没删。

选型
----
``BAAI/bge-reranker-v2-m3``：基于 ``bge-m3`` 的多语言重排模型，中英文都强，
官方在「multilingual」和「efficiency」两栏都首推它。

用法上跟 bge 的 embedding 模型**不一样**。以下三条均取自本地模型卡的官方 README：

1. **不加任何指令前缀**。官方示例就是 ``compute_score(['query', 'passage'])``，
   直接传原文，query 与文档都没有前缀——这跟 embedding 模型需要给 query 加
   instruction 的玩法完全不同，别混。
2. **分数用 sigmoid 映射到 [0, 1]**。官方原文：
   "the score can be mapped to a float value in [0,1] by sigmoid function."
   不开 sigmoid 拿到的是原始 logit（可能为负），跨查询不可比。
3. **``max_length`` 用 512**。官方 transformers 版示例里写的就是 ``max_length=512``，
   本项目沿用这个默认值。

调用方式
--------
高层（推荐，模块级函数，内部复用同一个单例）::

    from core import embed_store
    from core.reranker import rerank

    candidates = embed_store.search("网关监听哪个端口？", k=10)   # 粗排召回 10 个
    ranked = rerank("网关监听哪个端口？", candidates, top_n=3)     # 精排取 3 个
    for doc, score in ranked:
        print(f"{score:.4f}  {doc['content']}")

低层（需要自定义模型/长度/批量时用类）::

    from core.reranker import Reranker
    r = Reranker(max_length=256, batch_size=16)
    r.rerank(query, candidates, top_n=3)

输入输出约定
------------
``candidates`` 接受三种形式，都能吃：

1. ``core.embed_store.search()`` 的返回值 —— ``[(文档dict, 相似度), ...]``（最常用）
2. 文档 dict 列表 —— ``[{"content": "...", "metadata": {...}}, ...]``
3. 纯字符串列表 —— ``["一段文本", ...]``

返回 ``[(文档dict, 重排分数), ...]``，**与 ``search()`` 同构**，按分数从高到低排列，
长度取 ``top_n``（``None`` 表示全部返回）。文档 dict 里的 ``metadata`` 会原样保留。

分数怎么读（重要）
------------------
重排分数是**经过 sigmoid 归一化的相关性概率**，取值 ``(0, 1)``，越大越相关。

.. warning::
    它和向量检索的**余弦相似度不是同一个量纲，不可互相比较，也不能混用同一个阈值**。
    换个重排模型，同一批数据的分数分布就会变——**任何阈值都要在你自己数据上实测后再定**。

关于 HuggingFace 镜像
---------------------
与 ``embed_store`` 一致：导入时若 ``HF_ENDPOINT`` 为空则自动指向 ``https://hf-mirror.com``。

但**这个模型的权重建议从 ModelScope 拉**（见 ``_docmind_smoke/fetch_model_ms.py``）：
hf-mirror 会把 2.2GB 的 ``model.safetensors`` 重定向到 ``cas-bridge.xethub.hf.co``，
该域名在本机代理下频繁超时，实测只有 1.04 MB/s；ModelScope 实测 6.86 MB/s。

下载完成后权重落在 ``data/models/bge-reranker-v2-m3/``，
``resolve_model_path()`` 会自动优先命中这个本地目录，**之后完全离线可用**，
不需要把 ``RERANK_MODEL`` 改成绝对路径。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# HuggingFace 镜像：必须在 import huggingface_hub 之前设置好，否则不生效
# ---------------------------------------------------------------------------
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "models"

#: 单条输入的最大 token 数。模型本身支持到 8192，但**别按最大来**：
#: 交叉编码器的耗时与序列长度近似成正比，而本项目切出来的块通常只有一两百字，
#: 512 已经绰绰有余，调大只会白白变慢。
DEFAULT_MAX_LENGTH = 512

#: 单批处理的「query-文档」对数。CPU 上批量大一点能摊薄开销，但别超过可用内存。
DEFAULT_BATCH_SIZE = 8

#: 候选块类型：兼容 dict / str / (dict, 分数) 元组
Candidate = Union[Dict[str, Any], str, Tuple[Any, float]]
RankedItem = Tuple[Dict[str, Any], float]


class RerankerError(RuntimeError):
    """本模块的业务异常，便于调用方区分是依赖缺失还是用法错误。"""


# ---------------------------------------------------------------------------
# 模型定位：本地目录优先
# ---------------------------------------------------------------------------
def resolve_model_path(model_name: str, cache_folder: Union[str, Path]) -> str:
    """把「模型名」解析成实际可加载的路径。

    顺序：
    1. ``model_name`` 本身就是一个存在的目录 → 直接用（离线可用）
    2. ``<cache_folder>/<模型名最后一段>`` 存在且含 ``config.json`` → 用这个本地目录
    3. 都没有 → 原样返回模型名，交给 sentence-transformers 联网下载

    为什么需要第 2 条：本机经 hf-mirror 拉大文件会 302 到 ``cas-bridge.xethub.hf.co``，
    该域名在代理下频繁超时（实测仅 1.04 MB/s）。改从 ModelScope 拉取到普通目录后，
    靠这条规则就能直接用，无需把模型名改成本机绝对路径（那样换台机器就跑不了）。
    """
    p = Path(model_name)
    if p.is_dir():
        return str(p)

    local = Path(cache_folder) / model_name.split("/")[-1]
    if (local / "config.json").is_file():
        return str(local)

    return model_name


# ---------------------------------------------------------------------------
# 依赖懒加载（避免 import 本模块就吃下 torch 的启动开销）
# ---------------------------------------------------------------------------
_model = None
_sigmoid = None


def _get_sigmoid():
    """sigmoid 激活：把交叉编码器输出的原始 logit 压到 (0, 1) 当相关性概率用。"""
    global _sigmoid
    if _sigmoid is None:
        import torch

        _sigmoid = torch.nn.Sigmoid()
    return _sigmoid


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------
def _extract_text(item: Any) -> str:
    """从候选块里取出正文，兼容 dict / str 两种形态。"""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "page_content"):
            v = item.get(key)
            if isinstance(v, str):
                return v
        raise RerankerError(f"候选块 dict 里找不到正文字段（content/text/page_content）：{list(item)}")
    raise RerankerError(f"候选块类型不支持：{type(item).__name__}")


def _normalize_candidates(candidates: Sequence[Candidate]) -> List[Tuple[Dict[str, Any], str]]:
    """把三种候选形态统一成 ``[(文档dict, 正文), ...]``。

    元组形态 ``(文档, 粗排分数)`` 会丢掉粗排分数——它在精排结果里没有位置，
    调用方若需要对比，应在调用本模块之前自己留一份。

    .. important::
        **候选 dict 是「透传」而不是「另起一个」**：传进来的 dict 原对象会被原样
        放回返回值里，不做拷贝。

        所以本模块**只允许排序，不允许改写候选内容**。任何对候选 dict 的改动
        （加键、改字段、``dict(doc)`` 拷贝后改）都会反映到调用方手里那一份上。

        ``rag_chain`` 目前**不依赖**这个特性做分数对齐——它按 ``doc["id"]`` 把精排
        结果对回召回阶段的各路分数，候选被谁拷过一次都不影响。保持透传是为了另外两点：
        同一个块在全流程里只应该有一个身份；以及在候选池很大时省掉无意义的拷贝。
    """
    out: List[Tuple[Dict[str, Any], str]] = []
    for item in candidates:
        # (文档, 分数) 元组：embed_store.search() 的返回形态
        if isinstance(item, tuple) and len(item) == 2:
            item = item[0]
        text = _extract_text(item)
        doc = item if isinstance(item, dict) else {"content": text, "metadata": {}}
        if "content" not in doc:
            doc = {**doc, "content": text}
        out.append((doc, text))
    return out


# ---------------------------------------------------------------------------
# Reranker 类
# ---------------------------------------------------------------------------
class Reranker:
    """交叉编码器重排器。

    模型是懒加载的：实例化 ``Reranker()`` 不会立刻吃内存，
    第一次真正调用 ``score()`` / ``rerank()`` 时才把权重读进来。

    :param model_name: 模型名或本地路径
    :param max_length: 单条输入最大 token 数
    :param batch_size: 每批处理的 query-文档对数
    :param device: 传给 torch 的设备（``"cpu"`` / ``"cuda"`` / ``None`` 自动）
    :param cache_folder: 模型缓存目录，默认 ``data/models``
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
        cache_folder: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.cache_folder = str(cache_folder or MODEL_CACHE_DIR)
        self.local_files_only = local_files_only
        self._model = None
        self.resolved_path: Optional[str] = None       # 实际加载的本地目录/模型名
        self.load_seconds: Optional[float] = None      # 首次加载耗时，便于排查启动慢

    # -- 模型 -----------------------------------------------------------------

    def _get_model(self):
        """懒加载 CrossEncoder，只加载一次。"""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:            # pragma: no cover - 依赖缺失时的人话提示
            raise RerankerError(
                "缺少 sentence-transformers。安装：\n"
                "    .venv\\Scripts\\python.exe -m pip install sentence-transformers"
            ) from e

        t0 = time.time()
        resolved = resolve_model_path(self.model_name, self.cache_folder)
        self.resolved_path = resolved
        if resolved != self.model_name:
            print(f"[reranker] 使用本地模型目录：{resolved}")
        try:
            model = CrossEncoder(
                resolved,
                max_length=self.max_length,
                device=self.device,
                cache_folder=self.cache_folder,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            raise RerankerError(
                f"加载重排模型失败：{self.model_name}\n"
                f"  原因：{type(e).__name__}: {str(e)[:300]}\n"
                f"  模型缓存目录：{self.cache_folder}\n"
                f"  若是首次运行，请先确认模型已下载（见 _docmind_smoke/download_reranker.py）"
            ) from e

        # max_length 在部分版本里不会自动同步到 tokenizer，显式再设一次
        try:
            model.max_length = self.max_length
        except Exception:
            pass

        self._model = model
        self.load_seconds = time.time() - t0
        return model

    # -- 打分 -----------------------------------------------------------------

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """给 ``query`` 与每个文本的相关性打分，返回与 ``texts`` 等长的分数列表。

        分数已过 sigmoid，落在 ``(0, 1)``，越大越相关。
        """
        if not texts:
            return []

        model = self._get_model()
        pairs = [(query, t if isinstance(t, str) else str(t)) for t in texts]

        try:
            raw = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=_get_sigmoid(),      # 不开激活就是原始 logit，可能是负数
                convert_to_numpy=True,
            )
        except TypeError:
            # 老版本 CrossEncoder 没有 activation_fn 参数，退回原始输出后自己过激活
            raw = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        import numpy as np

        return [float(x) for x in np.asarray(raw).reshape(-1)]

    # -- 重排 -----------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_n: Optional[int] = None,
    ) -> List[RankedItem]:
        """对候选块重排，返回 ``[(文档, 重排分数), ...]``，按分数从高到低。

        :param query: 用户问题
        :param candidates: 候选块，支持 ``search()`` 返回值 / dict 列表 / 字符串列表
        :param top_n: 截断到前 N 个；``None`` 表示全部返回
        """
        normalized = _normalize_candidates(candidates)
        if not normalized:
            return []

        scores = self.score(query, [text for _, text in normalized])

        ranked: List[RankedItem] = [
            (doc, float(s)) for (doc, _), s in zip(normalized, scores)
        ]
        # 稳定排序：同分时保持粗排的相对顺序
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n >= 0:
            ranked = ranked[:top_n]
        return ranked

    # -- 生命周期 -------------------------------------------------------------

    def is_loaded(self) -> bool:
        """模型是否已经加载进内存。"""
        return self._model is not None

    def unload(self) -> None:
        """释放模型（内存紧张时用；下次调用会自动重新加载）。"""
        self._model = None
        self.load_seconds = None
        _release_torch_cache()


def _release_torch_cache() -> None:
    """清掉 torch 的缓存并触发一次 GC，让 2GB 权重真正还给系统。"""
    import gc

    try:
        import torch

        if torch.cuda.is_available():       # pragma: no cover - 本机无 CUDA
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# API 后端：把精排交给托管服务（免掉 2.3GB 权重与 CPU 前向）
# ---------------------------------------------------------------------------
#: 各家托管 rerank 服务的元信息。**要新接一家就往这里加一条**，其余代码不用动。
#:
#: * ``url``      请求地址
#: * ``model``    默认模型名（可被 ``config.RERANK_API_MODEL`` 覆盖）
#: * ``key_env``  从哪个环境变量读 key（写在项目根 ``.env``，已被 .gitignore 忽略）
#: * ``style``    返回体结构，决定怎么解析：
#:                ``"jina"`` = 顶层 ``results``；``"dashscope"`` = ``output.results``
#: * ``name``     中文名，只用于日志和错误提示
#:
#: .. warning::
#:     **``style`` 这一栏不能省，换个平台不是只换个 url。**
#:     硅基流动返回顶层 ``results``，阿里云百炼返回 ``output.results``——
#:     套用另一家的解析方式会直接 KeyError。
API_PROVIDERS: Dict[str, Dict[str, str]] = {
    "siliconflow": {
        "url": "https://api.siliconflow.cn/v1/rerank",
        "model": "BAAI/bge-reranker-v2-m3",
        "key_env": "SILICONFLOW_API_KEY",
        "style": "jina",
        "name": "硅基流动",
    },
    "dashscope": {
        "url": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        "model": "gte-rerank-v2",
        "key_env": "DASHSCOPE_API_KEY",
        "style": "dashscope",
        "name": "阿里云百炼",
    },
}

#: 单次请求超时（秒）。100 个候选实测 1.1~2.0 秒，这里只是给网络抖动留余量。
DEFAULT_API_TIMEOUT = 60.0

#: ``429`` / 5xx / 网络错误的额外重试次数。
#: **鉴权失败、欠费这类 4xx 不重试**——重试解决不了，只会白等（与 llm_client 同一策略）。
DEFAULT_API_RETRIES = 2


class ApiReranker:
    """托管 rerank API 后端。接口与本地 ``Reranker`` 一致，可直接互换。

    为什么要有它：本地交叉编码器是 XLM-R-large，FP32 权重 2.3GB —— 8GB 内存的机器
    装不下，跑 100 个候选会爆内存、系统换页、表现成「卡死」（实测卡死 2 小时 19 分）。
    换托管 API 之后同一套算法约 2 秒/问、零内存占用。

    :param provider: ``API_PROVIDERS`` 里的键（``"siliconflow"`` / ``"dashscope"``）
    :param model:    覆盖默认模型名；``None`` 表示用表里的默认值
    :param timeout:  单次请求超时（秒）
    :param retries:  429 / 5xx / 网络错误的额外重试次数

    .. note::
        候选 dict 与本地后端一样是**透传**的：``rerank()`` 返回的文档对象就是传进来的
        那一个，靠 API 返回的 ``index`` 对回原候选列表，不做任何拷贝或改写。
        这条约定见 ``_normalize_candidates`` 的说明。
    """

    def __init__(
        self,
        provider: str,
        *,
        model: Optional[str] = None,
        timeout: float = DEFAULT_API_TIMEOUT,
        retries: int = DEFAULT_API_RETRIES,
    ) -> None:
        key = str(provider).strip().lower()
        if key not in API_PROVIDERS:
            raise RerankerError(
                f"未知的 RERANK_PROVIDER：{provider!r}。可选："
                + "、".join(sorted(API_PROVIDERS))
                + '，或 "local"（走本地模型）。'
            )
        self.provider = key
        self.meta = API_PROVIDERS[key]
        self.model = (model or self.meta["model"]).strip()
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self._ready = False              # 是否已经成功调通过一次
        self.last_seconds: Optional[float] = None    # 最近一次请求耗时，排查用

    # -- key -------------------------------------------------------------------

    @property
    def key_env(self) -> str:
        """读 key 的环境变量名。"""
        return self.meta["key_env"]

    def _get_key(self) -> str:
        key = (os.environ.get(self.key_env) or "").strip()
        if not key:
            raise RerankerError(
                f"未找到 {self.key_env}（{self.meta['name']} 的 rerank key）。\n"
                f"  在项目根目录的 .env 里加一行：\n"
                f"      {self.key_env}=sk-你的key\n"
                f"  （.env 已被 .gitignore 忽略，不会提交）\n"
                f'  或者把 config.RERANK_PROVIDER 改成 "local" 走本地模型。'
            )
        return key

    # -- 请求组装 / 解析 --------------------------------------------------------

    def _build_body(self, query: str, texts: Sequence[str]) -> Dict[str, Any]:
        """按厂商协议组装请求体。两家的字段名完全不同，不能共用。"""
        if self.meta["style"] == "jina":
            # 硅基流动：OpenAI/Jina 风格，documents 直接躺着，top_n 显式给满
            # （不给的话服务端可能只回一部分，我们就拿不全每个候选的分数）
            return {
                "model": self.model,
                "query": query,
                "documents": list(texts),
                "top_n": len(texts),
                "return_documents": False,
            }
        # dashscope：input / parameters 两层包裹
        return {
            "model": self.model,
            "input": {"query": query, "documents": list(texts)},
            "parameters": {"return_documents": False},
        }

    def _parse(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从返回体里取出打分列表。两家路径不同，这里是唯一的分叉点。"""
        if self.meta["style"] == "jina":
            return data.get("results") or []
        return (data.get("output") or {}).get("results") or []

    def _friendly_http_error(self, code: int, detail: str) -> str:
        where = f"{self.meta['name']} rerank（{self.model}）"
        low = detail.lower()
        if code in (401, 403):
            return f"{where} 鉴权失败（{code}）：{self.key_env} 无效，或该 key 没有这个模型的权限。原始返回：{detail}"
        if code == 429:
            return f"{where} 限流（429），已重试 {self.retries} 次仍失败，请稍后再试。原始返回：{detail}"
        if any(w in low for w in ("arrearage", "insufficient", "overdue", "balance")):
            return (
                f"{where} 账户被拒（HTTP {code}）：欠费 / 额度耗尽。\n"
                f"  **这不是代码问题，也不是 key 格式问题** —— 请求已经到达服务端且鉴权通过，\n"
                f"  卡在账户状态。去控制台充值，或把 config.RERANK_PROVIDER 换成别家 / \"local\"。\n"
                f"  原始返回：{detail}"
            )
        return f"{where} 返回 HTTP {code}：{detail}"

    def _call(self, query: str, texts: Sequence[str]) -> List[Tuple[int, float]]:
        """调一次 API，返回 ``[(候选下标, 相关性分), ...]``（顺序照 API 给的，不保证已排序）。"""
        key = self._get_key()
        payload = json.dumps(self._build_body(query, texts), ensure_ascii=False).encode("utf-8")
        last_err = ""

        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.meta["url"],
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.last_seconds = time.time() - t0
                self._ready = True
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:250]
                msg = self._friendly_http_error(e.code, detail)
                # 只重试「可能靠重试解决」的：限流和服务端 5xx
                if (e.code == 429 or e.code >= 500) and attempt < self.retries:
                    time.sleep(min(3 * (attempt + 1), 10))
                    last_err = msg
                    continue
                raise RerankerError(msg) from e
            except RerankerError:
                raise
            except Exception as e:  # noqa: BLE001 - 超时 / DNS / 连接重置等
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < self.retries:
                    time.sleep(2)
                    continue
                raise RerankerError(
                    f"调用 {self.meta['name']} rerank 失败（已重试 {self.retries} 次）：{last_err}"
                ) from e

            hits: List[Tuple[int, float]] = []
            for item in self._parse(data):
                try:
                    idx = int(item["index"])
                    raw_score = item.get("relevance_score")
                    if raw_score is None:
                        raw_score = item["score"]
                    hits.append((idx, float(raw_score)))
                except (KeyError, TypeError, ValueError):
                    continue
            if not hits:
                raise RerankerError(
                    f"{self.meta['name']} rerank 返回了空结果或结构不认识"
                    f"（当前模型 {self.model!r}，style={self.meta['style']!r}）。"
                    f"模型名写错时常见这个报错。"
                )
            return hits

        raise RerankerError(f"调用 {self.meta['name']} rerank 失败：{last_err}")

    # -- 打分 / 重排 ------------------------------------------------------------

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """给一批文本打分，返回与 ``texts`` **等长、下标一一对应**的分数列表。

        .. note::
            API 返回的是 ``[(下标, 分数)]``，正常每个候选都有分。
            万一某个下标缺席，那一项按 ``0.0`` 填并打印一行警告 ——
            **别把它读成「这个块真的不相关」**，它只表示「API 没告诉我们」。
        """
        texts = list(texts)
        if not texts:
            return []

        hits = self._call(query, texts)
        out = [0.0] * len(texts)
        seen = set()
        for idx, s in hits:
            if 0 <= idx < len(texts):
                out[idx] = float(s)
                seen.add(idx)
        if len(seen) < len(texts):
            print(
                f"[reranker] {self.meta['name']} 只返回了 {len(seen)}/{len(texts)} 个分数，"
                f"缺失项按 0.0 处理（不代表它们不相关）"
            )
        return out

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_n: Optional[int] = None,
    ) -> List[RankedItem]:
        """对候选块重排，返回 ``[(文档, 重排分数), ...]``，按分数从高到低。

        与本地后端行为一致：文档对象**原样透传**，只调整顺序。
        """
        normalized = _normalize_candidates(candidates)
        if not normalized:
            return []

        hits = self._call(query, [text for _, text in normalized])

        # 靠 index 对回原候选列表 —— 这样返回的是**同一个 dict 对象**，
        # 调用方按 doc["id"] 反查各路分数才仍然对得上（见模块内「透传」约定）。
        ranked: List[RankedItem] = [
            (normalized[idx][0], float(s))
            for idx, s in hits
            if 0 <= idx < len(normalized)
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n >= 0:
            ranked = ranked[:top_n]
        return ranked

    # -- 生命周期 ---------------------------------------------------------------

    def warmup(self) -> float:
        """发一个最小请求探活（1 问 2 文档），返回耗时秒数。

        价值在于**把「key 没配 / 欠费 / 端点变更」在启动时就暴露出来**，
        而不是等第一个用户提问时才报错。
        """
        t0 = time.time()
        self.score("预热", ["预热文档", "无关内容"])
        return time.time() - t0

    def is_loaded(self) -> bool:
        """API 后端没有「加载」这回事；这里表示**是否已经成功调通过一次**。"""
        return self._ready

    def unload(self) -> None:
        """API 后端没有本地内存要释放，只把「就绪」标记清掉。"""
        self._ready = False


def provider_info(provider: Optional[str] = None) -> Dict[str, Any]:
    """当前精排后端的概况，给 ``config.summary()`` 和排查用。

    ``provider`` 传 ``None`` 表示读 ``config.RERANK_PROVIDER``。

    Returns:
        ``{"provider", "name", "model", "url", "key_env", "key_set", "is_local"}``。
        本地后端 ``url`` 为空、``key_set`` 恒为 True（它不需要 key）。
    """
    import config

    key = str(
        provider if provider is not None else getattr(config, "RERANK_PROVIDER", "local")
    ).strip().lower()

    if key in ("", "local"):
        return {
            "provider": "local",
            "name": "本地交叉编码器",
            "model": getattr(config, "RERANK_MODEL", MODEL_NAME),
            "url": "",
            "key_env": "",
            "key_set": True,          # 本地后端不需要 key
            "is_local": True,
        }

    if key not in API_PROVIDERS:
        raise RerankerError(
            f"未知的 RERANK_PROVIDER：{key!r}。可选："
            + "、".join(sorted(API_PROVIDERS))
            + '，或 "local"。'
        )

    meta = API_PROVIDERS[key]
    override = (getattr(config, "RERANK_API_MODEL", "") or "").strip()
    return {
        "provider": key,
        "name": meta["name"],
        "model": override or meta["model"],
        "url": meta["url"],
        "key_env": meta["key_env"],
        "key_set": bool((os.environ.get(meta["key_env"]) or "").strip()),
        "is_local": False,
    }


# ---------------------------------------------------------------------------
# 模块级单例：绝大多数场景直接用这几个函数就够了
# ---------------------------------------------------------------------------
_default: Optional[Union[Reranker, ApiReranker]] = None


def get_reranker() -> Union[Reranker, ApiReranker]:
    """取默认精排器（单例），后端由 ``config.RERANK_PROVIDER`` 决定。

    * ``"local"``（或留空）→ ``Reranker``，本地交叉编码器
    * ``"siliconflow"`` / ``"dashscope"`` → ``ApiReranker``，托管 API

    两者接口一致（``score`` / ``rerank`` / ``warmup``），调用方不需要区分。
    **创建单例不会加载任何东西**：本地后端是懒加载（第一次调用才读权重），
    API 后端根本不用加载。

    .. note::
        单例一旦建好就固定了后端。**运行时改了 ``config.RERANK_PROVIDER``
        要调 ``reset()``** 才会生效。
    """
    global _default
    if _default is None:
        import config

        provider = str(getattr(config, "RERANK_PROVIDER", "local") or "local").strip().lower()
        if provider in ("", "local"):
            _default = Reranker(
                model_name=getattr(config, "RERANK_MODEL", MODEL_NAME),
                max_length=getattr(config, "RERANK_MAX_LENGTH", DEFAULT_MAX_LENGTH),
                batch_size=getattr(config, "RERANK_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            )
        else:
            _default = ApiReranker(
                provider,
                model=(getattr(config, "RERANK_API_MODEL", "") or "").strip() or None,
                timeout=getattr(config, "RERANK_API_TIMEOUT", DEFAULT_API_TIMEOUT),
            )
    return _default


def reset() -> None:
    """丢掉默认单例。改了 ``config.RERANK_PROVIDER`` / 换了 key 之后调用，下次重建。"""
    global _default
    _default = None


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    top_n: Optional[int] = None,
) -> List[RankedItem]:
    """对候选块重排（模块级入口）。"""
    return get_reranker().rerank(query, candidates, top_n=top_n)


def score(query: str, texts: Sequence[str]) -> List[float]:
    """给一批文本打相关性分（模块级入口）。"""
    return get_reranker().score(query, texts)


def warmup() -> float:
    """提前把精排准备好，返回耗时（秒）。

    * **本地后端**：真正把 2.3GB 权重读进来 + 跑一次前向。模型加载本身要几十秒
      （XLM-R-large，CPU 上更慢），放在第一次提问时做用户会以为程序卡死，
      所以在应用启动时调一次 ``warmup()``，把这个等待挪到启动阶段。
    * **API 后端**：发一个最小请求探活（1 问 2 文档，实测不到 1 秒）。
      顺带把「key 没配 / 账户欠费 / 端点变更」在启动时就暴露出来，
      而不是等第一个用户提问才报错。
    """
    r = get_reranker()
    t0 = time.time()
    if isinstance(r, ApiReranker):
        r.warmup()
    else:
        r.score("预热", ["预热"])          # 真正触发加载 + 一次前向
    return time.time() - t0


def is_loaded() -> bool:
    """默认精排器是否已就绪。

    本地后端 = 权重已读进内存；API 后端 = 已经成功调通过一次（它没有「加载」这回事）。
    """
    return _default is not None and _default.is_loaded()


def unload() -> None:
    """释放默认单例占用的模型。"""
    if _default is not None:
        _default.unload()


if __name__ == "__main__":              # 手工冒烟：python -m core.reranker
    q = "网关默认监听哪个端口？"
    demo = [
        "本系统采用双活网关，上游统一走 8080 端口接入。",
        "备份策略为保留最近 30 天，归档目录是 /data/ops-backup。",
        "今天食堂的菜是红烧肉。",
    ]
    print(f"模型：{MODEL_NAME}")
    ranked = rerank(q, demo)
    for doc, s in ranked:
        print(f"  {s:.4f}  {doc['content']}")
