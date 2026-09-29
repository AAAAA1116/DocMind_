# -*- coding: utf-8 -*-
"""DocMind_ Web 界面（Streamlit，第四阶段）

启动::

    streamlit run app.py

界面分两块：
    左侧  上传文档 + 已入库文件清单（上传后自动入库，只处理新文件 / 内容有变化的文件）
    右侧  聊天区，提问后展示回答与来源片段

设计取向：**给不懂技术的人用**。界面上只留「上传」和「提问」两个动作；
参数、分数、名次、重建索引、日志这些排查用的东西全部收进左侧底部的「高级」折叠面板，
默认不打扰使用者。
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Streamlit 以 app.py 所在目录为当前目录，这里再兜一次，保证 config 一定能导入
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from core import embed_store  # noqa: E402
from core.llm_client import LLMError, chat as llm_smoke_test  # noqa: E402
from core.logger import log_error, log_qa, tail as log_tail  # noqa: E402
from core.rag_chain import answer as rag_answer  # noqa: E402

st.set_page_config(page_title="DocMind_ 企业知识助手", page_icon="📚", layout="wide")
config.ensure_dirs()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("flash", None)   # 重跑后还要显示的提示（st.rerun 会吞掉本次的提示）


# ---------------------------------------------------------------------------
# 启动预热：把重排模型读进内存
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def warmup_reranker() -> tuple[float, str]:
    """启动时把精排准备好，返回 ``(耗时秒数, 错误信息)``。失败时耗时为 -1，不阻断启动。

    干什么取决于 ``config.RERANK_PROVIDER``：

    * **本地后端** —— 真正加载权重。重排模型是 XLM-R-large（约 2.2GB），
      光把权重读进来就要二三十秒；不预热的话这笔开销会砸在用户**第一次提问**上，
      叠加重排本身的前向耗时，第一问能等到一分钟以上，用户会以为程序卡死。
    * **API 后端（1.8 起默认）** —— 发一个最小请求探活（1 问 2 文档，不到 1 秒）。
      启动就能暴露「key 没配 / 账户欠费 / 端点变更」，而不是等第一个用户提问才报错。

    ``st.cache_resource`` 是必须的：Streamlit 每次交互都会从头执行一遍脚本，
    不加缓存就会每次重建后端对象、每次探活。挂在「资源」缓存上，整个进程只执行一次。

    .. note::
        这里**不能碰 ``st.session_state``**——Streamlit 明令禁止在缓存函数内访问
        会话状态（缓存命中时函数体压根不执行，写进去的东西也就没了）。
        所以错误信息是当返回值交给调用方，由界面那边去展示。
    """
    if not config.RERANK_ENABLED:
        return 0.0, ""
    try:
        from core.reranker import warmup as _warmup

        return _warmup(), ""
    except Exception as e:  # noqa: BLE001
        # 加载失败不该让整个界面打不开：rag_chain.retrieve 里还有一层降级，
        # 真到检索时重排挂了会自动退回纯向量，问答本身仍然可用。
        try:
            log_error("加载重排模型", e)
        except Exception:  # noqa: BLE001
            pass
        return -1.0, f"{type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# 增量索引：入库清单（data/index/ingested.json 记录每个文件的 sha1）
#   sha1 没变      -> 跳过，不重新编码
#   sha1 变了      -> 先按 metadata.source 删掉旧块，再重新入库
#                     （不删的话新旧块会同时留在索引里，检索结果自相矛盾）
# ---------------------------------------------------------------------------
def file_sha1(path: Path) -> str:
    """算文件内容哈希，用来判断内容有没有变。"""
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    """读取已入库文件清单。文件损坏时返回空清单并记日志，不让界面崩掉。"""
    path = config.MANIFEST_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        log_error("读取已入库清单", e)
        return {}


def save_manifest(manifest: dict) -> None:
    """写回清单。目录不存在时先建。"""
    try:
        config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        log_error("写入已入库清单", e)


def ingest_file(path: Path, manifest: dict, force: bool = False) -> dict:
    """把单个文件解析、切分、向量化入库。

    Returns:
        ``{"name","status","chunks","detail"}``，
        status ∈ ``new`` / ``updated`` / ``skipped`` / ``error``
    """
    from splitter import split_document

    name = path.name

    try:
        digest = file_sha1(path)
    except Exception as e:  # noqa: BLE001
        log_error(f"读取文件 {name}", e)
        return {"name": name, "status": "error", "chunks": 0, "detail": f"读取失败：{e}"}

    record = manifest.get(name) or {}
    if not force and record.get("sha1") == digest:
        return {"name": name, "status": "skipped", "chunks": record.get("chunks", 0), "detail": "内容未变化"}

    try:
        chunks = split_document(
            str(path),
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.OVERLAP,
        )
    except ValueError as e:
        # loader 对不支持的扩展名 / 空内容抛 ValueError，信息可以直接展示给用户
        log_error(f"解析文件 {name}", e)
        return {"name": name, "status": "error", "chunks": 0, "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        log_error(f"解析文件 {name}", e)
        return {"name": name, "status": "error", "chunks": 0,
                "detail": f"文件解析失败（{type(e).__name__}）：{e}"}

    # 纯空白块（空文件、只有空行的文件）不该进索引
    chunks = [c for c in chunks if c and c.strip()]
    if not chunks:
        return {"name": name, "status": "error", "chunks": 0,
                "detail": "解析后没有可用文本（空文件、只有空行，或扫描版 PDF）"}

    docs = [
        {
            # 块内容前带上来源文件名。制度/章程类文档正文一律自称「公司」，
            # 块内不含公司名，跨文档检索时无法与问题里的公司名对齐——实测
            # 「北辰实业的董事任期」会命中新宝/龙旗的同构条款（命中 1/6），
            # 带上来源后同样的查询能正确锚定到北辰（命中 4/6）。
            "content": f"《{name}》\n{chunk}",
            "metadata": {
                "source": name,
                "chunk_index": i,
                "file_sha1": digest,
                "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        if record:
            # 内容变了：先清掉该来源的旧块，避免新旧混杂
            embed_store.get_store().collection.delete(where={"source": name})
        embed_store.add_documents(docs)
    except Exception as e:  # noqa: BLE001
        log_error(f"写入索引 {name}", e)
        return {"name": name, "status": "error", "chunks": 0,
                "detail": f"写入向量库失败（{type(e).__name__}）：{e}"}

    manifest[name] = {
        "sha1": digest,
        "chunks": len(docs),
        "size": path.stat().st_size,
        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.OVERLAP,
    }
    return {"name": name, "status": "updated" if record else "new", "chunks": len(docs), "detail": ""}


def ingest_files(paths, manifest: dict, force: bool = False) -> list:
    """批量入库。"""
    return [ingest_file(Path(p), manifest, force=force) for p in paths]


def stored_files() -> list:
    """列出 data/uploads 下所有受支持的文件。"""
    if not config.UPLOAD_DIR.exists():
        return []
    return sorted(
        p for p in config.UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS
    )


def rebuild(manifest: dict) -> list:
    """清空索引，用 uploads 目录里的文件全量重建。"""
    embed_store.reset_index()
    manifest.clear()
    results = ingest_files(stored_files(), manifest, force=True)
    save_manifest(manifest)
    return results


def store_uploads(uploaded) -> list:
    """把 Streamlit 上传的文件落到 data/uploads/，返回落盘后的路径列表。

    内容没变就不重写磁盘，省一次 IO（上传控件每次重跑都会把文件重发一遍）。
    """
    saved = []
    for uf in uploaded:
        target = config.UPLOAD_DIR / uf.name
        try:
            data = bytes(uf.getbuffer())
            changed = True
            if target.exists() and target.stat().st_size == len(data):
                changed = file_sha1(target) != hashlib.sha1(data).hexdigest()
            if changed:
                with open(target, "wb") as f:
                    f.write(data)
            saved.append(target)
        except Exception as e:  # noqa: BLE001
            log_error(f"保存上传文件 {uf.name}", e)
            st.error(f"保存「{uf.name}」失败：{e}")
    return saved


def _channel(name: str, score, rank) -> str:
    """把「某一路召回结果」渲染成短标签；没召回到就写明「未召回」。

    注意：``score`` 为 None 才是「没召回」，而不是 ``rank`` 为 None——
    纯向量路径（关了混合检索）只有分数、没有名次。
    """
    if score is None:
        return f"{name} 未召回"
    if rank is None:
        return f"{name} {score:.4f}"
    return f"{name} #{rank}（{score:.4f}）"


def render_sources(sources) -> None:
    """把来源片段渲染成一排折叠面板。

    默认**只显示来自哪个文件**——提问的人关心的是「依据是什么」，不是分数多少。
    分数 / 名次 / 块号这些诊断字段，只有勾了「高级」里的开关才显示。

    为什么要留那个开关而不是直接删掉：这些字段是判断
    「某个块是哪一路捞上来的」的唯一入口——

    * ``向量 未召回`` 而 ``BM25 #1`` → BM25 补上的盲区（精确字面量）
    * ``BM25 未召回`` 而 ``向量 #1`` → 语义命中，BM25 帮不上忙（转述 / 同义改写）

    排查时有用，平时是噪音。所以收进「高级」，默认关。
    """
    if not sources:
        return

    show_diag = bool(st.session_state.get("show_diag", False))

    for i, src in enumerate(sources, 1):
        label = f"依据 {i}：{src['source']}"
        if show_diag:
            label += f"　最终分 {src['score']:.4f}"
            if src.get("chunk_index") is not None:
                label += f"　第 {src['chunk_index']} 块"

        with st.expander(label):
            if show_diag:
                tags = []
                if src.get("rerank_score") is not None:
                    tags.append(f"精排 {src['rerank_score']:.4f}")
                if src.get("rrf_score") is not None:
                    tags.append(f"RRF #{src['rrf_rank']}（{src['rrf_score']:.5f}）")
                    # 走了混合检索才存在「哪一路召回」这回事；
                    # 纯向量路径下不该显示「BM25 未召回」，那会误导成 BM25 试过但没捞到
                    tags.append(_channel("向量", src.get("vector_score"), src.get("vector_rank")))
                    tags.append(_channel("BM25", src.get("bm25_score"), src.get("bm25_rank")))
                elif src.get("vector_score") is not None:
                    tags.append(f"向量 {src['vector_score']:.4f}")
                if tags:
                    st.caption("　·　".join(tags))
            st.markdown(src["content"])


# ---------------------------------------------------------------------------
# 左侧栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("知识库")
    st.caption(f"DocMind_ v{config.DOCMIND_VERSION}")

    if config.RERANK_ENABLED:
        # v1.8 起精排默认走托管 API，所以这里不再是「加载 2.3GB 权重」，
        # 而是「发一个最小请求探活」—— 把 key 没配 / 账户欠费这类问题在启动时就暴露。
        _backend = (
            "本地模型"
            if config.RERANK_PROVIDER == "local"
            else f"{config.RERANK_PROVIDER} API"
        )
        with st.spinner(f"正在检查精排后端（{_backend}）…"):
            _rerank_secs, _rerank_err = warmup_reranker()
        if _rerank_err:
            st.warning(
                "精排不可用，本次会话改为按 RRF 名次取候选（问答仍然可用）。"
            )
            st.caption(_rerank_err)
        elif _rerank_secs > 0:
            st.caption(f"精排已就绪（{_backend}）　{_rerank_secs:.1f}s")

    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))

    uploaded = st.file_uploader(
        "上传文档（可多选）",
        type=[ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS],
        accept_multiple_files=True,
        help=f"支持 {'、'.join(config.ALLOWED_EXTENSIONS)}。"
             f"上传后自动入库，内容没变的文件会跳过；有变化的文件会先删旧块再重新入库。",
    )

    manifest = load_manifest()

    if uploaded:
        files = store_uploads(uploaded)
        if files:
            with st.spinner(f"处理 {len(files)} 个文件…"):
                results = ingest_files(files, manifest, force=False)
            save_manifest(manifest)
            for r in results:
                if r["status"] == "new":
                    st.success(f"「{r['name']}」已入库，{r['chunks']} 个文本块")
                elif r["status"] == "updated":
                    st.info(f"「{r['name']}」内容有变化，已重新入库（{r['chunks']} 块）")
                elif r["status"] == "error":
                    st.error(f"「{r['name']}」处理失败：{r['detail']}")

    st.divider()

    st.subheader("已入库文件")
    if not manifest:
        st.caption("还没有文件，先上传一个吧。")
    else:
        for name, info in manifest.items():
            exists = (config.UPLOAD_DIR / name).exists()
            suffix = "" if exists else "　⚠️ 文件已不在 uploads 目录"
            st.markdown(f"- **{name}** — {info.get('chunks', 0)} 块{suffix}")
    st.caption(f"索引内共 {embed_store.count()} 个文本块")

    st.divider()

    # 排查用的东西全部收进这一个折叠面板：默认折叠，不干扰使用者。
    # 「简易」不等于「删功能」——是把不常用的收起来，需要时还在原位。
    with st.expander("高级"):
        st.caption(
            f"v{config.DOCMIND_VERSION}　"
            f"混合检索 {'开' if config.HYBRID_ENABLED else '关'}　"
            f"精排 {'开' if config.RERANK_ENABLED else '关'}"
            + (
                f"（{config.RERANK_PROVIDER}　池 {config.RERANK_CANDIDATES} → {config.TOP_K}）"
                if config.RERANK_ENABLED
                else ""
            )
        )
        st.caption(
            f"切分 {config.CHUNK_SIZE} / 重叠 {config.OVERLAP}　检索 top{config.TOP_K}　"
            f"阈值 {config.THRESHOLD}　模型 {config.MODEL_NAME}"
        )

        st.checkbox(
            "在「依据」里显示检索诊断",
            key="show_diag",
            help="显示每个片段的最终分，以及它是向量还是 BM25 捞上来的。排查检索问题时才需要开。",
        )
        st.divider()

        if st.button("清空并重建索引", use_container_width=True):
            with st.spinner("重建中…"):
                results = rebuild(manifest)
            st.session_state["messages"] = []
            failed = [r for r in results if r["status"] == "error"]
            for r in failed:
                st.error(f"「{r['name']}」：{r['detail']}")
            st.session_state["flash"] = (
                f"已用 uploads 目录里的 {len(results)} 个文件重建索引"
                + (f"，{len(failed)} 个失败" if failed else "")
            )
            st.rerun()

        if st.button("测试 DeepSeek 连通性", use_container_width=True):
            try:
                st.success(f"连接正常，模型回复：{llm_smoke_test('你是一个助手。', '只回复两个字：正常')}")
            except LLMError as e:
                st.error(str(e))

        st.divider()
        st.caption("最近日志")
        lines = log_tail(5)
        if lines:
            for line in lines:
                st.text(line)
        else:
            st.caption(f"暂无日志（{config.LOG_FILE}）")


# ---------------------------------------------------------------------------
# 右侧主区
# ---------------------------------------------------------------------------
st.title("DocMind_ 企业知识助手")
st.caption("回答只依据你上传的资料；资料里没有的，它会如实告诉你「没找到」。")

# 把底部输入框画成一个「看得见的框」。
#
# 为什么必须做这一步：``st.chat_input`` 固定在视口最底部，而主区上半部分是
# 一大片留白。第一次打开的人扫一眼，视线停在标题上，根本不会注意到最下面
# 那条几乎无色的灰线是个输入框——实测反馈就是「没看见问答的地方」。
#
# 选择器只用 ``data-testid``（Streamlit 为自动化测试保留的稳定钩子），
# 不去碰 ``st-emotion-cache-*`` 那种每次构建都会变的哈希类名。
st.markdown(
    """
    <style>
    [data-testid="stChatInput"] {
        border: 2px solid #4C8BF5 !important;
        border-radius: 14px !important;
        box-shadow: 0 2px 16px rgba(76, 139, 245, .20);
    }
    [data-testid="stChatInput"] textarea { font-size: 1rem; }
    [data-testid="stBottom"] { padding-bottom: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not manifest:
    st.info("知识库还是空的。先在左边上传文档，然后就可以提问了。")
elif not st.session_state["messages"]:
    # 空状态引导：把「怎么用」直接摆出来，顺手告诉使用者输入框在哪。
    # 只在还没问过问题时显示——一旦有了问答记录，这块让位给对话本身。
    with st.container(border=True):
        st.markdown("#### 怎么用")
        st.markdown(
            "1. **传文档** —— 在左侧「知识库」里选择文件，上传后自动入库"
            f"（当前已入库 **{len(manifest)}** 个文件、{embed_store.count()} 个文本块）。\n"
            "2. **提问** —— 在**页面最下方**那个蓝色边框的输入框里写下问题，按回车发送。\n"
            "3. **看依据** —— 回答下方会列出它参考了哪些文件。资料里没有的内容，"
            f"它会直接说「{config.REFUSAL_TEXT}」，不会编。"
        )
        # 这里只教「怎么问」，不举具体问题当例子。
        # 早先写过「试试问三家公司的注册资本分别是多少」——那正好是评测集里
        # 答不出的 q06（答案块排名 81，进不了 top-8）。界面亲自推荐一个会失败的
        # 问题，等于自己拆自己的台。示例改成提问方式，不承诺任何具体答案。
        st.caption(
            "小提示：提问时带上文档里出现的原词（公司名、条款里的说法），检索命中率更高。"
        )

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
        if message.get("refuse_reason"):
            st.caption(f"（{message['refuse_reason']}）")

if prompt := st.chat_input("在这里输入你的问题，按回车发送"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = None
        with st.spinner("检索并生成回答…"):
            try:
                result = rag_answer(prompt)
            except LLMError as e:
                log_error("调用 DeepSeek", e)
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                log_error("问答流程", e)
                st.error(f"问答失败（{type(e).__name__}）：{e}")

        if result is not None:
            st.markdown(result["answer"])
            render_sources(result["sources"])
            if result.get("refuse_reason"):
                st.caption(f"（{result['refuse_reason']}）")

            log_qa(
                question=result["question"],
                sources=result["sources"],
                answer=result["answer"],
                refused=result["refused"],
            )
            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "refuse_reason": result.get("refuse_reason", ""),
            })
        else:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "（回答失败，详见上方错误提示）",
                "sources": [],
            })
