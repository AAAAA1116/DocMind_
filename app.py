# -*- coding: utf-8 -*-
"""DocMind_ Web 界面（Streamlit，第四阶段）

启动::

    streamlit run app.py

界面分两块：
    左侧  已入库文件列表 + 多文件上传（上传后自动入库，只处理新文件 / 内容有变化的文件）
    右侧  聊天区，提问后展示回答与来源片段
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
            "content": chunk,
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


def render_sources(sources) -> None:
    """把来源片段渲染成一排折叠面板。"""
    for src in sources:
        label = f"来源：{src['source']}　相似度 {src['score']:.4f}"
        if src.get("chunk_index") is not None:
            label += f"　第 {src['chunk_index']} 块"
        with st.expander(label):
            st.markdown(src["content"])


# ---------------------------------------------------------------------------
# 左侧栏
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("知识库")

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
    st.subheader("当前参数")
    st.caption(
        f"切分 {config.CHUNK_SIZE} / 重叠 {config.OVERLAP}　检索 top{config.TOP_K}　"
        f"阈值 {config.THRESHOLD}　模型 {config.MODEL_NAME}"
    )
    if config.THRESHOLD > 0:
        st.caption(
            "关于阈值：bge 的相似度绝对值不能当相关性判据，官方原文说「大于 0.5 并不代表两句相似」。"
            "实测本机样例里正确命中块的分数在 0.51~0.56，所以阈值设 0.6 会把正确结果全过滤掉。"
            "改 config.THRESHOLD（设 0 即关闭过滤）。"
        )

    with st.expander("敏感操作"):
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

    with st.expander("最近日志"):
        lines = log_tail(10)
        if lines:
            for line in lines:
                st.text(line)
        else:
            st.caption(f"暂无日志（{config.LOG_FILE}）")


# ---------------------------------------------------------------------------
# 右侧主区
# ---------------------------------------------------------------------------
st.title("DocMind_ 企业知识助手")
st.caption(
    f"索引 {embed_store.count()} 个文本块 · 回答只依据你上传的资料 · "
    f"资料里没有的内容会说「{config.REFUSAL_TEXT}」"
)

if not manifest:
    st.info("先在左侧上传文档，然后再提问。")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))
        if message.get("refuse_reason"):
            st.caption(f"（{message['refuse_reason']}）")

if prompt := st.chat_input("向知识库提问…"):
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
