# -*- coding: utf-8 -*-
"""DocMind_ 全局配置

所有可调参数集中在这里，其它模块一律 ``from config import ...``，
不要在业务代码里写散落的常量。

改动注意
--------
``CHUNK_SIZE`` / ``OVERLAP`` 改了以后，**必须重建索引**（旧索引是按老参数切的），
因为向量库只存文本块，不存切分参数，改了参数旧块不会自动重切。
Web 界面左侧有「重建索引」按钮，或调用 ``core.embed_store.reset_index()``。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"                       # 所有本地数据（已被 .gitignore 忽略）
UPLOAD_DIR = DATA_DIR / "uploads"                  # 上传的原始文件
INDEX_DIR = DATA_DIR / "index"                     # ChromaDB 持久化目录
MODEL_CACHE_DIR = DATA_DIR / "models"              # sentence-transformers 模型缓存
LOG_DIR = BASE_DIR / "logs"                        # 问答日志
MANIFEST_PATH = INDEX_DIR / "ingested.json"        # 增量索引用：已入库文件清单

# ---------------------------------------------------------------------------
# 切分参数（第二阶段 splitter.py）
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500        # 每块最大字符数
OVERLAP = 80            # 相邻块重叠字符数

# ---------------------------------------------------------------------------
# 检索参数（第三阶段 core/embed_store.py）
# ---------------------------------------------------------------------------
TOP_K = 3               # 每次检索返回的候选块数量
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

#: 相似度阈值。低于它的候选块会被丢弃；若全部低于阈值，直接回复「知识库中未找到相关信息」。
#:
#: .. warning::
#:     **在调大这个值之前，先看你自己数据上的实际分数分布。**
#:     bge 官方模型卡明确说：「相似度大于 0.5 并不代表两句相似」，
#:     「真正重要的是分数的相对顺序，而不是绝对值」。
#:     实测本机样例文档（运维手册）的正确命中块，余弦相似度集中在 **0.51 ~ 0.56**，
#:     所以 **0.6 会把所有正确结果都过滤掉，导致每条问题都拒答**。
#:     设成 0.0 或负数等于关闭阈值过滤（只靠排序 + 让大模型自己判断资料够不够）。
THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# 大模型（DeepSeek）
# ---------------------------------------------------------------------------
#: 注意：这里是**对话模型**名，不是 embedding 模型。
#: embedding 模型是上面的 EMBED_MODEL_NAME，两者不要混。
MODEL_NAME = "deepseek-chat"

LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"      # 从环境变量 / 项目根 .env 读取
LLM_TIMEOUT = 60.0                         # 单次请求超时（秒）
LLM_MAX_RETRIES = 1                        # 超时后重试次数：1 表示最多请求 2 次
LLM_TEMPERATURE = 0.2                      # 问答场景要稳定，温度调低
LLM_MAX_TOKENS = 1024                      # 单次回答长度上限

#: 模型答不出时的固定话术，与 rag_chain 的 system_prompt 保持一致
REFUSAL_TEXT = "知识库中未找到相关信息"

# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
#: 允许上传的扩展名，与 loader.py 支持的一致
ALLOWED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
LOG_FILE = LOG_DIR / "qa.log"
LOG_BACKUP_DAYS = 30        # 按天滚动，保留 30 天
LOG_SNIPPET_LEN = 50        # 日志里每条检索片段只记前 50 字

# ---------------------------------------------------------------------------
# 读取 .env（放在最后，保证上面的 os.environ.get 能拿到值之后不会再被覆盖）
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:      # python-dotenv 未安装时静默跳过，不阻塞启动
    pass


def ensure_dirs() -> None:
    """创建运行所需目录。启动时调一次即可，重复调用无害。"""
    for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR, MODEL_CACHE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """配置摘要，启动时打印一份，方便确认参数是否生效。"""
    lines = [
        "=" * 68,
        "DocMind_ 配置",
        "=" * 68,
        f"  切分        CHUNK_SIZE={CHUNK_SIZE}  OVERLAP={OVERLAP}",
        f"  检索        TOP_K={TOP_K}  THRESHOLD={THRESHOLD}",
        f"  embedding   {EMBED_MODEL_NAME}",
        f"  对话模型    {MODEL_NAME}  ({LLM_BASE_URL})",
        f"  API Key     {'已设置' if os.environ.get(LLM_API_KEY_ENV) else '未设置（.env 里填 ' + LLM_API_KEY_ENV + '）'}",
        f"  索引目录    {INDEX_DIR}",
        f"  日志文件    {LOG_FILE}",
        "=" * 68,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs()
    print(summary())
