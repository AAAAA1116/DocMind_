# -*- coding: utf-8 -*-
"""问答日志（第四阶段）

把每次问答落盘到 ``logs/qa.log``，**按天滚动**（每天零点切一个新文件，
保留 ``config.LOG_BACKUP_DAYS`` 天，滚动后的文件名形如 ``qa.log.2026-09-18``）。

每条记录一行，字段固定为：

    时间 | 级别 | 问题 | 检索片段（每段前 50 字） | 回答 | 是否拒答

一行一条是为了方便后续 ``grep`` / 导入 Excel 分析，所以写入前会把
换行符替换成 ``\\n`` 字面量。

用法::

    from core.logger import log_qa

    log_qa(question="网关监听哪个端口？", sources=result["sources"],
           answer=result["answer"], refused=result["refused"])
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import LOG_BACKUP_DAYS, LOG_DIR, LOG_FILE, LOG_SNIPPET_LEN  # noqa: E402

_LOGGER_NAME = "docmind.qa"
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """返回配置好的 logger（进程内只配置一次）。

    用 ``propagate = False`` 避免日志被 root logger 再往控制台刷一遍；
    Streamlit 的重跑机制下重复调用也安全。
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        handler = TimedRotatingFileHandler(
            filename=str(LOG_FILE),
            when="midnight",          # 每天零点滚动
            interval=1,
            backupCount=LOG_BACKUP_DAYS,
            encoding="utf-8",
            delay=True,               # 首次写入才建文件，避免空文件
        )
        # 滚动文件的日期后缀
        handler.suffix = "%Y-%m-%d"
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)

        # 需要同时在终端看日志时，设 DOCMIND_LOG_CONSOLE=1
        import os
        if os.environ.get("DOCMIND_LOG_CONSOLE") not in (None, "", "0"):
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
            logger.addHandler(console)

    _logger = logger
    return logger


def _one_line(text: Any) -> str:
    """压成单行，保证「一条记录 = 一行」。"""
    return " ".join(str(text).replace("\r", " ").replace("\n", "\\n").split())


def _snippets(sources: Iterable[Any]) -> str:
    """把各检索片段的前 N 字拼起来；没有片段时给个占位符。"""
    parts: List[str] = []
    for src in sources or []:
        if isinstance(src, dict):
            text = src.get("content") or src.get("snippet") or ""
        else:
            text = str(src)
        text = _one_line(text)[:LOG_SNIPPET_LEN]
        if text:
            parts.append(text)
    return " § ".join(parts) if parts else "（无）"


def log_qa(
    question: str,
    sources: Optional[Iterable[Any]] = None,
    answer: str = "",
    refused: bool = False,
    extra: Optional[str] = None,
) -> None:
    """记录一条问答。

    Args:
        question: 用户问题
        sources:  检索到的片段列表（dict 含 content/snippet，或直接给字符串）
        answer:   模型回答（拒答时就是那句固定话术）
        refused:  是否判定为拒答
        extra:    附加说明，例如错误原因
    """
    message = (
        f"问答 | 问题={_one_line(question)} "
        f"| 检索片段={_snippets(sources)} "
        f"| 回答={_one_line(answer)} "
        f"| 是否拒答={'是' if refused else '否'}"
    )
    if extra:
        message += f" | 备注={_one_line(extra)}"
    get_logger().info(message)


def log_error(stage: str, error: Any) -> None:
    """记录异常（文件解析失败、API 报错等），同样一行一条。"""
    get_logger().error(f"异常 | 阶段={_one_line(stage)} | 详情={_one_line(error)}")


def tail(n: int = 20) -> List[str]:
    """读取日志最后 n 行（给界面/排查用）。文件不存在返回空列表。"""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    return lines[-n:]


# ---------------------------------------------------------------------------
# 自测：python -m core.logger
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import ensure_dirs

    ensure_dirs()
    log_qa(
        question="网关监听哪个端口？",
        sources=[{"content": "网关默认以单进程模式运行，监听 8080 端口。生产环境建议前置 Nginx。"}],
        answer="网关默认监听 8080 端口。",
        refused=False,
    )
    log_qa(
        question="今天天气怎么样？",
        sources=[],
        answer="知识库中未找到相关信息",
        refused=True,
    )
    log_error("测试阶段", "这是一个假的异常，用于验证日志格式")

    print(f"日志文件: {LOG_FILE}")
    print(f"存在    : {LOG_FILE.exists()}")
    print("-" * 70)
    for line in tail(5):
        print(line)
