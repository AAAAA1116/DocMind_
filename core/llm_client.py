# -*- coding: utf-8 -*-
"""DeepSeek 大模型调用封装（第四阶段）

职责很窄，只做三件事：
    1. 接收 ``system_prompt`` 与 ``user_message``，调用 DeepSeek 对话接口
    2. **超时自动重试一次**（共最多 2 次尝试），鉴权/参数错误不重试
    3. 返回模型回答的纯文本

用法::

    from core.llm_client import chat

    text = chat("你是一个助手", "你好")

关于重试策略
------------
只对**可能靠重试解决**的错误重试：超时、连接失败、限流、服务端 5xx。
以下错误立即抛出，重试没意义还浪费时间：

- 401 鉴权失败（key 错或没设置）
- 400 参数错误
- 余额不足

重试次数由 ``config.LLM_MAX_RETRIES`` 控制，默认 1（即最多请求 2 次）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让 `python -m core.llm_client` 之类的用法也能找到根目录的 config
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (  # noqa: E402
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MODEL_NAME,
)


class LLMError(RuntimeError):
    """调用大模型失败。message 是可以直接展示给用户看的中文说明。"""


# 进程内单例，避免每次问答都重建连接池
_client = None


def _get_client():
    """创建并缓存 OpenAI 兼容客户端（DeepSeek 用的是 OpenAI 协议）。"""
    global _client
    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as e:
        raise LLMError("缺少 openai 依赖，请先执行：pip install openai") from e

    api_key = os.environ.get(LLM_API_KEY_ENV) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMError(
            f"未找到 {LLM_API_KEY_ENV}。请在项目根目录的 .env 文件里写一行：\n"
            f"    {LLM_API_KEY_ENV}=sk-你的key\n"
            f"（.env 已被 .gitignore 忽略，不会被提交）"
        )

    _client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
    return _client


def reset_client() -> None:
    """丢弃已缓存的客户端（换 key 后调用）。"""
    global _client
    _client = None


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得重试。"""
    try:
        import openai
    except ImportError:
        return False

    retryable_types = tuple(
        t for t in (
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
        ) if isinstance(t, type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        return True

    # 5xx 也算服务端问题，可以重试
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return False


def _friendly_error(exc: Exception) -> str:
    """把 SDK 异常翻译成用户能看懂的中文提示。"""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    if status == 401 or "Authentication" in name:
        return "DeepSeek 鉴权失败（401）：API Key 无效或已失效，请检查 .env 里的 " + LLM_API_KEY_ENV
    if status == 402 or "InsufficientQuota" in name:
        return "DeepSeek 账户余额不足（402），请先充值"
    if status == 429 or "RateLimit" in name:
        return "请求过于频繁或超出配额（429），请稍后重试"
    if status == 400 or "BadRequest" in name:
        return f"请求参数有误（400）：{exc}"
    if "Timeout" in name:
        return f"调用 DeepSeek 超时（{LLM_TIMEOUT:.0f} 秒），已重试仍失败，请检查网络或稍后再试"
    if "Connection" in name:
        return "无法连接 DeepSeek 服务，请检查网络或代理设置"
    if status:
        return f"DeepSeek 接口返回错误 {status}：{exc}"
    return f"调用 DeepSeek 失败（{name}）：{exc}"


def chat_with_usage(
    system_prompt: str,
    user_message: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retries: Optional[int] = None,
) -> Dict[str, Any]:
    """调用 DeepSeek 并返回回答 + token 用量。

    Args:
        system_prompt: 系统提示词
        user_message:  用户消息正文
        temperature:   覆盖默认温度（默认取 config.LLM_TEMPERATURE）
        max_tokens:    覆盖默认长度上限
        retries:       覆盖默认重试次数（0 表示不重试）

    Returns:
        ``{"text": 回答文本, "usage": {"prompt_tokens":..,"completion_tokens":..,"total_tokens":..},
        "attempts": 实际请求次数, "model": 实际模型名}``

    Raises:
        LLMError: 依赖缺失、key 缺失，或请求在重试后仍失败
    """
    if not system_prompt or not str(system_prompt).strip():
        raise LLMError("system_prompt 不能为空")
    if not user_message or not str(user_message).strip():
        raise LLMError("user_message 不能为空")

    client = _get_client()
    max_attempts = max(1, (LLM_MAX_RETRIES if retries is None else retries) + 1)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": str(system_prompt)},
                    {"role": "user", "content": str(user_message)},
                ],
                temperature=LLM_TEMPERATURE if temperature is None else temperature,
                max_tokens=LLM_MAX_TOKENS if max_tokens is None else max_tokens,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 - 统一翻译成 LLMError
            last_exc = exc
            if attempt < max_attempts and _is_retryable(exc):
                # 指数退避，第一次重试等 1 秒
                time.sleep(min(2 ** (attempt - 1), 4))
                continue
            raise LLMError(_friendly_error(exc)) from exc

        text = _extract_text(response)
        if not text:
            raise LLMError("DeepSeek 返回了空回答，请重试")

        usage = getattr(response, "usage", None)
        return {
            "text": text,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            },
            "attempts": attempt,
            "model": getattr(response, "model", MODEL_NAME),
        }

    raise LLMError(_friendly_error(last_exc) if last_exc else "调用 DeepSeek 失败")


def chat(
    system_prompt: str,
    user_message: str,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retries: Optional[int] = None,
) -> str:
    """调用 DeepSeek，只返回回答文本（第四阶段提示词要求的接口）。"""
    return chat_with_usage(
        system_prompt,
        user_message,
        temperature=temperature,
        max_tokens=max_tokens,
        retries=retries,
    )["text"]


def _extract_text(response: Any) -> str:
    """从返回体里稳妥地取出正文。

    正常是 ``response.choices[0].message.content``。这里逐层判空，
    避免 DeepSeek 偶尔返回的异常结构直接抛 AttributeError 把界面打崩。
    """
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None:
        # 少数情况下模型只给了 reasoning_content
        content = getattr(message, "reasoning_content", None)
    return (content or "").strip()


# ---------------------------------------------------------------------------
# 自测：python -m core.llm_client
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "用一句话说明什么是文本切分"
    print(f"模型 : {MODEL_NAME}")
    print(f"问题 : {msg}")
    print("-" * 60)
    try:
        result = chat_with_usage("你是一个简洁的助手，回答不超过 50 字。", msg)
    except LLMError as e:
        print(f"[失败] {e}")
        sys.exit(1)
    print(f"回答 : {result['text']}")
    print(f"用量 : {result['usage']}   请求次数: {result['attempts']}")
