"""Unified LLM client for all conditions."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Sequence

import logging

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ModelConfig, LLMCallMode
from src.llm.prompt_logging import log_prompt
from src.runner.requests.manager import get_global_manager

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


logger = logging.getLogger(__name__)


def beutifyOutput(output: Any) -> str:
    """Return prompt-safe text from a model response or content blocks."""

    content = output.content if isinstance(output, AIMessage) else output

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        cleaned_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text = block.strip()
                if text:
                    cleaned_parts.append(text)
                continue

            if isinstance(block, dict):
                block_type = str(block.get("type", "")).strip().lower()
                text = block.get("text")
                if text is None:
                    text = block.get("content")

                if text is None:
                    continue

                if block_type and block_type not in {"text", "output_text"} and "text" not in block and "content" not in block:
                    continue

                text_value = str(text).strip()
                if text_value:
                    cleaned_parts.append(text_value)
                continue

            text = getattr(block, "text", None)
            if text is None:
                text = getattr(block, "content", None)
            if text is None:
                continue

            text_value = str(text).strip()
            if text_value:
                cleaned_parts.append(text_value)

        return "\n".join(cleaned_parts).strip()

    return str(content).strip()


def _normalize_ai_message(response: AIMessage) -> AIMessage:
    """Ensure model output content is plain text before callers consume it."""

    cleaned_content = beutifyOutput(response)
    if isinstance(response.content, str) and response.content.strip() == cleaned_content:
        return response

    try:
        return response.model_copy(update={"content": cleaned_content})
    except AttributeError:
        response.content = cleaned_content
        return response


def _resolve_api_key(cfg: ModelConfig) -> str:
    """Return the first usable API key for direct execution."""

    if cfg.api_keys:
        return cfg.api_keys[0]
    if cfg.api_key:
        return cfg.api_key
    return ""

def _build_direct_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI:
    """Construct a chat model instance for direct execution."""

    cfg = cfg or ModelConfig()
    return ChatGoogleGenerativeAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        google_api_key=_resolve_api_key(cfg),
        max_retries=0,
    )


def _ensure_queue_manager(cfg: ModelConfig):
    """Return the active queue manager or fail fast."""

    manager = get_global_manager()
    if manager is None or not manager._worker_thread.is_alive():
        raise RuntimeError("ModelConfig.call_mode='queue' requires an active RequestsManager.")
    return manager


def invoke_llm(
    messages: list[Any],
    cfg: ModelConfig,
    *,
    tools: Sequence[BaseTool] | None = None,
    **invoke_kwargs: Any,
) -> AIMessage:
    """Invoke the configured model using direct or queued execution."""

    if cfg.call_mode == LLMCallMode.QUEUE:
        manager = _ensure_queue_manager(cfg)
        log_prompt(
            logger,
            messages,
            model_name=cfg.model_name,
            call_mode=cfg.call_mode.value,
            prefix="[LLM] QUEUED PROMPT",
        )
        future = manager.submit(messages, tools=tools, invoke_kwargs=invoke_kwargs)
        return _normalize_ai_message(future.result())

    model = _build_direct_model(cfg)
    log_prompt(
        logger,
        messages,
        model_name=cfg.model_name,
        call_mode=cfg.call_mode.value,
        prefix="[LLM] DIRECT PROMPT",
    )
    if tools:
        model = model.bind_tools(tools)
    return _normalize_ai_message(model.invoke(messages, **invoke_kwargs))


def invoke_llm_timed(
    messages: list[Any],
    cfg: ModelConfig | None,
    *,
    tools: Sequence[BaseTool] | None = None,
    **invoke_kwargs: Any,
) -> tuple[AIMessage, float]:
    """Invoke the model and return ``(response, elapsed_ms)``."""

    start = time.perf_counter()
    response = invoke_llm(messages, cfg or ModelConfig(), tools=tools, **invoke_kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


async def ainvoke_llm(
    messages: list[Any],
    cfg: ModelConfig,
    *,
    tools: Sequence[BaseTool] | None = None,
    **invoke_kwargs: Any,
) -> AIMessage:
    """Async wrapper around :func:`invoke_llm`."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: invoke_llm(messages, cfg, tools=tools, **invoke_kwargs),
    )


def get_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI:
    """Return a configured direct chat model."""

    return _build_direct_model(cfg)


def get_model_with_tools(
    tools: Sequence[BaseTool],
    cfg: ModelConfig | None = None,
) -> Runnable[Any, AIMessage]:
    """Return a configured direct chat model with the given tools bound."""

    model = _build_direct_model(cfg)
    return model.bind_tools(tools)
