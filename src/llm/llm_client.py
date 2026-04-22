"""Unified LLM client factory for all conditions."""

from __future__ import annotations

import logging
import time
from itertools import count
from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ModelConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


api_logger = logging.getLogger("maat.api")
if not api_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    api_logger.addHandler(_handler)
api_logger.setLevel(logging.INFO)
api_logger.propagate = False

_CALL_COUNTER = count(1)


def _truncate(value: str, max_len: int = 80) -> str:
    single_line = " ".join(value.splitlines()).strip()
    if len(single_line) <= max_len:
        return single_line
    return f"{single_line[:max_len]}..."


def _summarize_request(payload: Any) -> str:
    if isinstance(payload, list):
        msg_count = len(payload)
        chars = 0
        last_role = ""
        for msg in payload:
            last_role = str(getattr(msg, "type", msg.__class__.__name__))
            chars += len(str(getattr(msg, "content", "")))
        return f"msgs={msg_count} chars={chars} last={last_role}"
    return f"payload={_truncate(str(type(payload).__name__), 20)}"


def _summarize_response(response: Any) -> str:
    content = getattr(response, "content", response)
    usage = getattr(response, "usage_metadata", {}) or {}
    tool_calls = getattr(response, "tool_calls", None) or []
    in_tokens = int(usage.get("input_tokens", 0) or 0)
    out_tokens = int(usage.get("output_tokens", 0) or 0)
    return (
        f"in={in_tokens} out={out_tokens} tools={len(tool_calls)} "
        f"text='{_truncate(str(content), 80)}'"
    )


class LoggedModelRunnable(Runnable[Any, AIMessage]):
    """Delegating wrapper that logs each model call and its response."""

    def __init__(self, runnable: Any, *, model_name: str, has_tools: bool) -> None:
        self._runnable = runnable
        self._model_name = model_name
        self._has_tools = has_tools

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        call_id = next(_CALL_COUNTER)
        started = time.perf_counter()
        api_logger.info(
            "[call=%s] request model=%s tools=%s payload=%s",
            call_id,
            self._model_name,
            self._has_tools,
            _summarize_request(input),
        )
        try:
            response = self._runnable.invoke(input, config=config, **kwargs)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            api_logger.exception(
                "[call=%s] response model=%s status=error elapsed_ms=%.1f",
                call_id,
                self._model_name,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        api_logger.info(
            "[call=%s] response model=%s status=ok elapsed_ms=%.1f %s",
            call_id,
            self._model_name,
            elapsed_ms,
            _summarize_response(response),
        )
        return response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        call_id = next(_CALL_COUNTER)
        started = time.perf_counter()
        api_logger.info(
            "[call=%s] request model=%s tools=%s payload=%s",
            call_id,
            self._model_name,
            self._has_tools,
            _summarize_request(input),
        )
        try:
            response = await self._runnable.ainvoke(input, config=config, **kwargs)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            api_logger.exception(
                "[call=%s] response model=%s status=error elapsed_ms=%.1f",
                call_id,
                self._model_name,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        api_logger.info(
            "[call=%s] response model=%s status=ok elapsed_ms=%.1f %s",
            call_id,
            self._model_name,
            elapsed_ms,
            _summarize_response(response),
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def _build_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI:
    """Construct a ``ChatGoogleGenerativeAI`` from *cfg*."""

    cfg = cfg or ModelConfig()

    if not cfg.api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    return ChatGoogleGenerativeAI(
        model=cfg.model_name,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        google_api_key=cfg.api_key,
        # Options: "minimal", "high"
        # thinking_level="minimal",
    )


def get_model(cfg: ModelConfig | None = None) -> Runnable[Any, AIMessage]:
    """Return a configured chat model (no tool binding)."""

    model = _build_model(cfg)
    return LoggedModelRunnable(
        model,
        model_name=model.model,
        has_tools=False,
    )


def get_model_with_tools(
    tools: Sequence[BaseTool],
    cfg: ModelConfig | None = None,
) -> Runnable[Any, AIMessage]:
    """Return a chat model with the given *tools* bound for function-calling."""

    model = _build_model(cfg)
    bound = model.bind_tools(tools)
    return LoggedModelRunnable(
        bound,
        model_name=model.model,
        has_tools=True,
    )
