"""Unified LLM client factory for all conditions."""

from __future__ import annotations

import logging
import time
import asyncio
from itertools import count
from typing import TYPE_CHECKING, Any, Sequence

from google.genai import types
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ModelConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ─── LOGGING & RATE LIMITING SETUP ──────────────────────────────────────────

# Rate limiter — imported lazily to avoid circular deps and
# to stay backward-compatible when running without the runner.
_rate_limiter = None

def _get_limiter():
    """Lazily fetch the global rate limiter (None if not configured)."""
    global _rate_limiter
    if _rate_limiter is None:
        try:
            from src.runner.limiting.rate_limiter import get_rate_limiter
            _rate_limiter = get_rate_limiter()
        except Exception:
            pass
    return _rate_limiter


# Configure dedicated logger for API calls
api_logger = logging.getLogger("maat.api")
if not api_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    api_logger.addHandler(_handler)
api_logger.setLevel(logging.INFO)
api_logger.propagate = False

# Suppress noisy third-party logging from the underlying SDKs
for _logger_name in ("httpx", "httpcore", "google_genai"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Global counter to uniquely identify interleaved API calls
_CALL_COUNTER = count(1)


# ─── FORMATTING HELPERS ─────────────────────────────────────────────────────

def _truncate(value: str, max_len: int = 40) -> str:
    """Truncate a string to max_len, appending '...' if shortened."""
    single_line = " ".join(value.splitlines()).strip()
    if len(single_line) <= max_len:
        return single_line
    return f"{single_line[:max_len-3]}..."

def _summarize_request(payload: Any) -> str:
    """Provide a compact summary of the request payload (number of messages, total characters)."""
    if isinstance(payload, list):
        chars = sum(len(str(getattr(msg, "content", ""))) for msg in payload)
        return f"msgs={len(payload)} chars={chars}"
    return f"type={type(payload).__name__}"

def _summarize_response(response: Any) -> str:
    """Provide a compact summary of the model response (token usage, tool calls, or short text)."""
    usage = getattr(response, "usage_metadata", {}) or {}
    tool_calls = getattr(response, "tool_calls", None) or []
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)

    parts = [f"in={in_tok}", f"out={out_tok}"]
    if tool_calls:
        parts.append(f"tools={len(tool_calls)}")
    else:
        content = str(getattr(response, "content", "")).strip()
        if content:
            parts.append(f"text='{_truncate(content, 30)}'")

    return " ".join(parts)


# ─── CLIENT WRAPPER ─────────────────────────────────────────────────────────

class LoggedModelRunnable(Runnable[Any, AIMessage]):
    """Delegating wrapper that intercepts, logs, and rate-limits each model call."""

    def __init__(self, runnable: Any, *, model_name: str, has_tools: bool) -> None:
        self._runnable = runnable
        self._model_name = model_name
        self._has_tools = has_tools

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        """Synchronously invoke the underlying model with logging and rate limiting."""
        limiter = _get_limiter()
        call_id = next(_CALL_COUNTER)
        short_model = self._model_name.split("/")[-1]
        tools_flag = "T" if self._has_tools else "F"

        max_attempts = 5
        base_delay = 2.0

        for attempt in range(1, max_attempts + 1):
            if limiter:
                limiter.acquire()

            api_logger.info(
                "[%s] ↗ req (attempt %d/%d): %s tools=%s | %s",
                call_id,
                attempt,
                max_attempts,
                short_model,
                tools_flag,
                _summarize_request(input),
            )
            started = time.perf_counter()
            
            try:
                response = self._runnable.invoke(input, config=config, **kwargs)
                
                elapsed_ms = (time.perf_counter() - started) * 1000
                api_logger.info(
                    "[%s] ↘ ok: %.0fms | %s",
                    call_id,
                    elapsed_ms,
                    _summarize_response(response),
                )

                if limiter:
                    usage = getattr(response, "usage_metadata", {}) or {}
                    limiter.record_tokens(
                        prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                        completion_tokens=int(usage.get("output_tokens", 0) or 0),
                    )

                return response

            except Exception as e:
                elapsed_ms = (time.perf_counter() - started) * 1000
                is_last_attempt = attempt == max_attempts
                
                if is_last_attempt:
                    api_logger.exception("[%s] ↘ err: %.0fms | Final attempt failed.", call_id, elapsed_ms)
                    raise
                else:
                    delay = base_delay * (2 ** (attempt - 1))
                    api_logger.warning(
                        "[%s] ↘ err: %.0fms | %s: %s | Retrying in %.1fs...",
                        call_id, elapsed_ms, type(e).__name__, _truncate(str(e), 100), delay
                    )
                    time.sleep(delay)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        """Asynchronously invoke the underlying model with logging and rate limiting."""
        limiter = _get_limiter()
        call_id = next(_CALL_COUNTER)
        short_model = self._model_name.split("/")[-1]
        tools_flag = "T" if self._has_tools else "F"

        max_attempts = 5
        base_delay = 2.0

        for attempt in range(1, max_attempts + 1):
            if limiter:
                limiter.acquire()

            api_logger.info(
                "[%s] ↗ req (attempt %d/%d): %s tools=%s | %s",
                call_id,
                attempt,
                max_attempts,
                short_model,
                tools_flag,
                _summarize_request(input),
            )
            started = time.perf_counter()
            
            try:
                response = await self._runnable.ainvoke(input, config=config, **kwargs)
                
                elapsed_ms = (time.perf_counter() - started) * 1000
                api_logger.info(
                    "[%s] ↘ ok: %.0fms | %s",
                    call_id,
                    elapsed_ms,
                    _summarize_response(response),
                )

                if limiter:
                    usage = getattr(response, "usage_metadata", {}) or {}
                    limiter.record_tokens(
                        prompt_tokens=int(usage.get("input_tokens", 0) or 0),
                        completion_tokens=int(usage.get("output_tokens", 0) or 0),
                    )

                return response

            except Exception as e:
                elapsed_ms = (time.perf_counter() - started) * 1000
                is_last_attempt = attempt == max_attempts
                
                if is_last_attempt:
                    api_logger.exception("[%s] ↘ err: %.0fms | Final attempt failed.", call_id, elapsed_ms)
                    raise
                else:
                    delay = base_delay * (2 ** (attempt - 1))
                    api_logger.warning(
                        "[%s] ↘ err: %.0fms | %s: %s | Retrying in %.1fs...",
                        call_id, elapsed_ms, type(e).__name__, _truncate(str(e), 100), delay
                    )
                    await asyncio.sleep(delay)

    def __getattr__(self, name: str) -> Any:
        # Delegate any other method/attribute accesses to the underlying runnable
        return getattr(self._runnable, name)


# ─── FACTORY METHODS ────────────────────────────────────────────────────────

def _build_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI:
    """Construct a raw ChatGoogleGenerativeAI instance based on configuration."""
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
    """Return a configured, logged chat model (no tool bindings)."""
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
    """Return a configured, logged chat model with the given tools bound."""
    model = _build_model(cfg)
    bound = model.bind_tools(tools)
    return LoggedModelRunnable(
        bound,
        model_name=model.model,
        has_tools=True,
    )
