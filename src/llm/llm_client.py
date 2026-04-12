"""Unified LLM client factory for all conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import ModelConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


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


def get_model(cfg: ModelConfig | None = None) -> ChatGoogleGenerativeAI:
    """Return a configured chat model (no tool binding)."""

    return _build_model(cfg)


def get_model_with_tools(
    tools: Sequence[BaseTool],
    cfg: ModelConfig | None = None,
) -> Runnable[Any, AIMessage]:
    """Return a chat model with the given *tools* bound for function-calling."""

    model = _build_model(cfg)
    return model.bind_tools(tools)
