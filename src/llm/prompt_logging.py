"""Helpers for logging prompts before they are sent to an LLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import logging


def _message_role(message: Any) -> str:
    """Return a stable, human-readable label for a chat message."""

    role = getattr(message, "type", None)
    if isinstance(role, str) and role.strip():
        return role

    class_name = message.__class__.__name__
    return class_name.removesuffix("Message").lower()


def _message_content(message: Any) -> str:
    """Return a printable version of a chat message content payload."""

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def format_prompt_messages(messages: Sequence[Any]) -> str:
    """Render a request payload as a multiline prompt log."""

    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = _message_role(message)
        content = _message_content(message)
        lines.append(f"{index}. {role}: {content}")
    return "\n".join(lines)


def log_prompt(
    logger: logging.Logger,
    messages: Sequence[Any],
    *,
    model_name: str,
    call_mode: str,
    prefix: str = "[Prompt]",
) -> None:
    """Log the full prompt payload before model invocation."""

    logger.info(
        "%s model=%s mode=%s\n%s",
        prefix,
        model_name,
        call_mode,
        format_prompt_messages(messages),
    )