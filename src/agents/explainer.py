"""LLM-based error explainer — used by Condition E."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import load_prompt
from src.llm.llm_client import get_model
from src.config import ModelConfig

import chess


class ExplainerResult(TypedDict):
    explanation: str
    prompt_tokens: int
    completion_tokens: int


def explain_error(
    *,
    fen: str,
    proposed_move: str,
    error_type: str,
    error_reason: str,
    model_config: ModelConfig | None = None,
) -> ExplainerResult:
    """Translate a symbolic validation error into pedagogical feedback.

    Returns a rich natural-language explanation plus token usage data.
    """

    board = chess.Board(fen)
    board_ascii = str(board)

    template = load_prompt("explainer.txt")
    prompt_text = template.format(
        fen=fen,
        board_ascii=board_ascii,
        proposed_move=proposed_move,
        error_type=error_type,
        error_reason=error_reason,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content="You are a chess rules teacher."),
        HumanMessage(content=prompt_text),
    ]

    response = model.invoke(messages)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    return {
        "explanation": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
