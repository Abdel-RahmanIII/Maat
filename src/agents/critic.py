"""LLM-based move critic — used by Condition C."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import load_prompt
from src.llm.llm_client import get_model
from src.config import ModelConfig

import chess


class CriticResult(TypedDict):
    valid: bool
    reasoning: str
    suggestion: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int


def critique_move(
    *,
    fen: str,
    proposed_move: str,
    model_config: ModelConfig | None = None,
) -> CriticResult:
    """Ask the LLM critic to evaluate whether *proposed_move* is legal.

    Returns a structured ``CriticResult``.  Falls back to ``valid=False``
    if the LLM response cannot be parsed as JSON.
    """

    board = chess.Board(fen)
    board_ascii = str(board)

    template = load_prompt("critic.txt")
    prompt_text = template.format(
        fen=fen,
        board_ascii=board_ascii,
        proposed_move=proposed_move,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content="You are a chess rules validation expert."),
        HumanMessage(content=prompt_text),
    ]

    response = model.invoke(messages)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    # Parse JSON from the response
    parsed = _parse_critic_json(raw)

    return {
        "valid": parsed.get("valid", False),
        "reasoning": parsed.get("reasoning", raw),
        "suggestion": parsed.get("suggestion", ""),
        "raw_output": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


def _parse_critic_json(raw: str) -> dict[str, Any]:
    """Best-effort parse of the critic's JSON response."""

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON within the text (e.g., surrounded by markdown fences)
    import re

    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: mark as invalid
    return {"valid": False, "reasoning": raw, "suggestion": ""}
