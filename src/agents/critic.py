"""LLM-based move critic — used by Condition C."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_side_to_move, load_agent_prompt
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


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
    move_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
) -> CriticResult:
    """Ask the LLM critic to evaluate whether *proposed_move* is legal.

    Returns a structured ``CriticResult``.  Falls back to ``valid=False``
    if the LLM response cannot be parsed as JSON.
    """

    board = chess.Board(fen)
    ascii_board = str(board)
    color = get_side_to_move(fen)
    history_str = " ".join(move_history or []) if move_history else "(none)"

    system_text = load_agent_prompt("critic", input_mode, "system")
    user_template = load_agent_prompt("critic", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
        proposed_move=proposed_move,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content=system_text),
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

    # Try fenced JSON first.
    import re

    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: parse from first opening brace to last closing brace.
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first:last + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: mark as invalid
    return {"valid": False, "reasoning": raw, "suggestion": ""}
