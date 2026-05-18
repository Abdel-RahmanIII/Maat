"""Strategist agent — planner in the Planner-Actor MAS extension."""

from __future__ import annotations

from typing import Any, TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    get_side_to_move,
    load_agent_prompt,
)
from src.llm.llm_client import invoke_llm_timed
from src.config import ModelConfig
from src.state import InputMode


class StrategistResult(TypedDict):
    plan: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float
    turn_messages: list[Any]


def create_plan_from_observation(
    *,
    move_history: list[str],
    color: str,
    observation_summary: str,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    conversation_history: list[Any] | None = None,
) -> StrategistResult:
    """Generate a strategic plan from an observer's board description."""
                             
    system_text = load_agent_prompt("strategist", input_mode, "system")
    user_template = load_agent_prompt("strategist", input_mode, "user")
    prompt_text = user_template.format(
        move_history = move_history,
        color=color,
        observation_summary=observation_summary,)

    messages: list[Any] = [SystemMessage(content=system_text)]
    if conversation_history:
        messages.extend(conversation_history)
    human_msg = HumanMessage(content=prompt_text)
    messages.append(human_msg)

    response, elapsed_ms = invoke_llm_timed(messages, model_config)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    return {
        "plan": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "elapsed_ms": elapsed_ms,
        "turn_messages": [human_msg, response],
    }


def create_plan(
    *,
    fen: str,
    move_history: list[str],
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    conversation_history: list[Any] | None = None,
) -> StrategistResult:
    """Generate a natural-language strategic plan for the current position.

    The plan is NOT a move — it describes what the position calls for
    and which piece to consider moving.  The Tactician then converts
    this plan into a concrete UCI move.
    """

    color = get_side_to_move(fen)
    board = chess.Board(fen)
    ascii_board = str(board)
    history_str = " ".join(move_history) if move_history else "(none)"

    system_text = load_agent_prompt("strategist", input_mode, "system")
    user_template = load_agent_prompt("strategist", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
    )

    messages: list[Any] = [SystemMessage(content=system_text)]
    if conversation_history:
        messages.extend(conversation_history)
    human_msg = HumanMessage(content=prompt_text)
    messages.append(human_msg)

    response, elapsed_ms = invoke_llm_timed(messages, model_config)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    return {
        "plan": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "elapsed_ms": elapsed_ms,
        "turn_messages": [human_msg, response],
    }
