"""Tactician agent — actor in the Planner-Actor MAS extension."""

from __future__ import annotations

from typing import Any

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    format_feedback_block,
    get_side_to_move,
    load_agent_prompt,
)
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


def execute_plan(
    *,
    fen: str,
    move_history: list[str],
    strategic_plan: str,
    feedback_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    conversation_history: list[Any] | None = None,
) -> dict[str, Any]:
    """Convert a strategic plan into a concrete UCI move.

    Returns a dict with ``raw_output``, ``prompt_tokens``,
    ``completion_tokens``, and ``turn_messages``.
    """

    color = get_side_to_move(fen)
    board = chess.Board(fen)
    ascii_board = str(board)
    feedback_block = format_feedback_block(feedback_history or [])
    history_str = " ".join(move_history) if move_history else "(none)"

    system_text = load_agent_prompt("tactician", input_mode, "system")
    user_template = load_agent_prompt("tactician", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
        strategic_plan=strategic_plan,
        feedback_block=feedback_block,
    )

    model = get_model(model_config)

    messages: list[Any] = [SystemMessage(content=system_text)]
    if conversation_history:
        messages.extend(conversation_history)
    human_msg = HumanMessage(content=prompt_text)
    messages.append(human_msg)

    response = model.invoke(messages)
    usage = response.usage_metadata or {}

    return {
        "raw_output": response.content.strip() if isinstance(response.content, str) else str(response.content).strip(),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "turn_messages": [human_msg, response],
    }
