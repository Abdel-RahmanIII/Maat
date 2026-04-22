"""Strategist agent — planner in the Planner-Actor MAS extension."""

from __future__ import annotations

from typing import TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    get_side_to_move,
    load_agent_prompt,
)
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


class StrategistResult(TypedDict):
    plan: str
    prompt_tokens: int
    completion_tokens: int


def create_plan(
    *,
    fen: str,
    move_history: list[str],
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
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

    model = get_model(model_config)
    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=prompt_text),
    ]

    response = model.invoke(messages)
    raw = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    usage = response.usage_metadata or {}

    return {
        "plan": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
