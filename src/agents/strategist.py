"""Strategist agent — planner in the Planner-Actor MAS extension."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    build_board_representation,
    get_side_to_move,
    load_prompt,
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
    board_repr = build_board_representation(fen, input_mode, move_history)
    history_str = " ".join(move_history) if move_history else "(none)"

    template = load_prompt("strategist.txt")
    prompt_text = template.format(
        color=color,
        board_representation=board_repr,
        move_history=history_str,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content="You are a chess strategist."),
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
