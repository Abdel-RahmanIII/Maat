"""Tactician agent — actor in the Planner-Actor MAS extension."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    build_board_representation,
    format_feedback_block,
    get_side_to_move,
    load_prompt,
)
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


_TACTICIAN_TEMPLATE = """\
You are a chess tactician. You are playing as {color}.

{board_representation}

Move history (UCI): {move_history}

A strategist has provided the following plan:
--- STRATEGIC PLAN ---
{strategic_plan}
--- END PLAN ---
{feedback_block}
Based on this plan, select the best concrete move.
Output exactly one move in UCI format (e.g., e2e4, g1f3).
Respond with ONLY the UCI move, no explanation."""


def execute_plan(
    *,
    fen: str,
    move_history: list[str],
    strategic_plan: str,
    feedback_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Convert a strategic plan into a concrete UCI move.

    Returns a dict with ``raw_output``, ``prompt_tokens``, and ``completion_tokens``.
    """

    color = get_side_to_move(fen)
    board_repr = build_board_representation(fen, input_mode, move_history)
    feedback_block = format_feedback_block(feedback_history or [])
    history_str = " ".join(move_history) if move_history else "(none)"

    prompt_text = _TACTICIAN_TEMPLATE.format(
        color=color,
        board_representation=board_repr,
        move_history=history_str,
        strategic_plan=strategic_plan,
        feedback_block=feedback_block,
    )

    model = get_model(model_config)
    messages = [
        SystemMessage(content="You are a chess-playing assistant executing a strategic plan."),
        HumanMessage(content=prompt_text),
    ]

    response = model.invoke(messages)
    usage = response.usage_metadata or {}

    return {
        "raw_output": response.content.strip() if isinstance(response.content, str) else str(response.content).strip(),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }
