"""Executor agent — move generation in the Observer-Executor MAS strategy."""

from __future__ import annotations

from typing import Any

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    format_feedback_block,
    get_side_to_move,
    load_agent_prompt,
)
from src.config import ModelConfig
from src.llm.llm_client import invoke_llm_timed
from src.state import InputMode


def execute_from_observation(
    *,
    fen: str,
    move_history: list[str],
    observation_summary: str,
    feedback_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    conversation_history: list[Any] | None = None,
) -> dict[str, Any]:
    """Generate a single move based on the Observer's board description.

    The Executor treats the Observer's summary as the authoritative
    representation of the board and bases its decision entirely on
    that description.  It is explicitly discouraged from
    re-interpreting the raw board independently.

    Returns a dict with ``raw_output``, ``prompt_tokens``,
    ``completion_tokens``, ``elapsed_ms``, and ``turn_messages``.
    """

    color = get_side_to_move(fen)
    board = chess.Board(fen)
    ascii_board = str(board)
    feedback_block = format_feedback_block(feedback_history or [])
    history_str = " ".join(move_history) if move_history else "(none)"

    system_text = load_agent_prompt("executor", input_mode, "system")
    user_template = load_agent_prompt("executor", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
        observation_summary=observation_summary,
        feedback_block=feedback_block,
    )

    messages: list[Any] = [SystemMessage(content=system_text)]
    if conversation_history:
        messages.extend(conversation_history)
    human_msg = HumanMessage(content=prompt_text)
    messages.append(human_msg)

    response, elapsed_ms = invoke_llm_timed(messages, model_config)
    usage = response.usage_metadata or {}

    return {
        "raw_output": (
            response.content.strip()
            if isinstance(response.content, str)
            else str(response.content).strip()
        ),
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "elapsed_ms": elapsed_ms,
        "turn_messages": [human_msg, response],
    }
