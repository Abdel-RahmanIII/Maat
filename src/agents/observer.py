"""Observer agent — board description in the Observer-Executor MAS strategy."""

from __future__ import annotations

from typing import Any, TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    get_side_to_move,
    load_agent_prompt,
)
from src.config import ModelConfig
from src.llm.llm_client import invoke_llm_timed
from src.state import InputMode


class ObserverResult(TypedDict):
    summary: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float
    turn_messages: list[Any]


def observe_position(
    *,
    fen: str,
    move_history: list[str],
    feedback_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
    conversation_history: list[Any] | None = None,
) -> ObserverResult:
    """Produce a comprehensive natural-language description of the board.

    The Observer describes piece positions, control of key squares,
    pawn structure, material balance, king safety, and tactical features.
    It does **not** suggest moves, evaluate options, or express intent.

    Returns a dict with ``summary``, ``prompt_tokens``,
    ``completion_tokens``, ``elapsed_ms``, and ``turn_messages``.
    """

    color = get_side_to_move(fen)
    board = chess.Board(fen)
    ascii_board = str(board)
    history_str = " ".join(move_history) if move_history else "(none)"

    # Build feedback block for the observer (past feedback from failed attempts)
    feedback_block = ""
    if feedback_history:
        lines = ["", "Previous move attempts were rejected for these reasons:"]
        for idx, fb in enumerate(feedback_history, 1):
            lines.append(f"  Attempt {idx}: {fb}")
        lines.append("")
        lines.append(
            "Include any positional details that might help avoid repeating "
            "these errors."
        )
        feedback_block = "\n".join(lines)

    system_text = load_agent_prompt("observer", input_mode, "system")
    user_template = load_agent_prompt("observer", input_mode, "user")
    prompt_text = user_template.format(
        color=color,
        fen=fen,
        ascii_board=ascii_board,
        move_history=history_str,
        feedback_block=feedback_block,
    )

    messages: list[Any] = [SystemMessage(content=system_text)]
    if conversation_history:
        messages.extend(conversation_history)
    human_msg = HumanMessage(content=prompt_text)
    messages.append(human_msg)

    response, elapsed_ms = invoke_llm_timed(messages, model_config)
    raw = (
        response.content.strip()
        if isinstance(response.content, str)
        else str(response.content).strip()
    )
    usage = response.usage_metadata or {}

    return {
        "summary": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "elapsed_ms": elapsed_ms,
        "turn_messages": [human_msg, response],
    }
