"""Phase router agent — classifies the game phase for the Router-Specialists extension."""

from __future__ import annotations

from typing import Literal, TypedDict

import chess
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (
    load_agent_prompt,
)
from src.llm.llm_client import get_model
from src.config import ModelConfig
from src.state import InputMode


Phase = Literal["opening", "middlegame", "endgame"]


class RouterResult(TypedDict):
    phase: Phase
    raw_output: str
    prompt_tokens: int
    completion_tokens: int


def classify_phase(
    *,
    fen: str,
    move_history: list[str],
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
) -> RouterResult:
    """Classify the current game phase using the LLM.

    Returns one of ``'opening'``, ``'middlegame'``, or ``'endgame'``.
    Falls back to ``'middlegame'`` if the response is unparseable.
    """

    board = chess.Board(fen)
    ascii_board = str(board)
    history_str = " ".join(move_history) if move_history else "(none)"

    system_text = load_agent_prompt("router", input_mode, "system")
    user_template = load_agent_prompt("router", input_mode, "user")
    prompt_text = user_template.format(
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

    phase = _parse_phase(raw)

    return {
        "phase": phase,
        "raw_output": raw,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
    }


def _parse_phase(raw: str) -> Phase:
    """Extract a valid phase from the LLM response."""

    normalized = raw.strip().lower()
    for phase in ("opening", "middlegame", "endgame"):
        if phase in normalized:
            return phase  # type: ignore[return-value]
    return "middlegame"
