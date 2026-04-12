"""Phase-specific specialist generators — used by the Router-Specialists extension."""

from __future__ import annotations

from typing import Any

from src.agents.generator import generate_move
from src.config import ModelConfig
from src.state import InputMode


# Mapping from phase → specialist prompt file
_PHASE_PROMPTS: dict[str, str] = {
    "opening": "opening_specialist.txt",
    "middlegame": "middlegame_specialist.txt",
    "endgame": "endgame_specialist.txt",
}


def generate_specialist_move(
    *,
    phase: str,
    fen: str,
    move_history: list[str],
    feedback_history: list[str] | None = None,
    input_mode: InputMode = "fen",
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Generate a move using the phase-specific specialist prompt.

    Delegates to :func:`generate_move` with the appropriate prompt template.
    Falls back to the default generator if the phase is unknown.
    """

    prompt_template = _PHASE_PROMPTS.get(phase, "generator.txt")

    return generate_move(
        fen=fen,
        move_history=move_history,
        feedback_history=feedback_history,
        input_mode=input_mode,
        model_config=model_config,
        prompt_template=prompt_template,
    )
