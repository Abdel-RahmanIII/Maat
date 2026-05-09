"""Centralized condition dispatcher.

Maps a condition name (``"A"`` – ``"F"``) to the corresponding
``run_condition_*`` function and executes one turn, returning the
final :class:`~src.state.TurnState`.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from src.config import ModelConfig
from src.context import ConversationContext
from src.state import InputMode, TurnState

logger = logging.getLogger(__name__)

# ── Lazy imports ─────────────────────────────────────────────────────────
# Condition modules are imported lazily to avoid circular-import issues
# and to keep startup fast when only a subset of conditions is needed.


def _import_runner(condition: str):  # noqa: ANN202
    """Return the ``run_condition_X`` callable for *condition*."""

    if condition == "A":
        from src.graph.condition_a import run_condition_a
        return run_condition_a
    if condition == "B":
        from src.graph.condition_b import run_condition_b
        return run_condition_b
    if condition == "C":
        from src.graph.condition_c import run_condition_c
        return run_condition_c
    if condition == "D":
        from src.graph.condition_d import run_condition_d
        return run_condition_d
    if condition == "E":
        from src.graph.condition_e import run_condition_e
        return run_condition_e
    if condition == "F":
        from src.graph.condition_f import run_condition_f
        return run_condition_f

    raise ValueError(f"Unknown condition: {condition!r}")


# ── Public dispatcher ────────────────────────────────────────────────────


def dispatch_turn(
    condition: str,
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    generation_strategy: str = "generator_only",
    model_config: ModelConfig | None = None,
    max_react_steps: int = 6,
    context: ConversationContext | None = None,
) -> TurnState:
    """Route to the correct condition runner and return the final TurnState.

    Parameters
    ----------
    condition:
        One of ``"A"`` – ``"F"``.
    fen:
        Current board position in FEN notation.
    move_history:
        All UCI moves played so far.
    move_number:
        Full-move number (1-indexed).
    game_id:
        Unique identifier for the game / position.
    input_mode:
        ``"fen"`` (Exp 1 & 2) or ``"history"`` (Exp 3).
    generation_strategy:
        ``"generator_only"``, ``"planner_actor"``, or
        ``"threat_analyst"``.
    model_config:
        LLM model configuration.
    max_react_steps:
        Maximum reasoning steps for Condition F.
    context:
        Optional :class:`ConversationContext` for multi-turn memory.

    Returns
    -------
    TurnState
        The completed state dict after the condition graph has run.
    """

    runner = cast(Any, _import_runner(condition))
    history = list(move_history or [])

    # Condition F uses a LangGraph StateGraph like B-E but takes
    # max_steps instead of generation_strategy.
    if condition == "F":
        return runner(
            fen=fen,
            move_history=history,
            move_number=move_number,
            game_id=game_id,
            input_mode=input_mode,
            max_steps=max_react_steps,
            model_config=model_config,
            context=context,
        )

    return runner(
        fen=fen,
        move_history=history,
        move_number=move_number,
        game_id=game_id,
        input_mode=input_mode,
        generation_strategy=generation_strategy,
        model_config=model_config,
        context=context,
    )



