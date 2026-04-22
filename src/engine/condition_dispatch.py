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
        ``"router_specialists"``.
    model_config:
        LLM model configuration.
    max_react_steps:
        Maximum reasoning steps for Condition F.

    Returns
    -------
    TurnState
        The completed state dict after the condition graph has run.
    """

    runner = cast(Any, _import_runner(condition))
    history = list(move_history or [])

    # Condition F has a unique signature (max_steps instead of
    # generation_strategy).
    if condition == "F":
        return runner(
            fen=fen,
            move_history=history,
            move_number=move_number,
            game_id=game_id,
            input_mode=input_mode,
            max_steps=max_react_steps,
            model_config=model_config,
        )

    return runner(
        fen=fen,
        move_history=history,
        move_number=move_number,
        game_id=game_id,
        input_mode=input_mode,
        generation_strategy=generation_strategy,
        model_config=model_config,
    )


# ── Retry-aware dispatcher with exponential backoff ──────────────────────


def dispatch_turn_with_backoff(
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
    max_api_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
) -> TurnState:
    """Like :func:`dispatch_turn` but with exponential backoff on API errors.

    Retries on ``Exception`` subclasses that look like transient API
    issues (rate-limit 429, server 5xx, timeout).  Non-transient
    exceptions are re-raised immediately.

    Parameters
    ----------
    max_api_retries:
        Maximum number of retry attempts before giving up.
    base_delay:
        Initial backoff delay in seconds.
    max_delay:
        Cap on the exponential backoff delay.
    """

    last_exc: Exception | None = None

    for attempt in range(1, max_api_retries + 1):
        try:
            return dispatch_turn(
                condition,
                fen=fen,
                move_history=move_history,
                move_number=move_number,
                game_id=game_id,
                input_mode=input_mode,
                generation_strategy=generation_strategy,
                model_config=model_config,
                max_react_steps=max_react_steps,
            )
        except Exception as exc:
            last_exc = exc
            exc_text = str(exc).lower()

            # Determine if the error is transient
            transient = any(
                kw in exc_text
                for kw in ("429", "rate", "quota", "resource exhausted",
                            "500", "502", "503", "504", "timeout", "unavailable")
            )

            if not transient:
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "[backoff] Attempt %d/%d for %s failed (%s). "
                "Retrying in %.1fs …",
                attempt,
                max_api_retries,
                game_id,
                exc.__class__.__name__,
                delay,
            )
            time.sleep(delay)

    # Exhausted all retries
    raise RuntimeError(
        f"All {max_api_retries} API retries exhausted for {game_id}"
    ) from last_exc
