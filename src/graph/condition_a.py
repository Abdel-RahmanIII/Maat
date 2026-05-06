"""Condition A — Single LLM baseline (no LangGraph).

Direct API call → parse → validate → accept or forfeit.
No retries.  Establishes the raw LLM capability floor.
"""

from __future__ import annotations

from typing import Any, cast

from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import snapshot_turn_result
from src.graph.generation import build_generation_subgraph
from src.state import InputMode, TurnState, create_initial_turn_state


def run_condition_a(
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    generation_strategy: str = "generator_only",
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> TurnState:
    """Execute one turn under Condition A.

    This invokes the generation subgraph directly — no parent LangGraph
    wrapper.  It mirrors the structure of conditions B–E for comparison
    but deliberately avoids framework overhead beyond the generation
    subgraph itself.
    """

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="A",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=0,
    )
    state["generation_strategy"] = generation_strategy

    # ── Run the generation subgraph ──
    gen_subgraph = build_generation_subgraph(generation_strategy, model_config, context)
    state = cast(TurnState, gen_subgraph.invoke(state))

    # ── Finalize ──
    if not state["is_valid"]:
        state["game_status"] = "forfeit"
    else:
        state["game_status"] = "ongoing"

    state["turn_results"].append(snapshot_turn_result(state))
    return state
