"""Condition A — Single LLM baseline (no LangGraph).

Direct API call → parse → validate → accept or forfeit.
No retries.  Establishes the raw LLM capability floor.
"""

from __future__ import annotations

from typing import cast

from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import persist_successful_turn_context, snapshot_turn_result
from src.graph.generation import build_generation_subgraph
from src.state import InputMode, TurnState, create_initial_turn_state
from src.validators.symbolic import validate_move


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
        result = validate_move(state["board_fen"], state["proposed_move"])
        error_types = list(state["error_types"])
        if not result["valid"] and result["error_type"]:
            error_types.append(result["error_type"])

        state["is_valid"] = result["valid"]
        if state["total_attempts"] == 1:
            state["first_try_valid"] = result["valid"]
        state["ground_truth_verdict"] = result["valid"]
        state["error_reason"] = result["reason"]
        state["error_types"] = error_types
        state["game_status"] = "ongoing" if result["valid"] else "forfeit"
        if result["valid"]:
            persist_successful_turn_context(state, context)

    state["turn_results"].append(snapshot_turn_result(state))
    return state
