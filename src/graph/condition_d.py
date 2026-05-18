"""Condition D — MAS + Symbolic Validator.

Generator proposes a move → python-chess validates → if invalid and
retries remain, terse machine-generated feedback is sent back to the
Generator → retry up to N times → forfeit.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import persist_successful_turn_context, snapshot_turn_result
from src.graph.generation import build_generation_subgraph
from src.state import InputMode, TurnState, create_initial_turn_state
from src.validators.symbolic import validate_move
from src.error_taxonomy import ErrorType


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
    generation_strategy: str = "generator_only",
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build and return the compiled Condition D graph."""

    gen_subgraph = build_generation_subgraph(generation_strategy, model_config, context)

    def _route_after_generate(state: TurnState) -> str:
        return "ground_truth"

    def _ground_truth_node(state: TurnState) -> dict[str, Any]:
        parse_error_types = (ErrorType.PARSE_ERROR.value, ErrorType.NO_OUTPUT.value)
        error_types = list(state.get("error_types", []))
        last_error = error_types[-1] if error_types else None

        if not state["is_valid"] and last_error in parse_error_types:
            feedback = list(state["feedback_history"])
            feedback.append(f"Invalid UCI: {state.get('proposed_move', '')}")

            updates: dict[str, Any] = {
                "is_valid": False,
                "ground_truth_verdict": False,
                "error_reason": state.get("error_reason", ""),
                "feedback_history": feedback,
            }
            if state["total_attempts"] == 1:
                updates["first_try_valid"] = False
            if state["total_attempts"] <= state["max_retries"]:
                updates["retry_count"] = state["retry_count"] + 1
            return updates

        result = validate_move(state["board_fen"], state["proposed_move"])

        error_types = list(state["error_types"])
        if not result["valid"] and result["error_type"]:
            error_types.append(result["error_type"])

        updates: dict[str, Any] = {
            "is_valid": result["valid"],
            "ground_truth_verdict": result["valid"],
            "error_reason": result["reason"],
            "error_types": error_types,
        }
        if state["total_attempts"] == 1:
            updates["first_try_valid"] = result["valid"]

        if not result["valid"]:
            error_reason = result["reason"].strip()
            feedback = list(state["feedback_history"])
            if error_reason:
                feedback.append(
                    f"Illegal move {state['proposed_move']}: {error_reason}"
                )
            else:
                feedback.append(
                    f"Illegal move {state['proposed_move']}."
                )
            if state["total_attempts"] <= state["max_retries"]:
                updates["retry_count"] = state["retry_count"] + 1
            updates["feedback_history"] = feedback

        return updates

    def _route_after_ground_truth(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        if state["total_attempts"] <= state["max_retries"]:
            return "generate"
        return "forfeit"



    def _accept_node(state: TurnState) -> dict[str, Any]:
        return {
            "game_status": "ongoing",
            "turn_results": state["turn_results"] + [snapshot_turn_result(state)],
        }

    def _forfeit_node(state: TurnState) -> dict[str, Any]:
        return {
            "game_status": "forfeit",
            "turn_results": state["turn_results"] + [snapshot_turn_result(state)],
        }

    graph = StateGraph(TurnState)

    graph.add_node("generate", gen_subgraph)
    graph.add_node("ground_truth", _ground_truth_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("generate")

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"ground_truth": "ground_truth"},
    )
    graph.add_conditional_edges(
        "ground_truth",
        _route_after_ground_truth,
        {"accept": "accept", "generate": "generate", "forfeit": "forfeit"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("forfeit", END)

    return graph.compile()


def run_condition_d(
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
    """Execute one turn under Condition D."""

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="D",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=3,
    )
    state["generation_strategy"] = generation_strategy

    compiled = build_graph(model_config, generation_strategy, context)
    state = cast(TurnState, compiled.invoke(state))
    
    if state.get("is_valid"):
        persist_successful_turn_context(state, context)
    
    return state
