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
from src.graph.base_graph import snapshot_turn_result
from src.graph.generation import build_generation_subgraph
from src.state import InputMode, TurnState, create_initial_turn_state


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
    generation_strategy: str = "generator_only",
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build and return the compiled Condition D graph."""

    gen_subgraph = build_generation_subgraph(generation_strategy, model_config, context)

    def _route_after_generate(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        if state["retry_count"] < state["max_retries"]:
            return "terse_feedback"
        return "forfeit"

    def _terse_feedback_node(state: TurnState) -> dict[str, Any]:
        """Build a terse machine-generated feedback message and increment retries."""

        # Construct feedback from the error info captured by the subgraph
        error_types = state.get("error_types", [])
        last_error = error_types[-1] if error_types else "UNKNOWN"

        feedback = list(state["feedback_history"])
        feedback.append(
            f"Illegal move {state['proposed_move']}: {last_error}"
        )

        return {
            "retry_count": state["retry_count"] + 1,
            "feedback_history": feedback,
            "is_valid": False,
        }

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
    graph.add_node("terse_feedback", _terse_feedback_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("generate")

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"accept": "accept", "terse_feedback": "terse_feedback", "forfeit": "forfeit"},
    )
    graph.add_edge("terse_feedback", "generate")
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
    return cast(TurnState, compiled.invoke(state))
