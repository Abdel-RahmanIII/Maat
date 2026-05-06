"""Condition E — MAS + Symbolic Validator + LLM Explainer.

Generator proposes → python-chess validates → if invalid and retries
remain, the LLM Explainer translates the symbolic error into rich
pedagogical feedback → Generator retries → up to N times → forfeit.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.explainer import explain_error
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
    """Build and return the compiled Condition E graph."""

    cfg = model_config
    gen_subgraph = build_generation_subgraph(generation_strategy, model_config, context)

    def _route_after_generate(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        if state["retry_count"] < state["max_retries"]:
            return "explainer"
        return "forfeit"

    def _explainer_node(state: TurnState) -> dict[str, Any]:
        """LLM Explainer translates the symbolic error into pedagogical feedback."""

        # Get the most recent error info from the subgraph's parse+validate
        error_types = state.get("error_types", [])
        error_type = error_types[-1] if error_types else "UNKNOWN"

        result = explain_error(
            fen=state["board_fen"],
            attempted_move=state["proposed_move"],
            error_type=error_type,
            engine_error_message=f"Illegal move: {error_type}",
            move_history=state["move_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
        )

        feedback = list(state["feedback_history"])
        feedback.append(result["feedback_text"])

        return {
            "retry_count": state["retry_count"] + 1,
            "feedback_history": feedback,
            "is_valid": False,
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": state["tokens_this_turn"] + result["prompt_tokens"] + result["completion_tokens"],
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
    graph.add_node("explainer", _explainer_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("generate")

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"accept": "accept", "explainer": "explainer", "forfeit": "forfeit"},
    )
    graph.add_edge("explainer", "generate")
    graph.add_edge("accept", END)
    graph.add_edge("forfeit", END)

    return graph.compile()


def run_condition_e(
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
    """Execute one turn under Condition E."""

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="E",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=3,
    )
    state["generation_strategy"] = generation_strategy

    compiled = build_graph(model_config, generation_strategy, context)
    return cast(TurnState, compiled.invoke(state))
