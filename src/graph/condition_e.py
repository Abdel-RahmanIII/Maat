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
from src.graph.base_graph import parse_and_validate, run_generation, snapshot_turn_result
from src.state import InputMode, TurnState, create_initial_turn_state


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Build and return the compiled Condition E graph."""

    cfg = model_config

    def _generate_node(state: TurnState) -> dict[str, Any]:
        """Generate a move and run parse + symbolic validation."""

        gen = run_generation(state, cfg)
        pv = parse_and_validate(gen["raw_output"], state["board_fen"])

        attempts = state["total_attempts"] + 1
        llm_calls = state["llm_calls_this_turn"] + 1 + gen.get("extra_llm_calls", 0)
        tokens = state["tokens_this_turn"] + gen["prompt_tokens"] + gen["completion_tokens"]

        error_types = list(state["error_types"])
        if not pv["is_valid"] and pv["error_type"]:
            error_types.append(pv["error_type"])

        return {
            "proposed_move": pv["proposed_move"],
            "is_valid": pv["is_valid"],
            "first_try_valid": pv["is_valid"] if attempts == 1 else state["first_try_valid"],
            "total_attempts": attempts,
            "llm_calls_this_turn": llm_calls,
            "tokens_this_turn": tokens,
            "prompt_token_count": state["prompt_token_count"] + gen["prompt_tokens"],
            "error_types": error_types,
            "strategic_plan": gen.get("strategic_plan", ""),
            "routed_phase": gen.get("routed_phase", ""),
            "_error_type": pv["error_type"] if not pv["is_valid"] else "",
            "_error_reason": pv["error_reason"] if not pv["is_valid"] else "",
        }

    def _route_after_generate(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        if state["retry_count"] < state["max_retries"]:
            return "explainer"
        return "forfeit"

    def _explainer_node(state: TurnState) -> dict[str, Any]:
        """LLM Explainer translates the symbolic error into pedagogical feedback."""

        error_type = state.get("_error_type", "UNKNOWN")
        error_reason = state.get("_error_reason", "Move is illegal.")

        result = explain_error(
            fen=state["board_fen"],
            proposed_move=state["proposed_move"],
            error_type=error_type,
            error_reason=error_reason,
            model_config=cfg,
        )

        feedback = list(state["feedback_history"])
        feedback.append(result["explanation"])

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

    graph.add_node("generate", _generate_node)
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

    compiled = build_graph(model_config)
    return cast(TurnState, compiled.invoke(state))
