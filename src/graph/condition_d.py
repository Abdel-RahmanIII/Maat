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
from src.graph.base_graph import parse_and_validate, run_generation, snapshot_turn_result
from src.state import InputMode, TurnState, create_initial_turn_state


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Build and return the compiled Condition D graph."""

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
            # Stash error info for feedback construction
            "_error_reason": pv["error_reason"] if not pv["is_valid"] else "",
        }

    def _route_after_generate(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
        if state["retry_count"] < state["max_retries"]:
            return "terse_feedback"
        return "forfeit"

    def _terse_feedback_node(state: TurnState) -> dict[str, Any]:
        """Build a terse machine-generated feedback message and increment retries."""

        error_reason = state.get("_error_reason", "Move is illegal.")

        feedback = list(state["feedback_history"])
        feedback.append(
            f"Illegal move {state['proposed_move']}: {error_reason}"
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

    graph.add_node("generate", _generate_node)
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

    compiled = build_graph(model_config)
    return cast(TurnState, compiled.invoke(state))
