"""Condition C — MAS + LLM Critic.

Generator proposes a move → LLM Critic evaluates legality →
if Critic approves, ground-truth check → accept or forfeit.
If Critic rejects and retries remain, Generator retries with feedback.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.critic import critique_move
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import persist_successful_turn_context, snapshot_turn_result
from src.graph.generation import build_generation_subgraph
from src.state import InputMode, TurnState, create_initial_turn_state
from src.validators.symbolic import validate_move


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
    generation_strategy: str = "generator_only",
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build and return the compiled Condition C graph."""

    cfg = model_config
    gen_subgraph = build_generation_subgraph(generation_strategy, model_config, context)

    def _route_after_generate(state: TurnState) -> str:
        """If parse succeeded, go to critic; if parse failed, forfeit."""

        if state["is_valid"]:
            return "critic"

        return "forfeit"

    def _critic_node(state: TurnState) -> dict[str, Any]:
        """LLM Critic evaluates the proposed move."""

        result = critique_move(
            fen=state["board_fen"],
            proposed_move=state["proposed_move"],
            move_history=state["move_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
        )

        updates: dict[str, Any] = {
            "critic_verdict": result["valid"],
            "error_reason": result["reasoning"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": state["tokens_this_turn"] + result["prompt_tokens"] + result["completion_tokens"],
            "wall_clock_ms": state["wall_clock_ms"] + result["elapsed_ms"],
        }

        if not result["valid"]:
            feedback = list(state["feedback_history"])
            reason = result["reasoning"].strip()
            if reason:
                feedback.append(
                    f"Critic rejected move {state['proposed_move']}: {reason}"
                )
            else:
                feedback.append(
                    f"Critic rejected move {state['proposed_move']}. Please try a different legal move."
                )

            updates.update(
                {
                    "retry_count": state["retry_count"] + 1,
                    "feedback_history": feedback,
                    "is_valid": False,
                }
            )

        return updates

    def _route_after_critic(state: TurnState) -> str:
        """If critic says valid → ground-truth check.  Otherwise retry or forfeit."""

        if state["critic_verdict"]:
            return "ground_truth"

        # Critic rejected — retry if possible
        if state["retry_count"] <= state["max_retries"]:
            return "generate"
        return "forfeit"

    def _ground_truth_node(state: TurnState) -> dict[str, Any]:
        """Check the proposed move against python-chess ground truth.

        This runs AFTER the Critic approves.  If ground truth disagrees,
        the game forfeits (no retry — the Critic already passed it).
        """

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
        return updates

    def _route_after_ground_truth(state: TurnState) -> str:
        if state["is_valid"]:
            return "accept"
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
    graph.add_node("critic", _critic_node)
    graph.add_node("ground_truth", _ground_truth_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("generate")

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"critic": "critic", "forfeit": "forfeit"},
    )
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"ground_truth": "ground_truth", "generate": "generate", "forfeit": "forfeit"},
    )
    graph.add_conditional_edges(
        "ground_truth",
        _route_after_ground_truth,
        {"accept": "accept", "forfeit": "forfeit"},
    )
    graph.add_edge("accept", END)
    graph.add_edge("forfeit", END)

    return graph.compile()


def run_condition_c(
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
    """Execute one turn under Condition C."""

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="C",
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
