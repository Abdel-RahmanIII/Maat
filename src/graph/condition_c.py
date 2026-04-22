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
from src.graph.base_graph import parse_and_validate, run_generation, snapshot_turn_result
from src.state import InputMode, TurnState, create_initial_turn_state
from src.validators.symbolic import validate_move


# ── Graph builder ─────────────────────────────────────────────────────────


def build_graph(
    model_config: ModelConfig | None = None,
) -> CompiledStateGraph[TurnState, None, TurnState, TurnState]:
    """Build and return the compiled Condition C graph."""

    cfg = model_config

    def _generate_node(state: TurnState) -> dict[str, Any]:
        """Generate a move via the configured generation strategy."""

        gen = run_generation(state, cfg)
        pv = parse_and_validate(gen["raw_output"], state["board_fen"])

        attempts = state["total_attempts"] + 1
        llm_calls = state["llm_calls_this_turn"] + 1 + gen.get("extra_llm_calls", 0)
        tokens = state["tokens_this_turn"] + gen["prompt_tokens"] + gen["completion_tokens"]

        error_types = list(state["error_types"])

        # If parse failed, record error now (critic can't evaluate unparseable output)
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
        }

    def _route_after_generate(state: TurnState) -> str:
        """If parse succeeded, go to critic; if parse failed, check retries."""

        if state["is_valid"]:
            # Parse succeeded and move is syntactically valid — send to critic
            return "critic"

        # Parse failed
        if state["retry_count"] < state["max_retries"]:
            return "retry_generate"
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

        return {
            "critic_verdict": result["valid"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": state["tokens_this_turn"] + result["prompt_tokens"] + result["completion_tokens"],
        }

    def _route_after_critic(state: TurnState) -> str:
        """If critic says valid → ground-truth check.  Otherwise retry or forfeit."""

        if state["critic_verdict"]:
            return "ground_truth"

        # Critic rejected — build feedback and retry if possible
        if state["retry_count"] < state["max_retries"]:
            return "retry_generate"
        return "forfeit"

    def _retry_node(state: TurnState) -> dict[str, Any]:
        """Increment retry counter and add feedback for the generator."""

        feedback = list(state["feedback_history"])
        feedback.append(
            f"Move {state['proposed_move']} was rejected. Please try a different legal move."
        )

        return {
            "retry_count": state["retry_count"] + 1,
            "feedback_history": feedback,
            "is_valid": False,
        }

    def _ground_truth_node(state: TurnState) -> dict[str, Any]:
        """Check the proposed move against python-chess ground truth.

        This runs AFTER the Critic approves.  If ground truth disagrees,
        the game forfeits (no retry — the Critic already passed it).
        """

        result = validate_move(state["board_fen"], state["proposed_move"])

        error_types = list(state["error_types"])
        if not result["valid"] and result["error_type"]:
            error_types.append(result["error_type"])

        return {
            "is_valid": result["valid"],
            "ground_truth_verdict": result["valid"],
            "error_types": error_types,
        }

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

    graph.add_node("generate", _generate_node)
    graph.add_node("critic", _critic_node)
    graph.add_node("retry_generate", _retry_node)
    graph.add_node("ground_truth", _ground_truth_node)
    graph.add_node("accept", _accept_node)
    graph.add_node("forfeit", _forfeit_node)

    graph.set_entry_point("generate")

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"critic": "critic", "retry_generate": "retry_generate", "forfeit": "forfeit"},
    )
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"ground_truth": "ground_truth", "retry_generate": "retry_generate", "forfeit": "forfeit"},
    )
    graph.add_edge("retry_generate", "generate")
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

    compiled = build_graph(model_config)
    return cast(TurnState, compiled.invoke(state))
