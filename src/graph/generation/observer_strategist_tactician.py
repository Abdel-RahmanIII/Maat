"""Observer-Strategist-Tactician subgraph — Observer → Strategist → Tactician → parse + validate."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.observer import observe_position
from src.agents.strategist import create_plan_from_observation
from src.agents.tactician import execute_plan_from_observation
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import parse_only
from src.state import TurnState
from src.agents.base import get_side_to_move


def build_observer_strategist_tactician_subgraph(
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build the three-agent observation → strategy → tactics generation subgraph."""

    cfg = model_config
    ctx = context

    def _observer(state: TurnState) -> dict[str, Any]:
        """Observer produces the authoritative board description."""

        history = (
            ctx.get_history(
                "observer",
                tokens_used_so_far=state["prompt_token_count"],
                tokenizer_name=cfg.model_name if cfg else None,
            )
            if ctx
            else None
        )

        result = observe_position(
            fen=state["board_fen"],
            move_history=state["move_history"],
            feedback_history=state["feedback_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
            conversation_history=history,
        )
        return {
            "observation_summary": result["summary"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + result["prompt_tokens"]
                + result["completion_tokens"]
            ),
            "prompt_token_count": state["prompt_token_count"] + result["prompt_tokens"],
            "wall_clock_ms": state["wall_clock_ms"] + result["elapsed_ms"],
        }

    def _strategist(state: TurnState) -> dict[str, Any]:
        """Strategist turns the observer summary into a strategic plan."""

        history = (
            ctx.get_history(
                "strategist",
                tokens_used_so_far=state["prompt_token_count"],
                tokenizer_name=cfg.model_name if cfg else None,
            )
            if ctx
            else None
        )
        color = get_side_to_move(state["board_fen"])

        result = create_plan_from_observation(
            move_history=state["move_history"],
            color=color,
            observation_summary=state["observation_summary"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
            conversation_history=history,
        )
        pending = dict(state.get("messages", {}))
        pending["strategist"] = list(result.get("turn_messages", []))
        return {
            "strategic_plan": result["plan"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + result["prompt_tokens"]
                + result["completion_tokens"]
            ),
            "prompt_token_count": state["prompt_token_count"] + result["prompt_tokens"],
            "wall_clock_ms": state["wall_clock_ms"] + result["elapsed_ms"],
            "messages": pending,
        }

    def _tactician(state: TurnState) -> dict[str, Any]:
        """Tactician converts the plan into a UCI move using the raw board."""

        history = (
            ctx.get_history(
                "tactician",
                tokens_used_so_far=state["prompt_token_count"],
                tokenizer_name=cfg.model_name if cfg else None,
            )
            if ctx
            else None
        )

        result = execute_plan_from_observation(
            fen=state["board_fen"],
            move_history=state["move_history"],
            observation_summary=state["observation_summary"],
            strategic_plan=state["strategic_plan"],
            feedback_history=state["feedback_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
            conversation_history=history,
        )
        return {
            "raw_llm_response": result["raw_output"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + result["prompt_tokens"]
                + result["completion_tokens"]
            ),
            "prompt_token_count": state["prompt_token_count"] + result["prompt_tokens"],
            "wall_clock_ms": state["wall_clock_ms"] + result["elapsed_ms"],
            "_gen_raw_output": result["raw_output"],
        }

    def _parse_validate(state: TurnState) -> dict[str, Any]:
        """Parse the raw LLM output (no legality validation)."""

        pv = parse_only(state["_gen_raw_output"], state["board_fen"])
        attempts = state["total_attempts"] + 1

        error_types = list(state["error_types"])
        if not pv["is_valid"] and pv["error_type"]:
            error_types.append(pv["error_type"])

        return {
            "proposed_move": pv["proposed_move"],
            "is_valid": pv["is_valid"],
            "error_reason": pv["error_reason"],
            "total_attempts": attempts,
            "error_types": error_types,
        }

    graph = StateGraph(TurnState)
    graph.add_node("observer", _observer)
    graph.add_node("strategist", _strategist)
    graph.add_node("tactician", _tactician)
    graph.add_node("parse_validate", _parse_validate)

    graph.set_entry_point("observer")
    graph.add_edge("observer", "strategist")
    graph.add_edge("strategist", "tactician")
    graph.add_edge("tactician", "parse_validate")
    graph.add_edge("parse_validate", END)

    return graph.compile()