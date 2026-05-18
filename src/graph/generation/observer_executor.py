"""Observer-Executor subgraph — Observer → Executor → parse + validate.

Decomposes generation along the Observation → Execution axis:
- The Observer produces a comprehensive natural-language description of the
  board state — piece positions, control of key squares, pawn structure,
  material balance, king safety, and tactical features.  It does not suggest
  moves, evaluate options, or express intent.
- The Executor receives the Observer's summary and treats it as the
  authoritative board representation.  It generates a single move based
  entirely on that description, explicitly discouraged from re-interpreting
  the raw board independently.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.executor import execute_from_observation
from src.agents.observer import observe_position
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import parse_only
from src.state import TurnState


def build_observer_executor_subgraph(
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build the Observer-Executor three-node generation subgraph.

    Nodes: observer → executor → parse_validate

    If *context* is provided, agent nodes will read/write conversation
    history for their respective agent IDs.
    """

    cfg = model_config
    ctx = context

    def _observer(state: TurnState) -> dict[str, Any]:
        """Observer produces a comprehensive board description."""

        history = ctx.get_history("observer") if ctx else None

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
            "strategic_plan": "",  # Clear PA field
        }

    def _executor(state: TurnState) -> dict[str, Any]:
        """Executor selects a move based on the Observer's description."""

        history = ctx.get_history("executor") if ctx else None

        result = execute_from_observation(
            fen=state["board_fen"],
            move_history=state["move_history"],
            observation_summary=state["observation_summary"],
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

    # ── Conditional entry point for Observer caching ──
    def _should_skip_observer(state: TurnState) -> str:
        """Skip Observer if summary is already cached (retry scenario)."""
        if state.get("observation_summary"):
            return "executor"
        return "observer"

    graph = StateGraph(TurnState)
    graph.add_node("observer", _observer)
    graph.add_node("executor", _executor)
    graph.add_node("parse_validate", _parse_validate)

    graph.set_conditional_entry_point(
        _should_skip_observer,
        {"observer": "observer", "executor": "executor"},
    )
    graph.add_edge("observer", "executor")
    graph.add_edge("executor", "parse_validate")
    graph.add_edge("parse_validate", END)

    return graph.compile()
