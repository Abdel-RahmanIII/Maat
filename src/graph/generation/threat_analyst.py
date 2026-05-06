"""Threat-Analyst subgraph — Threat Analyst → Constrained Generator → parse + validate.

Decomposes generation along the Analysis → Execution axis:
- The Threat Analyst surfaces constraints (pins, checks, blocked pieces)
  that directly target LEAVES_IN_CHECK, INVALID_PIECE, and ILLEGAL_DESTINATION errors.
- The Constrained Generator selects a move respecting every constraint.

This is orthogonal to the Planner-Actor's Strategy → Tactics axis.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.constrained_generator import generate_constrained_move
from src.agents.threat_analyst import analyze_threats
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import parse_and_validate
from src.state import TurnState


def build_threat_analyst_subgraph(
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build the Threat-Analyst three-node generation subgraph.

    Nodes: threat_analyst → constrained_generator → parse_validate

    If *context* is provided, agent nodes will read/write conversation
    history for their respective agent IDs.
    """

    cfg = model_config
    ctx = context

    def _threat_analyst(state: TurnState) -> dict[str, Any]:
        """Threat Analyst produces a structured board constraint report."""

        history = ctx.get_history("threat_analyst") if ctx else None

        result = analyze_threats(
            fen=state["board_fen"],
            move_history=state["move_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
            conversation_history=history,
        )
        return {
            "threat_report": result["report"],
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + result["prompt_tokens"]
                + result["completion_tokens"]
            ),
            "prompt_token_count": state["prompt_token_count"] + result["prompt_tokens"],
            "strategic_plan": "",  # Clear PA field
        }

    def _constrained_generator(state: TurnState) -> dict[str, Any]:
        """Constrained Generator selects a move respecting the threat report."""

        history = ctx.get_history("constrained_generator") if ctx else None

        result = generate_constrained_move(
            fen=state["board_fen"],
            move_history=state["move_history"],
            threat_report=state["threat_report"],
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
            "_gen_raw_output": result["raw_output"],
        }

    def _parse_validate(state: TurnState) -> dict[str, Any]:
        """Parse the raw LLM output and validate it against python-chess."""

        pv = parse_and_validate(state["_gen_raw_output"], state["board_fen"])
        attempts = state["total_attempts"] + 1

        error_types = list(state["error_types"])
        if not pv["is_valid"] and pv["error_type"]:
            error_types.append(pv["error_type"])

        return {
            "proposed_move": pv["proposed_move"],
            "is_valid": pv["is_valid"],
            "first_try_valid": (
                pv["is_valid"] if attempts == 1 else state["first_try_valid"]
            ),
            "total_attempts": attempts,
            "error_types": error_types,
        }

    # ── Phase 3: conditional entry point for Agent 1 caching ──
    def _should_skip_analyst(state: TurnState) -> str:
        """Skip Threat Analyst if report is already cached (retry scenario)."""
        if state.get("threat_report"):
            return "constrained_generator"
        return "threat_analyst"

    graph = StateGraph(TurnState)
    graph.add_node("threat_analyst", _threat_analyst)
    graph.add_node("constrained_generator", _constrained_generator)
    graph.add_node("parse_validate", _parse_validate)

    graph.set_conditional_entry_point(
        _should_skip_analyst,
        {"threat_analyst": "threat_analyst", "constrained_generator": "constrained_generator"},
    )
    graph.add_edge("threat_analyst", "constrained_generator")
    graph.add_edge("constrained_generator", "parse_validate")
    graph.add_edge("parse_validate", END)

    return graph.compile()
