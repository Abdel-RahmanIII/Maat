"""Generator-only subgraph — single LLM call → parse + validate."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.generator import generate_move
from src.config import ModelConfig
from src.context import ConversationContext
from src.graph.base_graph import parse_and_validate
from src.state import TurnState


def build_generator_only_subgraph(
    model_config: ModelConfig | None = None,
    context: ConversationContext | None = None,
) -> CompiledStateGraph:
    """Build a two-node generation subgraph: generate → parse_validate."""

    cfg = model_config
    ctx = context

    def _generate(state: TurnState) -> dict[str, Any]:
        """Call the Generator LLM to produce a raw move string."""

        history = ctx.get_history("generator") if ctx else None

        result = generate_move(
            fen=state["board_fen"],
            move_history=state["move_history"],
            feedback_history=state["feedback_history"],
            input_mode=state.get("input_mode", "fen"),
            model_config=cfg,
            conversation_history=history,
        )
        return {
            "raw_llm_response": result["raw_output"],
            "prompt_token_count": state["prompt_token_count"] + result["prompt_tokens"],
            "tokens_this_turn": (
                state["tokens_this_turn"]
                + result["prompt_tokens"]
                + result["completion_tokens"]
            ),
            "llm_calls_this_turn": state["llm_calls_this_turn"] + 1,
            "strategic_plan": "",
            "threat_report": "",
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

    graph = StateGraph(TurnState)
    graph.add_node("generate", _generate)
    graph.add_node("parse_validate", _parse_validate)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "parse_validate")
    graph.add_edge("parse_validate", END)

    return graph.compile()
