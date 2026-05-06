"""Shared graph utilities used across all LangGraph condition graphs."""

from __future__ import annotations

from typing import Any

from src.state import TurnState
from src.validators.move_parser import parse_uci_move
from src.validators.symbolic import validate_move


# ── Parse + validate helper ──────────────────────────────────────────────


def parse_and_validate(raw_output: str, fen: str) -> dict[str, Any]:
    """Parse a raw LLM output and validate the resulting move.

    Returns a dict with ``proposed_move``, ``is_valid``, ``error_type``,
    ``error_reason``, and ``used_fallback``.
    """

    parse_result = parse_uci_move(raw_output)

    if not parse_result["is_valid"]:
        return {
            "proposed_move": raw_output.strip()[:20],
            "is_valid": False,
            "error_type": parse_result["error_type"],
            "error_reason": parse_result["reason"] or "Could not parse move.",
            "used_fallback": parse_result["used_fallback"],
        }

    uci_move = parse_result["move_uci"]
    val_result = validate_move(fen, uci_move)

    return {
        "proposed_move": uci_move,
        "is_valid": val_result["valid"],
        "error_type": val_result["error_type"],
        "error_reason": val_result["reason"],
        "used_fallback": parse_result["used_fallback"],
    }


# ── Turn result snapshot ──────────────────────────────────────────────────


def snapshot_turn_result(state: TurnState) -> dict[str, Any]:
    """Create a turn-result record from the current state."""

    return {
        "move_number": state["move_number"],
        "proposed_move": state["proposed_move"],
        "is_valid": state["is_valid"],
        "first_try_valid": state["first_try_valid"],
        "total_attempts": state["total_attempts"],
        "error_types": list(state["error_types"]),
        "retry_count": state["retry_count"],
        "llm_calls_this_turn": state["llm_calls_this_turn"],
        "tokens_this_turn": state["tokens_this_turn"],
        "prompt_token_count": state["prompt_token_count"],
        "tool_calls": list(state["tool_calls"]),
        "critic_verdict": state.get("critic_verdict"),
        "ground_truth_verdict": state.get("ground_truth_verdict"),
        "generation_strategy": state.get("generation_strategy", "generator_only"),
        "strategic_plan": state.get("strategic_plan", ""),
        "threat_report": state.get("threat_report", ""),
        "feedback_history": list(state["feedback_history"]),
        "wall_clock_ms": state.get("wall_clock_ms", 0.0),
        "game_phase": state.get("game_phase", ""),
        "board_fen": state["board_fen"],
        "raw_llm_response": state.get("raw_llm_response", ""),
    }

