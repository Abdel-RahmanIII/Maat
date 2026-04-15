"""Shared graph utilities used across all LangGraph condition graphs."""

from __future__ import annotations

from typing import Any

from src.agents.generator import generate_move
from src.agents.strategist import create_plan
from src.agents.tactician import execute_plan
from src.agents.router import classify_phase
from src.agents.specialists import generate_specialist_move
from src.config import GenerationStrategy, ModelConfig
from src.state import TurnState
from src.validators.move_parser import parse_uci_move
from src.validators.symbolic import validate_move


# ── Generation dispatch ──────────────────────────────────────────────────


def run_generation(
    state: TurnState,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Execute the generation stage using the configured strategy.

    Returns a dict of state updates including ``raw_output`` and token
    counts.  The caller is responsible for merging these into ``state``.
    """

    strategy = state.get("generation_strategy", "generator_only")
    fen = state["board_fen"]
    move_history = state["move_history"]
    feedback_history = state["feedback_history"]
    input_mode = state.get("input_mode", "fen")

    if strategy == GenerationStrategy.PLANNER_ACTOR.value:
        return _run_planner_actor(
            fen=fen,
            move_history=move_history,
            feedback_history=feedback_history,
            input_mode=input_mode,
            model_config=model_config,
        )

    if strategy == GenerationStrategy.ROUTER_SPECIALISTS.value:
        return _run_router_specialists(
            fen=fen,
            move_history=move_history,
            feedback_history=feedback_history,
            input_mode=input_mode,
            model_config=model_config,
        )

    # Default: generator_only
    result = generate_move(
        fen=fen,
        move_history=move_history,
        feedback_history=feedback_history,
        input_mode=input_mode,
        model_config=model_config,
    )
    return {
        "raw_output": result["raw_output"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "strategic_plan": "",
        "routed_phase": "",
        "extra_llm_calls": 0,
    }


def _run_planner_actor(
    *,
    fen: str,
    move_history: list[str],
    feedback_history: list[str],
    input_mode: str,
    model_config: ModelConfig | None,
) -> dict[str, Any]:
    """Planner (strategist) → Actor (tactician) pipeline."""

    plan_result = create_plan(
        fen=fen,
        move_history=move_history,
        input_mode=input_mode,
        model_config=model_config,
    )

    tactic_result = execute_plan(
        fen=fen,
        move_history=move_history,
        strategic_plan=plan_result["plan"],
        feedback_history=feedback_history,
        input_mode=input_mode,
        model_config=model_config,
    )

    return {
        "raw_output": tactic_result["raw_output"],
        "prompt_tokens": plan_result["prompt_tokens"] + tactic_result["prompt_tokens"],
        "completion_tokens": plan_result["completion_tokens"] + tactic_result["completion_tokens"],
        "strategic_plan": plan_result["plan"],
        "routed_phase": "",
        "extra_llm_calls": 1,  # strategist is extra
    }


def _run_router_specialists(
    *,
    fen: str,
    move_history: list[str],
    feedback_history: list[str],
    input_mode: str,
    model_config: ModelConfig | None,
) -> dict[str, Any]:
    """Router → Phase-specific specialist pipeline."""

    router_result = classify_phase(
        fen=fen,
        move_history=move_history,
        input_mode=input_mode,
        model_config=model_config,
    )

    specialist_result = generate_specialist_move(
        phase=router_result["phase"],
        fen=fen,
        move_history=move_history,
        feedback_history=feedback_history,
        input_mode=input_mode,
        model_config=model_config,
    )

    return {
        "raw_output": specialist_result["raw_output"],
        "prompt_tokens": router_result["prompt_tokens"] + specialist_result["prompt_tokens"],
        "completion_tokens": router_result["completion_tokens"] + specialist_result["completion_tokens"],
        "strategic_plan": "",
        "routed_phase": router_result["phase"],
        "extra_llm_calls": 1,  # router is extra
    }


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
        "game_phase": state.get("game_phase", ""),
        "wall_clock_ms": state.get("wall_clock_ms", 0.0),
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
        "routed_phase": state.get("routed_phase", ""),
        "feedback_history": list(state["feedback_history"]),
    }
