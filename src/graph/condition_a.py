"""Condition A — Single LLM baseline (no LangGraph).

Direct API call → parse → validate → accept or forfeit.
No retries.  Establishes the raw LLM capability floor.
"""

from __future__ import annotations

from typing import Any

from src.config import ModelConfig
from src.graph.base_graph import parse_and_validate, run_generation, snapshot_turn_result
from src.state import InputMode, TurnState, create_initial_turn_state


def run_condition_a(
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    generation_strategy: str = "generator_only",
    model_config: ModelConfig | None = None,
) -> TurnState:
    """Execute one turn under Condition A.

    This is a plain function — no LangGraph graph.  It mirrors the
    structure of conditions B–E for comparison but deliberately avoids
    any framework overhead.
    """

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="A",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=0,
    )
    state["generation_strategy"] = generation_strategy

    # ── Generate ──
    gen_result = run_generation(state, model_config)

    state["llm_calls_this_turn"] = 1 + gen_result.get("extra_llm_calls", 0)
    state["tokens_this_turn"] = gen_result["prompt_tokens"] + gen_result["completion_tokens"]
    state["prompt_token_count"] = gen_result["prompt_tokens"]
    state["total_attempts"] = 1

    # ── Parse + validate ──
    pv = parse_and_validate(gen_result["raw_output"], fen)

    state["proposed_move"] = pv["proposed_move"]
    state["is_valid"] = pv["is_valid"]
    state["first_try_valid"] = pv["is_valid"]

    if not pv["is_valid"]:
        state["error_types"].append(pv["error_type"] or "UNKNOWN")
        state["game_status"] = "forfeit"
    else:
        state["game_status"] = "ongoing"

    state["turn_results"].append(snapshot_turn_result(state))
    return state
