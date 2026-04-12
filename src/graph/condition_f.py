"""Condition F — ReAct + Tool Calling.

The agent autonomously reasons, optionally invokes chess tools,
and eventually calls ``submit_move``.  The submitted move is checked
against python-chess ground truth.  No retries after submission.
"""

from __future__ import annotations

from typing import Any

from src.agents.react_agent import run_react_loop
from src.config import ModelConfig
from src.graph.base_graph import parse_and_validate, snapshot_turn_result
from src.state import InputMode, TurnState, create_initial_turn_state


def run_condition_f(
    *,
    fen: str,
    move_history: list[str] | None = None,
    move_number: int = 1,
    game_id: str = "",
    input_mode: InputMode = "fen",
    max_steps: int = 6,
    model_config: ModelConfig | None = None,
) -> TurnState:
    """Execute one turn under Condition F.

    The ReAct loop runs inside :func:`run_react_loop` which manages
    the think → tool → observe cycle.  This function handles the
    ground-truth check and state finalization.
    """

    state = create_initial_turn_state(
        board_fen=fen,
        game_id=game_id,
        condition="F",
        input_mode=input_mode,
        move_history=list(move_history or []),
        move_number=move_number,
        max_retries=0,
    )
    state["generation_strategy"] = "generator_only"  # ReAct is its own paradigm

    # ── Run the ReAct loop ──
    react_result = run_react_loop(
        fen=fen,
        move_history=list(move_history or []),
        input_mode=input_mode,
        max_steps=max_steps,
        model_config=model_config,
    )

    state["tool_calls"] = react_result["tool_calls_log"]
    state["llm_calls_this_turn"] = react_result["steps_taken"]
    state["tokens_this_turn"] = (
        react_result["total_prompt_tokens"] + react_result["total_completion_tokens"]
    )
    state["prompt_token_count"] = react_result["total_prompt_tokens"]
    state["total_attempts"] = 1

    if react_result["forfeited"]:
        # Agent failed to submit a move within max_steps
        state["proposed_move"] = ""
        state["is_valid"] = False
        state["first_try_valid"] = False
        state["error_types"].append("NO_OUTPUT")
        state["game_status"] = "forfeit"
    else:
        # Agent submitted a move — ground-truth check
        submitted = react_result["submitted_move"]
        pv = parse_and_validate(submitted, fen)

        state["proposed_move"] = pv["proposed_move"]
        state["is_valid"] = pv["is_valid"]
        state["first_try_valid"] = pv["is_valid"]

        if pv["is_valid"]:
            state["game_status"] = "ongoing"
        else:
            if pv["error_type"]:
                state["error_types"].append(pv["error_type"])
            state["game_status"] = "forfeit"

    state["turn_results"].append(snapshot_turn_result(state))
    return state
