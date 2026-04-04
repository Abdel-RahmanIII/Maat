from __future__ import annotations

from typing import Any, TypedDict, cast

from src.state import InputMode, TurnState
from src.tools.chess_tools import get_board_state
from src.validators.move_parser import ParseResult, parse_uci_move
from src.validators.symbolic import ValidationResult, validate_move


class MoveEvaluation(TypedDict):
    raw_output: str | None
    parse_result: ParseResult
    validation_result: ValidationResult
    proposed_move: str
    is_valid: bool


def copy_turn_state(state: TurnState) -> TurnState:
    """Return a shallow copy of state with list fields detached."""

    cloned: dict[str, Any] = dict(state)

    for field in (
        "move_history",
        "feedback_history",
        "messages",
        "error_types",
        "tool_calls",
        "turn_results",
    ):
        cloned[field] = list(state[field])

    return cast(TurnState, cloned)


def format_move_history(move_history: list[str]) -> str:
    if not move_history:
        return "(none)"
    return " ".join(move_history)


def build_board_representation(
    *,
    board_fen: str,
    input_mode: InputMode,
) -> str:
    """Build board text inserted into prompts based on experiment input mode."""

    if input_mode == "fen":
        return get_board_state(board_fen)

    return "Board state is hidden for this run. Use move history only."


def build_base_turn_prompt(
    *,
    color: str,
    board_fen: str,
    move_history: list[str],
    input_mode: InputMode,
) -> str:
    """Build the shared per-turn prompt context block."""

    board_representation = build_board_representation(
        board_fen=board_fen,
        input_mode=input_mode,
    )

    return (
        f"You are playing chess as {color}.\n\n"
        f"{board_representation}\n\n"
        f"Move history (UCI): {format_move_history(move_history)}"
    )


def evaluate_model_output(board_fen: str, raw_output: str | None) -> MoveEvaluation:
    """Parse model output and, when possible, run symbolic legality validation."""

    parse_result = parse_uci_move(raw_output)
    move_uci = parse_result["move_uci"]

    if parse_result["is_valid"]:
        validation_result = validate_move(board_fen, move_uci)
    else:
        validation_result = {
            "valid": False,
            "error_type": parse_result["error_type"],
            "reason": parse_result["reason"] or "Could not parse model output.",
        }

    proposed_move = move_uci or ""

    return {
        "raw_output": raw_output,
        "parse_result": parse_result,
        "validation_result": validation_result,
        "proposed_move": proposed_move,
        "is_valid": validation_result["valid"],
    }


def reset_turn_fields(state: TurnState) -> None:
    """Reset per-turn mutable fields before executing a new turn."""

    state["proposed_move"] = ""
    state["is_valid"] = False
    state["retry_count"] = 0
    state["feedback_history"] = []
    state["messages"] = []
    state["first_try_valid"] = False
    state["error_types"] = []
    state["tool_calls"] = []
    state["total_attempts"] = 0
    state["llm_calls_this_turn"] = 0
    state["tokens_this_turn"] = 0
    state["prompt_token_count"] = 0
    state["critic_verdict"] = None
    state["ground_truth_verdict"] = None


def set_proposed_move(state: TurnState, move_uci: str | None) -> None:
    state["proposed_move"] = (move_uci or "").strip().lower()


def append_feedback(state: TurnState, feedback: str | None) -> None:
    if feedback is None:
        return

    text = feedback.strip()
    if text:
        state["feedback_history"].append(text)


def add_error_type(state: TurnState, error_type: str | None) -> None:
    if error_type is None:
        return

    state["error_types"].append(error_type)


def register_attempt(
    state: TurnState,
    *,
    is_valid: bool,
    error_type: str | None,
) -> None:
    """Record one model attempt and update first-try validity + error tracking."""

    state["total_attempts"] += 1
    state["is_valid"] = is_valid

    if state["total_attempts"] == 1:
        state["first_try_valid"] = is_valid

    if not is_valid:
        add_error_type(state, error_type)


def increment_retry_count(state: TurnState) -> None:
    state["retry_count"] += 1


def can_retry(state: TurnState) -> bool:
    return state["retry_count"] < state["max_retries"]


def record_llm_usage(
    state: TurnState,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    calls: int = 1,
) -> None:
    """Accumulate per-turn usage stats for later metric aggregation."""

    state["llm_calls_this_turn"] += calls
    state["prompt_token_count"] += prompt_tokens
    state["tokens_this_turn"] += prompt_tokens + completion_tokens


def record_tool_call(
    state: TurnState,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    state["tool_calls"].append(
        {
            "tool": tool_name,
            "arguments": dict(arguments),
            "result": result,
        }
    )


def append_turn_result(state: TurnState, result: dict[str, Any]) -> None:
    state["turn_results"].append(dict(result))


def mark_accepted_move(state: TurnState, move_uci: str) -> None:
    state["proposed_move"] = move_uci.strip().lower()
    state["is_valid"] = True


def mark_forfeit(state: TurnState, reason: str | None = None) -> None:
    state["is_valid"] = False
    state["game_status"] = "forfeit"
    append_feedback(state, reason)
