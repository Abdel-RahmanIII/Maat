from __future__ import annotations

from src.state import TurnState, create_initial_turn_state


EXPECTED_TURN_STATE_FIELDS = {
    "board_fen",
    "move_history",
    "move_number",
    "input_mode",
    "proposed_move",
    "is_valid",
    "retry_count",
    "max_retries",
    "feedback_history",
    "messages",
    "first_try_valid",
    "error_types",
    "tool_calls",
    "total_attempts",
    "llm_calls_this_turn",
    "tokens_this_turn",
    "prompt_token_count",
    "critic_verdict",
    "ground_truth_verdict",
    "generation_strategy",
    "strategic_plan",
    "routed_phase",
    "game_id",
    "condition",
    "turn_results",
    "game_status",
}


def test_turn_state_annotations_cover_core_fields() -> None:
    annotations = TurnState.__annotations__
    missing = EXPECTED_TURN_STATE_FIELDS.difference(annotations)
    assert not missing, f"Missing TurnState fields: {sorted(missing)}"


def test_create_initial_turn_state_defaults() -> None:
    payload = create_initial_turn_state(
        board_fen="start_fen",
        game_id="game-1",
        condition="B",
        move_history=["e2e4"],
        max_retries=3,
    )

    assert payload["board_fen"] == "start_fen"
    assert payload["game_id"] == "game-1"
    assert payload["condition"] == "B"
    assert payload["input_mode"] == "fen"
    assert payload["move_history"] == ["e2e4"]
    assert payload["retry_count"] == 0
    assert payload["max_retries"] == 3
    assert payload["game_status"] == "ongoing"


def test_create_initial_turn_state_copies_move_history() -> None:
    move_history = ["e2e4"]
    payload = create_initial_turn_state(
        board_fen="start_fen",
        game_id="game-2",
        condition="C",
        move_history=move_history,
    )

    move_history.append("e7e5")
    assert payload["move_history"] == ["e2e4"]
