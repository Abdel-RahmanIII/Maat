from __future__ import annotations

from typing import Any, Literal, TypedDict

InputMode = Literal["fen", "history"]
GameStatus = Literal[
    "ongoing",
    "checkmate",
    "stalemate",
    "draw",
    "forfeit",
    "max_moves",
]


class TurnState(TypedDict):
    # Position
    board_fen: str
    move_history: list[str]
    move_number: int

    # Input mode
    input_mode: InputMode

    # Current turn
    proposed_move: str
    is_valid: bool
    retry_count: int
    max_retries: int
    feedback_history: list[str]

    # Messages
    messages: list[Any]

    # Turn metrics
    first_try_valid: bool
    error_types: list[str]
    tool_calls: list[dict[str, Any]]
    total_attempts: int
    llm_calls_this_turn: int
    tokens_this_turn: int
    prompt_token_count: int
    wall_clock_ms: float
    game_phase: str

    # Critic-specific (Condition C)
    critic_verdict: bool | None
    ground_truth_verdict: bool | None

    # Generation strategy metadata
    generation_strategy: str
    strategic_plan: str  # NL plan from strategist (planner_actor)
    routed_phase: str  # Phase chosen by router (router_specialists)

    # Game-level (accumulated)
    game_id: str
    condition: str
    turn_results: list[dict[str, Any]]
    game_status: GameStatus


def create_initial_turn_state(
    *,
    board_fen: str,
    game_id: str,
    condition: str,
    input_mode: InputMode = "fen",
    move_history: list[str] | None = None,
    move_number: int = 1,
    max_retries: int = 0,
) -> TurnState:
    """Create a default turn state payload used by LangGraph conditions."""

    return {
        "board_fen": board_fen,
        "move_history": list(move_history or []),
        "move_number": move_number,
        "input_mode": input_mode,
        "proposed_move": "",
        "is_valid": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "feedback_history": [],
        "messages": [],
        "first_try_valid": False,
        "error_types": [],
        "tool_calls": [],
        "total_attempts": 0,
        "llm_calls_this_turn": 0,
        "tokens_this_turn": 0,
        "prompt_token_count": 0,
        "wall_clock_ms": 0.0,
        "game_phase": "",
        "critic_verdict": None,
        "ground_truth_verdict": None,
        "generation_strategy": "generator_only",
        "strategic_plan": "",
        "routed_phase": "",
        "game_id": game_id,
        "condition": condition,
        "turn_results": [],
        "game_status": "ongoing",
    }
