from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValidationErrorCode(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    ILLEGAL_MOVE = "illegal_move"
    WRONG_TURN = "wrong_turn"
    GAME_ALREADY_TERMINAL = "game_already_terminal"
    UNSUPPORTED_FORMAT = "unsupported_format"


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    normalized_move_uci: str | None
    error_code: ValidationErrorCode | None
    message: str


@dataclass(frozen=True)
class StateSnapshot:
    fen: str
    side_to_move: str
    halfmove_clock: int
    fullmove_number: int
    is_checkmate: bool
    is_stalemate: bool
    is_insufficient_material: bool
    can_claim_threefold_repetition: bool
    can_claim_fifty_moves: bool
    is_terminal: bool
    outcome: str | None


@dataclass(frozen=True)
class AttemptLogRecord:
    schema_version: str
    turn_id: int
    attempt_id: int
    input_move: str
    normalized_move: str | None
    validator_result: str
    validator_message: str
    state_before_fen: str
    state_after_fen: str | None
    terminal_flag: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "attempt_id": self.attempt_id,
            "input_move": self.input_move,
            "normalized_move": self.normalized_move,
            "validator_result": self.validator_result,
            "validator_message": self.validator_message,
            "state_before_fen": self.state_before_fen,
            "state_after_fen": self.state_after_fen,
            "terminal_flag": self.terminal_flag,
        }
