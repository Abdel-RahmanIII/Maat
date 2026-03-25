from __future__ import annotations

import chess

from core.schemas import ValidationErrorCode, ValidationResult
from core.state_manager import StateManager


class RuleValidator:
    """Performs syntax normalization and legality checks."""

    def __init__(self, state_manager: StateManager) -> None:
        self._state_manager = state_manager

    def validate_move(self, raw_move: str) -> ValidationResult:
        board = self._state_manager.board

        if board.is_game_over(claim_draw=True):
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                error_code=ValidationErrorCode.GAME_ALREADY_TERMINAL,
                message="Game is terminal; no further moves are allowed.",
            )

        text = raw_move.strip()
        if not text:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                error_code=ValidationErrorCode.SYNTAX_ERROR,
                message="Empty move input.",
            )

        # UCI is canonical. SAN is accepted and normalized to UCI.
        move: chess.Move | None = None
        try:
            if len(text) in (4, 5):
                candidate = chess.Move.from_uci(text)
                move = candidate
            else:
                move = board.parse_san(text)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                error_code=ValidationErrorCode.UNSUPPORTED_FORMAT,
                message="Move must be valid UCI or SAN notation.",
            )

        if move not in board.legal_moves:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                error_code=ValidationErrorCode.ILLEGAL_MOVE,
                message=f"Move is illegal in this position: {text}",
            )

        return ValidationResult(
            is_valid=True,
            normalized_move_uci=move.uci(),
            error_code=None,
            message="Move is valid.",
        )
