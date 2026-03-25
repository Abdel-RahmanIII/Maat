from __future__ import annotations

import chess

from core.state_manager import StateManager
from schemas.move import ValidationResult


class RuleValidator:
    """Two-stage move validation: syntax first, legality second."""

    def __init__(self, state_manager: StateManager) -> None:
        self._state_manager = state_manager

    def validate_move(self, raw_move: str) -> ValidationResult:
        board = self._state_manager.board

        # Guard: reject any move if the game is already over.
        if board.is_game_over(claim_draw=True):
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                validation_stage="syntax",
                error_code="game_already_terminal",
                message="Game is terminal; no further moves are allowed.",
            )

        # ── Stage 1: Syntax ─────────────────────────────────────────────────
        # Is the input a non-empty string that can be parsed as UCI or SAN?
        # Failures here mean the agent produced something that isn't a move
        # string at all — empty output, extra tokens, plain English, etc.

        text = raw_move.strip()
        if not text:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                validation_stage="syntax",
                error_code="syntax_error",
                message="Empty move input.",
            )

        move: chess.Move | None = None
        try:
            if len(text) in (4, 5):
                # Attempt UCI parse. From-uci never checks board legality,
                # so a structurally valid UCI token always succeeds here.
                candidate = chess.Move.from_uci(text)
                move = candidate
            else:
                # SAN requires the board for disambiguation, so a parse error
                # here is still a syntax failure (wrong notation format).
                move = board.parse_san(text)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                validation_stage="syntax",
                error_code="unsupported_format",
                message=(
                    f"Move '{text}' could not be parsed as UCI or SAN. "
                    "Expected formats: UCI (e.g. 'e2e4', 'e7e8q') or SAN (e.g. 'e4', 'Nf3')."
                ),
            )

        # ── Stage 2: Legality ────────────────────────────────────────────────
        # The string is a syntactically valid move token. Is it legal on this
        # board? Failures here mean the agent understood the format but
        # proposed a move that violates chess rules in the current position.

        if move not in board.legal_moves:
            return ValidationResult(
                is_valid=False,
                normalized_move_uci=None,
                validation_stage="legality",
                error_code="illegal_move",
                message=(
                    f"Move '{text}' is syntactically valid but illegal in this position."
                ),
            )

        return ValidationResult(
            is_valid=True,
            normalized_move_uci=move.uci(),
            validation_stage=None,
            error_code=None,
            message="Move is valid.",
        )