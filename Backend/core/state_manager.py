from __future__ import annotations

import chess

from core.exceptions import IllegalMoveError, InvalidFENError
from schemas.game import GameStatus, StateSnapshot


class StateManager:
    """Owns a single chess.Board and controls all state transitions."""

    def __init__(self, start_fen: str | None = None) -> None:
        if start_fen:
            try:
                self._board = chess.Board(start_fen)
            except ValueError as exc:
                raise InvalidFENError(start_fen) from exc
        else:
            self._board = chess.Board()

    @property
    def board(self) -> chess.Board:
        """Exposed for RuleValidator only. Graph nodes must use snapshot()."""
        return self._board

    def current_fen(self) -> str:
        return self._board.fen()

    def snapshot(self) -> StateSnapshot:
        """Build an immutable snapshot of all fields graph nodes may need."""
        outcome = self._board.outcome(claim_draw=True)

        if self._board.is_checkmate():
            status = GameStatus.CHECKMATE
        elif self._board.is_stalemate():
            status = GameStatus.STALEMATE
        elif self._board.is_insufficient_material():
            status = GameStatus.DRAW_INSUFFICIENT
        elif self._board.can_claim_fifty_moves():
            status = GameStatus.DRAW_FIFTY_MOVES
        elif self._board.can_claim_threefold_repetition():
            status = GameStatus.DRAW_THREEFOLD
        else:
            status = GameStatus.ONGOING

        return StateSnapshot(
            fen=self._board.fen(),
            side_to_move="white" if self._board.turn == chess.WHITE else "black",
            halfmove_clock=self._board.halfmove_clock,
            fullmove_number=self._board.fullmove_number,
            is_terminal=self._board.is_game_over(claim_draw=True),
            game_status=status,
            outcome=str(outcome.result()) if outcome else None,
            legal_move_count=self._board.legal_moves.count(),
        )

    def apply_validated_move_uci(self, uci_move: str) -> None:
        """Apply a UCI move that has already passed validation."""
        move = chess.Move.from_uci(uci_move)
        if move not in self._board.legal_moves:
            raise IllegalMoveError(move=uci_move, fen=self._board.fen())
        self._board.push(move)