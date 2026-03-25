from __future__ import annotations

import chess

from core.schemas import StateSnapshot


class StateManager:
    """Owns a single chess board and controls state transitions."""

    def __init__(self, start_fen: str | None = None) -> None:
        self._board = chess.Board(start_fen) if start_fen else chess.Board()

    @property
    def board(self) -> chess.Board:
        return self._board

    def initialize_from_fen(self, fen: str) -> None:
        self._board = chess.Board(fen)

    def current_fen(self) -> str:
        return self._board.fen()

    def snapshot(self) -> StateSnapshot:
        outcome = self._board.outcome(claim_draw=True)
        return StateSnapshot(
            fen=self._board.fen(),
            side_to_move="white" if self._board.turn == chess.WHITE else "black",
            halfmove_clock=self._board.halfmove_clock,
            fullmove_number=self._board.fullmove_number,
            is_checkmate=self._board.is_checkmate(),
            is_stalemate=self._board.is_stalemate(),
            is_insufficient_material=self._board.is_insufficient_material(),
            can_claim_threefold_repetition=self._board.can_claim_threefold_repetition(),
            can_claim_fifty_moves=self._board.can_claim_fifty_moves(),
            is_terminal=self._board.is_game_over(claim_draw=True),
            outcome=str(outcome.result()) if outcome else None,
        )

    def apply_validated_move_uci(self, uci_move: str) -> None:
        move = chess.Move.from_uci(uci_move)
        if move not in self._board.legal_moves:
            raise ValueError(f"Move is not legal in current state: {uci_move}")
        self._board.push(move)
