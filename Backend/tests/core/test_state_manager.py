import pytest
import chess

from core.exceptions import IllegalMoveError, InvalidFENError
from core.state_manager import StateManager
from schemas.game import GameStatus


def test_initial_snapshot_has_white_to_move() -> None:
    manager = StateManager()
    snapshot = manager.snapshot()

    assert snapshot.side_to_move == "white"
    assert snapshot.is_terminal is False
    assert snapshot.game_status == GameStatus.ONGOING


def test_snapshot_includes_legal_move_count() -> None:
    """legal_move_count must be present — generator prompts will use it."""
    manager = StateManager()
    snapshot = manager.snapshot()

    assert snapshot.legal_move_count == 20  # standard starting position


def test_initialize_from_fen() -> None:
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    manager = StateManager(fen)

    assert manager.current_fen().startswith("8/8/8/8/8/8/8/K6k")


def test_invalid_fen_raises_typed_exception() -> None:
    with pytest.raises(InvalidFENError):
        StateManager("this is not a fen")


def test_apply_validated_move_uci_updates_board() -> None:
    manager = StateManager()
    manager.apply_validated_move_uci("e2e4")

    piece = manager.board.piece_at(chess.E4)
    assert piece is not None
    assert piece.symbol() == "P"


def test_apply_illegal_move_raises_illegal_move_error() -> None:
    """apply_validated_move_uci must raise IllegalMoveError, not ValueError."""
    manager = StateManager()

    with pytest.raises(IllegalMoveError) as exc_info:
        manager.apply_validated_move_uci("e2e5")

    assert exc_info.value.move == "e2e5"
    assert exc_info.value.fen == manager.current_fen()


def test_snapshot_detects_checkmate() -> None:
    manager = StateManager()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        manager.apply_validated_move_uci(move)

    snapshot = manager.snapshot()

    assert snapshot.is_terminal is True
    assert snapshot.game_status == GameStatus.CHECKMATE
    assert snapshot.outcome == "0-1"