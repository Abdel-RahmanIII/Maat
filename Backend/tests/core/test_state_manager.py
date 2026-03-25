import chess

from core.state_manager import StateManager


def test_state_manager_initial_snapshot_has_white_to_move() -> None:
    manager = StateManager()

    snapshot = manager.snapshot()

    assert snapshot.side_to_move == "white"
    assert snapshot.is_terminal is False


def test_state_manager_initialize_from_fen() -> None:
    fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    manager = StateManager(fen)

    assert manager.current_fen().startswith("8/8/8/8/8/8/8/K6k")


def test_apply_validated_move_updates_board() -> None:
    manager = StateManager()

    manager.apply_validated_move_uci("e2e4")

    piece = manager.board.piece_at(chess.E4)
    assert piece is not None
    assert piece.symbol() == "P"
