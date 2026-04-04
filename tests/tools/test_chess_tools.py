from __future__ import annotations

import chess
import pytest

from src.tools.chess_tools import (
    get_attacked_squares,
    get_board_state,
    get_legal_moves,
    get_piece_moves,
    validate_move,
)


def test_tool_validate_move_returns_valid_for_legal_move() -> None:
    result = validate_move(chess.STARTING_FEN, "e2e4")
    assert result["valid"] is True


def test_tool_validate_move_returns_reason_for_illegal_move() -> None:
    result = validate_move(chess.STARTING_FEN, "e2e5")
    assert result["valid"] is False
    assert "destination" in result["reason"].lower() or "legal" in result["reason"].lower()


def test_get_board_state_contains_fen_and_board() -> None:
    board_state = get_board_state(chess.STARTING_FEN)
    assert "FEN:" in board_state
    assert "Board:" in board_state


def test_get_legal_moves_starting_position() -> None:
    legal_moves = get_legal_moves(chess.STARTING_FEN)
    assert len(legal_moves) == 20
    assert "e2e4" in legal_moves


def test_get_piece_moves_for_pawn() -> None:
    moves = get_piece_moves(chess.STARTING_FEN, "e2")
    assert "e2e3" in moves
    assert "e2e4" in moves


def test_get_piece_moves_empty_square_returns_empty_list() -> None:
    moves = get_piece_moves(chess.STARTING_FEN, "e5")
    assert moves == []


def test_get_attacked_squares_for_white() -> None:
    attacked = get_attacked_squares(chess.STARTING_FEN, "white")
    assert "e3" in attacked


def test_get_attacked_squares_rejects_invalid_color() -> None:
    with pytest.raises(ValueError):
        get_attacked_squares(chess.STARTING_FEN, "green")
