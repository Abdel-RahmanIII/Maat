from __future__ import annotations

import chess

from src.error_taxonomy import ErrorType
from src.validators.symbolic import validate_move


def test_validate_move_accepts_legal_move() -> None:
    result = validate_move(chess.STARTING_FEN, "e2e4")
    assert result["valid"] is True
    assert result["error_type"] is None


def test_validate_move_invalid_piece_for_empty_source() -> None:
    result = validate_move(chess.STARTING_FEN, "e3e4")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.INVALID_PIECE.value


def test_validate_move_illegal_destination() -> None:
    result = validate_move(chess.STARTING_FEN, "a1a3")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.ILLEGAL_DESTINATION.value


def test_validate_move_leaves_king_in_check() -> None:
    fen = "4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1"
    result = validate_move(fen, "e2f2")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.LEAVES_IN_CHECK.value


def test_validate_move_castling_violation() -> None:
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    result = validate_move(fen, "e1g1")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.CASTLING_VIOLATION.value


def test_validate_move_en_passant_violation() -> None:
    fen = "k7/8/8/3pP3/8/8/8/4K3 w - - 0 1"
    result = validate_move(fen, "e5d6")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.EN_PASSANT_VIOLATION.value


def test_validate_move_promotion_error() -> None:
    fen = "k7/4P3/8/8/8/8/8/4K3 w - - 0 1"
    result = validate_move(fen, "e7e8")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.PROMOTION_ERROR.value


def test_validate_move_non_pawn_promotion_error() -> None:
    fen = "4k3/8/8/8/8/8/4N3/4K3 w - - 0 1"
    result = validate_move(fen, "e2e4q")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.PROMOTION_ERROR.value


def test_validate_move_parse_error() -> None:
    result = validate_move(chess.STARTING_FEN, "not-a-move")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.PARSE_ERROR.value


def test_validate_move_no_output() -> None:
    result = validate_move(chess.STARTING_FEN, "")
    assert result["valid"] is False
    assert result["error_type"] == ErrorType.NO_OUTPUT.value
