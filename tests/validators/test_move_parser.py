from __future__ import annotations

import chess

from src.error_taxonomy import ErrorType
from src.validators.move_parser import parse_san_move, parse_uci_move


def test_parse_uci_move_accepts_strict_uci() -> None:
    result = parse_uci_move("e2e4")
    assert result["is_valid"] is True
    assert result["move_uci"] == "e2e4"
    assert result["used_fallback"] is False


def test_parse_uci_move_uses_regex_fallback() -> None:
    result = parse_uci_move("I choose move e7e8q right now")
    assert result["is_valid"] is True
    assert result["move_uci"] == "e7e8q"
    assert result["used_fallback"] is True


def test_parse_uci_move_rejects_empty_output() -> None:
    result = parse_uci_move("   ")
    assert result["is_valid"] is False
    assert result["error_type"] == ErrorType.NO_OUTPUT.value


def test_parse_uci_move_rejects_non_uci_output() -> None:
    result = parse_uci_move("castle kingside please")
    assert result["is_valid"] is False
    assert result["error_type"] == ErrorType.PARSE_ERROR.value


def test_parse_uci_move_normalizes_case() -> None:
    result = parse_uci_move("E2E4")
    assert result["is_valid"] is True
    assert result["move_uci"] == "e2e4"


def test_parse_uci_move_rejects_null_move() -> None:
    result = parse_uci_move("0000")
    assert result["is_valid"] is False
    assert result["error_type"] == ErrorType.PARSE_ERROR.value


def test_parse_san_move_accepts_strict_san() -> None:
    board = chess.Board(chess.STARTING_FEN)
    result = parse_san_move("e4", board)
    assert result["is_valid"] is True
    assert result["move_uci"] == "e2e4"
    assert result["used_fallback"] is False


def test_parse_san_move_accepts_sentinel_san() -> None:
    board = chess.Board(chess.STARTING_FEN)
    result = parse_san_move("Thoughts... MOVE: Nf3", board)
    assert result["is_valid"] is True
    assert result["move_uci"] == "g1f3"
    assert result["used_fallback"] is False


def test_parse_san_move_uses_regex_fallback() -> None:
    board = chess.Board(chess.STARTING_FEN)
    result = parse_san_move("I think e4 is best", board)
    assert result["is_valid"] is True
    assert result["move_uci"] == "e2e4"
    assert result["used_fallback"] is True


def test_parse_san_move_normalizes_case() -> None:
    board = chess.Board("8/8/8/8/R7/8/8/4K2k w - - 0 1")
    result = parse_san_move("ra6", board)
    assert result["is_valid"] is True
    assert result["move_uci"] == "a4a6"


def test_parse_san_move_rejects_invalid_san() -> None:
    board = chess.Board(chess.STARTING_FEN)
    result = parse_san_move("Na9", board)
    assert result["is_valid"] is False
    assert result["error_type"] == ErrorType.PARSE_ERROR.value
