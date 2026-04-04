from __future__ import annotations

from typing import TypedDict

import chess

from src.validators.symbolic import validate_move as symbolic_validate_move


class ToolValidationResult(TypedDict):
    valid: bool
    reason: str


def _board_from_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError as exc:
        raise ValueError("Invalid FEN provided.") from exc


def _parse_color(color: str) -> chess.Color:
    normalized = color.strip().lower()
    if normalized in {"white", "w"}:
        return chess.WHITE
    if normalized in {"black", "b"}:
        return chess.BLACK
    raise ValueError("Color must be one of: white, black, w, b.")


def validate_move(fen: str, uci_move: str) -> ToolValidationResult:
    """Validate move legality via symbolic validator tool."""

    result = symbolic_validate_move(fen, uci_move)
    return {
        "valid": result["valid"],
        "reason": result["reason"],
    }


def get_board_state(fen: str) -> str:
    """Return a readable board state that can be injected into prompts."""

    board = _board_from_fen(fen)
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    board_ascii = str(board)
    return (
        f"FEN: {board.fen()}\n"
        f"Side to move: {side_to_move}\n"
        f"Fullmove: {board.fullmove_number}, Halfmove clock: {board.halfmove_clock}\n"
        f"Board:\n{board_ascii}"
    )


def get_legal_moves(fen: str) -> list[str]:
    """Return all legal moves in UCI format for the position."""

    board = _board_from_fen(fen)
    return [move.uci() for move in board.legal_moves]


def get_piece_moves(fen: str, square: str) -> list[str]:
    """Return legal moves for the piece on a given square."""

    board = _board_from_fen(fen)
    try:
        source_square = chess.parse_square(square.strip().lower())
    except ValueError as exc:
        raise ValueError("Square must be in algebraic form such as e2.") from exc

    piece = board.piece_at(source_square)
    if piece is None:
        return []

    return [
        move.uci()
        for move in board.legal_moves
        if move.from_square == source_square
    ]


def get_attacked_squares(fen: str, color: str) -> list[str]:
    """Return all squares attacked by the requested color."""

    board = _board_from_fen(fen)
    attacker = _parse_color(color)

    attacked = [
        chess.square_name(square)
        for square in chess.SQUARES
        if board.is_attacked_by(attacker, square)
    ]

    return attacked
