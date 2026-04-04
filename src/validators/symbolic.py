from __future__ import annotations

from typing import TypedDict

import chess

from src.error_taxonomy import ErrorType


class ValidationResult(TypedDict):
    valid: bool
    error_type: str | None
    reason: str


def _is_castling_attempt(piece: chess.Piece, move: chess.Move) -> bool:
    return (
        piece.piece_type == chess.KING
        and chess.square_file(move.from_square) == 4
        and chess.square_file(move.to_square) in (2, 6)
        and chess.square_rank(move.from_square) == chess.square_rank(move.to_square)
    )


def _promotion_error_reason(board: chess.Board, move: chess.Move, piece: chess.Piece) -> str | None:
    to_rank = chess.square_rank(move.to_square)
    reaches_back_rank = to_rank in (0, 7)

    if piece.piece_type == chess.PAWN:
        if reaches_back_rank and move.promotion is None:
            return "Pawn must include a promotion piece when moving to the back rank."
        if move.promotion is not None and not reaches_back_rank:
            return "Promotion is only allowed when a pawn reaches the back rank."
        return None

    if move.promotion is not None:
        return "Only pawns can promote."

    return None


def _is_invalid_en_passant_attempt(board: chess.Board, move: chess.Move, piece: chess.Piece) -> bool:
    if piece.piece_type != chess.PAWN:
        return False

    from_file = chess.square_file(move.from_square)
    to_file = chess.square_file(move.to_square)
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)

    rank_step = 1 if piece.color == chess.WHITE else -1

    if abs(to_file - from_file) != 1:
        return False
    if to_rank - from_rank != rank_step:
        return False
    if board.piece_at(move.to_square) is not None:
        return False

    if board.ep_square is None:
        return True

    return move.to_square != board.ep_square


def validate_move(fen: str, move_uci: str | None) -> ValidationResult:
    """Validate UCI move legality and classify invalid attempts."""

    if move_uci is None or not move_uci.strip():
        return {
            "valid": False,
            "error_type": ErrorType.NO_OUTPUT.value,
            "reason": "No move was provided.",
        }

    try:
        board = chess.Board(fen)
    except ValueError:
        return {
            "valid": False,
            "error_type": ErrorType.PARSE_ERROR.value,
            "reason": "Invalid FEN was provided to validator.",
        }

    normalized_move = move_uci.strip().lower()

    try:
        move = chess.Move.from_uci(normalized_move)
    except ValueError:
        return {
            "valid": False,
            "error_type": ErrorType.PARSE_ERROR.value,
            "reason": "Move is not valid UCI notation.",
        }

    if move == chess.Move.null():
        return {
            "valid": False,
            "error_type": ErrorType.PARSE_ERROR.value,
            "reason": "Null move is not allowed.",
        }

    piece = board.piece_at(move.from_square)
    if piece is None:
        return {
            "valid": False,
            "error_type": ErrorType.INVALID_PIECE.value,
            "reason": f"No piece on source square {chess.square_name(move.from_square)}.",
        }

    if piece.color != board.turn:
        return {
            "valid": False,
            "error_type": ErrorType.INVALID_PIECE.value,
            "reason": (
                f"Piece on {chess.square_name(move.from_square)} belongs to the opponent."
            ),
        }

    promotion_error = _promotion_error_reason(board, move, piece)
    if promotion_error is not None:
        return {
            "valid": False,
            "error_type": ErrorType.PROMOTION_ERROR.value,
            "reason": promotion_error,
        }

    if _is_castling_attempt(piece, move) and not board.is_legal(move):
        return {
            "valid": False,
            "error_type": ErrorType.CASTLING_VIOLATION.value,
            "reason": "Castling attempt is illegal in the current position.",
        }

    if _is_invalid_en_passant_attempt(board, move, piece):
        return {
            "valid": False,
            "error_type": ErrorType.EN_PASSANT_VIOLATION.value,
            "reason": "En passant capture is not available for this move.",
        }

    if board.is_legal(move):
        return {
            "valid": True,
            "error_type": None,
            "reason": "Move is legal.",
        }

    if board.is_pseudo_legal(move):
        return {
            "valid": False,
            "error_type": ErrorType.LEAVES_IN_CHECK.value,
            "reason": "Move leaves the king in check.",
        }

    return {
        "valid": False,
        "error_type": ErrorType.ILLEGAL_DESTINATION.value,
        "reason": "Piece cannot move to the destination square.",
    }
