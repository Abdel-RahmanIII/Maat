"""Chess tool implementations used by the ReAct condition (F).

Tools are split into two groups:
- Always available tools (all experiments / input modes).
- FEN-dependent tools (only when board state is available).
"""

from __future__ import annotations

import json
from typing import Any

import chess
from langchain_core.tools import tool

from src.validators.symbolic import validate_move as _symbolic_validate


def _board_from_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError as exc:
        raise ValueError("Invalid FEN provided.") from exc


def _parse_square(square: str) -> chess.Square:
    try:
        return chess.parse_square(square.strip().lower())
    except ValueError as exc:
        raise ValueError("Square must be in algebraic form such as e4.") from exc


def _parse_color(color: str) -> chess.Color:
    normalized = color.strip().lower()
    if normalized in {"white", "w"}:
        return chess.WHITE
    if normalized in {"black", "b"}:
        return chess.BLACK
    raise ValueError("Color must be one of: white, black, w, b.")


def _piece_code(piece: chess.Piece) -> str:
    prefix = "w" if piece.color == chess.WHITE else "b"
    return f"{prefix}{piece.symbol().upper()}"


def _serialize_piece_entry(board: chess.Board, square: chess.Square) -> dict[str, str] | None:
    piece = board.piece_at(square)
    if piece is None:
        return None

    return {
        "square": chess.square_name(square),
        "piece": _piece_code(piece),
        "color": "white" if piece.color == chess.WHITE else "black",
    }


def _rule_ref_from_error_type(error_type: str | None) -> str:
    if not error_type:
        return "LEGAL"
    return error_type


def _pgn_from_uci_history(move_history: list[str]) -> str:
    board = chess.Board()
    san_moves: list[str] = []

    for index, raw_move in enumerate(move_history, start=1):
        normalized_move = raw_move.strip().lower()
        try:
            move = chess.Move.from_uci(normalized_move)
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move at ply {index}: '{raw_move}'.") from exc

        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at ply {index}: '{raw_move}'.")

        san_moves.append(board.san(move))
        board.push(move)

    if not san_moves:
        return ""

    chunks: list[str] = []
    for i in range(0, len(san_moves), 2):
        move_number = (i // 2) + 1
        if i + 1 < len(san_moves):
            chunks.append(f"{move_number}. {san_moves[i]} {san_moves[i + 1]}")
        else:
            chunks.append(f"{move_number}. {san_moves[i]}")

    return " ".join(chunks)


# ── Tool definitions ─────────────────────────────────────────────────────


@tool
def validate_move(fen: str, move_uci: str) -> str:
    """Check if a move is legal.

    Returns JSON: {legal: bool, reason: str, rule_ref: str, error_type: str | null}.
    """

    result = _symbolic_validate(fen, move_uci)
    return json.dumps(
        {
            "legal": result["valid"],
            "reason": result["reason"],
            "rule_ref": _rule_ref_from_error_type(result["error_type"]),
            "error_type": result["error_type"],
        }
    )


@tool
def is_in_check(fen: str) -> str:
    """Return whether side-to-move is in check and attacker squares.

    Returns JSON: {in_check: bool, checking_squares: list[str]}.
    """

    board = _board_from_fen(fen)
    if not board.is_check():
        return json.dumps({"in_check": False, "checking_squares": []})

    checking_squares = sorted(chess.square_name(square) for square in board.checkers())
    return json.dumps({"in_check": True, "checking_squares": checking_squares})


@tool
def get_game_phase(move_history: list[str]) -> str:
    """Infer game phase from move-history ply count."""

    ply_count = len(move_history)
    if ply_count <= 20:
        return "opening"
    if ply_count <= 80:
        return "middlegame"
    return "endgame"


@tool
def get_move_history_pgn(move_history: list[str]) -> str:
    """Convert UCI move history to PGN move text from standard start position."""

    return _pgn_from_uci_history(move_history)


@tool
def get_board_visual(fen: str) -> str:
    """Return an ASCII 8x8 board representation for the given position."""

    board = _board_from_fen(fen)
    return str(board)


@tool
def get_piece_at(fen: str, square: str) -> str:
    """Return piece code (e.g., wN) or 'empty' for a square."""

    board = _board_from_fen(fen)
    target_square = _parse_square(square)
    piece = board.piece_at(target_square)
    if piece is None:
        return "empty"
    return _piece_code(piece)


@tool
def get_attackers(fen: str, square: str) -> str:
    """Return all pieces (either color) attacking a square."""

    board = _board_from_fen(fen)
    target_square = _parse_square(square)

    entries: list[dict[str, str]] = []
    for color in (chess.WHITE, chess.BLACK):
        for attacker_square in board.attackers(color, target_square):
            entry = _serialize_piece_entry(board, attacker_square)
            if entry is not None:
                entries.append(entry)

    entries.sort(key=lambda item: (item["color"], item["square"]))
    return json.dumps(entries)


@tool
def get_defenders(fen: str, square: str) -> str:
    """Return defenders of the occupant on a square."""

    board = _board_from_fen(fen)
    target_square = _parse_square(square)
    target_piece = board.piece_at(target_square)
    if target_piece is None:
        return json.dumps([])

    entries: list[dict[str, str]] = []
    for defender_square in board.attackers(target_piece.color, target_square):
        entry = _serialize_piece_entry(board, defender_square)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda item: item["square"])
    return json.dumps(entries)


@tool
def is_square_safe(fen: str, square: str, color: str) -> str:
    """Return whether a square is safe for the given color.

    Returns JSON: {safe: bool, threats: list[str]}.
    """

    board = _board_from_fen(fen)
    target_square = _parse_square(square)
    own_color = _parse_color(color)
    opponent = not own_color

    threats = sorted(
        chess.square_name(attacker_square)
        for attacker_square in board.attackers(opponent, target_square)
    )

    return json.dumps({"safe": len(threats) == 0, "threats": threats})


@tool
def get_position_after_moves(fen: str, moves: list[str]) -> str:
    """Apply a UCI sequence to a position and return resulting FEN."""

    board = _board_from_fen(fen)

    for index, raw_move in enumerate(moves, start=1):
        normalized_move = raw_move.strip().lower()
        try:
            move = chess.Move.from_uci(normalized_move)
        except ValueError as exc:
            raise ValueError(f"Invalid UCI move at step {index}: '{raw_move}'.") from exc

        if move not in board.legal_moves:
            raise ValueError(f"Illegal move at step {index}: '{raw_move}'.")

        board.push(move)

    return board.fen()


@tool
def submit_move(uci_move: str) -> str:
    """Submit your final move. Call this ONLY when you are confident in your choice.

    The move will be validated against the rules. If it is illegal the game is forfeited.
    """
    return f"SUBMIT:{uci_move}"


# ── Tool catalogs for experiment/input-mode gating ───────────────────────

ALWAYS_AVAILABLE_TOOLS = [
    validate_move,
    is_in_check,
    get_game_phase,
    get_move_history_pgn,
    submit_move,
]

FEN_ONLY_TOOLS = [
    get_board_visual,
    get_piece_at,
    get_attackers,
    get_defenders,
    is_square_safe,
    get_position_after_moves,
]

# Backward-compatible alias: full catalog (fen mode / experiments 1 and 2)
ALL_TOOLS = [*ALWAYS_AVAILABLE_TOOLS, *FEN_ONLY_TOOLS]


def get_tools_for_input_mode(input_mode: str) -> list[Any]:
    """Return tools available for the given input mode.

    - ``fen`` mode (Experiments 1 and 2): all tools.
    - ``history`` mode (Experiment 3): restricted safe set only.
    """

    normalized = input_mode.strip().lower()
    if normalized == "fen":
        return ALL_TOOLS
    if normalized == "history":
        return ALWAYS_AVAILABLE_TOOLS
    raise ValueError("input_mode must be either 'fen' or 'history'.")
