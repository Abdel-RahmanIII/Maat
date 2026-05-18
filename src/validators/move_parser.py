from __future__ import annotations

import re
from typing import TypedDict

import chess

from src.error_taxonomy import ErrorType
from src.tools.san_resolver import _resolve_san_to_uci

UCI_PATTERN = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")

# Sentinel pattern: "MOVE: e2e4" (case-insensitive, optional whitespace)
MOVE_SENTINEL = re.compile(r"move:\s*([a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE)

SAN_CORE = r"(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)"
SAN_PATTERN = re.compile(SAN_CORE, re.IGNORECASE)
SAN_SENTINEL = re.compile(rf"move:\s*({SAN_CORE})", re.IGNORECASE)


class ParseResult(TypedDict):
    is_valid: bool
    move_uci: str | None
    error_type: str | None
    reason: str | None
    used_fallback: bool


def _try_parse_uci(candidate: str) -> str | None:
    normalized = candidate.strip().lower()
    try:
        move = chess.Move.from_uci(normalized)
    except ValueError:
        return None

    if move == chess.Move.null():
        return None

    return move.uci()


def _normalize_san(candidate: str) -> str:
    token = candidate.strip()
    lowered = token.lower()
    if lowered == "0-0":
        return "O-O"
    if lowered == "0-0-0":
        return "O-O-O"
    if token and token[0] in "kqrbn":
        return token[0].upper() + token[1:]
    return token


def _try_parse_san(candidate: str, board: chess.Board) -> str | None:
    token = _normalize_san(candidate)
    try:
        move = board.parse_san(token)
    except chess.IllegalMoveError:
        return _resolve_san_to_uci(token, board)
    except ValueError:
        return None

    return move.uci()




def parse_uci_move(raw_output: str | None) -> ParseResult:
    """Parse an LLM output into a normalized UCI move string.

    Extraction priority:
    1. Strict — entire output is a bare UCI string (e.g. ``e2e4``).
    2. Sentinel — output contains ``MOVE: <uci>`` (last occurrence wins).
    3. Fallback — last UCI-shaped token anywhere in the text.
    """

    if raw_output is None or not raw_output.strip():
        return {
            "is_valid": False,
            "move_uci": None,
            "error_type": ErrorType.NO_OUTPUT.value,
            "reason": "Model output was empty.",
            "used_fallback": False,
        }

    cleaned = raw_output.strip().lower()

    # 1. Strict: entire output is a bare UCI move
    strict_move = _try_parse_uci(cleaned)
    if strict_move is not None:
        return {
            "is_valid": True,
            "move_uci": strict_move,
            "error_type": None,
            "reason": None,
            "used_fallback": False,
        }

    # 2. Sentinel: look for "MOVE: <uci>" — use the LAST occurrence
    sentinel_matches = list(MOVE_SENTINEL.finditer(cleaned))
    if sentinel_matches:
        sentinel_move = _try_parse_uci(sentinel_matches[-1].group(1))
        if sentinel_move is not None:
            return {
                "is_valid": True,
                "move_uci": sentinel_move,
                "error_type": None,
                "reason": None,
                "used_fallback": False,
            }

    # 3. Fallback: last UCI-shaped token anywhere in output
    fallback_matches = list(UCI_PATTERN.finditer(cleaned))
    if fallback_matches:
        fallback_move = _try_parse_uci(fallback_matches[-1].group(0))
        if fallback_move is not None:
            return {
                "is_valid": True,
                "move_uci": fallback_move,
                "error_type": None,
                "reason": None,
                "used_fallback": True,
            }

    return {
        "is_valid": False,
        "move_uci": None,
        "error_type": ErrorType.PARSE_ERROR.value,
        "reason": "Could not extract a valid UCI move from model output.",
        "used_fallback": False,
    }


def parse_san_move(raw_output: str | None, board: chess.Board) -> ParseResult:
    """Parse an LLM output into a normalized UCI move string via SAN.

    Extraction priority:
    1. Strict — entire output is a bare SAN string (e.g. ``Nf3``).
    2. Sentinel — output contains ``MOVE: <san>`` (last occurrence wins).
    3. Fallback — last SAN-shaped token anywhere in the text.
    """

    if raw_output is None or not raw_output.strip():
        return {
            "is_valid": False,
            "move_uci": None,
            "error_type": ErrorType.NO_OUTPUT.value,
            "reason": "Model output was empty.",
            "used_fallback": False,
        }

    cleaned = raw_output.strip()

    # 1. Strict: entire output is a bare SAN move
    strict_move = _try_parse_san(cleaned, board)
    if strict_move is not None:
        return {
            "is_valid": True,
            "move_uci": strict_move,
            "error_type": None,
            "reason": None,
            "used_fallback": False,
        }

    # 2. Sentinel: look for "MOVE: <san>" — use the LAST occurrence
    sentinel_matches = list(SAN_SENTINEL.finditer(cleaned))
    if sentinel_matches:
        sentinel_move = _try_parse_san(sentinel_matches[-1].group(1), board)
        if sentinel_move is not None:
            return {
                "is_valid": True,
                "move_uci": sentinel_move,
                "error_type": None,
                "reason": None,
                "used_fallback": False,
            }

    # 3. Fallback: last SAN-shaped token anywhere in output
    fallback_matches = list(SAN_PATTERN.finditer(cleaned))
    if fallback_matches:
        fallback_move = _try_parse_san(fallback_matches[-1].group(0), board)
        if fallback_move is not None:
            return {
                "is_valid": True,
                "move_uci": fallback_move,
                "error_type": None,
                "reason": None,
                "used_fallback": True,
            }

    return {
        "is_valid": False,
        "move_uci": None,
        "error_type": ErrorType.PARSE_ERROR.value,
        "reason": "Could not extract a valid SAN move from model output.",
        "used_fallback": False,
    }
