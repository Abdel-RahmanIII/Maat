from __future__ import annotations

import re
from typing import TypedDict

import chess

from src.error_taxonomy import ErrorType

UCI_PATTERN = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


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


def parse_uci_move(raw_output: str | None) -> ParseResult:
    """Parse an LLM output into a normalized UCI move string."""

    if raw_output is None or not raw_output.strip():
        return {
            "is_valid": False,
            "move_uci": None,
            "error_type": ErrorType.NO_OUTPUT.value,
            "reason": "Model output was empty.",
            "used_fallback": False,
        }

    cleaned = raw_output.strip().lower()

    strict_move = _try_parse_uci(cleaned)
    if strict_move is not None:
        return {
            "is_valid": True,
            "move_uci": strict_move,
            "error_type": None,
            "reason": None,
            "used_fallback": False,
        }

    fallback_match = UCI_PATTERN.search(cleaned)
    if fallback_match is not None:
        fallback_move = _try_parse_uci(fallback_match.group(0))
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
