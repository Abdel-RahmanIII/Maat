from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class GameStatus(str, Enum):
    """Exhaustive terminal reasons, kept as a string enum so values
    serialise cleanly into JSONL and YAML config without extra mapping."""

    ONGOING = "ongoing"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    DRAW_FIFTY_MOVES = "draw_fifty_moves"
    DRAW_THREEFOLD = "draw_threefold_repetition"
    DRAW_INSUFFICIENT = "draw_insufficient_material"
    DRAW_AGREEMENT = "draw_agreement"  # reserved — not triggered by python-chess


class StateSnapshot(BaseModel):
    """Immutable board snapshot consumed by LangGraph nodes and the logger.

    Every field the GraphState TypedDict (graph/state.py) will need is
    present here.
    """

    model_config = {"frozen": True}

    fen: str
    side_to_move: str          # "white" | "black"
    halfmove_clock: int
    fullmove_number: int
    is_terminal: bool
    game_status: GameStatus
    outcome: str | None        # "1-0" | "0-1" | "1/2-1/2" | None
    legal_move_count: int      # useful context for the generator prompt