"""Shared utilities used by every agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import chess

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(template_name: str) -> str:
    """Read a prompt template from ``src/prompts/<template_name>``."""

    path = _PROMPTS_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_board_representation(
    fen: str,
    input_mode: Literal["fen", "history"],
    move_history: list[str] | None = None,
) -> str:
    """Build the board representation string injected into prompts.

    In ``fen`` mode the full FEN and an ASCII diagram are provided.
    In ``history`` mode only the move history is shown (FEN withheld).
    """

    if input_mode == "history":
        return "(Board state withheld — reason from the move history.)"

    board = chess.Board(fen)
    board_ascii = str(board)
    return (
        f"FEN: {fen}\n"
        f"Board:\n{board_ascii}"
    )


def format_feedback_block(feedback_history: list[str]) -> str:
    """Format accumulated feedback messages for re-injection into the prompt."""

    if not feedback_history:
        return ""

    lines = ["", "Previous attempts were invalid:"]
    for idx, feedback in enumerate(feedback_history, 1):
        lines.append(f"  Attempt {idx}: {feedback}")
    lines.append("")
    lines.append("Please try a different, legal move.")
    return "\n".join(lines)


def get_side_to_move(fen: str) -> str:
    """Return ``'white'`` or ``'black'`` based on the FEN."""

    board = chess.Board(fen)
    return "white" if board.turn == chess.WHITE else "black"
