"""Shared utilities used by every agent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import chess
import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=64)
def _load_prompt_yaml(agent_id: str) -> dict[str, object]:
    """Read and validate ``src/prompts/<agent_id>.yaml``."""

    path = _PROMPTS_DIR / f"{agent_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt YAML not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {path}")

    agent_name = raw.get("agent")
    if agent_name != agent_id:
        raise ValueError(
            f"Prompt YAML agent mismatch for {path}: expected '{agent_id}', got '{agent_name}'"
        )

    variants = raw.get("variants")
    if not isinstance(variants, dict):
        raise ValueError(f"Prompt YAML missing 'variants' mapping: {path}")

    for mode in ("fen", "history"):
        mode_payload = variants.get(mode)
        if not isinstance(mode_payload, dict):
            raise ValueError(f"Prompt YAML missing variants.{mode}: {path}")
        for role in ("system", "user"):
            text = mode_payload.get(role)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Prompt YAML missing variants.{mode}.{role}: {path}")

    return raw


def load_agent_prompt(
    agent_id: str,
    input_mode: Literal["fen", "history"],
    role: Literal["system", "user"],
) -> str:
    """Load an agent prompt text from YAML by mode and role."""

    raw = _load_prompt_yaml(agent_id)
    variants = cast(dict[str, Any], raw["variants"])
    mode_payload = cast(dict[str, Any], variants[input_mode])
    return str(mode_payload[role])


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
